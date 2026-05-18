#!/usr/bin/env bash
#
# sbatch-wrapper.sh
# ==================
# SLURM job submission wrapper that generates sbatch directives and runs a shell script payload.
#
# The wrapper handles all SLURM configuration and passes OUTPUT_DIR to the payload script.
# All output from the script should be directed to $OUTPUT_DIR.
#
# Usage:
#   sbatch-wrapper.sh [OPTIONS] -- <script> [script_args...]
#
# Examples:
#   # Basic job submission
#   sbatch-wrapper.sh -- ./run_singles_task_no_container.sh
#
#   # With job name, partition, time, and output directory
#   sbatch-wrapper.sh -N "dc-spect-run" -p cpu -t 02:00:00 -o /scratch/jobs -- ./run_singles_task.sh
#
#   # Array job with task parameters
#   sbatch-wrapper.sh -N "batch-100" -a "0-99%20" -o /scratch/results -- ./run_singles_task.sh
#
#   # With dependency and environment variables
#   sbatch-wrapper.sh -d "afterok:12345" -e "ALL,SOURCE_ACTIVITY=2e6" -o /tmp/logs -- ./merge_stats.sh
#
#   # Dry-run to see generated sbatch file
#   sbatch-wrapper.sh --dry-run -N "test" -- ./run_task.sh
#
# Options:
#   -N, --job-name NAME       Set job name
#   -p, --partition PART      Set partition (cpu, gpu, normal, interactive, etc.)
#   -t, --time TIME           Set wall-clock time (HH:MM:SS format)
#   -n, --ntasks N            Set number of tasks (default: 1)
#   -c, --cpus-per-task N     Set CPUs per task (default: 1)
#   -a, --array RANGE         Set array job range (e.g., 0-99%20)
#   -d, --dependency DEPEND   Set job dependency (e.g., afterok:12345)
#   -e, --export VARS         Comma-separated list of env vars to export (e.g., ALL,FOO=bar)
#   -o, --output-dir DIR      Set base output directory (runtime OUTPUT_DIR will be DIR/JOB_NAME/<array_job_id|job_id>; default base: /scratch/<u>/<user>)
#   --keep-sbatch             Keep generated sbatch file after submission (default: delete)
#   --dry-run                 Print generated sbatch file without submitting
#   -h, --help                Show this help message

set -euo pipefail

module load git

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Script directory
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Default values
JOB_NAME=""
PARTITION=""
TIME_LIMIT=""
NTASKS=""
CPUS_PER_TASK=""
ARRAY_RANGE=""
DEPENDENCY=""
EXPORT_VARS=""
# Default OUTPUT_BASE_DIR: /scratch/{first_letter_of_username}/{username}
readonly USERNAME=$(whoami)
readonly FIRST_LETTER="${USERNAME:0:1}"
OUTPUT_BASE_DIR="/scratch/${FIRST_LETTER}/${USERNAME}"
SLURM_LOG_SUBDIR="slurm_logs"
LOCAL_TMP_DIR="${SCRIPT_DIR}/.tmp"
KEEP_SBATCH=1
DRY_RUN=0
SCRIPT_PATH=""
SCRIPT_ARGS=()
PROJECT_ROOT=""

# ============================================================================
# Helper Functions
# ============================================================================

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# //'
}

error() {
  echo -e "${RED}Error: $*${NC}" >&2
  exit 1
}

warning() {
  echo -e "${YELLOW}Warning: $*${NC}" >&2
}

info() {
  echo -e "${BLUE}Info: $*${NC}"
}

success() {
  echo -e "${GREEN}Success: $*${NC}"
}

# ============================================================================
# Argument Parsing
# ============================================================================

parse_args() {
  local args_mode="options"

  while [[ $# -gt 0 ]]; do
    if [[ "$args_mode" == "options" && "$1" == "--" ]]; then
      args_mode="script"
      shift
      continue
    fi

    case "$args_mode" in
      options)
        case "$1" in
          -N | --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
          -p | --partition)
            PARTITION="$2"
            shift 2
            ;;
          -t | --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
          -n | --ntasks)
            NTASKS="$2"
            shift 2
            ;;
          -c | --cpus-per-task)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
          -a | --array)
            ARRAY_RANGE="$2"
            shift 2
            ;;
          -d | --dependency)
            DEPENDENCY="$2"
            shift 2
            ;;
          -e | --export)
            EXPORT_VARS="$2"
            shift 2
            ;;
          -o | --output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
          --keep-sbatch)
            KEEP_SBATCH=1
            shift
            ;;
          --dry-run)
            DRY_RUN=1
            shift
            ;;
          -h | --help)
            usage
            exit 0
            ;;
          *)
            error "Unknown option: $1"
            ;;
        esac
        ;;
      script)
        if [[ -z "$SCRIPT_PATH" ]]; then
          SCRIPT_PATH="$1"
        else
          SCRIPT_ARGS+=("$1")
        fi
        shift
        ;;
    esac
  done

  # Validation
  if [[ -z "$SCRIPT_PATH" ]]; then
    error "No script specified. Use: sbatch-wrapper.sh -- <script>"
  fi

  # Resolve relative paths to absolute
  if [[ "$SCRIPT_PATH" != /* ]]; then
    SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_PATH"
  fi

  if [[ ! -f "$SCRIPT_PATH" ]]; then
    error "Script not found: $SCRIPT_PATH"
  fi

  # Calculate PROJECT_ROOT using git rev-parse
  local script_parent
  script_parent="$(dirname "$SCRIPT_PATH")"
  PROJECT_ROOT=$(cd "$script_parent" && git rev-parse --show-toplevel 2>/dev/null || echo "")
  
  if [[ -z "$PROJECT_ROOT" ]]; then
    warning "Could not determine git root from $script_parent, using script parent directory"
    PROJECT_ROOT="$script_parent"
  fi

  # Resolve OUTPUT_BASE_DIR to absolute path
  if [[ "$OUTPUT_BASE_DIR" != /* ]]; then
    OUTPUT_BASE_DIR="$(cd "$OUTPUT_BASE_DIR" 2>/dev/null && pwd)" || OUTPUT_BASE_DIR="$(cd "$(dirname "$OUTPUT_BASE_DIR")" && pwd)/$(basename "$OUTPUT_BASE_DIR")"
  fi

  # Compute base output directory for this submission as OUTPUT_BASE_DIR/JOB_NAME.
  # At runtime the payload receives OUTPUT_DIR with run id suffix:
  #   <base>/<SLURM_ARRAY_JOB_ID> for array jobs
  #   <base>/<SLURM_JOB_ID> for non-array jobs
  # If no job name is set, use a default.
  local final_job_name="${JOB_NAME:-job}"
  # Avoid doubling the job name if the provided base already ends with it
  local base_leaf="${OUTPUT_BASE_DIR##*/}"
  if [[ "$base_leaf" == "$final_job_name" ]]; then
    readonly OUTPUT_DIR="${OUTPUT_BASE_DIR}"
  else
    readonly OUTPUT_DIR="${OUTPUT_BASE_DIR}/${final_job_name}"
  fi

  if [[ ! -x "$SCRIPT_PATH" ]]; then
    warning "Script is not executable: $SCRIPT_PATH. It may still run via bash/sh."
  fi
}

# ============================================================================
# Generate SBATCH File
# ============================================================================

generate_sbatch_file() {
  local tmp_sbatch
  mkdir -p "$LOCAL_TMP_DIR"
  tmp_sbatch=$(mktemp "$LOCAL_TMP_DIR/sbatch-wrapper.XXXXXX.sh")

  cat > "$tmp_sbatch" <<'SBATCH_HEADER'
#!/usr/bin/env bash

# Auto-generated by sbatch-wrapper.sh
# Do not edit - this file will be deleted after submission

SBATCH_HEADER

  # Add SBATCH directives without emitting empty lines for unset options.
  {
    echo "#SBATCH --job-name=${JOB_NAME:-job}"
    [[ -n "$PARTITION" ]] && echo "#SBATCH --partition=$PARTITION"
    [[ -n "$TIME_LIMIT" ]] && echo "#SBATCH --time=$TIME_LIMIT"
    [[ -n "$NTASKS" ]] && echo "#SBATCH --ntasks=$NTASKS"
    [[ -n "$CPUS_PER_TASK" ]] && echo "#SBATCH --cpus-per-task=$CPUS_PER_TASK"
    [[ -n "$ARRAY_RANGE" ]] && echo "#SBATCH --array=$ARRAY_RANGE"
    [[ -n "$DEPENDENCY" ]] && echo "#SBATCH --dependency=$DEPENDENCY"
  } >> "$tmp_sbatch"

  # Add export directive if provided
  if [[ -n "$EXPORT_VARS" ]]; then
    echo "#SBATCH --export=$EXPORT_VARS" >> "$tmp_sbatch"
  fi

  cat >> "$tmp_sbatch" <<'SBATCH_BODY'

SBATCH_BODY

  # Add output/error directives for sbatch stdout/stderr
  cat >> "$tmp_sbatch" <<SBATCH_LOGGING
#SBATCH --output=${OUTPUT_DIR}/${SLURM_LOG_SUBDIR}/%A_%a.out
#SBATCH --error=${OUTPUT_DIR}/${SLURM_LOG_SUBDIR}/%A_%a.err

SBATCH_LOGGING

  # Emit CAMPAIGN_ID (use SLURM_ARRAY_JOB_ID for array jobs, otherwise SLURM_JOB_ID)
  cat >> "$tmp_sbatch" <<'SBATCH_CAMPAIGN'
# Campaign id follows the SLURM job identity:
# - array jobs: SLURM_ARRAY_JOB_ID
# - single jobs: SLURM_JOB_ID

CAMPAIGN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
export CAMPAIGN_ID

SBATCH_CAMPAIGN

  # Add payload: run script with explicit output redirection and OUTPUT_DIR env var
  if [[ ${#SCRIPT_ARGS[@]} -gt 0 ]]; then
    local srun_command
    printf -v srun_command 'srun bash %q' "$SCRIPT_PATH"
    for arg in "${SCRIPT_ARGS[@]}"; do
      printf -v srun_command '%s %q' "$srun_command" "$arg"
    done

    cat >> "$tmp_sbatch" <<SBATCH_PAYLOAD

# Use CAMPAIGN_ID to build the runtime output dir

export OUTPUT_DIR="${OUTPUT_DIR}/\${CAMPAIGN_ID}"
export PROJECT_ROOT="${PROJECT_ROOT}"
$srun_command
SBATCH_PAYLOAD
  else
    cat >> "$tmp_sbatch" <<SBATCH_PAYLOAD

# Use CAMPAIGN_ID to build the runtime output dir

export OUTPUT_DIR="${OUTPUT_DIR}/\${CAMPAIGN_ID}"
export PROJECT_ROOT="${PROJECT_ROOT}"

# Run payload script with OUTPUT_DIR and PROJECT_ROOT environment variables

srun bash "$SCRIPT_PATH"
SBATCH_PAYLOAD
  fi

  echo "$tmp_sbatch"
}

# ============================================================================
# Print Configuration
# ============================================================================

print_config() {
  info "Job Configuration:"
  [[ -n "$JOB_NAME" ]] && echo "  Job Name:        $JOB_NAME" || echo "  Job Name:        (from script)"
  [[ -n "$PARTITION" ]] && echo "  Partition:       $PARTITION" || echo "  Partition:       (default)"
  [[ -n "$TIME_LIMIT" ]] && echo "  Time Limit:      $TIME_LIMIT" || echo "  Time Limit:      (default)"
  [[ -n "$NTASKS" ]] && echo "  Number of Tasks: $NTASKS" || echo "  Number of Tasks: 1"
  [[ -n "$CPUS_PER_TASK" ]] && echo "  CPUs per Task:   $CPUS_PER_TASK" || echo "  CPUs per Task:   1"
  [[ -n "$ARRAY_RANGE" ]] && echo "  Array Range:     $ARRAY_RANGE" || echo "  Array Range:     (none)"
  [[ -n "$DEPENDENCY" ]] && echo "  Dependency:      $DEPENDENCY" || echo "  Dependency:      (none)"
  [[ -n "$EXPORT_VARS" ]] && echo "  Export Vars:     $EXPORT_VARS" || echo "  Export Vars:     (none)"
  echo "  Base Output Dir: $OUTPUT_BASE_DIR"
  echo "  Submission Output Dir: $OUTPUT_DIR"
  echo "  Runtime Output Dir: ${OUTPUT_DIR}/<SLURM_ARRAY_JOB_ID|SLURM_JOB_ID|local>"
  echo "  Slurm Log Dir:   ${OUTPUT_DIR}/${SLURM_LOG_SUBDIR}"
  echo "  Project Root:    $PROJECT_ROOT"
  echo "  Script:          $SCRIPT_PATH"
  [[ ${#SCRIPT_ARGS[@]} -gt 0 ]] && echo "  Script Args:     ${SCRIPT_ARGS[*]}"
  echo ""
}

# ============================================================================
# Submit Job
# ============================================================================

submit_job() {
  local sbatch_file
  mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/$SLURM_LOG_SUBDIR"
  sbatch_file=$(generate_sbatch_file)

  info "Generated sbatch file: $sbatch_file"
  echo ""

  if [[ "$DRY_RUN" -eq 1 ]]; then
    info "Dry-run mode: showing generated sbatch file"
    echo ""
    cat "$sbatch_file"
    echo ""
    info "Job NOT submitted (dry-run mode)"
    rm -f "$sbatch_file"
    return 0
  fi

  info "Submitting job..."
  local job_output
  if job_output=$(sbatch "$sbatch_file" 2>&1); then
    success "Job submitted successfully!"
    echo "$job_output"

    # Extract job ID if possible
    if [[ "$job_output" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
      local job_id="${BASH_REMATCH[1]}"
      success "Job ID: $job_id"
      echo ""
      info "View job status: squeue -j $job_id"
      info "Cancel job: scancel $job_id"
      info "View logs: tail -f ${OUTPUT_DIR}/${SLURM_LOG_SUBDIR}/*_${job_id}_*.{out,err}"
      info "Output directory (runtime): ${OUTPUT_DIR}/<SLURM_ARRAY_JOB_ID|SLURM_JOB_ID|local>"
    fi

    # Clean up temp sbatch file unless --keep-sbatch specified
    if [[ "$KEEP_SBATCH" -eq 0 ]]; then
      rm -f "$sbatch_file"
    else
      info "Keeping sbatch file: $sbatch_file"
    fi
  else
    error "Failed to submit job:\n$job_output"
  fi
}

# ============================================================================
# Main
# ============================================================================

main() {
  parse_args "$@"
  print_config
  submit_job
}

main "$@"
