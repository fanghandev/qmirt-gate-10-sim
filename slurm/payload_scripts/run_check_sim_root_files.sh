#!/usr/bin/env bash
# Wrapper to run the Python ROOT-file inspector under the cluster Python module.
# Designed to be called by sbatch-wrapper.sh which provides OUTPUT_DIR and SLURM env vars.

set -euo pipefail

# Simple arg parsing: first positional argument is target folder to scan.
# Remaining args are forwarded to the Python checker.

# Load project Python (uproot installed on cluster)
module load Python-bundle-PyPI

# Setup Python virtual environment
# Determine PROJECT_ROOT if not set by sbatch-wrapper
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  # Fallback: assume script is in PROJECT_ROOT/slurm/payload_scripts/
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

# Setup or activate venv
VENV_PATH="${PROJECT_ROOT}/.venv"
if [[ ! -d "$VENV_PATH" ]]; then
  echo "Creating virtual environment at $VENV_PATH" >&2
  python -m venv "$VENV_PATH"
fi

# Activate venv
source "${VENV_PATH}/bin/activate"

# Check and install required packages
for pkg in uproot numpy; do
  if ! python -c "import $pkg" 2>/dev/null; then
    echo "Installing $pkg..." >&2
    pip install "$pkg"
  fi
done

# Ensure OUTPUT_DIR is set.
# Default behavior:
# - array job:  ./output/<SLURM_ARRAY_JOB_ID>
# - single job: ./output/<SLURM_JOB_ID>
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  :
else
  RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
  OUTPUT_DIR="./output/${RUN_ID}"
fi
mkdir -p "${OUTPUT_DIR}"

# Parse folder and forward args.
# Primary source is first positional arg; fallback can be provided via CHECK_ROOT_TARGET_DIR.
TARGET_FOLDER="${1:-${CHECK_ROOT_TARGET_DIR:-}}"
if [[ $# -gt 0 ]]; then
  shift || true
fi
PY_ARGS=("$@")

if [[ -z "$TARGET_FOLDER" ]]; then
  echo "Usage: $0 <folder> [python-args...]" >&2
  echo "Error: missing target folder argument." >&2
  echo "Hint: pass folder after script path, e.g. sbatch-wrapper.sh ... -- run_check_sim_root_files.sh /path/to/data --recursive" >&2
  echo "Hint: or export CHECK_ROOT_TARGET_DIR via --export." >&2
  exit 2
fi

# Helper to find if forwarded args contain a flag
has_arg() {
  local name="$1"
  for a in "${PY_ARGS[@]}"; do
    if [[ "$a" == "$name" || "$a" == $name=* ]]; then
      return 0
    fi
  done
  return 1
}

# Extract pattern and recursive flags from forwarded args (optional)
PATTERN='*.root'
RECURSIVE=0
for ((i=0;i<${#PY_ARGS[@]};i++)); do
  a=${PY_ARGS[i]}
  case "$a" in
    --pattern)
      PATTERN="${PY_ARGS[i+1]:-}"
      ;;
    --pattern=*)
      PATTERN="${a#--pattern=}"
      ;;
    --recursive)
      RECURSIVE=1
      ;;
  esac
done

# Determine total tasks and task id from environment if not explicitly set
# User can export TOTAL_TASKS and TASK_ID via sbatch-wrapper --export
TOTAL_TASKS=${TOTAL_TASKS:-}
TASK_ID=${TASK_ID:-}

if [[ -z "$TOTAL_TASKS" ]]; then
  if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
    TOTAL_TASKS=${SLURM_ARRAY_TASK_COUNT}
  elif [[ -n "${SLURM_NTASKS:-}" ]]; then
    TOTAL_TASKS=${SLURM_NTASKS}
  else
    TOTAL_TASKS=1
  fi
fi

if [[ -z "$TASK_ID" ]]; then
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    TASK_ID=${SLURM_ARRAY_TASK_ID}
  elif [[ -n "${SLURM_PROCID:-}" ]]; then
    TASK_ID=${SLURM_PROCID}
  else
    TASK_ID=0
  fi
fi

# Ensure integers
TOTAL_TASKS=$((TOTAL_TASKS+0))
TASK_ID=$((TASK_ID+0))

# Find matching files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_DIR="$(cd "$TARGET_FOLDER" 2>/dev/null && pwd || echo "")"
if [[ -z "$SEARCH_DIR" ]]; then
  echo "Error: target folder not found: $TARGET_FOLDER" >&2
  exit 2
fi

if [[ $RECURSIVE -eq 1 ]]; then
  mapfile -t ALL_FILES < <(find "$SEARCH_DIR" -type f -name "$PATTERN" -print | sort)
else
  mapfile -t ALL_FILES < <(find "$SEARCH_DIR" -maxdepth 1 -type f -name "$PATTERN" -print | sort)
fi

NUM_FILES=${#ALL_FILES[@]}
if [[ $NUM_FILES -eq 0 ]]; then
  echo "No files found in $SEARCH_DIR matching pattern $PATTERN" >&2
  exit 0
fi

# Compute task index for round-robin distribution
TASK_INDEX=$(( TASK_ID % TOTAL_TASKS ))

SUBSET=()
for i in "${!ALL_FILES[@]}"; do
  if (( i % TOTAL_TASKS == TASK_INDEX )); then
    SUBSET+=("${ALL_FILES[i]}")
  fi
done

if [[ ${#SUBSET[@]} -eq 0 ]]; then
  echo "Task ${TASK_ID} has no files to process (TOTAL_TASKS=${TOTAL_TASKS})" >&2
  exit 0
fi

# Write subset to temp file and call python with --file-list
TMP_LIST=$(mktemp -t check_sim_root_files.XXXXXX)
trap 'rm -f "$TMP_LIST"' EXIT
for f in "${SUBSET[@]}"; do
  printf "%s\n" "$f" >>"$TMP_LIST"
done

# Determine log file names using SLURM ids when available.
# For array jobs, prefer array job id over per-task job id.
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$$}}"
TASK_SUFFIX=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TASK_SUFFIX="_${SLURM_ARRAY_TASK_ID}"
fi
LOG_BASE="${OUTPUT_DIR}/check_sim_root_files_${JOB_ID}${TASK_SUFFIX}"
STDOUT_LOG="${LOG_BASE}.out"
STDERR_LOG="${LOG_BASE}.err"

PY_SCRIPT="${SCRIPT_DIR}/python/check_sim_root_files.py"
if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "Error: python script not found: $PY_SCRIPT" >&2
  exit 2
fi

# If user didn't request an explicit --output, create a per-task output path and pass it
if ! has_arg --output; then
  DEFAULT_OUT="${OUTPUT_DIR}/check_sim_root_files_results_${JOB_ID}${TASK_SUFFIX}.csv"
  PY_ARGS+=("--output" "$DEFAULT_OUT")
fi

# If user didn't specify format, default to csv (ensure arg present)
if ! has_arg --format; then
  PY_ARGS+=("--format" "csv")
fi

printf "Start:     %s\n" "$(date)" > "$STDOUT_LOG"
printf "Action:    Starting check_sim_root_files\n" >> "$STDOUT_LOG"
printf "Script:    %s\n" "$PY_SCRIPT" >> "$STDOUT_LOG"

printf "Progress:  Task %3d/%3d | Processing %3d files\n" \
    "$((TASK_ID + 1))" \
    "$TOTAL_TASKS" \
    "${#SUBSET[@]}" >> "$STDOUT_LOG"

# Run the checker on this subset, forwarding any user args
python "$PY_SCRIPT" --file-list "$TMP_LIST" "${PY_ARGS[@]}" >>"$STDOUT_LOG" 2>>"$STDERR_LOG"
EXIT_CODE=$?
  
printf "Finish:    %s" "$(date)" >> "$STDOUT_LOG"
printf "Exit code: %d\n" "$EXIT_CODE" >>"$STDOUT_LOG"
exit "$EXIT_CODE"
