#!/bin/bash
#
# run_gate_sim_dc_spect_slurm.sh
# ==============================
# Wrapper script that prepares per-job output directories and calls the
# DC-SPECT simulation Python payload inside a container.
#
# Usage:
#   ./run_gate_sim_dc_spect_slurm.sh
#
# Or via sbatch-wrapper:
#   sbatch-wrapper.sh -N "dc-spect" -p debug -a "0-10" -n 1 -o /scratch/f/fh890/test-ids-array -- ./run_gate_sim_dc_spect_slurm.sh
#
# Environment variables (set by SLURM or passed via sbatch-wrapper -e):
#   GATE_SIM_PREFIX: Path to the gate10mc repository root (default: auto-detected)
#   CONTAINER_URI: Container URI (default: docker://gitlab.partners.org:5050/rpil/qmirt/simulation/gate10mc:latest)
#   CONTAINER_SIF: Path to the apptainer .sif file (preferred, default: $GATE_SIM_PREFIX/gate10mc.sif)
#   OUTPUT_DIR: Base output directory (set by sbatch-wrapper)
#   SOURCE_ACTIVITY_BQ: Source activity in Bq (default: 1e6)
#   CHUNK_DURATION_S: Chunk duration in seconds (default: 1.0)
#   NUM_CHUNKS: Number of chunks (default: 1)
#   MAX_TASK_SECONDS: Task timeout in seconds (default: 0 = no timeout)

ml Apptainer

set -euo pipefail

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE_SIM_PREFIX="${GATE_SIM_PREFIX:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_SCRIPT_REL="python/gate_sim_dc_spect_slurm.py"

# Container configuration
export CONTAINER_URI="${CONTAINER_URI:-docker://gitlab.partners.org:5050/rpil/qmirt/simulation/gate10mc:latest}"
export CONTAINER_SIF="${CONTAINER_SIF:-${GATE_SIM_PREFIX}/gate10mc.sif}"

# Simulation configuration
export SOURCE_ACTIVITY_BQ="${SOURCE_ACTIVITY_BQ:-3.7e5}"
export CHUNK_DURATION_S="${CHUNK_DURATION_S:-1.0}"
export NUM_CHUNKS="${NUM_CHUNKS:-1}"
export NUM_THREADS="${NUM_THREADS:-1}"
export MAX_TASK_SECONDS="${MAX_TASK_SECONDS:-0}"

# SLURM context
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
JOB_ARRAY_ID="$RUN_ID"
JOB_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-${SLURM_PROCID:-0}}"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-array-id)
      JOB_ARRAY_ID="$2"
      shift 2
      ;;
    --job-array-task-id)
      JOB_ARRAY_TASK_ID="$2"
      shift 2
      ;;
    --source-activity-bq)
      SOURCE_ACTIVITY_BQ="$2"
      shift 2
      ;;
    --chunk-duration-s)
      CHUNK_DURATION_S="$2"
      shift 2
      ;;
    --num-chunks)
      NUM_CHUNKS="$2"
      shift 2
      ;;
    --max-task-seconds)
      MAX_TASK_SECONDS="$2"
      shift 2
      ;;
    *)
      echo "Warning: unknown argument '$1'" >&2
      shift
      ;;
  esac
done

# Prepare output directory.
# Default behavior:
# - array job:  ./output/<SLURM_ARRAY_JOB_ID>
# - single job: ./output/<SLURM_JOB_ID>
# If OUTPUT_DIR is already provided, use it as-is.
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  RUN_OUTPUT_DIR="${OUTPUT_DIR}"
else
  RUN_OUTPUT_DIR="./output/${RUN_ID}"
fi
mkdir -p "${RUN_OUTPUT_DIR}"

# Record task timing
TASK_START_TS="$(date +%s)"

# Determine container execution command
if [[ -f "$CONTAINER_SIF" ]]; then
  apptainer_exec=(apptainer exec --bind /scratch "$CONTAINER_SIF")
else
  apptainer_exec=(apptainer exec --bind /scratch "$CONTAINER_URI")
fi

echo "DC-SPECT Simulation Task"
echo "  Job Array ID: $JOB_ARRAY_ID"
echo "  Task ID: $JOB_ARRAY_TASK_ID"
echo "  Output Dir: $RUN_OUTPUT_DIR"
echo "  Source Activity: $SOURCE_ACTIVITY_BQ Bq"
echo "  Chunk Duration: $CHUNK_DURATION_S s"
echo "  Num Chunks: $NUM_CHUNKS"
echo "  Container: $CONTAINER_SIF"
echo ""

# Change to the repository root so relative paths work inside the container
cd "$GATE_SIM_PREFIX"

# Build the simulation command
sim_cmd=(
  "${apptainer_exec[@]}"
  python3 "${SCRIPT_DIR}/${PYTHON_SCRIPT_REL}"
  --output-dir "${RUN_OUTPUT_DIR}"
  --job-array-id "${JOB_ARRAY_ID}"
  --job-array-task-id "${JOB_ARRAY_TASK_ID}"
  --source-activity-bq "${SOURCE_ACTIVITY_BQ}"
  --chunk-duration-s "${CHUNK_DURATION_S}"
  --num-chunks "${NUM_CHUNKS}"
)

# Run simulation with optional timeout
if [[ "$MAX_TASK_SECONDS" =~ ^[0-9]+$ ]] && [[ "$MAX_TASK_SECONDS" -gt 0 ]]; then
  timeout --signal=TERM --kill-after=30 "$MAX_TASK_SECONDS" "${sim_cmd[@]}" || sim_exit=$?
else
  "${sim_cmd[@]}" || sim_exit=$?
fi

# Record timing
TASK_END_TS="$(date +%s)"
TASK_WALL_TIME_S="$((TASK_END_TS - TASK_START_TS))"

# Write wall time report
cat > "$RUN_OUTPUT_DIR/task_${JOB_ARRAY_TASK_ID}_wall_time.txt" <<EOF
job_array_id: $JOB_ARRAY_ID
job_array_task_id: $JOB_ARRAY_TASK_ID
start_epoch_s: $TASK_START_TS
end_epoch_s: $TASK_END_TS
wall_time_seconds: $TASK_WALL_TIME_S
source_activity_bq: $SOURCE_ACTIVITY_BQ
chunk_duration_s: $CHUNK_DURATION_S
num_chunks: $NUM_CHUNKS
EOF

echo "Task $JOB_ARRAY_TASK_ID wall time: ${TASK_WALL_TIME_S}s"
echo "Task $JOB_ARRAY_TASK_ID: completed"

exit "${sim_exit:-0}"
