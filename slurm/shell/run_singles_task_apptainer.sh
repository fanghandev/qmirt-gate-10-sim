#!/usr/bin/env bash
set -euo pipefail

# Required environment variables from parent SLURM script
: "${GATE_SIM_PREFIX:?Error: GATE_SIM_PREFIX must be set by parent SLURM script}"
: "${PYTHON_SCRIPT_DIR:?Error: PYTHON_SCRIPT_DIR must be set by parent SLURM script}"
: "${JOB_OUTPUT_ROOT:?Error: JOB_OUTPUT_ROOT must be set by parent SLURM script}"
: "${SOURCE_ACTIVITY_BQ:?Error: SOURCE_ACTIVITY_BQ must be set by parent SLURM script}"
: "${CHUNK_DURATION_S:?Error: CHUNK_DURATION_S must be set by parent SLURM script}"
: "${NUM_CHUNKS:?Error: NUM_CHUNKS must be set by parent SLURM script}"
: "${CONTAINER_URI:?Error: CONTAINER_URI must be set by parent SLURM script}"
: "${CONTAINER_SIF:?Error: CONTAINER_SIF must be set by parent SLURM script}"

JOB_ID="${SLURM_JOB_ID:-nojid}"
TASK_ID="${SLURM_PROCID:-${SLURM_ARRAY_TASK_ID:-0}}"
RUN_GROUP_ID="${RUN_GROUP_ID:-$TASK_ID}"
MAX_TASK_SECONDS="${MAX_TASK_SECONDS:-0}"
TASK_OUTPUT_ROOT="${JOB_OUTPUT_ROOT}/task_${TASK_ID}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id)
      JOB_ID="$2"
      shift 2
      ;;
    --task-id)
      TASK_ID="$2"
      shift 2
      ;;
    --run-group-id)
      RUN_GROUP_ID="$2"
      shift 2
      ;;
    *)
      echo "Error: unknown argument '$1'" >&2
      exit 2
      ;;
  esac
done

RUN_GROUP_ID="${RUN_GROUP_ID:-$TASK_ID}"

# Setup output directory
OUTDIR="$TASK_OUTPUT_ROOT"
mkdir -p "$OUTDIR"

TASK_START_TS="$(date +%s)"

# Determine container execution command
if [[ -f "$CONTAINER_SIF" ]]; then
  apptainer_exec=(apptainer exec "$CONTAINER_SIF")
else
  apptainer_exec=(apptainer exec "$CONTAINER_URI")
fi

echo "Job ID $JOB_ID, task ID $RUN_GROUP_ID"
echo "  Output: $OUTDIR"
echo "  Source Activity: $SOURCE_ACTIVITY_BQ Bq"
echo "  Chunks: $NUM_CHUNKS"
echo "  SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-none}"
echo "  Max Task Seconds: $MAX_TASK_SECONDS"

# Run simulation in container
cd "$GATE_SIM_PREFIX"
sim_cmd=(
  "${apptainer_exec[@]}"
  python3 "$PYTHON_SCRIPT_DIR/dc_spect_run_sim_singles_batch_slurm.py"
  --job-id "$JOB_ID"
  --task-id "$RUN_GROUP_ID"
  --source-activity-bq "$SOURCE_ACTIVITY_BQ"
  --chunk-duration-s "$CHUNK_DURATION_S"
  --num-chunks "$NUM_CHUNKS"
  --run-group-id "$RUN_GROUP_ID"
  --num-threads "${NUM_THREADS:-1}"
  --output-dir "$OUTDIR"
)

if [[ "$MAX_TASK_SECONDS" =~ ^[0-9]+$ ]] && [[ "$MAX_TASK_SECONDS" -gt 0 ]]; then
  timeout --signal=TERM --kill-after=30 "$MAX_TASK_SECONDS" "${sim_cmd[@]}"
else
  "${sim_cmd[@]}"
fi

TASK_END_TS="$(date +%s)"
TASK_WALL_TIME_S="$((TASK_END_TS - TASK_START_TS))"

cat > "$OUTDIR/run_group_${RUN_GROUP_ID}_wall_time.txt" <<EOF
task_id: $TASK_ID
run_group_id: $RUN_GROUP_ID
start_epoch_s: $TASK_START_TS
end_epoch_s: $TASK_END_TS
wall_time_seconds: $TASK_WALL_TIME_S
EOF

echo "Task $TASK_ID wall time: ${TASK_WALL_TIME_S}s"

echo "Task $TASK_ID: completed"
