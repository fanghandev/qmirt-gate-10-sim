#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if command -v module >/dev/null 2>&1; then
    module load Apptainer 2>/dev/null || true
fi

CONTAINER_SIF="${CONTAINER_SIF:-${REPO_ROOT}/submit_slurm/qmirt-gate-10-sim-sif_v1.0.0.sif}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${SLURM_PROCID:-0}}"
OUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/cardiac_spect/slurm/${JOB_ID}}"

export REPO_ROOT
export SIM_TYPE="cardiac"
export SIM_PYTHON_SCRIPT="payload/python/gate_sim_cardiac_spect_boolean.py"
export PYTHONPATH="$REPO_ROOT/qmirt/src${PYTHONPATH:+:$PYTHONPATH}"
export SOURCE_ACTIVITY_BQ="${SOURCE_ACTIVITY_BQ:-3.7e5}"
export CHUNK_DURATION_S="${CHUNK_DURATION_S:-1.0}"
export NUM_CHUNKS="${NUM_CHUNKS:-1}"
export MAX_TASK_SECONDS="${MAX_TASK_SECONDS:-0}"
export PROFILE_RESOURCES="${PROFILE_RESOURCES:-1}"
export PROFILE_INTERVAL_S="${PROFILE_INTERVAL_S:-5}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile-resources) PROFILE_RESOURCES=1; shift ;;
        --no-profile-resources) PROFILE_RESOURCES=0; shift ;;
        --profile-interval-s)
            [[ $# -ge 2 ]] || { echo "Missing value for --profile-interval-s" >&2; exit 2; }
            PROFILE_INTERVAL_S="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--profile-resources|--no-profile-resources] [--profile-interval-s SECONDS]"
            exit 0
            ;;
        *)
            echo "Unexpected argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "$CONTAINER_SIF" ]]; then
    echo "Error: Apptainer image not found at $CONTAINER_SIF"
    echo "Expected local SIF under submit_slurm/ from the README instructions."
    exit 1
fi

mkdir -p "$OUT_DIR"

TASK_START_TS="$(date +%s)"

if command -v apptainer >/dev/null 2>&1; then
    APPTAINER_CMD=(
        apptainer exec
        --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}"
        --bind "$REPO_ROOT:$REPO_ROOT"
        "$CONTAINER_SIF"
    )
else
    APPTAINER_CMD=()
fi

if [[ ${#APPTAINER_CMD[@]} -gt 0 ]]; then
    sim_cmd=(
        "${APPTAINER_CMD[@]}"
        python3
        "$REPO_ROOT/payload/python/gate_sim_cardiac_spect_boolean.py"
        -o "$OUT_DIR"
        -j "$JOB_ID"
        -k "$TASK_ID"
        --execution-environment slurm
        -n "${SLURM_CPUS_PER_TASK:-1}"
        -s "$SOURCE_ACTIVITY_BQ"
        -d "$CHUNK_DURATION_S"
        -c "$NUM_CHUNKS"
    )
else
    sim_cmd=(
        python3
        "$REPO_ROOT/payload/python/gate_sim_cardiac_spect_boolean.py"
        -o "$OUT_DIR"
        -j "$JOB_ID"
        -k "$TASK_ID"
        --execution-environment slurm
        -n "${SLURM_CPUS_PER_TASK:-1}"
        -s "$SOURCE_ACTIVITY_BQ"
        -d "$CHUNK_DURATION_S"
        -c "$NUM_CHUNKS"
    )
fi

echo "Starting SLURM job $JOB_ID task $TASK_ID..."
echo "Output dir: $OUT_DIR"
echo "Source activity: ${SOURCE_ACTIVITY_BQ} Bq"
echo "Chunk duration: ${CHUNK_DURATION_S} s"
echo "Num chunks: ${NUM_CHUNKS}"

if [[ "$PROFILE_RESOURCES" == "1" ]]; then
    profile_cmd=(bash "$SCRIPT_DIR/profile_resources.sh" "$OUT_DIR" "$PROFILE_INTERVAL_S" "${sim_cmd[@]}")
else
    profile_cmd=("${sim_cmd[@]}")
fi

if [[ "$MAX_TASK_SECONDS" =~ ^[0-9]+$ ]] && [[ "$MAX_TASK_SECONDS" -gt 0 ]]; then
    timeout --signal=TERM --kill-after=30 "$MAX_TASK_SECONDS" "${profile_cmd[@]}" || sim_exit=$?
else
    "${profile_cmd[@]}" || sim_exit=$?
fi

TASK_END_TS="$(date +%s)"
TASK_WALL_TIME_S="$((TASK_END_TS - TASK_START_TS))"

cat > "$OUT_DIR/task_${TASK_ID}_wall_time.txt" <<EOF
job_array_id: $JOB_ID
job_array_task_id: $TASK_ID
start_epoch_s: $TASK_START_TS
end_epoch_s: $TASK_END_TS
wall_time_seconds: $TASK_WALL_TIME_S
source_activity_bq: $SOURCE_ACTIVITY_BQ
chunk_duration_s: $CHUNK_DURATION_S
num_chunks: $NUM_CHUNKS
EOF

echo "Task $TASK_ID wall time: ${TASK_WALL_TIME_S}s"
echo "Task $TASK_ID: completed"

exit "${sim_exit:-0}"
