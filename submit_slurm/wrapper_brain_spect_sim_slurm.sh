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
OUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/brain_spect/slurm/${JOB_ID}}"

# SCRATCH_ROOT must be exported from parent submission script
# Fail loudly if it's missing (parent script should have set it)
if [[ -z "${SCRATCH_ROOT:-}" ]]; then
    echo "Error: SCRATCH_ROOT not exported from parent submission script"
    echo "This should be set by run_spect_sim_slurm.sh based on cluster detection."
    exit 1
fi
export SCRATCH_ROOT 
export REPO_ROOT
export PYTHONPATH="$REPO_ROOT/qmirt/src${PYTHONPATH:+:$PYTHONPATH}"
export SOURCE_ACTIVITY_BQ="${SOURCE_ACTIVITY_BQ:-3.7e5}"
export CHUNK_DURATION_S="${CHUNK_DURATION_S:-1.0}"
export NUM_CHUNKS="${NUM_CHUNKS:-1}"
export NUM_LOOPS="${NUM_LOOPS:-1}"
export SPARSE_SRM="${SPARSE_SRM:-0}"
export SRM_FOV_SIZE_MM="${SRM_FOV_SIZE_MM:-210}"
export LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-${SLURM_TMPDIR:-${TMPDIR:-$SCRATCH_ROOT}}}"
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
        --sparse-srm) SPARSE_SRM=1; shift ;;
        --num-loops)
            [[ $# -ge 2 ]] || { echo "Missing value for --num-loops" >&2; exit 2; }
            NUM_LOOPS="$2"
            shift 2
            ;;
        --srm-fov-size-mm)
            [[ $# -ge 2 ]] || { echo "Missing value for --srm-fov-size-mm" >&2; exit 2; }
            SRM_FOV_SIZE_MM="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--sparse-srm] [--num-loops N] [--srm-fov-size-mm VALUE] [--profile-resources|--no-profile-resources] [--profile-interval-s SECONDS]"
            exit 0
            ;;
        *)
            echo "Unexpected argument: $1" >&2
            exit 2
            ;;
    esac
done

if ! [[ "$NUM_LOOPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "num_loops must be a positive integer" >&2
    exit 2
fi

if ! [[ "$NUM_CHUNKS" =~ ^[1-9][0-9]*$ ]]; then
    echo "num_chunks must be a positive integer" >&2
    exit 2
fi
if ! [[ "$SRM_FOV_SIZE_MM" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$(awk -v size="$SRM_FOV_SIZE_MM" 'BEGIN { print (size > 0) ? 1 : 0 }')" != "1" ]]; then
    echo "srm_fov_size_mm must be a positive number" >&2
    exit 2
fi

if [[ ! -f "$CONTAINER_SIF" ]]; then
    echo "Error: Apptainer image not found at $CONTAINER_SIF"
    echo "Expected local SIF under submit_slurm/ from the README instructions."
    exit 1
fi

mkdir -p "$OUT_DIR"

if [[ "$SPARSE_SRM" == "1" ]]; then
    LOCAL_RUN_ROOT="${LOCAL_SCRATCH_ROOT}/qmirt/${JOB_ID}/${TASK_ID}"
    CHUNK_OUTPUT_DIR="${OUT_DIR}/srm_chunks"
    mkdir -p "$LOCAL_RUN_ROOT" "$CHUNK_OUTPUT_DIR"
else
    LOCAL_RUN_ROOT="$OUT_DIR"
fi

TASK_START_TS="$(date +%s)"

if command -v apptainer >/dev/null 2>&1; then
    APPTAINER_CMD=(
        apptainer exec
        --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}"
        --bind "$REPO_ROOT:$REPO_ROOT"
        --bind "$LOCAL_RUN_ROOT:$LOCAL_RUN_ROOT"
        "$CONTAINER_SIF"
    )
else
    APPTAINER_CMD=()
fi

build_sim_command() {
    local output_dir="$1"
    local task_id="$2"
    if [[ ${#APPTAINER_CMD[@]} -gt 0 ]]; then
        sim_cmd=(
            "${APPTAINER_CMD[@]}"
            python3
            "$REPO_ROOT/payload/python/gate_sim_brain_spect_boolean.py"
            -o "$output_dir"
            -j "$JOB_ID"
            -k "$task_id"
            --execution-environment slurm
            -n "${SLURM_CPUS_PER_TASK:-1}"
            -s "$SOURCE_ACTIVITY_BQ"
            -d "$CHUNK_DURATION_S"
            -c "$NUM_CHUNKS"
        )
    else
        sim_cmd=(
            python3
            "$REPO_ROOT/payload/python/gate_sim_brain_spect_boolean.py"
            -o "$output_dir"
            -j "$JOB_ID"
            -k "$task_id"
            --execution-environment slurm
            -n "${SLURM_CPUS_PER_TASK:-1}"
            -s "$SOURCE_ACTIVITY_BQ"
            -d "$CHUNK_DURATION_S"
            -c "$NUM_CHUNKS"
        )
    fi
}

build_sparse_worker_command() {
    local input_dir="$1"
    local output_dir="$2"
    if [[ ${#APPTAINER_CMD[@]} -gt 0 ]]; then
        sparse_worker_cmd=(
            "${APPTAINER_CMD[@]}"
            python3
            "$REPO_ROOT/payload/python/generate_brain_sparse_srm.py"
            --input-dir "$input_dir"
            --output-dir "$output_dir"
            --fov-size-mm "${SRM_FOV_SIZE_MM:-210}"
            --job-id "$JOB_ID"
            --task-id "$TASK_ID"
            --loop-id "$CURRENT_LOOP_ID"
        )
    else
        sparse_worker_cmd=(
            python3
            "$REPO_ROOT/payload/python/generate_brain_sparse_srm.py"
            --input-dir "$input_dir"
            --output-dir "$output_dir"
            --fov-size-mm "${SRM_FOV_SIZE_MM:-210}"
            --job-id "$JOB_ID"
            --task-id "$TASK_ID"
            --loop-id "$CURRENT_LOOP_ID"
        )
    fi
}

echo "Starting SLURM job $JOB_ID task $TASK_ID..."
echo "Output dir: $OUT_DIR"
echo "Source activity: ${SOURCE_ACTIVITY_BQ} Bq"
echo "Chunk duration: ${CHUNK_DURATION_S} s"
echo "Num chunks: ${NUM_CHUNKS}"
echo "Sparse SRM mode: ${SPARSE_SRM}"
echo "Num loops: ${NUM_LOOPS}"

run_sparse_workflow() {
    for ((loop_index = 0; loop_index < NUM_LOOPS; loop_index++)); do
        CURRENT_LOOP_ID="$(printf '%05d' "$loop_index")"
        loop_dir="${LOCAL_RUN_ROOT}/loop_${CURRENT_LOOP_ID}"
        loop_srm_dir="${loop_dir}/srm"
        mkdir -p "$loop_dir"

        build_sim_command "$loop_dir" "${TASK_ID}_loop_${CURRENT_LOOP_ID}"
        echo "Starting sparse SRM loop ${CURRENT_LOOP_ID}/${NUM_LOOPS}..."
        if [[ "$PROFILE_RESOURCES" == "1" ]]; then
            profile_cmd=(bash "$SCRIPT_DIR/profile_resources.sh" "$loop_dir" "$PROFILE_INTERVAL_S" "${sim_cmd[@]}")
        else
            profile_cmd=("${sim_cmd[@]}")
        fi
        if [[ "$MAX_TASK_SECONDS" =~ ^[0-9]+$ ]] && [[ "$MAX_TASK_SECONDS" -gt 0 ]]; then
            timeout --signal=TERM --kill-after=30 "$MAX_TASK_SECONDS" "${profile_cmd[@]}"
        else
            "${profile_cmd[@]}"
        fi

        build_sparse_worker_command "$loop_dir" "$loop_srm_dir"
        "${sparse_worker_cmd[@]}"

        for resolution_label in 1mm 1p5mm 2mm; do
            cp "$loop_srm_dir/srm_${resolution_label}.npz" \
                "$CHUNK_OUTPUT_DIR/srm_${resolution_label}_loop_${CURRENT_LOOP_ID}.npz"
        done
        cp "$loop_srm_dir/srm_metadata.json" \
            "$CHUNK_OUTPUT_DIR/srm_metadata_loop_${CURRENT_LOOP_ID}.json"
        for stats_file in "$loop_dir"/*_sim_stats.txt; do
            [[ -f "$stats_file" ]] || continue
            cp "$stats_file" "$CHUNK_OUTPUT_DIR/$(basename "$stats_file" .txt)_loop_${CURRENT_LOOP_ID}.txt"
        done
        cp "$loop_dir/resource_profile.tsv" "$CHUNK_OUTPUT_DIR/resource_profile_loop_${CURRENT_LOOP_ID}.tsv" 2>/dev/null || true
        cp "$loop_dir/resource_profile_summary.txt" "$CHUNK_OUTPUT_DIR/resource_profile_summary_loop_${CURRENT_LOOP_ID}.txt" 2>/dev/null || true
        rm -rf "$loop_dir"
    done

    if [[ ${#APPTAINER_CMD[@]} -gt 0 ]]; then
        combine_cmd=(
            "${APPTAINER_CMD[@]}"
            python3
            "$REPO_ROOT/payload/python/combine_brain_sparse_srm.py"
            --input-dir "$CHUNK_OUTPUT_DIR"
            --output-dir "$OUT_DIR"
            --expected-loops "$NUM_LOOPS"
        )
    else
        combine_cmd=(
            python3
            "$REPO_ROOT/payload/python/combine_brain_sparse_srm.py"
            --input-dir "$CHUNK_OUTPUT_DIR"
            --output-dir "$OUT_DIR"
            --expected-loops "$NUM_LOOPS"
        )
    fi
    "${combine_cmd[@]}"
}

if [[ "$SPARSE_SRM" == "1" ]]; then
    run_sparse_workflow
else
    build_sim_command "$OUT_DIR" "$TASK_ID"
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