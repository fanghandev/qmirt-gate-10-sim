#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if command -v module >/dev/null 2>&1; then
    # Expanse ships singularitypro rather than Apptainer.
    module load Apptainer 2>/dev/null || module load singularitypro 2>/dev/null || true
fi
CONTAINER_EXEC="$(command -v apptainer || command -v singularity || true)"

CONTAINER_SIF="${CONTAINER_SIF:-${REPO_ROOT}/submit_slurm/qmirt-gate-10-sim-sif_v1.0.0.sif}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${SLURM_PROCID:-0}}"
CAMPAIGN_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/brain_spect/slurm/${JOB_ID}}"
# Array elements share CAMPAIGN_DIR, so every task write must stay under OUT_DIR.
OUT_DIR="${CAMPAIGN_DIR}/task_${TASK_ID}"

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
# Gate's Python messages (seed, START/STOP, progress) must reach the per-process
# logs; buffered output written to a file is lost at exit.
export PYTHONUNBUFFERED=1

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
        --no-sparse-srm) SPARSE_SRM=0; shift ;;
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
            echo "Usage: $0 [--sparse-srm|--no-sparse-srm] [--num-loops N] [--srm-fov-size-mm VALUE] [--profile-resources|--no-profile-resources] [--profile-interval-s SECONDS]"
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

if [[ -n "$CONTAINER_EXEC" ]]; then
    APPTAINER_CMD=(
        "$CONTAINER_EXEC" exec
        --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}"
        --bind "$REPO_ROOT:$REPO_ROOT"
        --bind "$LOCAL_RUN_ROOT:$LOCAL_RUN_ROOT"
        "$CONTAINER_SIF"
    )
else
    APPTAINER_CMD=()
fi

# Shield geometry (SHIELD_MODEL): stl (the 186,772-facet STL), pieces (STL pieces
# in SHIELD_PIECES_DIR, relative to the repo root, e.g. the 0.05 mm simplified
# shield) or csg (analytic tiles, fastest; dev/python/brain_shield_csg.md).
# A pieces dir without SHIELD_MODEL keeps the old behaviour (pieces).
SHIELD_MODEL="${SHIELD_MODEL:-}"
if [[ -z "$SHIELD_MODEL" ]]; then
    SHIELD_MODEL=$([[ -n "${SHIELD_PIECES_DIR:-}" ]] && echo pieces || echo stl)
fi
SHIELD_ARGS=(--shield-model "$SHIELD_MODEL")
case "$SHIELD_MODEL" in
    pieces)
        if [[ ! -f "$REPO_ROOT/${SHIELD_PIECES_DIR:-}/manifest.json" ]]; then
            echo "Error: SHIELD_MODEL=pieces needs SHIELD_PIECES_DIR with a manifest.json" >&2
            exit 1
        fi
        SHIELD_ARGS+=(--shield-pieces-dir "$REPO_ROOT/$SHIELD_PIECES_DIR")
        ;;
    stl|csg) ;;
    *) echo "Error: SHIELD_MODEL must be stl, pieces or csg (got '$SHIELD_MODEL')" >&2; exit 1 ;;
esac
# Geant4's overlap check is off in production loops; CHECK_OVERLAPS=1 turns it on
if [[ "${CHECK_OVERLAPS:-0}" == "1" ]]; then
    SHIELD_ARGS+=(--check-overlaps)
fi

# One Gate process per NUMA domain (NUMA_SPLIT=auto|off|N; numa_layout.py). A single
# process spread over the 8 NUMA domains of an Expanse node was 3.3x slower per node
# than one pinned process per domain (dev/python/brain_shield_csg.md).
NUMA_SPLIT="${NUMA_SPLIT:-auto}"
NUMA_GROUPS=()
if [[ "$NUMA_SPLIT" != "off" ]]; then
    # run in the image like the simulation (it sees the host's /sys and CPU affinity)
    mapfile -t NUMA_GROUPS < <("${APPTAINER_CMD[@]}" python3 "$SCRIPT_DIR/numa_layout.py" --split "$NUMA_SPLIT")
    if (( ${#NUMA_GROUPS[@]} > 1 )) && ! command -v numactl >/dev/null 2>&1; then
        echo "Warning: numactl not found; running one Gate process per loop." >&2
        NUMA_GROUPS=()
    fi
    if (( ${#NUMA_GROUPS[@]} < 2 )); then
        NUMA_GROUPS=()
    fi
fi

build_sim_command() {
    local output_dir="$1"
    local task_id="$2"
    local threads="${3:-${SLURM_CPUS_PER_TASK:-1}}"
    if [[ ${#APPTAINER_CMD[@]} -gt 0 ]]; then
        sim_cmd=(
            "${APPTAINER_CMD[@]}"
            python3
            "$REPO_ROOT/payload/python/gate_sim_brain_spect_boolean.py"
            -o "$output_dir"
            -j "$JOB_ID"
            -k "$task_id"
            --execution-environment slurm
            -n "$threads"
            -s "$SOURCE_ACTIVITY_BQ"
            -d "$CHUNK_DURATION_S"
            -c "$NUM_CHUNKS"
            --fov-shape sphere
            --fov-size-mm "${SRM_FOV_SIZE_MM:-210}"
            --actor-layout "${ACTOR_LAYOUT:-merged}"
            --physics-list "${PHYSICS_LIST:-QGSP_BERT_EMV}"
            "${SHIELD_ARGS[@]}"
        )
    else
        sim_cmd=(
            python3
            "$REPO_ROOT/payload/python/gate_sim_brain_spect_boolean.py"
            -o "$output_dir"
            -j "$JOB_ID"
            -k "$task_id"
            --execution-environment slurm
            -n "$threads"
            -s "$SOURCE_ACTIVITY_BQ"
            -d "$CHUNK_DURATION_S"
            -c "$NUM_CHUNKS"
            --fov-shape sphere
            --fov-size-mm "${SRM_FOV_SIZE_MM:-210}"
            --actor-layout "${ACTOR_LAYOUT:-merged}"
            --physics-list "${PHYSICS_LIST:-QGSP_BERT_EMV}"
            "${SHIELD_ARGS[@]}"
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
            "$REPO_ROOT/payload/python/create_spect_sparse_srm_from_batch_root.py"
            --input-dir "$input_dir"
            --output-dir "$output_dir"
            --fov-size-mm "${SRM_FOV_SIZE_MM:-210}"
            --allow-empty
            --job-id "$JOB_ID"
            --task-id "$TASK_ID"
            --loop-id "$CURRENT_LOOP_ID"
        )
    else
        sparse_worker_cmd=(
            python3
            "$REPO_ROOT/payload/python/create_spect_sparse_srm_from_batch_root.py"
            --input-dir "$input_dir"
            --output-dir "$output_dir"
            --fov-size-mm "${SRM_FOV_SIZE_MM:-210}"
            --allow-empty
            --job-id "$JOB_ID"
            --task-id "$TASK_ID"
            --loop-id "$CURRENT_LOOP_ID"
        )
    fi
}

write_campaign_geometry_provenance() {
    local provenance_cmd=(
        python3
        "$REPO_ROOT/payload/python/write_brain_spect_geometry_provenance.py"
        --output "$CAMPAIGN_DIR/geometry_provenance.json"
        --fov-size-mm "$SRM_FOV_SIZE_MM"
        --shield-model "$SHIELD_MODEL"
    )
    if [[ "$SHIELD_MODEL" == "pieces" ]]; then
        provenance_cmd+=(--shield-pieces-dir "$REPO_ROOT/$SHIELD_PIECES_DIR")
    fi
    if [[ ${#APPTAINER_CMD[@]} -gt 0 ]]; then
        provenance_cmd=("${APPTAINER_CMD[@]}" "${provenance_cmd[@]}")
    fi
    "${provenance_cmd[@]}"
}

write_campaign_geometry_provenance

echo "Starting SLURM job $JOB_ID task $TASK_ID..."
echo "Campaign dir: $CAMPAIGN_DIR"
echo "Task output dir: $OUT_DIR"
echo "Source activity: ${SOURCE_ACTIVITY_BQ} Bq"
echo "Chunk duration: ${CHUNK_DURATION_S} s"
echo "Num chunks: ${NUM_CHUNKS}"
echo "Sparse SRM mode: ${SPARSE_SRM}"
echo "Actor layout: ${ACTOR_LAYOUT:-merged}"
echo "Physics list: ${PHYSICS_LIST:-QGSP_BERT_EMV}"
echo "Shield model: ${SHIELD_MODEL}${SHIELD_PIECES_DIR:+ (pieces: $SHIELD_PIECES_DIR)}"
echo "Overlap check: ${CHECK_OVERLAPS:-0}"
echo "Num loops: ${NUM_LOOPS}"
if (( ${#NUMA_GROUPS[@]} > 1 )); then
    echo "NUMA split (${NUMA_SPLIT}): ${#NUMA_GROUPS[@]} Gate processes per loop (memory nodes, CPUs, threads):"
    printf '    %s\n' "${NUMA_GROUPS[@]}"
else
    echo "NUMA split (${NUMA_SPLIT}): one Gate process per loop"
fi

# Sole signal that a task's outputs are complete and safe for the workstation to pull.
write_task_complete_marker() {
    local marker="${OUT_DIR}/TASK_COMPLETE.json"
    local marker_tmp="${marker}.tmp"
    local first=1
    {
        printf '{\n'
        printf '  "job_id": "%s",\n' "$JOB_ID"
        printf '  "task_id": "%s",\n' "$TASK_ID"
        printf '  "num_loops": %s,\n' "$NUM_LOOPS"
        printf '  "completed_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '  "files": {\n'
        for srm_file in "$OUT_DIR"/final_srm_*.npz; do
            [[ -f "$srm_file" ]] || continue
            [[ "$first" -eq 1 ]] || printf ',\n'
            first=0
            printf '    "%s": {"sha256": "%s", "size_bytes": %s}' \
                "$(basename "$srm_file")" \
                "$(sha256sum "$srm_file" | cut -d' ' -f1)" \
                "$(stat -c %s "$srm_file")"
        done
        printf '\n  }\n}\n'
    } > "$marker_tmp"

    if [[ "$first" -eq 1 ]]; then
        echo "Error: no final_srm_*.npz found in $OUT_DIR; not marking task complete." >&2
        rm -f "$marker_tmp"
        return 1
    fi
    mv "$marker_tmp" "$marker"
    echo "Wrote completion marker: $marker"
}

# Build sim_cmd for one loop: a single Gate process, or one numactl-pinned Gate
# process per NUMA group run by run_parallel_commands.sh (each with its own task id,
# hence its own seed, output folder and log).
build_loop_command() {
    local loop_dir="$1"
    local loop_task_id="$2"
    if (( ${#NUMA_GROUPS[@]} < 2 )); then
        build_sim_command "$loop_dir" "$loop_task_id"
        return
    fi
    local command_file="${loop_dir}/parallel_commands.txt"
    local index mem cpus count bind
    : > "$command_file"
    for index in "${!NUMA_GROUPS[@]}"; do
        read -r mem cpus count <<<"${NUMA_GROUPS[$index]}"
        build_sim_command "${loop_dir}/numa${index}" "${loop_task_id}_n${index}" "$count"
        bind=(numactl "--physcpubind=${cpus}")
        if [[ "$mem" != "-" ]]; then
            bind+=("--membind=${mem}")
        fi
        mkdir -p "${loop_dir}/numa${index}"
        printf '%q ' "${bind[@]}" "${sim_cmd[@]}" >> "$command_file"
        printf '> %q 2>&1\n' "${loop_dir}/numa${index}.log" >> "$command_file"
    done
    sim_cmd=(bash "$SCRIPT_DIR/run_parallel_commands.sh" "$command_file")
}

# Per-process logs stay on node-local scratch; keep a short Geant4 warning summary
# per loop and show the tail of any log after a failure.
summarize_loop_logs() {
    local loop_dir="$1"
    local failed="$2"
    local log
    shopt -s nullglob
    local logs=("$loop_dir"/numa*.log)
    shopt -u nullglob
    (( ${#logs[@]} )) || return 0
    local warnings="${CHUNK_OUTPUT_DIR}/geant4_warnings_loop_${CURRENT_LOOP_ID}.txt"
    for log in "${logs[@]}"; do
        local n
        n="$(grep -a -c "G4Exception-START" "$log" || true)"
        if [[ "$n" -gt 0 ]]; then
            {
                echo "== $(basename "$log"): ${n} G4Exception(s)"
                grep -a -A8 "G4Exception-START" "$log" | head -60
            } >> "$warnings"
        fi
        if [[ "$failed" == "1" ]]; then
            # node-local scratch is lost with the job: keep the log tails with the task
            local keep="${CHUNK_OUTPUT_DIR}/failed_loop_${CURRENT_LOOP_ID}"
            mkdir -p "$keep"
            tr '\r' '\n' < "$log" | grep -av "^\s*$" | tail -n 200 > "${keep}/$(basename "$log")" || true
            echo "---- last lines of $(basename "$log") ($(wc -c < "$log") bytes; tail kept in ${keep}) ----" >&2
            tail -n 15 "${keep}/$(basename "$log")" >&2
        fi
    done
    if [[ -f "$warnings" ]]; then
        echo "Loop ${CURRENT_LOOP_ID}: Geant4 warnings summarized in ${warnings}"
    fi
}

run_sparse_workflow() {
    completed_loops=0
    for ((loop_index = 0; loop_index < NUM_LOOPS; loop_index++)); do
        CURRENT_LOOP_ID="$(printf '%05d' "$loop_index")"
        loop_dir="${LOCAL_RUN_ROOT}/loop_${CURRENT_LOOP_ID}"
        loop_srm_dir="${loop_dir}/srm"
        mkdir -p "$loop_dir"

        build_loop_command "$loop_dir" "${TASK_ID}_loop_${CURRENT_LOOP_ID}"
        echo "Starting sparse SRM loop ${CURRENT_LOOP_ID}/${NUM_LOOPS}..."
        if [[ "$PROFILE_RESOURCES" == "1" ]]; then
            profile_cmd=(bash "$SCRIPT_DIR/profile_resources.sh" "$loop_dir" "$PROFILE_INTERVAL_S" "${sim_cmd[@]}")
        else
            profile_cmd=("${sim_cmd[@]}")
        fi
        if [[ "$MAX_TASK_SECONDS" =~ ^[0-9]+$ ]] && [[ "$MAX_TASK_SECONDS" -gt 0 ]]; then
            if ! timeout --signal=TERM --kill-after=30 "$MAX_TASK_SECONDS" "${profile_cmd[@]}"; then
                echo "Loop ${CURRENT_LOOP_ID} simulation failed; combining completed loops." >&2
                summarize_loop_logs "$loop_dir" 1
                rm -rf "$loop_dir"
                break
            fi
        else
            if ! "${profile_cmd[@]}"; then
                echo "Loop ${CURRENT_LOOP_ID} simulation failed; combining completed loops." >&2
                summarize_loop_logs "$loop_dir" 1
                rm -rf "$loop_dir"
                break
            fi
        fi
        summarize_loop_logs "$loop_dir" 0

        build_sparse_worker_command "$loop_dir" "$loop_srm_dir"
        if ! "${sparse_worker_cmd[@]}"; then
            echo "Loop ${CURRENT_LOOP_ID} SRM conversion failed; combining completed loops." >&2
            rm -rf "$loop_dir"
            break
        fi

        for resolution_label in 1mm 1p5mm 2mm; do
            cp "$loop_srm_dir/srm_${resolution_label}.npz" \
                "$CHUNK_OUTPUT_DIR/srm_${resolution_label}_loop_${CURRENT_LOOP_ID}.npz"
        done
        cp "$loop_srm_dir/srm_metadata.json" \
            "$CHUNK_OUTPUT_DIR/srm_metadata_loop_${CURRENT_LOOP_ID}.json"
        for stats_file in "$loop_dir"/*_sim_stats.txt "$loop_dir"/numa*/*_sim_stats.txt; do
            [[ -f "$stats_file" ]] || continue
            # report_campaign_progress.py globs "*_sim_stats_loop_*.txt".
            cp "$stats_file" "$CHUNK_OUTPUT_DIR/$(basename "$stats_file" _sim_stats.txt)_sim_stats_loop_${CURRENT_LOOP_ID}.txt"
        done
        for manifest_file in "$loop_dir"/*_run_manifest.json "$loop_dir"/numa*/*_run_manifest.json; do
            [[ -f "$manifest_file" ]] || continue
            cp "$manifest_file" "$CHUNK_OUTPUT_DIR/$(basename "$manifest_file")"
        done
        cp "$loop_dir/resource_profile.tsv" "$CHUNK_OUTPUT_DIR/resource_profile_loop_${CURRENT_LOOP_ID}.tsv" 2>/dev/null || true
        cp "$loop_dir/resource_profile_summary.txt" "$CHUNK_OUTPUT_DIR/resource_profile_summary_loop_${CURRENT_LOOP_ID}.txt" 2>/dev/null || true
        completed_loops=$((completed_loops + 1))
        rm -rf "$loop_dir"
    done

    if [[ "$completed_loops" -eq 0 ]]; then
        echo "Error: no completed loops produced SRM inputs." >&2
        return 1
    fi

    # Sum actual completed-loop primaries so combined_srm_metadata.json's
    # simulated_primaries matches exactly the loops folded into the SRM
    # (loops that never finished never got a stats file copied here).
    SIMULATED_PRIMARIES="$(python3 - "$CHUNK_OUTPUT_DIR" <<'PY'
import glob, json, os, sys

total = 0
for path in glob.glob(os.path.join(sys.argv[1], "*_sim_stats_loop_*.txt")):
    try:
        with open(path) as handle:
            total += int(json.load(handle)["events"]["value"])
    except (OSError, ValueError, KeyError, TypeError):
        continue
print(total)
PY
)"

    if [[ ${#APPTAINER_CMD[@]} -gt 0 ]]; then
        combine_cmd=(
            "${APPTAINER_CMD[@]}"
            python3
            "$REPO_ROOT/payload/python/combine_spect_sparse_srm.py"
            --input-dir "$CHUNK_OUTPUT_DIR"
            --output-dir "$OUT_DIR"
            --expected-inputs "$NUM_LOOPS"
            --no-split-per-head
            --simulated-primaries "$SIMULATED_PRIMARIES"
        )
    else
        combine_cmd=(
            python3
            "$REPO_ROOT/payload/python/combine_spect_sparse_srm.py"
            --input-dir "$CHUNK_OUTPUT_DIR"
            --output-dir "$OUT_DIR"
            --expected-inputs "$NUM_LOOPS"
            --no-split-per-head
            --simulated-primaries "$SIMULATED_PRIMARIES"
        )
    fi
    "${combine_cmd[@]}"
    if [[ "$completed_loops" -eq "$NUM_LOOPS" ]]; then
        write_task_complete_marker
    else
        echo "Wrote partial SRMs for ${completed_loops}/${NUM_LOOPS} completed loops; task remains incomplete."
    fi
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