#!/bin/bash
# Execute-node worker: simulate, convert to a partial sparse SRM, discard the ROOT files.
set -euo pipefail

CLUSTER_ID="${1:?Usage: $0 <cluster_id> <proc_id> [num_loops] [source_activity_bq] [num_chunks] [chunk_duration_s] [fov_size_mm] [resolutions]}"
PROC_ID="${2:?Usage: $0 <cluster_id> <proc_id> ...}"
NUM_LOOPS="${3:-1}"
SOURCE_ACTIVITY_BQ="${4:-5e6}"
NUM_CHUNKS="${5:-100}"
CHUNK_DURATION_S="${6:-1.0}"
FOV_SIZE_MM="${7:-210.0}"
RESOLUTIONS_MM="${8:-1,1.5,2}"

export PYTHONPATH="$PWD/qmirt/src${PYTHONPATH:+:$PYTHONPATH}"
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-1}"

# With OSDF distribution the sandbox receives one tarball instead of three
# directories; unpack it before anything looks for payload/ or persistent_data/.
for payload_archive in qmirt-cardiac-payload-*.tar.gz; do
    [[ -f "$payload_archive" ]] || continue
    echo "Unpacking payload: $payload_archive"
    tar -xzf "$payload_archive"
    rm -f "$payload_archive"
    break
done

if [[ ! -d payload/python ]]; then
    echo "Error: payload/python missing; neither OSDF tarball nor staged dirs arrived" >&2
    exit 1
fi

SRM_CONVERTER="payload/python/create_spect_sparse_srm_from_batch_root.py"
if [[ ! -f "$SRM_CONVERTER" ]]; then
    echo "Error: new SRM converter missing from job payload: $SRM_CONVERTER" >&2
    exit 1
fi

TASK_TAG="c_${CLUSTER_ID}_p_${PROC_ID}"
SRM_DIR="srm_${TASK_TAG}"
STATS_DIR="${SRM_DIR}/stats"
RESULT_ARCHIVE="srm_${TASK_TAG}.tar.gz"

mkdir -p "$SRM_DIR" "$STATS_DIR"

# HTCondor only retrieves the declared output file, so always leave one behind:
# an empty archive is how a failed job reports itself rather than going held.
archive_outputs() {
    local exit_code="$1"
    printf '%s\n' "$exit_code" > "${SRM_DIR}/exit_code.txt"
    tar -czf "$RESULT_ARCHIVE" "$SRM_DIR"
    exit "$exit_code"
}
trap 'archive_outputs $?' EXIT

echo "Cardiac sparse SRM job ${CLUSTER_ID}.${PROC_ID}"
echo "  loops=${NUM_LOOPS} activity=${SOURCE_ACTIVITY_BQ} Bq chunks=${NUM_CHUNKS} x ${CHUNK_DURATION_S}s"
echo "  fov=${FOV_SIZE_MM} mm resolutions=${RESOLUTIONS_MM} mm"

for ((loop_index = 0; loop_index < NUM_LOOPS; loop_index++)); do
    LOOP_ID="$(printf '%05d' "$loop_index")"
    LOOP_DIR="loop_${LOOP_ID}"
    mkdir -p "$LOOP_DIR"

    sim_cmd=(
        python3 payload/python/gate_sim_cardiac_spect_boolean.py
        -o "$LOOP_DIR"
        -j "$CLUSTER_ID"
        -k "${PROC_ID}_loop_${LOOP_ID}"
        -n 1
        --fov-shape sphere
        --fov-size-mm "$FOV_SIZE_MM"
        -c "$NUM_CHUNKS"
        -d "$CHUNK_DURATION_S"
        -s "$SOURCE_ACTIVITY_BQ"
    )
    "${sim_cmd[@]}"

    python3 "$SRM_CONVERTER" \
        --input-dir "$LOOP_DIR" \
        --output-dir "$SRM_DIR" \
        --resolutions-mm "$RESOLUTIONS_MM" \
        --fov-size-mm "$FOV_SIZE_MM" \
        --pixels-per-head 625 \
        --allow-empty \
        --output-stem "srm_${TASK_TAG}_loop_${LOOP_ID}" \
        --job-id "$CLUSTER_ID" \
        --task-id "$PROC_ID" \
        --loop-id "$LOOP_ID"

    # Stats names carry the task tag because every job unpacks into one shared dir.
    mv "$SRM_DIR/srm_metadata.json" \
        "${STATS_DIR}/srm_metadata_${TASK_TAG}_loop_${LOOP_ID}.json"
    for stats_file in "$LOOP_DIR"/*_sim_stats.txt; do
        [[ -f "$stats_file" ]] || continue
        cp "$stats_file" "${STATS_DIR}/sim_stats_${TASK_TAG}_loop_${LOOP_ID}.txt"
    done

    # ROOT files stay on the execute node; only the sparse SRM is returned.
    rm -rf "$LOOP_DIR"
done

echo "Finished ${NUM_LOOPS} loop(s); partial SRMs in ${SRM_DIR}:"
ls -1 "$SRM_DIR"
