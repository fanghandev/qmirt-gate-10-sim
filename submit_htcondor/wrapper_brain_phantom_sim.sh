#!/bin/bash
# OSPool job: one time slice of a brain SPECT phantom acquisition
# (gate_sim_brain_spect_boolean.py --mode phantom), reduced to list-mode in the job; the
# singles ROOT file never leaves the execute node (unless KEEP_ROOT=1).
#
#   wrapper_brain_phantom_sim.sh <cluster_id> <proc_id> <campaign.env>
#
# campaign.env (written by run_brain_phantom_batch.sh) sets:
#   PHANTOM_SPEC     spec name in the payload (payload/phantom_specs/<name>.json)
#   ACTIVITY_BQ      decays/s at time 0 in ACTIVITY_REGION ("all" or "brain")
#   ACTIVITY_REGION
#   SLICE_S          acquisition seconds per job; job p covers [p*SLICE_S, (p+1)*SLICE_S) + SLICE_OFFSET
#   SLICE_OFFSET     index of the first slice (default 0; for adding jobs to a campaign)
#   CHUNK_S          GATE run length (EventID range per run); SLICE_S must be a multiple
#   PHYSICS_LIST     G4EmStandardPhysics_option4 (as the 288 mm SRM)
#   KEEP_ROOT        1 to also return the singles ROOT file
set -euo pipefail

CLUSTER_ID="${1:?Usage: $0 <cluster_id> <proc_id> <campaign.env>}"
PROC_ID="${2:?Usage: $0 <cluster_id> <proc_id> <campaign.env>}"
CAMPAIGN_ENV="${3:?Usage: $0 <cluster_id> <proc_id> <campaign.env>}"
# shellcheck disable=SC1090
source "$CAMPAIGN_ENV"

for payload_archive in qmirt-brain-phantom-payload-*.tar.gz; do
    [[ -f "$payload_archive" ]] || continue
    echo "Unpacking payload: $payload_archive"
    tar -xzf "$payload_archive"
    rm -f "$payload_archive"
    break
done
[[ -d payload/python ]] || { echo "Error: payload/python missing" >&2; exit 1; }

export PYTHONPATH="$PWD/qmirt/src${PYTHONPATH:+:$PYTHONPATH}"
export QMIRT_PHANTOM_DATA="$PWD/payload/phantom_sources"
export POLARS_MAX_THREADS=1

TAG="c_${CLUSTER_ID}_p_${PROC_ID}"
OUT_DIR="phantom_${TAG}"
RESULT_ARCHIVE="phantom_${TAG}.tar.gz"
mkdir -p "$OUT_DIR"

# HTCondor only retrieves the declared output, so always leave one behind: an archive
# with exit_code.txt is how a failed job reports itself rather than going held.
archive_results() {
    local exit_code="$1"
    printf '%s\n' "$exit_code" > "$OUT_DIR/exit_code.txt"
    if [[ "${KEEP_ROOT:-0}" != "1" && -f "$OUT_DIR/listmode.npz" ]]; then
        rm -f "$OUT_DIR"/pixel_singles_*.root
    fi
    rm -rf "$OUT_DIR/phantom_image"
    tar -czf "$RESULT_ARCHIVE" "$OUT_DIR"
    exit "$exit_code"
}
trap 'archive_results $?' EXIT

SLICE=$((PROC_ID + ${SLICE_OFFSET:-0}))
NUM_CHUNKS=$(python3 -c "print(round(${SLICE_S} / ${CHUNK_S}))")
TIME_START=$(python3 -c "print(${SLICE} * ${SLICE_S})")
echo "Job $CLUSTER_ID.$PROC_ID: ${PHANTOM_SPEC}, ${ACTIVITY_BQ} Bq in ${ACTIVITY_REGION:-all}, slice ${SLICE}: t = ${TIME_START} s + ${NUM_CHUNKS} x ${CHUNK_S} s"
START=$(date +%s)

python3 payload/python/gate_sim_brain_spect_boolean.py \
    -o "$OUT_DIR" -j "$CLUSTER_ID" -k "$PROC_ID" \
    --execution-environment ospool -n 1 \
    -d "$CHUNK_S" -c "$NUM_CHUNKS" --time-start-s "$TIME_START" \
    --fov-shape sphere --fov-size-mm 288 \
    --actor-layout merged --shield-model csg \
    --physics-list "${PHYSICS_LIST:-G4EmStandardPhysics_option4}" \
    --mode phantom --phantom-spec "payload/phantom_specs/${PHANTOM_SPEC}.json" \
    --phantom-activity-bq "$ACTIVITY_BQ" --phantom-activity-region "${ACTIVITY_REGION:-all}" \
    --energy-resolution 0

python3 payload/python/reduce_phantom_singles.py "$OUT_DIR"
echo "{\"wall_s\": $(( $(date +%s) - START )), \"host\": \"$(hostname)\", \"cpu\": \"$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)\"}" > "$OUT_DIR/job_info.json"
