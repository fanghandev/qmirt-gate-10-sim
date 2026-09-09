#!/bin/bash
# Workstation aggregation for an OSPool cardiac-SPECT campaign.
#
# Reads the per-job partial SRM tarballs from the sshfs-mounted OSPool data area,
# unpacks and combines them on local disk, and writes 80 per-head CSR matrices.
# Nothing is written back to the mount: sshfs is slow and a campaign is O(10k)
# partials.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OSPOOL_MOUNT="${OSPOOL_MOUNT:-$HOME/ospool}"
CAMPAIGN_SUBDIR="${CAMPAIGN_SUBDIR:-cardiac_spect_srm}"
LOCAL_ROOT="${LOCAL_ROOT:-/data/fanghan/opengate_sim/data/cardiac_spect}"

SOURCE_DIR=""
BATCH_ID=""
USE_LATEST=0
LOCAL_DIR=""
OUTPUT_DIR=""
RESOLUTIONS_MM="1,1.5,2"
NUM_HEADS="80"
PIXELS_PER_HEAD="625"
SHARD_COUNT="16"
SHARD_WORKERS="${SHARD_WORKERS:-8}"
MIN_INPUTS="1"
EXPECTED_PARTIALS="0"
CONDOR_CLUSTER_ID="${CONDOR_CLUSTER_ID:-}"
EXTRACT=1
FORCE=0

usage() {
    echo "Usage: $0 (--latest | --batch BATCH_ID | --source-dir DIR) [--local-dir DIR]"
    echo "          [--output-dir DIR] [--resolutions-mm 1,1.5,2] [--num-heads N]"
    echo "          [--pixels-per-head N] [--shard-count N] [--min-inputs N]"
    echo "          [--expected-partials N] [--no-extract] [--force]"
    echo ""
    echo "Reads srm_c_*_p_*.tar.gz from the OSPool mount (${OSPOOL_MOUNT}),"
    echo "extracts to <local-dir>/partials, and writes final_srm_<label>_head_01..NN.npz."
    echo "Extraction is incremental, so re-running only unpacks jobs that finished since."
    echo "A combine is skipped when no new partials arrived; --force overrides that."
    echo ""
    echo "--shard-count N does a tree reduction: N partial merges, then one final merge."
    echo "Use it when a single pass over all partials will not fit in memory."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --latest) USE_LATEST=1; shift ;;
        --batch) BATCH_ID="$2"; shift 2 ;;
        --source-dir) SOURCE_DIR="$2"; shift 2 ;;
        --local-dir) LOCAL_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --resolutions-mm) RESOLUTIONS_MM="$2"; shift 2 ;;
        --num-heads) NUM_HEADS="$2"; shift 2 ;;
        --pixels-per-head) PIXELS_PER_HEAD="$2"; shift 2 ;;
        --shard-count) SHARD_COUNT="$2"; shift 2 ;;
        --min-inputs) MIN_INPUTS="$2"; shift 2 ;;
        --expected-partials) EXPECTED_PARTIALS="$2"; shift 2 ;;
        --no-extract) EXTRACT=0; shift ;;
        --force) FORCE=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unexpected argument: $1" >&2; usage; exit 2 ;;
    esac
done

CAMPAIGN_ROOT="${OSPOOL_MOUNT}/${CAMPAIGN_SUBDIR}"

if [[ -z "$SOURCE_DIR" ]]; then
    if [[ ! -d "$OSPOOL_MOUNT" ]]; then
        echo "Error: OSPool mount is not readable (stale sshfs?): $OSPOOL_MOUNT" >&2
        exit 1
    fi
    if [[ "$USE_LATEST" -eq 1 ]]; then
        BATCH_ID="$(find "$CAMPAIGN_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'batch_*' \
            -printf '%f\n' 2>/dev/null | sort | tail -1)"
        if [[ -z "$BATCH_ID" ]]; then
            echo "Error: no batch_* campaigns under $CAMPAIGN_ROOT" >&2
            exit 1
        fi
    fi
    if [[ -z "$BATCH_ID" ]]; then
        echo "Error: one of --latest, --batch or --source-dir is required" >&2
        exit 2
    fi
    SOURCE_DIR="${CAMPAIGN_ROOT}/${BATCH_ID}"
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: campaign source directory not found: $SOURCE_DIR" >&2
    exit 1
fi
SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"
BATCH_ID="${BATCH_ID:-$(basename "$SOURCE_DIR")}"

if ! [[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "shard_count must be a positive integer" >&2
    exit 2
fi

LOCAL_DIR="${LOCAL_DIR:-${LOCAL_ROOT}/${BATCH_ID}}"
if [[ "$LOCAL_DIR" == "$OSPOOL_MOUNT"* ]]; then
    echo "Error: refusing to write results onto the OSPool mount ($LOCAL_DIR)" >&2
    exit 2
fi
mkdir -p "$LOCAL_DIR"
LOCAL_DIR="$(cd "$LOCAL_DIR" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$LOCAL_DIR}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
PARTIAL_DIR="${LOCAL_DIR}/partials"

export PYTHONPATH="$REPO_ROOT/qmirt/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Batch:          $BATCH_ID"
echo "Source (mount): $SOURCE_DIR"
echo "Local dir:      $LOCAL_DIR"
echo "Output dir:     $OUTPUT_DIR"
echo "Heads:          ${NUM_HEADS} x ${PIXELS_PER_HEAD} pixels"

if [[ -f "${SOURCE_DIR}/campaign_manifest.json" ]]; then
    cp "${SOURCE_DIR}/campaign_manifest.json" "${LOCAL_DIR}/campaign_manifest.json"
    if [[ "$EXPECTED_PARTIALS" == "0" ]]; then
        EXPECTED_PARTIALS="$(python3 -c "
import json, sys
print(json.load(open(sys.argv[1])).get('expected_partials', 0))
" "${LOCAL_DIR}/campaign_manifest.json" 2>/dev/null || echo 0)"
    fi
fi

if [[ -z "$CONDOR_CLUSTER_ID" ]]; then
    CONDOR_CLUSTER_ID="$(find "$SOURCE_DIR" -maxdepth 1 -type f -name 'srm_c_*_p_*.tar.gz' -printf '%f\n' \
        | sed -nE 's/^srm_c_([0-9]+)_p_[0-9]+\.tar\.gz$/\1/p' | sort -u | head -n 1)"
fi

if [[ "$EXTRACT" -eq 1 ]]; then
    mkdir -p "$PARTIAL_DIR"
    extracted=0
    skipped=0
    failed=0
    # Markers keep re-runs incremental while a campaign is still draining.
    while IFS= read -r -d '' archive; do
        marker="${PARTIAL_DIR}/.$(basename "${archive%.tar.gz}").extracted"
        if [[ -f "$marker" ]]; then
            skipped=$((skipped + 1))
            continue
        fi
        if tar -xzf "$archive" -C "$PARTIAL_DIR" --strip-components=1 2>/dev/null; then
            touch "$marker"
            extracted=$((extracted + 1))
        else
            echo "Warning: failed to extract $(basename "$archive")" >&2
            failed=$((failed + 1))
        fi
    done < <(find "$SOURCE_DIR" -maxdepth 1 -name 'srm_c_*_p_*.tar.gz' -print0)
    echo "Extracted ${extracted}, already present ${skipped}, failed ${failed}."
fi

FIRST_RESOLUTION="${RESOLUTIONS_MM%%,*}"
FIRST_LABEL="${FIRST_RESOLUTION/./p}mm"
PARTIAL_COUNT="$(find "$PARTIAL_DIR" -name "srm_c_*_${FIRST_LABEL}.npz" 2>/dev/null | wc -l)"
echo "Partials (${FIRST_LABEL}): ${PARTIAL_COUNT}"
if [[ "$EXPECTED_PARTIALS" != "0" ]]; then
    echo "Expected partials:  ${EXPECTED_PARTIALS}"
fi

SRM_LABELS="$(python3 - "$RESOLUTIONS_MM" <<'PY'
import sys

print(",".join(
    f"{item.strip().replace('.', 'p')}mm"
    for item in sys.argv[1].split(",")
    if item.strip()
))
PY
)"

write_report() {
    report_cmd=(python3 "$REPO_ROOT/payload/python/report_campaign_progress.py" \
        --campaign-dir "$LOCAL_DIR" \
        --srm-dir "$OUTPUT_DIR" \
        --srm-labels "$SRM_LABELS" \
        --task-layout ospool \
        --expected-tasks "$EXPECTED_PARTIALS" \
        --output "${LOCAL_DIR}/progress.json")
    if [[ -n "$CONDOR_CLUSTER_ID" ]]; then
        report_cmd+=(--condor-cluster-id "$CONDOR_CLUSTER_ID")
    fi
    "${report_cmd[@]}"
    echo "Dashboard report: ${LOCAL_DIR}/progress.json"
}

if [[ "$PARTIAL_COUNT" -eq 0 ]]; then
    # Still publish a zero-progress report: the dashboard picks the newest campaign
    # that has one, so without this a freshly submitted campaign stays invisible and
    # the old one keeps being displayed.
    mkdir -p "$PARTIAL_DIR"
    echo "No partial SRMs yet in $PARTIAL_DIR; publishing zero progress."
    write_report
    exit 0
fi

# Combining rereads every partial, so skip the work when no new ones arrived.
STAMP_FILE="${LOCAL_DIR}/.last_combined_partials"
LAST_COUNT="$(cat "$STAMP_FILE" 2>/dev/null || echo 0)"
if [[ "$FORCE" -eq 0 && "$PARTIAL_COUNT" == "$LAST_COUNT" ]] \
    && compgen -G "${OUTPUT_DIR}/final_srm_*_head_*.npz" > /dev/null; then
    echo "No new partials since the last combine (${LAST_COUNT}); skipping."
    write_report
    exit 0
fi

# Combining rereads every partial, so skip the work when no new ones arrived.
STAMP_FILE="${LOCAL_DIR}/.last_combined_partials"
LAST_COUNT="$(cat "$STAMP_FILE" 2>/dev/null || echo 0)"
if [[ "$FORCE" -eq 0 && "$PARTIAL_COUNT" == "$LAST_COUNT" ]] \
    && compgen -G "${OUTPUT_DIR}/final_srm_*_head_*.npz" > /dev/null; then
    echo "No new partials since the last combine (${LAST_COUNT}); skipping."
    exit 0
fi

# Stragglers are normal, so the SRM is only interpretable against the primaries
# that actually produced it. Sum them from the same job tarballs that were combined.
SIMULATED_PRIMARIES="$(python3 - "$PARTIAL_DIR/stats" <<'PY'
import glob, json, os, sys

total = 0
for path in glob.glob(os.path.join(sys.argv[1], "sim_stats_*.txt")):
    try:
        with open(path) as handle:
            total += int(json.load(handle)["events"]["value"])
    except (OSError, ValueError, KeyError, TypeError):
        continue
print(total)
PY
)"
echo "Simulated primaries: ${SIMULATED_PRIMARIES}"

python3 "$REPO_ROOT/payload/python/incremental_sparse_srm_aggregate.py" \
    --input-dir "$PARTIAL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --input-glob 'srm_c_*_{label}.npz' \
    --resolutions-mm "$RESOLUTIONS_MM" \
    --num-heads "$NUM_HEADS" \
    --pixels-per-head "$PIXELS_PER_HEAD" \
    --shard-count "$SHARD_COUNT" \
    --workers "$SHARD_WORKERS" \
    --simulated-primaries "$SIMULATED_PRIMARIES"

echo "Done. Per-head SRMs in $OUTPUT_DIR"
printf '%s\n' "$PARTIAL_COUNT" > "$STAMP_FILE"

write_report
