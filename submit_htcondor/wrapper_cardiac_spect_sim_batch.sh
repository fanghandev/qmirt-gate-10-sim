#!/bin/bash
set -euo pipefail

CLUSTER_ID="${1:?Usage: $0 <cluster_id> <proc_id>}"
PROC_ID="${2:?Usage: $0 <cluster_id> <proc_id>}"

# Ensure the qmirt module in the working directory is discoverable.
export PYTHONPATH="$PWD/qmirt/src${PYTHONPATH:+:$PYTHONPATH}"

# Define and create a unique output directory for this specific job.
OUT_DIR="output_${CLUSTER_ID}_${PROC_ID}"
RESULT_ARCHIVE="results_${CLUSTER_ID}_${PROC_ID}.tar.gz"

archive_results() {
    local exit_code="$1"

    if [[ -d "$OUT_DIR" ]]; then
        tar -czf "$RESULT_ARCHIVE" "$OUT_DIR"
    else
        tar -czf "$RESULT_ARCHIVE" --files-from /dev/null
    fi

    exit "$exit_code"
}

trap 'archive_results $?' EXIT

mkdir -p "$OUT_DIR"

echo "Starting job $CLUSTER_ID task $PROC_ID..."

# Optimized for 500 million primaries (safely under the 12-hour eviction window)
python3 "payload/python/gate_sim_cardiac_spect_no_boolean_geometry.py" \
    -o "$OUT_DIR" \
    -j "$CLUSTER_ID" \
    -k "$PROC_ID" \
    -n 1 \
    -m "box" \
    -t "Gamma-140" \
    -c 1 \
    -d 1.0 \
    -s 500000000