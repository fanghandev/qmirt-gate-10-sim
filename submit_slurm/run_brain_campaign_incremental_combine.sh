#!/usr/bin/env bash
# Incrementally aggregate completed brain-SPECT array-task SRMs on local storage.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CAMPAIGN_DIR="${1:?Usage: $0 CAMPAIGN_DIR [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-$CAMPAIGN_DIR}"
SHARD_COUNT="${SHARD_COUNT:-16}"

CAMPAIGN_DIR="$(cd "$CAMPAIGN_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

SIMULATED_PRIMARIES="$(python3 - "$CAMPAIGN_DIR" <<'PY'
import glob
import json
import os
import sys

total = 0
for path in glob.glob(os.path.join(sys.argv[1], "task_*", "srm_chunks", "*_sim_stats_loop_*.txt")):
    task_dir = os.path.dirname(os.path.dirname(path))
    if not os.path.isfile(os.path.join(task_dir, "final_srm_1mm.npz")):
        continue
    try:
        with open(path) as handle:
            total += int(json.load(handle)["events"]["value"])
    except (OSError, ValueError, KeyError, TypeError):
        continue
print(total)
PY
)"

python3 "$REPO_ROOT/payload/python/incremental_sparse_srm_aggregate.py" \
    --input-dir "$CAMPAIGN_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --input-glob 'task_*/final_srm_{label}.npz' \
    --resolutions-mm 1,1.5,2 \
    --num-heads 73 \
    --pixels-per-head 625 \
    --shard-count "$SHARD_COUNT" \
    --simulated-primaries "$SIMULATED_PRIMARIES"