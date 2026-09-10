#!/usr/bin/env bash
# Incrementally aggregate all parts of one grouped brain-SPECT campaign.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $0 GROUP_ROOT GROUP_ID [OUTPUT_DIR]"
    echo "Selects campaign parts by campaign_group_id and incrementally combines their SRMs."
    exit 0
fi
GROUP_ROOT="${1:?Usage: $0 GROUP_ROOT GROUP_ID [OUTPUT_DIR]}"
GROUP_ID="${2:?Usage: $0 GROUP_ROOT GROUP_ID [OUTPUT_DIR]}"
OUTPUT_DIR="${3:-${GROUP_ROOT}/group_${GROUP_ID}_aggregate}"
SHARD_COUNT="${SHARD_COUNT:-16}"

GROUP_ROOT="$(cd "$GROUP_ROOT" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

mapfile -t PARTS < <(python3 - "$GROUP_ROOT" "$GROUP_ID" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
group_id = sys.argv[2]
parts = []
for manifest_path in sorted(root.glob("*/campaign_manifest.json")):
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    if manifest.get("campaign_group_id") == group_id:
        parts.append(manifest_path.parent)
for path in parts:
    print(path)
if not parts:
    raise SystemExit(f"no campaign parts found for group {group_id!r} under {root}")
PY
)

INPUT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/qmirt-group-input.XXXXXX")"
cleanup() { rm -rf "$INPUT_ROOT"; }
trap cleanup EXIT

part_index=0
for part in "${PARTS[@]}"; do
    ln -s "$part" "$INPUT_ROOT/part_$(printf '%03d' "$part_index")"
    part_index=$((part_index + 1))
done

SIMULATED_PRIMARIES="$(python3 - "$INPUT_ROOT" <<'PY'
import glob
import json
import os
import sys

total = 0
for path in glob.glob(os.path.join(sys.argv[1], "part_*", "task_*", "srm_chunks", "*_sim_stats_loop_*.txt")):
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
    --input-dir "$INPUT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --input-glob 'part_*/task_*/final_srm_{label}.npz' \
    --resolutions-mm 1,1.5,2 \
    --num-heads 73 \
    --pixels-per-head 625 \
    --shard-count "$SHARD_COUNT" \
    --simulated-primaries "$SIMULATED_PRIMARIES"

printf 'Grouped campaign: %s\n' "$GROUP_ID"
printf 'Parts: %d\n' "${#PARTS[@]}"
printf 'Output: %s\n' "$OUTPUT_DIR"
printf 'Simulated primaries: %s\n' "$SIMULATED_PRIMARIES"
