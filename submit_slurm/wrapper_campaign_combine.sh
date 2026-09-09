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
CAMPAIGN_DIR="${CAMPAIGN_DIR:-${OUTPUT_DIR:-}}"
EXPECTED_TASKS="${EXPECTED_TASKS:-0}"
MIN_TASKS="${MIN_TASKS:-1}"
REQUIRE_COMPLETE="${REQUIRE_COMPLETE:-0}"
INPUT_STAGE="${INPUT_STAGE:-tasks}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
SHARD_COUNT="${SHARD_COUNT:-1}"
COMBINE_OUTPUT_DIR="${COMBINE_OUTPUT_DIR:-}"
# Shard outputs are inputs to a later merge, so they must stay in the coords format.
SPLIT_PER_HEAD="${SPLIT_PER_HEAD:-auto}"

usage() {
    echo "Usage: $0 --campaign-dir DIR [--input-stage tasks|groups] [--shard-index N] [--shard-count N]"
    echo "          [--output-dir DIR] [--expected-tasks N] [--min-tasks N] [--require-complete]"
    echo "          [--split-per-head|--no-split-per-head]"
    echo ""
    echo "Aggregates sparse SRMs into higher-level final SRMs. With --input-stage tasks it"
    echo "reads task_*/final_srm_*.npz; with --input-stage groups it reads group_*/final_srm_*.npz."
    echo "By default it combines whatever succeeded and records completeness in"
    echo "combined_srm_metadata.json."
    echo ""
    echo "--split-per-head writes one CSR SRM per detector head, shape (pixels, voxels),"
    echo "instead of a single final_srm_<label>.npz. It is the default for a final merge"
    echo "(--shard-count 1) and is disabled automatically for shard combines."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --campaign-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --campaign-dir" >&2; exit 2; }
            CAMPAIGN_DIR="$2"
            shift 2
            ;;
        --expected-tasks)
            [[ $# -ge 2 ]] || { echo "Missing value for --expected-tasks" >&2; exit 2; }
            EXPECTED_TASKS="$2"
            shift 2
            ;;
        --min-tasks)
            [[ $# -ge 2 ]] || { echo "Missing value for --min-tasks" >&2; exit 2; }
            MIN_TASKS="$2"
            shift 2
            ;;
        --require-complete) REQUIRE_COMPLETE=1; shift ;;
        --split-per-head) SPLIT_PER_HEAD=1; shift ;;
        --no-split-per-head) SPLIT_PER_HEAD=0; shift ;;
        --input-stage)
            [[ $# -ge 2 ]] || { echo "Missing value for --input-stage" >&2; exit 2; }
            INPUT_STAGE="$2"
            shift 2
            ;;
        --shard-index)
            [[ $# -ge 2 ]] || { echo "Missing value for --shard-index" >&2; exit 2; }
            SHARD_INDEX="$2"
            shift 2
            ;;
        --shard-count)
            [[ $# -ge 2 ]] || { echo "Missing value for --shard-count" >&2; exit 2; }
            SHARD_COUNT="$2"
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --output-dir" >&2; exit 2; }
            COMBINE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unexpected argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$CAMPAIGN_DIR" ]]; then
    echo "Error: --campaign-dir (or CAMPAIGN_DIR/OUTPUT_DIR) is required" >&2
    exit 2
fi
if [[ ! -d "$CAMPAIGN_DIR" ]]; then
    echo "Error: campaign directory not found: $CAMPAIGN_DIR" >&2
    exit 1
fi
if ! [[ "$EXPECTED_TASKS" =~ ^[0-9]+$ ]]; then
    echo "expected_tasks must be a non-negative integer" >&2
    exit 2
fi
if ! [[ "$MIN_TASKS" =~ ^[1-9][0-9]*$ ]]; then
    echo "min_tasks must be a positive integer" >&2
    exit 2
fi
if ! [[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "shard_count must be a positive integer" >&2
    exit 2
fi
if ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] || [[ "$SHARD_INDEX" -ge "$SHARD_COUNT" ]]; then
    echo "shard_index must be in [0, shard_count)" >&2
    exit 2
fi

case "$INPUT_STAGE" in
    tasks) INPUT_PREFIX="task" ;;
    groups) INPUT_PREFIX="group" ;;
    *) echo "input_stage must be 'tasks' or 'groups'" >&2; exit 2 ;;
esac

CAMPAIGN_DIR="$(cd "$CAMPAIGN_DIR" && pwd)"
COMBINE_OUTPUT_DIR="${COMBINE_OUTPUT_DIR:-$CAMPAIGN_DIR}"
mkdir -p "$COMBINE_OUTPUT_DIR"
COMBINE_OUTPUT_DIR="$(cd "$COMBINE_OUTPUT_DIR" && pwd)"
export PYTHONPATH="$REPO_ROOT/qmirt/src${PYTHONPATH:+:$PYTHONPATH}"

AVAILABLE_INPUTS="$(find "$CAMPAIGN_DIR" -maxdepth 2 -name 'final_srm_1mm.npz' -path "*/${INPUT_PREFIX}_*" | wc -l)"
echo "Campaign dir: $CAMPAIGN_DIR"
echo "Input stage: $INPUT_STAGE (${INPUT_PREFIX}_*)"
echo "Inputs with final SRMs: $AVAILABLE_INPUTS"
if [[ "$SHARD_COUNT" -gt 1 ]]; then
    echo "Shard: ${SHARD_INDEX}/${SHARD_COUNT}"
fi
echo "Output dir: $COMBINE_OUTPUT_DIR"
if [[ "$EXPECTED_TASKS" -gt 0 ]]; then
    echo "Inputs expected: $EXPECTED_TASKS"
fi

if [[ "$SPLIT_PER_HEAD" == "auto" ]]; then
    if [[ "$SHARD_COUNT" -gt 1 ]]; then SPLIT_PER_HEAD=0; else SPLIT_PER_HEAD=1; fi
fi
echo "Split per head: $SPLIT_PER_HEAD"

combine_cmd=(
    python3
    "$REPO_ROOT/payload/python/combine_spect_sparse_srm.py"
    --input-dir "$CAMPAIGN_DIR"
    --output-dir "$COMBINE_OUTPUT_DIR"
    --input-glob "${INPUT_PREFIX}_*/final_srm_{label}.npz"
    --expected-inputs "$EXPECTED_TASKS"
    --min-inputs "$MIN_TASKS"
    --shard-index "$SHARD_INDEX"
    --shard-count "$SHARD_COUNT"
)

# A campaign is combined while tasks are still finishing, so record the primaries
# behind these inputs; counts alone cannot be normalized.
if [[ "$SPLIT_PER_HEAD" == "1" ]]; then
    SIMULATED_PRIMARIES="$(python3 - "$CAMPAIGN_DIR" <<'PY'
import glob, json, os, sys

total = 0
pattern = os.path.join(sys.argv[1], "*", "srm_chunks", "*_sim_stats_loop_*.txt")
for path in glob.glob(pattern):
    try:
        with open(path) as handle:
            total += int(json.load(handle)["events"]["value"])
    except (OSError, ValueError, KeyError, TypeError):
        continue
print(total)
PY
)"
    echo "Simulated primaries: ${SIMULATED_PRIMARIES}"
    combine_cmd+=(--simulated-primaries "$SIMULATED_PRIMARIES")
fi
if [[ "$REQUIRE_COMPLETE" == "1" ]]; then
    combine_cmd+=(--require-complete)
fi
if [[ "$SPLIT_PER_HEAD" == "1" ]]; then
    combine_cmd+=(--split-per-head)
else
    combine_cmd+=(--no-split-per-head)
fi

if [[ -f "$CONTAINER_SIF" ]] && [[ -n "$CONTAINER_EXEC" ]]; then
    "$CONTAINER_EXEC" exec \
        --bind "$REPO_ROOT:$REPO_ROOT" \
        --bind "$CAMPAIGN_DIR:$CAMPAIGN_DIR" \
        --bind "$COMBINE_OUTPUT_DIR:$COMBINE_OUTPUT_DIR" \
        "$CONTAINER_SIF" \
        "${combine_cmd[@]}"
else
    "${combine_cmd[@]}"
fi

echo "Combine stage '${INPUT_STAGE}': completed"
