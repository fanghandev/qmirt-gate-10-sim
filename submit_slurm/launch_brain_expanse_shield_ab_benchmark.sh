#!/usr/bin/env bash
# Submit the Expanse A/B benchmark for the brain SRM production settings.
#
# Three configurations at the 288 mm FOV and production activity, each as a
# one-task, one-loop job at two loop sizes (NUM_CHUNKS_SMALL, NUM_CHUNKS_LARGE) so
# the cost per primary is a slope and start-up costs cancel:
#   legacy  per-head actors + full STL             (the 210 mm campaigns' setup)
#   pieces  merged actors + 0.05 mm simplified STL (production scripts before CSG)
#   csg     merged actors + analytic CSG shield    (current production default)
# Run from an Expanse login node; then, once the jobs finish:
#   python3 submit_slurm/analyze_brain_shield_ab_benchmark.py <PROJECT_DIR>/brain_spect_sim/brain_ab_*_<STAMP>
# Rough cost: ~600 SU in total (most of it the legacy configuration).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
NUM_CHUNKS_SMALL="${NUM_CHUNKS_SMALL:-2}"
NUM_CHUNKS_LARGE="${NUM_CHUNKS_LARGE:-8}"
SIMPLIFIED_DIR="persistent_data/brain_spect/stl/BrainFrame.008.Lead_Shield.simplified_0.05mm"
DRY_RUN="${DRY_RUN:-0}"

# name  actor layout  shield model  pieces dir  time limit
CONFIGS=(
    "legacy per-head stl - 05:00:00"
    "pieces merged pieces ${SIMPLIFIED_DIR} 02:00:00"
    "csg merged csg - 01:00:00"
)

if [[ "$(hostname -s)" != login* && "$DRY_RUN" != "1" ]]; then
    echo "Run this from an Expanse login node (or set DRY_RUN=1)." >&2
    exit 1
fi

for config in "${CONFIGS[@]}"; do
    read -r name layout model pieces time_limit <<<"$config"
    [[ "$pieces" == "-" ]] && pieces=""
    for chunks in "$NUM_CHUNKS_SMALL" "$NUM_CHUNKS_LARGE"; do
        batch_id="brain_ab_${name}_c${chunks}_${STAMP}"
        echo "Submitting ${batch_id}: actors=${layout} shield=${model} chunks=${chunks}"
        if [[ "$DRY_RUN" == "1" ]]; then
            continue
        fi
        BATCH_ID="$batch_id" ACTOR_LAYOUT="$layout" SHIELD_MODEL="$model" \
            SHIELD_PIECES_DIR="$pieces" NUM_CHUNKS="$chunks" TIME_LIMIT="$time_limit" \
            bash "${SCRIPT_DIR}/run_brain_expanse_127_benchmark.sh"
    done
done
echo "Benchmark stamp: ${STAMP}"
