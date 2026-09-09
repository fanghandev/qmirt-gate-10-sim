#!/usr/bin/env bash
# Submit one reproducible brain-SPECT production batch on Expanse shared nodes.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ACCOUNT="${ACCOUNT:-mgh102}"
PROJECT_DIR="${PROJECT_DIR:-/expanse/lustre/projects/mgh102/${USER}}"
JOB_COUNT="${JOB_COUNT:-100}"
CONCURRENT_LIMIT="${CONCURRENT_LIMIT:-64}"
NUM_LOOPS="${NUM_LOOPS:-10}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"

if [[ "$(hostname -s)" != login* ]]; then
    echo "Run this from an Expanse login node." >&2
    exit 1
fi
if (( CONCURRENT_LIMIT > 64 )); then
    echo "CONCURRENT_LIMIT cannot exceed the shared-normal 64-node user limit." >&2
    exit 1
fi

# Each loop targets 7.9375e9 primaries. Defaults target 7.9375e12 primaries.
# Aggregate completed batches locally and sum their stats-file primary counts.
exec bash "${REPO_ROOT}/submit_slurm/run_spect_sim_slurm.sh" brain \
    --cluster expanse \
    --account "${ACCOUNT}" \
    --project-dir "${PROJECT_DIR}" \
    --partition shared \
    --job-count "${JOB_COUNT}" \
    --concurrent-limit "${CONCURRENT_LIMIT}" \
    --cpus-per-task 127 \
    --mem-gb 220 \
    --time-limit "${TIME_LIMIT}" \
    --source-activity-bq 6.25e6 \
    --chunk-duration-s 1 \
    --num-chunks 10 \
    --num-loops "${NUM_LOOPS}" \
    --sparse-srm \
    --srm-fov-size-mm 210 \
    --profile-resources \
    --profile-interval-s 10 \
    --auto-report \
    --report-interval-s 60 \
    --report-partition shared \
    --report-cpus 1 \
    --report-mem-gb 2 \
    --report-time-limit 2-00:00:00