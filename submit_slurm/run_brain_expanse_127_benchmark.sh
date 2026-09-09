#!/usr/bin/env bash
# Benchmark the production brain-SPECT physics at Expanse's shared-QOS CPU cap.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ACCOUNT="${ACCOUNT:-mgh102}"
PROJECT_DIR="${PROJECT_DIR:-/expanse/lustre/projects/mgh102/${USER}}"
JOB_COUNT="${JOB_COUNT:-2}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"

if [[ "$(hostname -s)" != login* ]]; then
    echo "Run this from an Expanse login node." >&2
    exit 1
fi

# One loop is 127 threads x 10 chunks x 1 s x 6.25e6 Bq = 7.9375e9 primaries.
exec bash "${REPO_ROOT}/submit_slurm/run_spect_sim_slurm.sh" brain \
    --cluster expanse \
    --account "${ACCOUNT}" \
    --project-dir "${PROJECT_DIR}" \
    --partition shared \
    --job-count "${JOB_COUNT}" \
    --concurrent-limit "${JOB_COUNT}" \
    --cpus-per-task 127 \
    --mem-gb 220 \
    --time-limit "${TIME_LIMIT}" \
    --source-activity-bq 6.25e6 \
    --chunk-duration-s 1 \
    --num-chunks 10 \
    --num-loops 1 \
    --sparse-srm \
    --srm-fov-size-mm 210 \
    --profile-resources \
    --profile-interval-s 5 \
    --auto-report \
    --report-interval-s 60 \
    --report-partition shared \
    --report-cpus 1 \
    --report-mem-gb 2 \
    --report-time-limit 08:00:00