#!/usr/bin/env bash
# Benchmark production brain-SPECT physics on one Expanse scheduler partition.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ACCOUNT="${ACCOUNT:-mgh102}"
PROJECT_DIR="${PROJECT_DIR:-/expanse/lustre/projects/mgh102/${USER}}"
PARTITION="${PARTITION:-shared}"
JOB_COUNT="${JOB_COUNT:-1}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"

if [[ "$(hostname -s)" != login* ]]; then
    echo "Run this from an Expanse login node." >&2
    exit 1
fi

case "$PARTITION" in
    shared) CPUS_PER_TASK=127 ;;
    compute) CPUS_PER_TASK=128 ;;
    *)
        echo "PARTITION must be shared or compute for this benchmark." >&2
        exit 1
        ;;
esac

# One loop targets ~1.3e9 primaries, expected to take approximately 30 minutes.
exec bash "${REPO_ROOT}/submit_slurm/run_spect_sim_slurm.sh" brain \
    --cluster expanse \
    --account "${ACCOUNT}" \
    --project-dir "${PROJECT_DIR}" \
    --partition "${PARTITION}" \
    --job-count "${JOB_COUNT}" \
    --concurrent-limit "${JOB_COUNT}" \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem-gb 220 \
    --time-limit "${TIME_LIMIT}" \
    --source-activity-bq 1e6 \
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
    --report-time-limit 02:00:00