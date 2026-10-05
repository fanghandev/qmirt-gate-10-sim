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
SRM_FOV_SIZE_MM="${SRM_FOV_SIZE_MM:-288}"
# What to benchmark (see launch_brain_expanse_shield_ab_benchmark.sh for the A/B set):
# ACTOR_LAYOUT merged|per-head, SHIELD_MODEL csg|pieces|stl (pieces needs
# SHIELD_PIECES_DIR). Production activity per thread; one loop of NUM_CHUNKS 1 s
# chunks. Two NUM_CHUNKS values give the cost per primary as a slope.
export ACTOR_LAYOUT="${ACTOR_LAYOUT:-merged}"
export SHIELD_MODEL="${SHIELD_MODEL:-csg}"
export SHIELD_PIECES_DIR="${SHIELD_PIECES_DIR:-}"
export CHECK_OVERLAPS="${CHECK_OVERLAPS:-0}"
# One Gate process per NUMA domain of the node (numa_layout.py): 3.3x the node
# throughput of a single process on Expanse. NUMA_SPLIT=off restores one process.
export NUMA_SPLIT="${NUMA_SPLIT:-auto}"
SOURCE_ACTIVITY_BQ="${SOURCE_ACTIVITY_BQ:-6.25e6}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"
AUTO_REPORT_ARGS=(--auto-report --report-interval-s 60 --report-partition shared
                  --report-cpus 1 --report-mem-gb 2 --report-time-limit "${TIME_LIMIT}")
if [[ "${AUTO_REPORT:-1}" != "1" ]]; then
    AUTO_REPORT_ARGS=()
fi

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

# One loop of NUM_CHUNKS x SOURCE_ACTIVITY_BQ x 1 s x CPUS_PER_TASK primaries.
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
    --source-activity-bq "${SOURCE_ACTIVITY_BQ}" \
    --chunk-duration-s 1 \
    --num-chunks "${NUM_CHUNKS}" \
    --num-loops 1 \
    --sparse-srm \
    --srm-fov-size-mm "${SRM_FOV_SIZE_MM}" \
    --profile-resources \
    --profile-interval-s 5 \
    "${AUTO_REPORT_ARGS[@]}"
