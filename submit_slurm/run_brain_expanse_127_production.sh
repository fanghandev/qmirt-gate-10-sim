#!/usr/bin/env bash
# Submit one reproducible brain-SPECT production batch on Expanse shared nodes.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ACCOUNT="${ACCOUNT:-mgh102}"
PROJECT_DIR="${PROJECT_DIR:-/expanse/lustre/projects/mgh102/${USER}}"
JOB_COUNT="${JOB_COUNT:-64}"
CONCURRENT_LIMIT="${CONCURRENT_LIMIT:-64}"
NUM_LOOPS="${NUM_LOOPS:-5}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
# Source sphere diameter and SRM cube side. 288 mm is the largest round size inside
# the collimator nozzle tips (r = 144.57 mm, max D = 289.14 mm) that splits evenly
# into 1, 1.5 and 2 mm voxels. See dev/python/brain_spect_max_fov.py.
SRM_FOV_SIZE_MM="${SRM_FOV_SIZE_MM:-288}"
# Merged singles digitizer (one chain for all heads, identical singles) and the
# analytic CSG shield (73 G4Sphere tiles minus the apertures). Production-like
# benchmark at 64 threads (2026-10-02): 9 us/primary/thread vs 30 with the 0.05 mm
# simplified STL and 403 with the full STL; crystal counts within 0.1% of the STL.
# See dev/python/brain_shield_csg.md. SHIELD_MODEL=pieces (with SHIELD_PIECES_DIR)
# or stl restore the older shields.
export ACTOR_LAYOUT="${ACTOR_LAYOUT:-merged}"
export SHIELD_MODEL="${SHIELD_MODEL:-csg}"
export SHIELD_PIECES_DIR="${SHIELD_PIECES_DIR:-}"
# Geant4's overlap check is off in production loops (it reruns every loop); the CSG
# production geometry (288 mm FOV) was checked once with --check-overlaps: none.
export CHECK_OVERLAPS="${CHECK_OVERLAPS:-0}"
# One Gate process per NUMA domain of the node (numa_layout.py): 3.3x the node
# throughput of a single process on Expanse. NUMA_SPLIT=off restores one process.
export NUMA_SPLIT="${NUMA_SPLIT:-auto}"
# 1 s chunks of 6.25e6 Bq per thread keep each chunk at 7.9e8 events (32-bit EventID
# limit 2.1e9); loop length is set by the number of chunks.
NUM_CHUNKS="${NUM_CHUNKS:-40}"
REPORT_TIME_LIMIT="${REPORT_TIME_LIMIT:-2-00:00:00}"
ARRAY_DEPENDENCY="${ARRAY_DEPENDENCY:-}"
CAMPAIGN_GROUP_ID="${CAMPAIGN_GROUP_ID:-}"
CAMPAIGN_PART_INDEX="${CAMPAIGN_PART_INDEX:-}"
CAMPAIGN_PART_COUNT="${CAMPAIGN_PART_COUNT:-}"

if [[ "$(hostname -s)" != login* ]]; then
    echo "Run this from an Expanse login node." >&2
    exit 1
fi
if (( CONCURRENT_LIMIT > 64 )); then
    echo "CONCURRENT_LIMIT cannot exceed the shared-normal 64-node user limit." >&2
    exit 1
fi

# Each loop is NUM_CHUNKS x 7.9375e8 primaries (6.25e6 Bq x 1 s x 127 threads):
# 3.175e10 with 40 chunks, 4x the 210 mm campaign's loop, so the fixed per-loop
# costs (Gate start-up, geometry build, ROOT to SRM conversion) stay small now that
# a primary is much cheaper. Defaults are one campaign part: 64 tasks x 5 loops =
# 1.016e13 primaries; the 288 mm campaign is 10 parts = 1.016e14 primaries (98.5% of
# the 4e13 / 210 mm primary density scaled to 288 mm). The cost per loop on Expanse
# is not measured yet: run launch_brain_expanse_shield_ab_benchmark.sh first and
# set NUM_CHUNKS / NUM_LOOPS / TIME_LIMIT from it.
# Aggregate completed batches locally and sum their stats-file primary counts.
ARRAY_DEPENDENCY_ARGS=()
if [[ -n "$ARRAY_DEPENDENCY" ]]; then
    ARRAY_DEPENDENCY_ARGS=(--array-dependency "$ARRAY_DEPENDENCY")
fi
export CAMPAIGN_GROUP_ID CAMPAIGN_PART_INDEX CAMPAIGN_PART_COUNT
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
    --num-chunks "${NUM_CHUNKS}" \
    --num-loops "${NUM_LOOPS}" \
    --sparse-srm \
    --srm-fov-size-mm "${SRM_FOV_SIZE_MM}" \
    --profile-resources \
    --profile-interval-s 10 \
    --auto-report \
    --report-interval-s 60 \
    --report-partition shared \
    --report-cpus 1 \
    --report-mem-gb 2 \
    --report-time-limit "${REPORT_TIME_LIMIT}" \
    "${ARRAY_DEPENDENCY_ARGS[@]}"