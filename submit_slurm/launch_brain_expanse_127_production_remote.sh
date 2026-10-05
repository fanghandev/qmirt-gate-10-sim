#!/usr/bin/env bash
# Submit brain-SPECT production campaign(s) on Expanse from a workstation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${REMOTE_HOST:-expanse}"
REMOTE_REPO="${REMOTE_REPO:-/home/fhan1/qmirt-gate-10-sim}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${SCRIPT_DIR}/logs/remote_submissions}"

SUBMISSION_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
# One campaign group spans allocations. To continue it later, pass the same
# CAMPAIGN_GROUP_ID and set PART_INDEX_START to the next unsubmitted part, e.g.
#   CAMPAIGN_GROUP_ID=brain_288mm_csg_102t_20261002T120000Z PART_INDEX_START=4 BATCH_COUNT=3 ...
CAMPAIGN_GROUP_ID="${CAMPAIGN_GROUP_ID:-brain_288mm_csg_102t_${SUBMISSION_TIMESTAMP}}"
BATCH_PREFIX="${BATCH_PREFIX:-$CAMPAIGN_GROUP_ID}"
# Parts to submit now, numbered PART_INDEX_START..PART_INDEX_START+BATCH_COUNT-1.
BATCH_COUNT="${BATCH_COUNT:-1}"
PART_INDEX_START="${PART_INDEX_START:-1}"
# Parts in the full campaign: 10 x 64 tasks x 5 loops x 40 chunks x 7.9375e8 =
# 1.016e14 primaries, 98.5% of the 4e13 / 210 mm primary density scaled to 288 mm.
CAMPAIGN_PART_COUNT="${CAMPAIGN_PART_COUNT:-10}"
JOB_COUNT="${JOB_COUNT:-64}"
NUM_LOOPS="${NUM_LOOPS:-5}"
NUM_CHUNKS="${NUM_CHUNKS:-40}"
SHIELD_MODEL="${SHIELD_MODEL:-csg}"
ACTOR_LAYOUT="${ACTOR_LAYOUT:-merged}"
PHYSICS_LIST="${PHYSICS_LIST:-G4EmStandardPhysics_option4}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
SRM_FOV_SIZE_MM="${SRM_FOV_SIZE_MM:-288}"
REPORT_TIME_LIMIT="${REPORT_TIME_LIMIT:-2-00:00:00}"
CONCURRENT_LIMIT="${CONCURRENT_LIMIT:-}"
CHAIN_BATCHES="${CHAIN_BATCHES:-1}"
DRY_RUN="${DRY_RUN:-0}"
# SU per loop of NUM_CHUNKS chunks. Provisional, from the 2026-10-02 Expanse tests:
# merged actors + CSG shield with one Gate process per NUMA domain ran at ~15 us per
# primary per thread (vs 50.6 as one process), so a 40-chunk loop (2.5e8 primaries
# per thread) is ~1.04 h x 127 CPUs = ~135 SU. Replace with the value from
# analyze_brain_shield_ab_benchmark.py after the csg_numa validation run.
SU_PER_LOOP="${SU_PER_LOOP:-135}"

if ! [[ "$BATCH_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_COUNT must be a positive integer" >&2
    exit 2
fi
if ! [[ "$JOB_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "JOB_COUNT must be a positive integer" >&2
    exit 2
fi
if ! [[ "$PART_INDEX_START" =~ ^[1-9][0-9]*$ ]] || ! [[ "$CAMPAIGN_PART_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "PART_INDEX_START and CAMPAIGN_PART_COUNT must be positive integers" >&2
    exit 2
fi
PART_INDEX_END=$((PART_INDEX_START + BATCH_COUNT - 1))
if (( PART_INDEX_END > CAMPAIGN_PART_COUNT )); then
    echo "Parts ${PART_INDEX_START}-${PART_INDEX_END} exceed CAMPAIGN_PART_COUNT=${CAMPAIGN_PART_COUNT}" >&2
    exit 2
fi
if ! [[ "$NUM_LOOPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_LOOPS must be a positive integer" >&2
    exit 2
fi
if ! [[ "$DRY_RUN" =~ ^[01]$ ]]; then
    echo "DRY_RUN must be 0 or 1" >&2
    exit 2
fi
if ! [[ "$CHAIN_BATCHES" =~ ^[01]$ ]]; then
    echo "CHAIN_BATCHES must be 0 or 1" >&2
    exit 2
fi

if [[ -z "$CONCURRENT_LIMIT" ]]; then
    if [[ "$CHAIN_BATCHES" -eq 1 ]]; then
        CONCURRENT_LIMIT=64
    else
        CONCURRENT_LIMIT=$((64 / BATCH_COUNT))
    fi
fi
if (( CONCURRENT_LIMIT < 1 )); then
    echo "BATCH_COUNT cannot exceed the 64-node Expanse user limit" >&2
    exit 2
fi
if [[ "$CHAIN_BATCHES" -eq 0 ]] && (( BATCH_COUNT * CONCURRENT_LIMIT > 64 )); then
    echo "BATCH_COUNT * CONCURRENT_LIMIT must not exceed 64" >&2
    echo "This launcher submits all batches, so their array limits are concurrent." >&2
    exit 2
fi

mkdir -p "$LOCAL_LOG_DIR"
local_log="$LOCAL_LOG_DIR/brain_expanse_production_${SUBMISSION_TIMESTAMP}.log"

previous_job_id=""
loops_now=$((BATCH_COUNT * JOB_COUNT * NUM_LOOPS))
printf 'Campaign %s: parts %d-%d of %d, %d loops (%s primaries), ~%d SU estimated\n' \
    "$CAMPAIGN_GROUP_ID" "$PART_INDEX_START" "$PART_INDEX_END" "$CAMPAIGN_PART_COUNT" \
    "$loops_now" "$(awk -v n="$loops_now" -v c="$NUM_CHUNKS" 'BEGIN { printf "%.3e", n * c * 7.9375e8 }')" \
    "$((loops_now * SU_PER_LOOP))" | tee -a "$local_log"
for ((batch_index = PART_INDEX_START; batch_index <= PART_INDEX_END; batch_index++)); do
    printf -v batch_id '%s_%02d' "$BATCH_PREFIX" "$batch_index"
    array_dependency=""
    if [[ "$CHAIN_BATCHES" -eq 1 && -n "$previous_job_id" ]]; then
        array_dependency="afterany:${previous_job_id}"
    fi
    printf -v remote_command \
        'cd -- %q && BATCH_ID=%q JOB_COUNT=%q CONCURRENT_LIMIT=%q NUM_LOOPS=%q NUM_CHUNKS=%q SHIELD_MODEL=%q ACTOR_LAYOUT=%q PHYSICS_LIST=%q TIME_LIMIT=%q SRM_FOV_SIZE_MM=%q REPORT_TIME_LIMIT=%q ARRAY_DEPENDENCY=%q CAMPAIGN_GROUP_ID=%q CAMPAIGN_PART_INDEX=%q CAMPAIGN_PART_COUNT=%q bash submit_slurm/run_brain_expanse_127_production.sh' \
        "$REMOTE_REPO" "$batch_id" "$JOB_COUNT" "$CONCURRENT_LIMIT" "$NUM_LOOPS" "$NUM_CHUNKS" "$SHIELD_MODEL" "$ACTOR_LAYOUT" "$PHYSICS_LIST" "$TIME_LIMIT" "$SRM_FOV_SIZE_MM" "$REPORT_TIME_LIMIT" "$array_dependency" "$CAMPAIGN_GROUP_ID" "$batch_index" "$CAMPAIGN_PART_COUNT"

    printf 'Submitting part %d/%d as %s (jobs=%s, loops=%s x %s chunks, fov=%s mm, shield=%s, concurrent=%s)\n' \
        "$batch_index" "$CAMPAIGN_PART_COUNT" "$batch_id" "$JOB_COUNT" "$NUM_LOOPS" "$NUM_CHUNKS" "$SRM_FOV_SIZE_MM" "$SHIELD_MODEL" "$CONCURRENT_LIMIT" | tee -a "$local_log"
    printf 'ssh %q %q\n' "$REMOTE_HOST" "$remote_command" | tee -a "$local_log"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        remote_output="$(ssh -o BatchMode=yes "$REMOTE_HOST" "$remote_command")"
        printf '%s\n' "$remote_output" | tee -a "$local_log"
        if [[ "$CHAIN_BATCHES" -eq 1 ]]; then
            previous_job_id="$(printf '%s\n' "$remote_output" | sed -n 's/^Submitted progress reporter job \([0-9][0-9]*\).*/\1/p' | tail -1)"
            if [[ -z "$previous_job_id" ]]; then
                echo "Could not determine the reporter job ID; refusing to submit the next chained batch." >&2
                exit 1
            fi
        fi
    fi
done

printf 'Submission log: %s\n' "$local_log"
if [[ "$CHAIN_BATCHES" -eq 1 ]]; then
    printf 'Remote concurrency requested: %d nodes maximum (batches chained)\n' "$CONCURRENT_LIMIT"
    printf 'Batch sequencing: each array waits for the prior reporter job\n'
else
    printf 'Remote concurrency requested: %d nodes maximum\n' "$((BATCH_COUNT * CONCURRENT_LIMIT))"
fi