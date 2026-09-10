#!/usr/bin/env bash
# Submit brain-SPECT production campaign(s) on Expanse from a workstation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${REMOTE_HOST:-expanse}"
REMOTE_REPO="${REMOTE_REPO:-/home/fhan1/qmirt-gate-10-sim}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${SCRIPT_DIR}/logs/remote_submissions}"

SUBMISSION_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BATCH_PREFIX="${BATCH_PREFIX:-brain_40t_${SUBMISSION_TIMESTAMP}}"
CAMPAIGN_GROUP_ID="${CAMPAIGN_GROUP_ID:-$BATCH_PREFIX}"
BATCH_COUNT="${BATCH_COUNT:-1}"
JOB_COUNT="${JOB_COUNT:-126}"
NUM_LOOPS="${NUM_LOOPS:-5}"
TIME_LIMIT="${TIME_LIMIT:-20:00:00}"
REPORT_TIME_LIMIT="${REPORT_TIME_LIMIT:-2-00:00:00}"
CONCURRENT_LIMIT="${CONCURRENT_LIMIT:-}"
CHAIN_BATCHES="${CHAIN_BATCHES:-1}"
DRY_RUN="${DRY_RUN:-0}"

if ! [[ "$BATCH_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_COUNT must be a positive integer" >&2
    exit 2
fi
if ! [[ "$JOB_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "JOB_COUNT must be a positive integer" >&2
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
for ((batch_index = 1; batch_index <= BATCH_COUNT; batch_index++)); do
    printf -v batch_id '%s_%02d' "$BATCH_PREFIX" "$batch_index"
    array_dependency=""
    if [[ "$CHAIN_BATCHES" -eq 1 && -n "$previous_job_id" ]]; then
        array_dependency="afterany:${previous_job_id}"
    fi
    printf -v remote_command \
        'cd -- %q && BATCH_ID=%q JOB_COUNT=%q CONCURRENT_LIMIT=%q NUM_LOOPS=%q TIME_LIMIT=%q REPORT_TIME_LIMIT=%q ARRAY_DEPENDENCY=%q CAMPAIGN_GROUP_ID=%q CAMPAIGN_PART_INDEX=%q CAMPAIGN_PART_COUNT=%q bash submit_slurm/run_brain_expanse_127_production.sh' \
        "$REMOTE_REPO" "$batch_id" "$JOB_COUNT" "$CONCURRENT_LIMIT" "$NUM_LOOPS" "$TIME_LIMIT" "$REPORT_TIME_LIMIT" "$array_dependency" "$CAMPAIGN_GROUP_ID" "$batch_index" "$BATCH_COUNT"

    printf 'Submitting batch %d/%d as %s (jobs=%s, loops=%s, concurrent=%s)\n' \
        "$batch_index" "$BATCH_COUNT" "$batch_id" "$JOB_COUNT" "$NUM_LOOPS" "$CONCURRENT_LIMIT" | tee -a "$local_log"
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