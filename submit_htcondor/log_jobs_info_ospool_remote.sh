#!/usr/bin/env bash
# Logs submitted jobs info with given cluster id
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Make sure cluster ID is passed
if [[ $# -ne 1 ]]; then
    echo "Error: Cluster Id is required." >&2
    echo "Usage: $0 <Cluster Id>" >&2
    exit 1
fi

CLUSTER_ID="$1"

# Verify Cluster ID is a valid number using a regular expression
if ! [[ "$CLUSTER_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: '$CLUSTER_ID' is not a valid Cluster ID." >&2
    exit 1
fi

# Build the query string to run remotely (combines active and history queues)
# Append ',' to the autoformat flags (-af:h, and -af:,) to generate CSV output

printf -v CURRENT_CONDOR_JOB_QUERY "condor_q %s -af:h, ClusterId ProcId Cmd JobStatus 'formatTime(QDate)' 'formatTime(JobStartDate)' 'formatTime(CompletionDate)' ; condor_history %s -af:, ClusterId ProcId Cmd JobStatus 'formatTime(QDate)' 'formatTime(JobStartDate)' 'formatTime(CompletionDate)'" "$CLUSTER_ID" "$CLUSTER_ID"

# Define local paths
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${SCRIPT_DIR}/logs/remote_submissions}"
LOG_FILE_NAME="ospool_${CLUSTER_ID}_condor_jobs.log"
FULL_LOG_PATH="${LOCAL_LOG_DIR}/${LOG_FILE_NAME}"

# Ensure the local log directory exists before attempting to write to it
mkdir -p "$LOCAL_LOG_DIR"

echo "Querying OSPool for Cluster ID: $CLUSTER_ID..."
echo "Running: $CURRENT_CONDOR_JOB_QUERY"

# Execute remotely and save locally
ssh -o BatchMode=yes ospool "$CURRENT_CONDOR_JOB_QUERY" > "$FULL_LOG_PATH"

echo "Done! Job list saved to: $FULL_LOG_PATH"