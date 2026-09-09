#!/usr/bin/env bash
# Compare small production-physics benchmarks on Expanse shared and compute.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${REMOTE_HOST:-expanse}"
REMOTE_REPO="${REMOTE_REPO:-/home/fhan1/qmirt-gate-10-sim}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${SCRIPT_DIR}/logs/remote_submissions}"

mkdir -p "$LOCAL_LOG_DIR"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
local_log="$LOCAL_LOG_DIR/brain_expanse_partition_benchmark_${timestamp}.log"

for partition in shared compute; do
    batch_id="benchmark_${timestamp}_${partition}"
    printf -v remote_command \
        'cd -- %q && BATCH_ID=%q PARTITION=%q bash submit_slurm/run_brain_expanse_127_benchmark.sh' \
        "$REMOTE_REPO" "$batch_id" "$partition"
    printf 'Submitting %s benchmark as %s\n' "$partition" "$batch_id" | tee -a "$local_log"
    ssh -o BatchMode=yes "$REMOTE_HOST" "$remote_command" | tee -a "$local_log"
done

printf 'Remote submission log: %s\n' "$local_log"