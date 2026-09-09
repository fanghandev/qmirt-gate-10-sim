#!/usr/bin/env bash
# Launch the Expanse 127-thread brain-SPECT benchmark from this workstation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${REMOTE_HOST:-expanse}"
REMOTE_REPO="${REMOTE_REPO:-~/qmirt-gate-10-sim}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${SCRIPT_DIR}/logs/remote_submissions}"

mkdir -p "$LOCAL_LOG_DIR"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
local_log="$LOCAL_LOG_DIR/brain_expanse_127_benchmark_${timestamp}.log"

ssh -o BatchMode=yes "$REMOTE_HOST" \
    "cd $REMOTE_REPO && bash submit_slurm/run_brain_expanse_127_benchmark.sh" \
    | tee "$local_log"

printf 'Remote submission log: %s\n' "$local_log"