#!/usr/bin/env bash
# Launch an OSPool cardiac-SPECT batch from this workstation over key-only SSH.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${REMOTE_HOST:-ospool}"
REMOTE_REPO="${REMOTE_REPO:-/home/fang.han/qmirt-gate-10-sim}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${SCRIPT_DIR}/logs/remote_submissions}"

mkdir -p "$LOCAL_LOG_DIR"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
local_log="$LOCAL_LOG_DIR/cardiac_ospool_${timestamp}.log"

converter_path="payload/python/create_spect_sparse_srm_from_batch_root.py"
printf -v remote_command \
    'cd -- %q && if ! test -f %q; then echo %q >&2; exit 1; fi && bash submit_htcondor/run_cardiac_sparse_srm_batch.sh' \
    "$REMOTE_REPO" \
    "$converter_path" \
    "Error: remote checkout is missing $converter_path"
for argument in "$@"; do
    printf -v quoted_argument ' %q' "$argument"
    remote_command+="$quoted_argument"
done

ssh -o BatchMode=yes "$REMOTE_HOST" "$remote_command" 2>&1 | tee "$local_log"
printf 'Remote submission log: %s\n' "$local_log"