#!/bin/bash
# Point globus.env at a campaign batch, deriving every per-batch value from the mount.
#
# The four path keys share a common root and differ only in their trailing batch id, so
# the roots are reused from the existing config and only the batch component is rewritten.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/globus.env"
BATCH=""
USE_LATEST=0
LIST_ONLY=0

usage() {
    echo "Usage: $0 [--batch BATCH_ID | --latest | --list] [--config FILE]"
    echo ""
    echo "Rewrites QMIRT_SRC_PATH, QMIRT_MOUNT_PATH, QMIRT_DST_PATH, QMIRT_LOCAL_PATH and"
    echo "QMIRT_EXPECTED_TASKS in the config for the chosen campaign batch."
    echo "The task count is read from the campaign manifest on the mount."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch)
            [[ $# -ge 2 ]] || { echo "Missing value for --batch" >&2; exit 2; }
            BATCH="$2"; shift 2 ;;
        --latest) USE_LATEST=1; shift ;;
        --list) LIST_ONLY=1; shift ;;
        --config)
            [[ $# -ge 2 ]] || { echo "Missing value for --config" >&2; exit 2; }
            CONFIG="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unexpected argument: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config not found: $CONFIG" >&2
    echo "Copy submit_slurm/globus.env.example to submit_slurm/globus.env first." >&2
    exit 1
fi

read_key() {
    sed -n "s/^[[:space:]]*$1=//p" "$CONFIG" | tail -n 1 | tr -d '"'"'"''
}

MOUNT_PATH="$(read_key QMIRT_MOUNT_PATH)"
SRC_PATH="$(read_key QMIRT_SRC_PATH)"
DST_PATH="$(read_key QMIRT_DST_PATH)"
LOCAL_PATH="$(read_key QMIRT_LOCAL_PATH)"

for key in QMIRT_MOUNT_PATH:"$MOUNT_PATH" QMIRT_SRC_PATH:"$SRC_PATH" \
           QMIRT_DST_PATH:"$DST_PATH" QMIRT_LOCAL_PATH:"$LOCAL_PATH"; do
    if [[ -z "${key#*:}" ]]; then
        echo "Error: ${key%%:*} is not set in $CONFIG" >&2
        exit 1
    fi
done

MOUNT_ROOT="$(dirname "$MOUNT_PATH")"
if [[ ! -d "$MOUNT_ROOT" ]]; then
    echo "Error: mount root is not readable (stale mount?): $MOUNT_ROOT" >&2
    exit 1
fi

list_batches() {
    find "$MOUNT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'batch_*' -printf '%f\n' 2>/dev/null | sort
}

if [[ "$LIST_ONLY" -eq 1 ]]; then
    list_batches
    exit 0
fi

if [[ "$USE_LATEST" -eq 1 ]]; then
    BATCH="$(list_batches | tail -n 1)"
    [[ -n "$BATCH" ]] || { echo "Error: no batch_* directories under $MOUNT_ROOT" >&2; exit 1; }
fi

if [[ -z "$BATCH" ]]; then
    usage
    exit 2
fi

CAMPAIGN_MOUNT="${MOUNT_ROOT}/${BATCH}"
if [[ ! -d "$CAMPAIGN_MOUNT" ]]; then
    echo "Error: campaign not found on the mount: $CAMPAIGN_MOUNT" >&2
    echo "Available batches:" >&2
    list_batches >&2
    exit 1
fi

# Prefer the job_count recorded at submission over counting directories, which
# would undercount while the array is still running.
EXPECTED_TASKS=""
MANIFEST="${CAMPAIGN_MOUNT}/campaign_manifest.json"
if [[ -f "$MANIFEST" ]]; then
    EXPECTED_TASKS="$(sed -n 's/.*"job_count"[[:space:]]*:[[:space:]]*\([0-9]\{1,\}\).*/\1/p' "$MANIFEST" | head -n 1)"
fi
if [[ -z "$EXPECTED_TASKS" ]]; then
    EXPECTED_TASKS="$(find "$CAMPAIGN_MOUNT" -mindepth 1 -maxdepth 1 -type d -name 'task_*' | wc -l)"
    echo "Warning: no campaign manifest; falling back to ${EXPECTED_TASKS} observed task directories."
fi

set_key() {
    local key="$1" value="$2" file="$3"
    if grep -q "^[[:space:]]*${key}=" "$file"; then
        python3 - "$file" "$key" "$value" <<'PY'
import re, sys
path, key, value = sys.argv[1:4]
with open(path, encoding="utf-8") as handle:
    text = handle.read()
text = re.sub(rf"^\s*{re.escape(key)}=.*$", f"{key}={value}", text, flags=re.M)
with open(path, "w", encoding="utf-8") as handle:
    handle.write(text)
PY
    else
        printf '%s=%s\n' "$key" "$value" >> "$file"
    fi
}

TEMP_CONFIG="$(mktemp "${CONFIG}.XXXXXX")"
cp "$CONFIG" "$TEMP_CONFIG"
set_key QMIRT_MOUNT_PATH "${MOUNT_ROOT}/${BATCH}" "$TEMP_CONFIG"
set_key QMIRT_SRC_PATH "$(dirname "$SRC_PATH")/${BATCH}" "$TEMP_CONFIG"
set_key QMIRT_DST_PATH "$(dirname "$DST_PATH")/${BATCH}" "$TEMP_CONFIG"
set_key QMIRT_LOCAL_PATH "$(dirname "$LOCAL_PATH")/${BATCH}" "$TEMP_CONFIG"
set_key QMIRT_EXPECTED_TASKS "$EXPECTED_TASKS" "$TEMP_CONFIG"
mv "$TEMP_CONFIG" "$CONFIG"

READY="$(find "$CAMPAIGN_MOUNT" -mindepth 2 -maxdepth 2 -name TASK_COMPLETE.json | wc -l)"

echo "Campaign:       $BATCH"
echo "Mount path:     ${MOUNT_ROOT}/${BATCH}"
echo "Globus source:  $(dirname "$SRC_PATH")/${BATCH}"
echo "Local landing:  $(dirname "$LOCAL_PATH")/${BATCH}"
echo "Expected tasks: $EXPECTED_TASKS"
echo "Tasks ready:    $READY"
echo "Updated:        $CONFIG"
