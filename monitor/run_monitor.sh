#!/bin/bash
# Start the SRM dashboard backend for every campaign on this workstation.
#
# nginx (sites-available/srm-monitor) terminates TLS on 443 and proxies to this
# process, so it binds to loopback only: exposing the raw port would serve the
# dashboard unencrypted alongside the HTTPS vhost.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HOST="${MONITOR_HOST:-127.0.0.1}"
PORT="${MONITOR_PORT:-8765}"
INTERVAL_S="${MONITOR_INTERVAL_S:-30}"
DATA_ROOT="${MONITOR_DATA_ROOT:-/data/fanghan/opengate_sim/data}"
BRAIN_SUBDIR="${MONITOR_BRAIN_SUBDIR:-brain_spect}"
CARDIAC_SUBDIR="${MONITOR_CARDIAC_SUBDIR:-cardiac_spect}"
# Between campaigns there is nothing to serve; wait rather than exiting, so systemd
# does not restart-loop until the next combine produces a report.
WAIT_S="${MONITOR_WAIT_S:-60}"

# Newest batch_* directory that actually has a report to serve.
latest_progress() {
    local root="$1"
    [[ -d "$root" ]] || return 0
    local candidate
    candidate="$(find "$root" -mindepth 2 -maxdepth 2 -name progress.json \
        -printf '%h\n' 2>/dev/null | sort | tail -1)"
    [[ -n "$candidate" ]] && printf '%s/progress.json\n' "$candidate"
    # Finding nothing is normal between campaigns, so never fail under set -e.
    return 0
}

args=()

while true; do
    args=()

    BRAIN_JSON="${MONITOR_BRAIN_JSON:-$(latest_progress "${DATA_ROOT}/${BRAIN_SUBDIR}")}"
    if [[ -n "$BRAIN_JSON" ]]; then
        args+=(--campaign "Brain SPECT (Expanse)=${BRAIN_JSON}")
        echo "Brain campaign:   $BRAIN_JSON"
    fi

    CARDIAC_JSON="${MONITOR_CARDIAC_JSON:-$(latest_progress "${DATA_ROOT}/${CARDIAC_SUBDIR}")}"
    if [[ -n "$CARDIAC_JSON" ]]; then
        args+=(--campaign "Cardiac SPECT (OSPool)=${CARDIAC_JSON}")
        echo "Cardiac campaign: $CARDIAC_JSON"
    fi

    [[ ${#args[@]} -gt 0 ]] && break

    echo "No campaign progress.json under ${DATA_ROOT}; retrying in ${WAIT_S}s."
    sleep "$WAIT_S"
done

exec python3 "$REPO_ROOT/monitor/serve_monitor.py" \
    --host "$HOST" \
    --port "$PORT" \
    --interval-s "$INTERVAL_S" \
    "${args[@]}"
