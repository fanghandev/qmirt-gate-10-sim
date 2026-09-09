#!/bin/bash
# Start the SRM dashboard backend for every campaign on this workstation.
#
# nginx (sites-available/srm-monitor) terminates TLS on 443 and proxies to this
# process, so it binds to loopback only: exposing the raw port would serve the
# dashboard unencrypted alongside the HTTPS vhost.
#
# Campaigns are passed as directory roots rather than files, so the newest batch
# is re-resolved on every poll and a new campaign appears without a restart.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HOST="${MONITOR_HOST:-127.0.0.1}"
PORT="${MONITOR_PORT:-8765}"
INTERVAL_S="${MONITOR_INTERVAL_S:-30}"
DATA_ROOT="${MONITOR_DATA_ROOT:-/data/fanghan/opengate_sim/data}"
CARDIAC_SUBDIR="${MONITOR_CARDIAC_SUBDIR:-cardiac_spect}"

# Brain SPECT reads straight off the Expanse sshfs mount: run_spect_sim_slurm.sh
# --auto-report already regenerates progress.json on the login node every
# minute with full per-task detail, so going through the locally-pulled/derived
# copy under DATA_ROOT only adds staleness and loses detail.
BRAIN_MOUNT_ROOT="${MONITOR_BRAIN_MOUNT_ROOT:-/home/fanghan/sdsc-expanse/brain_spect_sim}"
CARDIAC_ROOT="${DATA_ROOT}/${CARDIAC_SUBDIR}"
mkdir -p "$CARDIAC_ROOT"

echo "Brain campaigns:   newest batch under $BRAIN_MOUNT_ROOT (Expanse mount)"
echo "Cardiac campaigns: newest batch under $CARDIAC_ROOT"

exec python3 "$REPO_ROOT/monitor/serve_monitor.py" \
    --host "$HOST" \
    --port "$PORT" \
    --interval-s "$INTERVAL_S" \
    --campaign-root "Brain SPECT (Expanse)=${BRAIN_MOUNT_ROOT}" \
    --campaign-root "Cardiac SPECT (OSPool)=${CARDIAC_ROOT}"
