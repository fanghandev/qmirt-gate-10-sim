#!/bin/bash
# Periodically regenerate a campaign's progress.json until its Slurm array job
# (and optional combine job) reaches a terminal state, then do one final refresh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if command -v module >/dev/null 2>&1; then
    module load Apptainer 2>/dev/null || true
fi

CONTAINER_SIF="${CONTAINER_SIF:-${REPO_ROOT}/submit_slurm/qmirt-gate-10-sim-sif_v1.0.0.sif}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-}"
EXPECTED_TASKS="${EXPECTED_TASKS:-0}"
JOB_ID="${JOB_ID:-}"
WATCH_JOB_ID="${WATCH_JOB_ID:-${JOB_ID}}"
OUTPUT_FILE="${OUTPUT_FILE:-}"
INTERVAL_S="${INTERVAL_S:-60}"
MAX_DURATION_S="${MAX_DURATION_S:-0}"
GEOMETRY_WRL="${GEOMETRY_WRL:-}"

usage() {
    echo "Usage: $0 --campaign-dir DIR --output FILE [--expected-tasks N] [--job-id ID]"
    echo "          [--watch-job-id ID] [--interval-s SECONDS] [--max-duration-s SECONDS]"
    echo "          [--geometry-wrl FILE]"
    echo ""
    echo "Regenerates progress.json every --interval-s seconds. If --watch-job-id is"
    echo "given (defaults to --job-id), stops once that Slurm job leaves the queue,"
    echo "after one final refresh. Intended to run in the background on the login node,"
    echo "e.g. launched with 'nohup ... &' by run_spect_sim_slurm.sh --auto-report."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --campaign-dir) CAMPAIGN_DIR="$2"; shift 2 ;;
        --expected-tasks) EXPECTED_TASKS="$2"; shift 2 ;;
        --job-id) JOB_ID="$2"; WATCH_JOB_ID="${WATCH_JOB_ID:-$2}"; shift 2 ;;
        --watch-job-id) WATCH_JOB_ID="$2"; shift 2 ;;
        --output) OUTPUT_FILE="$2"; shift 2 ;;
        --interval-s) INTERVAL_S="$2"; shift 2 ;;
        --max-duration-s) MAX_DURATION_S="$2"; shift 2 ;;
        --geometry-wrl) GEOMETRY_WRL="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unexpected argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$CAMPAIGN_DIR" || -z "$OUTPUT_FILE" ]]; then
    echo "Error: --campaign-dir and --output are required" >&2
    exit 2
fi
if ! [[ "$INTERVAL_S" =~ ^[0-9]+$ ]] || [[ "$INTERVAL_S" -lt 1 ]]; then
    echo "interval_s must be a positive integer" >&2
    exit 2
fi

# Refuse to start a second writer for the same output file: write_atomic()'s
# temp-file-then-rename only protects readers, not two concurrent writers.
LOCK_FILE="${OUTPUT_FILE}.reporter.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "Error: another progress reporter already holds the lock for $OUTPUT_FILE (see $LOCK_FILE)" >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT/qmirt/src${PYTHONPATH:+:$PYTHONPATH}"

report_cmd=(python3 "$REPO_ROOT/payload/python/report_campaign_progress.py"
    --campaign-dir "$CAMPAIGN_DIR" --expected-tasks "$EXPECTED_TASKS" --output "$OUTPUT_FILE")
[[ -n "$JOB_ID" ]] && report_cmd+=(--job-id "$JOB_ID")
[[ -n "$GEOMETRY_WRL" ]] && report_cmd+=(--geometry-wrl "$GEOMETRY_WRL")

if [[ -f "$CONTAINER_SIF" ]] && command -v apptainer >/dev/null 2>&1; then
    report_cmd=(apptainer exec --bind "$REPO_ROOT:$REPO_ROOT" --bind "$CAMPAIGN_DIR:$CAMPAIGN_DIR" \
        "$CONTAINER_SIF" "${report_cmd[@]}")
fi

job_is_active() {
    [[ -n "$WATCH_JOB_ID" ]] || return 1
    command -v squeue >/dev/null 2>&1 || return 1
    [[ -n "$(squeue -h -j "$WATCH_JOB_ID" 2>/dev/null)" ]]
}

start_ts="$(date +%s)"
echo "Progress reporter: campaign=$CAMPAIGN_DIR output=$OUTPUT_FILE interval=${INTERVAL_S}s watch_job=${WATCH_JOB_ID:-none}"
while true; do
    "${report_cmd[@]}" || echo "Warning: report generation failed, will retry next interval" >&2

    if [[ -n "$WATCH_JOB_ID" ]] && ! job_is_active; then
        echo "Watched job ${WATCH_JOB_ID} left the queue; writing one final report and exiting."
        "${report_cmd[@]}" || true
        break
    fi
    if [[ "$MAX_DURATION_S" =~ ^[0-9]+$ ]] && [[ "$MAX_DURATION_S" -gt 0 ]]; then
        elapsed="$(( $(date +%s) - start_ts ))"
        if [[ "$elapsed" -ge "$MAX_DURATION_S" ]]; then
            echo "Reached --max-duration-s (${MAX_DURATION_S}s); exiting."
            break
        fi
    fi
    sleep "$INTERVAL_S"
done
