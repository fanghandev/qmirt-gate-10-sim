#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"
CARDIAC_ROOT="${CARDIAC_ROOT:-/data/fanghan/opengate_sim/data/cardiac_spect}"
BRAIN_ROOT="${BRAIN_ROOT:-/data/fanghan/opengate_sim/data/brain_spect}"
# Cardiac tarball reads are storage-bound; avoid saturating the filesystem by default.
CARDIAC_WORKERS="${CARDIAC_WORKERS:-16}"
CARDIAC_GROUP_SIZE="${CARDIAC_GROUP_SIZE:-20}"
CARDIAC_LIMIT="${CARDIAC_LIMIT:-1000000}"
BRAIN_HEADS="${BRAIN_HEADS:-73}"
BRAIN_PIXELS="${BRAIN_PIXELS:-625}"

action="run"
usage() {
    cat <<EOF
Usage: $0 [options]

Runs the requested cardiac campaigns first, then the requested brain campaigns.

Options:
  --cardiac-workers N       Cardiac tarball reader threads (default: $CARDIAC_WORKERS)
  --cardiac-group-size N    Cardiac merge group size (default: $CARDIAC_GROUP_SIZE)
  --cardiac-limit N         Maximum cardiac jobs per campaign (default: $CARDIAC_LIMIT)
  --brain-heads N            Brain detector heads (default: $BRAIN_HEADS)
  --brain-pixels N           Brain pixels per head (default: $BRAIN_PIXELS)
  --python PATH              Python executable (default: $PYTHON)
  --dry-run                  Print commands without running them
  -h, --help                 Show this help

Environment overrides:
  CARDIAC_ROOT, BRAIN_ROOT, CARDIAC_WORKERS, CARDIAC_GROUP_SIZE,
  CARDIAC_LIMIT, BRAIN_HEADS, BRAIN_PIXELS, PYTHON
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cardiac-workers) CARDIAC_WORKERS="$2"; shift 2 ;;
        --cardiac-group-size) CARDIAC_GROUP_SIZE="$2"; shift 2 ;;
        --cardiac-limit) CARDIAC_LIMIT="$2"; shift 2 ;;
        --brain-heads) BRAIN_HEADS="$2"; shift 2 ;;
        --brain-pixels) BRAIN_PIXELS="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --dry-run) action="dry-run"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unexpected argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value_name in CARDIAC_WORKERS CARDIAC_GROUP_SIZE CARDIAC_LIMIT BRAIN_HEADS BRAIN_PIXELS; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value_name must be a positive integer: $value" >&2
        exit 2
    fi
done

run_command() {
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    if [[ "$action" == "run" ]]; then
        "$@"
    fi
}

run_cardiac() {
    local batch_name="$1"
    local cluster_id="$2"
    local campaign_dir="$CARDIAC_ROOT/$batch_name"
    local db_path="$campaign_dir/ospool_condor_jobs.db"
    local partial_dir="$campaign_dir/partials/targz"

    [[ -d "$campaign_dir" ]] || {
        echo "Missing cardiac campaign directory: $campaign_dir" >&2
        exit 1
    }
    [[ -f "$db_path" ]] || {
        echo "Missing cardiac database: $db_path" >&2
        exit 1
    }
    [[ -d "$partial_dir" ]] || {
        echo "Missing cardiac tarball directory: $partial_dir" >&2
        exit 1
    }

    echo "=== Cardiac: $batch_name (ClusterId $cluster_id) ==="
    run_command "$PYTHON" "$REPO_ROOT/payload/python/merge_cardiac_campaigns.py" \
        --name "$batch_name" \
        --cluster-id "$cluster_id" \
        --data-root "$CARDIAC_ROOT" \
        --workers "$CARDIAC_WORKERS" \
        --group-size "$CARDIAC_GROUP_SIZE" \
        --limit "$CARDIAC_LIMIT"
}

run_brain() {
    local campaign_name="$1"
    local campaign_dir="$BRAIN_ROOT/$campaign_name"

    [[ -d "$campaign_dir" ]] || {
        echo "Missing brain campaign directory: $campaign_dir" >&2
        exit 1
    }

    echo "=== Brain: $campaign_name ==="
    run_command "$PYTHON" "$REPO_ROOT/payload/python/merge_brain_campaigns.py" \
        --directory "$campaign_dir" \
        --output-dir "$campaign_dir" \
        --heads "$BRAIN_HEADS" \
        --pixels "$BRAIN_PIXELS"
}

run_cardiac batch_20260911_104934 15295120
run_cardiac batch_20260909_012610 14996562
run_brain brain_40t_20260910T020837Z_03
run_brain brain_40t_20260910T020837Z_02

echo "All requested campaign merges completed."
