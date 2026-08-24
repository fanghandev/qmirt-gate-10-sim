#!/bin/bash
# List progress-reporter lock files under a root directory and report whether each
# is actively held (a reporter is running) or stale (no reporter currently running).

set -euo pipefail

usage() {
    echo "Usage: $0 <scratch-root-or-campaign-dir> [<scratch-root-or-campaign-dir> ...]"
    echo ""
    echo "Finds every progress.json.reporter.lock under the given roots (e.g. your"
    echo "SCRATCH_ROOT/<sim_name> directory covering many campaigns) and prints whether"
    echo "each one is actively held by a running wrapper_generate_progress_report.sh."
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

printf '%-9s  %s\n' "STATUS" "LOCK FILE"
found_any=0
for root in "$@"; do
    while IFS= read -r -d '' lock_file; do
        found_any=1
        progress_file="${lock_file%.reporter.lock}"
        if flock -n "$lock_file" -c true 2>/dev/null; then
            printf '%-9s  %s\n' "STALE" "$lock_file"
        else
            printf '%-9s  %s\n' "ACTIVE" "$lock_file"
        fi
        [[ -f "$progress_file" ]] && echo "           -> $progress_file"
    done < <(find "$root" -name '*.reporter.lock' -print0 2>/dev/null)
done

if [[ "$found_any" -eq 0 ]]; then
    echo "No reporter lock files found under: $*"
fi
