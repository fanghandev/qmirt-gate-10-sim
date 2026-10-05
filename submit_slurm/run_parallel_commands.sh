#!/bin/bash
# Run every line of a command file as a background process and wait for all.
# Used by the brain wrapper to run one Gate process per NUMA domain (see
# numa_layout.py). Exits 0 only if every command succeeded; on TERM/INT (SLURM time
# limit, `timeout`) the whole process tree of every command is terminated.
set -uo pipefail

if [[ $# -ne 1 || ! -f "$1" ]]; then
    echo "Usage: $0 command_file" >&2
    exit 2
fi
COMMAND_FILE="$1"

pids=()

descendants() {
    local pid="$1" child
    for child in $(pgrep -P "$pid" 2>/dev/null); do
        descendants "$child"
    done
    printf '%s\n' "$pid"
}

terminate_all() {
    local pid
    for pid in "${pids[@]}"; do
        # children first, so nothing is re-parented and missed
        kill -TERM $(descendants "$pid") 2>/dev/null || true
    done
    wait 2>/dev/null
    exit 143
}
trap terminate_all TERM INT

while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -n "$line" ]] || continue
    bash -c "$line" &
    pids+=("$!")
done < "$COMMAND_FILE"

status=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "Parallel command $index failed: $(sed -n "$((index + 1))p" "$COMMAND_FILE" | cut -c1-240)" >&2
        status=1
    fi
done
exit "$status"
