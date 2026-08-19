#!/bin/bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 output_dir interval_seconds command [args...]" >&2
    exit 2
fi

OUTPUT_DIR="$1"
INTERVAL_S="$2"
shift 2

mkdir -p "$OUTPUT_DIR"
SAMPLE_FILE="$OUTPUT_DIR/resource_profile.tsv"
SUMMARY_FILE="$OUTPUT_DIR/resource_profile_summary.txt"
REQUESTED_CPUS="${SLURM_CPUS_PER_TASK:-1}"

printf 'epoch_s\telapsed_s\tthreads\tcpu_pct\tcpu_pct_of_allocation\trss_mb\tread_mb\twrite_mb\tfs_used_pct\n' > "$SAMPLE_FILE"

START_TS="$(date +%s)"
"$@" &
COMMAND_PID=$!

get_descendants() {
    local parent_pid="$1"
    local child_pid
    printf '%s\n' "$parent_pid"
    while read -r child_pid; do
        [[ -n "$child_pid" ]] || continue
        get_descendants "$child_pid"
    done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
}

sample_resources() {
    local now elapsed allocation cpu threads rss read_bytes write_bytes pid
    local -a pids
    now="$(date +%s)"
    elapsed=$((now - START_TS))
    allocation="$REQUESTED_CPUS"
    threads=0
    cpu=0
    rss=0
    read_bytes=0
    write_bytes=0

    mapfile -t pids < <(get_descendants "$COMMAND_PID" | sort -nu)
    for pid in "${pids[@]}"; do
        while read -r thread_cpu; do
            [[ "$thread_cpu" =~ ^[0-9]+([.][0-9]+)?$ ]] || continue
            cpu="$(awk -v total="$cpu" -v value="$thread_cpu" 'BEGIN { printf "%.2f", total + value }')"
            threads=$((threads + 1))
        done < <(ps -L -p "$pid" -o pcpu= 2>/dev/null || true)

        process_rss="$(awk '/^VmRSS:/ { print $2; exit }' "/proc/$pid/status" 2>/dev/null || true)"
        read -r process_read process_write < <(
            awk '
                /^read_bytes:/ { read=$2 }
                /^write_bytes:/ { write=$2 }
                END { printf "%s %s", read + 0, write + 0 }
            ' "/proc/$pid/io" 2>/dev/null || true
        ) || true
        rss=$((rss + ${process_rss:-0}))
        read_bytes=$((read_bytes + ${process_read:-0}))
        write_bytes=$((write_bytes + ${process_write:-0}))
    done

    local fs_used_pct
    fs_used_pct="$(df -P "$OUTPUT_DIR" | awk 'NR == 2 { gsub(/%/, "", $5); print $5 }')"
    awk -v epoch="$now" -v elapsed="$elapsed" -v threads="$threads" \
        -v cpu="$cpu" -v allocation="$allocation" -v rss="$rss" \
        -v read="$read_bytes" -v write="$write_bytes" -v fs="$fs_used_pct" \
        'BEGIN { printf "%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s\n", epoch, elapsed, threads, cpu, cpu / allocation, rss / 1024, read / 1048576, write / 1048576, fs }' \
        >> "$SAMPLE_FILE"
}

trap 'sample_resources' USR1
while kill -0 "$COMMAND_PID" 2>/dev/null; do
    sample_resources
    sleep "$INTERVAL_S"
done

wait "$COMMAND_PID" || COMMAND_EXIT=$?
sample_resources

awk -F '\t' -v requested="$REQUESTED_CPUS" '
    NR == 1 { next }
    { samples++; cpu += $4; cpu_alloc += $5; rss += $6; read = $7; write = $8; fs = $9; if ($3 > peak_threads) peak_threads = $3; if ($6 > peak_rss) peak_rss = $6 }
    END {
        if (samples == 0) exit 1
        printf "samples: %d\nrequested_cpus: %s\npeak_observed_threads: %d\naverage_cpu_pct: %.2f\naverage_cpu_pct_of_allocation: %.2f\npeak_rss_mb: %.2f\naverage_rss_mb: %.2f\nfinal_read_mb: %.2f\nfinal_write_mb: %.2f\nfinal_filesystem_used_pct: %s\n", samples, requested, peak_threads, cpu / samples, cpu_alloc / samples, peak_rss, rss / samples, read, write, fs
    }
' "$SAMPLE_FILE" > "$SUMMARY_FILE"

exit "${COMMAND_EXIT:-0}"