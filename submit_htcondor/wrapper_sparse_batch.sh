#!/bin/bash
set -e

CLUSTER_ID=$1
PROC_ID=$2
FOV_SIZE=${3:-150}
PROFILE_RESOURCES=${4:-1}
PROFILE_INTERVAL_S=${5:-5}
shift 5
VOXEL_SIZES=("$@")
if [[ ${#VOXEL_SIZES[@]} -eq 0 ]]; then
    VOXEL_SIZES=(1 1.5 2)
fi

export PYTHONPATH=$PWD/qmirt:$PYTHONPATH

OUTPUT_DIR="."
LOCAL_FILE_LIST="sparse_input_files_${CLUSTER_ID}_${PROC_ID}.txt"
PROFILE_TSV="job_${CLUSTER_ID}_${PROC_ID}_resource_profile.tsv"
PROFILE_SUMMARY="job_${CLUSTER_ID}_${PROC_ID}_resource_profile_summary.txt"

profile_command() {
    local interval_s="$1"
    shift

    local requested_cpus
    requested_cpus="${_CONDOR_NPROCS:-1}"

    printf 'epoch_s\telapsed_s\tthreads\tcpu_pct\tcpu_pct_of_allocation\trss_mb\tread_mb\twrite_mb\n' > "$PROFILE_TSV"

    local start_ts command_pid command_exit
    start_ts="$(date +%s)"
    "$@" &
    command_pid=$!

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
        local now elapsed cpu threads rss read_bytes write_bytes pid allocation
        local process_rss process_read process_write thread_cpu
        local -a pids

        now="$(date +%s)"
        elapsed=$((now - start_ts))
        allocation="$requested_cpus"
        threads=0
        cpu=0
        rss=0
        read_bytes=0
        write_bytes=0

        mapfile -t pids < <(get_descendants "$command_pid" | sort -nu)
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

        awk -v epoch="$now" -v elapsed="$elapsed" -v threads="$threads" \
            -v cpu="$cpu" -v allocation="$allocation" -v rss="$rss" \
            -v read="$read_bytes" -v write="$write_bytes" \
            'BEGIN { printf "%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n", epoch, elapsed, threads, cpu, cpu / allocation, rss / 1024, read / 1048576, write / 1048576 }' \
            >> "$PROFILE_TSV"
    }

    while kill -0 "$command_pid" 2>/dev/null; do
        sample_resources
        sleep "$interval_s"
    done

    wait "$command_pid" || command_exit=$?
    sample_resources

    awk -F '\t' -v requested="$requested_cpus" '
        NR == 1 { next }
        {
            samples++
            cpu += $4
            cpu_alloc += $5
            rss += $6
            read = $7
            write = $8
            if ($3 > peak_threads) peak_threads = $3
            if ($6 > peak_rss) peak_rss = $6
        }
        END {
            if (samples == 0) {
                printf "samples: 0\nrequested_cpus: %s\n", requested
                exit 0
            }
            printf "samples: %d\nrequested_cpus: %s\npeak_observed_threads: %d\naverage_cpu_pct: %.2f\naverage_cpu_pct_of_allocation: %.2f\npeak_rss_mb: %.2f\naverage_rss_mb: %.2f\nfinal_read_mb: %.2f\nfinal_write_mb: %.2f\n", samples, requested, peak_threads, cpu / samples, cpu_alloc / samples, peak_rss, rss / samples, read, write
        }
    ' "$PROFILE_TSV" > "$PROFILE_SUMMARY"

    return "${command_exit:-0}"
}

mkdir -p "$OUTPUT_DIR"

echo "Starting sparse worker job ${CLUSTER_ID} task ${PROC_ID}"

: > "$LOCAL_FILE_LIST"
for archive_path in ./*.tar.gz; do
    [ -e "$archive_path" ] || continue
    printf '%s\n' "$(basename "$archive_path")" >> "$LOCAL_FILE_LIST"
done

if [[ ! -s "$LOCAL_FILE_LIST" ]]; then
    echo "No staged tar.gz inputs found for job ${CLUSTER_ID}.${PROC_ID}"
    exit 1
fi

worker_cmd=(
    python3 payload/python/ospool_sparse_srm_worker.py
    --input-list "$LOCAL_FILE_LIST"
    --output-dir "$OUTPUT_DIR"
    --job-tag "job_${CLUSTER_ID}_${PROC_ID}"
    --fov-size "$FOV_SIZE"
    --voxel-sizes "${VOXEL_SIZES[@]}"
)

if [[ "$PROFILE_RESOURCES" == "1" ]]; then
    echo "Resource profiling enabled (interval ${PROFILE_INTERVAL_S}s)"
    profile_command "$PROFILE_INTERVAL_S" "${worker_cmd[@]}"
else
    printf 'epoch_s\telapsed_s\tthreads\tcpu_pct\tcpu_pct_of_allocation\trss_mb\tread_mb\twrite_mb\n' > "$PROFILE_TSV"
    printf 'profiling_enabled: 0\nreason: disabled by launcher option\n' > "$PROFILE_SUMMARY"
    "${worker_cmd[@]}"
fi

OUTPUT_ARCHIVE="job_${CLUSTER_ID}_${PROC_ID}_sparse_5d_srm_outputs.tar.gz"
tar -czf "$OUTPUT_ARCHIVE" job_${CLUSTER_ID}_${PROC_ID}_*_sparse_5d_srm.npz

echo "Worker outputs bundled in $OUTPUT_DIR/$OUTPUT_ARCHIVE"
echo "Resource profile written to $OUTPUT_DIR/$PROFILE_TSV"
