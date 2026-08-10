#!/bin/bash
set -e

CLUSTER_ID=$1
PROC_ID=$2

export PYTHONPATH=$PWD/qmirt:$PYTHONPATH

OUTPUT_DIR="."
LOCAL_FILE_LIST="sparse_input_files_${CLUSTER_ID}_${PROC_ID}.txt"

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

python3 payload/python/ospool_sparse_srm_worker.py \
    --input-list "$LOCAL_FILE_LIST" \
    --output-dir "$OUTPUT_DIR" \
    --job-tag "job_${CLUSTER_ID}_${PROC_ID}" \
    --hist-min -105 --hist-max 105 --hist-bins 210

echo "Worker output written to $OUTPUT_DIR/job_${CLUSTER_ID}_${PROC_ID}_sparse_5d_srm.npz"
