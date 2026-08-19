#!/bin/bash
set -e

CLUSTER_ID=$1
PROC_ID=$2
FOV_SIZE=${3:-150}
shift 3
VOXEL_SIZES=("$@")
if [[ ${#VOXEL_SIZES[@]} -eq 0 ]]; then
    VOXEL_SIZES=(1 1.5 2)
fi

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
    --fov-size "$FOV_SIZE" \
    --voxel-sizes "${VOXEL_SIZES[@]}"

OUTPUT_ARCHIVE="job_${CLUSTER_ID}_${PROC_ID}_sparse_5d_srm_outputs.tar.gz"
tar -czf "$OUTPUT_ARCHIVE" job_${CLUSTER_ID}_${PROC_ID}_*_sparse_5d_srm.npz

echo "Worker outputs bundled in $OUTPUT_DIR/$OUTPUT_ARCHIVE"
