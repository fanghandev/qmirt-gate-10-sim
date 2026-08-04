#!/bin/bash
set -e

CLUSTER_ID=$1
PROC_ID=$2

export PYTHONPATH=$PWD/qmirt:$PYTHONPATH
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
fi
. .venv/bin/activate
python -m pip install --no-input -r requirements_sparse.txt

FILE_LIST="filenames.txt"
OUTPUT_DIR="sparse_srm_chunks_${CLUSTER_ID}"
FILES_PER_JOB=100
START_INDEX=$((PROC_ID * FILES_PER_JOB))
END_INDEX=$((START_INDEX + FILES_PER_JOB))

mkdir -p "$OUTPUT_DIR"

echo "Starting sparse worker job ${CLUSTER_ID} task ${PROC_ID}"
echo "File slice: ${START_INDEX}..${END_INDEX}"

python3 payload/python/ospool_sparse_srm_worker.py \
    --input-list "$FILE_LIST" \
    --output-dir "$OUTPUT_DIR" \
    --start-index "$START_INDEX" \
    --end-index "$END_INDEX" \
    --job-tag "job_${CLUSTER_ID}_${PROC_ID}"

tar -czf "sparse_srm_chunks_${CLUSTER_ID}_${PROC_ID}.tar.gz" -C "$OUTPUT_DIR" "job_${CLUSTER_ID}_${PROC_ID}"
