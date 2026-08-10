#!/bin/bash
set -e

export PYTHONPATH=$PWD/qmirt:$PYTHONPATH

INPUT_DIR=$1
OUTPUT_DIR=$2
WORK_DIR="sparse_combine_work"
LOCAL_OUTPUT_DIR="sparse_combine_output"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
rm -rf "$LOCAL_OUTPUT_DIR"
mkdir -p "$LOCAL_OUTPUT_DIR"

echo "Combining sparse SRM chunks from $INPUT_DIR"

found_npz=false
for npz_file in "$INPUT_DIR"/*.npz; do
    [ -e "$npz_file" ] || continue
    cp "$npz_file" "$WORK_DIR"/
    found_npz=true
done

if [[ "$found_npz" != true ]]; then
    echo "No sparse NPZ files found in: $INPUT_DIR"
    exit 1
fi

python3 payload/python/ospool_sparse_srm_combine.py \
    --input-dir "$WORK_DIR" \
    --output-dir "$LOCAL_OUTPUT_DIR"

echo "Combined sparse SRM written to $LOCAL_OUTPUT_DIR/sparse_5d_histograms.npz for remap to $OUTPUT_DIR"
