#!/bin/bash
set -e

export PYTHONPATH=$PWD/qmirt:$PYTHONPATH

INPUT_DIR=$1
OUTPUT_DIR=$2
VOXEL_SIZE=${3:-}
WORK_DIR="sparse_combine_work"
LOCAL_OUTPUT_DIR="sparse_combine_output"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
rm -rf "$LOCAL_OUTPUT_DIR"
mkdir -p "$LOCAL_OUTPUT_DIR"

echo "Combining sparse SRM chunks from $INPUT_DIR"

found_npz=false
for npz_file in ./*.npz; do
    [ -e "$npz_file" ] || continue
    cp "$npz_file" "$WORK_DIR"/
    found_npz=true
done

for bundle_file in ./*.tar.gz; do
    [ -e "$bundle_file" ] || continue
    tar -xzf "$bundle_file" -C "$WORK_DIR"
    found_npz=true
done

if [[ "$found_npz" != true ]]; then
    echo "No sparse NPZ files found in: $INPUT_DIR"
    exit 1
fi

COMBINE_ARGS=(
    --input-dir "$WORK_DIR"
    --output-dir "$LOCAL_OUTPUT_DIR"
)
if [[ -n "$VOXEL_SIZE" ]]; then
    COMBINE_ARGS+=(--voxel-size "$VOXEL_SIZE")
fi

python3 payload/python/ospool_sparse_srm_combine.py "${COMBINE_ARGS[@]}"

cp "$LOCAL_OUTPUT_DIR/sparse_5d_histograms.npz" combined_sparse_5d_srm.npz

echo "Combined sparse SRM written to $LOCAL_OUTPUT_DIR/sparse_5d_histograms.npz for remap to $OUTPUT_DIR"
