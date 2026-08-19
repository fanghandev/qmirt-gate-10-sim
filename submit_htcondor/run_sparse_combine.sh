#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# 1. Pass the worker output directory explicitly.
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <worker_output_dir> [output_dir] [voxel_size]"
    exit 1
fi

INPUT_DIR="$1"
COMBINE_BATCH_ID="sparse_combine_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${2:-/ospool/ap40/data/fang.han/${COMBINE_BATCH_ID}}"
LOG_DIR="logs/${COMBINE_BATCH_ID}"
VOXEL_SIZE="${3:-}"

shopt -s nullglob
INPUT_PATHS=("$INPUT_DIR"/*.npz "$INPUT_DIR"/*.tar.gz)
shopt -u nullglob

if [[ ${#INPUT_PATHS[@]} -eq 0 ]]; then
    echo "No sparse SRM NPZ files found in: $INPUT_DIR"
    exit 1
fi

IFS=,; INPUT_FILES="${INPUT_PATHS[*]}"; unset IFS

mkdir -p "$LOG_DIR"

echo "Combining sparse SRM chunks from: $INPUT_DIR"
echo "Writing final sparse SRM to:      $OUTPUT_DIR"

condor_submit combine_sparse.sub \
    input_dir="$INPUT_DIR" \
    output_dir="$OUTPUT_DIR" \
    log_dir="$LOG_DIR" \
    input_files="$INPUT_FILES" \
    voxel_size="$VOXEL_SIZE"
