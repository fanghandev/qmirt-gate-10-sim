#!/bin/bash

# 1. Pass the worker output directory explicitly.
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <worker_output_dir>"
    exit 1
fi

INPUT_DIR="$1"
COMBINE_BATCH_ID="sparse_combine_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/ospool/ap40/data/fang.han/${COMBINE_BATCH_ID}"
LOG_DIR="logs/${COMBINE_BATCH_ID}"
INPUT_FILES=""

if compgen -G "$INPUT_DIR"'/*.npz' > /dev/null; then
    INPUT_FILES=$(printf '%s,' "$INPUT_DIR"/*.npz)
    INPUT_FILES=${INPUT_FILES%,}
fi

if [[ -z "$INPUT_FILES" ]]; then
    echo "No sparse SRM NPZ files found in: $INPUT_DIR"
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "Combining sparse SRM chunks from: $INPUT_DIR"
echo "Writing final sparse SRM to:      $OUTPUT_DIR"

condor_submit combine_sparse.sub input_dir="$INPUT_DIR" output_dir="$OUTPUT_DIR" log_dir="$LOG_DIR" input_files="$INPUT_FILES"
