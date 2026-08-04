#!/bin/bash

# 1. Pass the worker output directory and final output directory explicitly.
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <worker_output_dir> <final_output_dir>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)_combine"

mkdir -p "$LOG_DIR"

echo "Combining sparse SRM chunks from: $INPUT_DIR"
echo "Writing final sparse SRMs to:     $OUTPUT_DIR"

condor_submit combine_sparse.sub input_dir="$INPUT_DIR" output_dir="$OUTPUT_DIR" log_dir="$LOG_DIR"
