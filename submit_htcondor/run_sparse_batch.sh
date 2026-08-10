#!/bin/bash

# 1. Generate a unique batch ID
BATCH_ID="sparse_batch_$(date +%Y%m%d_%H%M%S)"
FILES_PER_JOB=100
JOB_COUNT=100

# 2. Define separate data and log directories
DATA_DIR="/ospool/ap40/data/fang.han/${BATCH_ID}"
LOG_DIR="logs/${BATCH_ID}"
BATCH_DIR="sparse_input_lists/${BATCH_ID}"
INPUT_MANIFEST="${BATCH_DIR}/input_files.txt"

# 3. Create both directories
mkdir -p "$DATA_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$BATCH_DIR"

mapfile -t SOURCE_FILES < filenames.txt
: > "$INPUT_MANIFEST"

for ((job_index = 0; job_index < JOB_COUNT; job_index++)); do
	start_index=$((job_index * FILES_PER_JOB))
	slice=(${SOURCE_FILES[@]:start_index:FILES_PER_JOB})
	[[ ${#slice[@]} -gt 0 ]] || break
	(IFS=,; echo "${slice[*]}") >> "$INPUT_MANIFEST"
done

echo "Created data folder: $DATA_DIR"
echo "Created log folder:  $LOG_DIR"
echo "Created input manifest: $INPUT_MANIFEST"

# 4. Submit the worker jobs
condor_submit batch_sparse.sub out_dir="$DATA_DIR" log_dir="$LOG_DIR" input_manifest="$INPUT_MANIFEST"
