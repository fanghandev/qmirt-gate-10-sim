#!/bin/bash

# 1. Generate a unique batch ID
BATCH_ID="sparse_batch_$(date +%Y%m%d_%H%M%S)"

# 2. Define separate data and log directories
DATA_DIR="/ospool/ap40/data/fang.han/${BATCH_ID}"
LOG_DIR="logs/${BATCH_ID}"

# 3. Create both directories
mkdir -p "$DATA_DIR"
mkdir -p "$LOG_DIR"

echo "Created data folder: $DATA_DIR"
echo "Created log folder:  $LOG_DIR"

# 4. Submit the worker jobs
condor_submit batch_sparse.sub out_dir="$DATA_DIR" log_dir="$LOG_DIR"
