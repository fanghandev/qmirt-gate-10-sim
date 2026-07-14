#!/bin/bash
set -e

CLUSTER_ID=$1
PROC_ID=$2

# Ensure the qmirt module in the working directory is discoverable
export PYTHONPATH=$PWD/qmirt:$PYTHONPATH

# Define and create a unique output directory for this specific job
OUT_DIR="output_${CLUSTER_ID}_${PROC_ID}"
mkdir -p $OUT_DIR

echo "Starting job $CLUSTER_ID task $PROC_ID..."

# Optimized for 500 million primaries (safely under the 12-hour eviction window)
python3 payload/python/gate_sim_dc_spect_pixelated_w_2mm_box.py \
    -o $OUT_DIR \
    -j $CLUSTER_ID \
    -k $PROC_ID \
    -t 1 \
    -c 1 \
    -d 1.0 \
    --source-activity-bq 500000000

# Compress the output folder to ensure clean file transfer back to the access point
tar -czf results_${CLUSTER_ID}_${PROC_ID}.tar.gz $OUT_DIR