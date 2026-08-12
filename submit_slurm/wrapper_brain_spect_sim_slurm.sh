#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${SLURM_PROCID:-0}}"
OUT_DIR="${OUTPUT_DIR:-output_${JOB_ID}_${TASK_ID}}"

mkdir -p "$OUT_DIR"

echo "Starting SLURM job $JOB_ID task $TASK_ID..."

python3 "payload/python/gate_sim_brain_spect_no_boolean.py" \
    -o "$OUT_DIR" \
    -j "$JOB_ID" \
    -k "$TASK_ID" \
    --execution-environment slurm \
    -n "${SLURM_CPUS_PER_TASK:-1}" \
    "$@"
