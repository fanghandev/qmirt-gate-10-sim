

#!/bin/bash
#
# test_job_task_identifiers.sh
# =============================
# Wrapper script that calls the Python payload.
#
# Usage:
#   ./test_job_task_identifiers.sh
#
# Or via sbatch-wrapper:
#   sbatch-wrapper.sh -N "slurm-id-test" -p normal -a "0-2" -o ./logs -- ./test_job_task_identifiers.sh
#
# Output will be saved to $OUTPUT_DIR/$SLURM_ARRAY_JOB_ID (set by sbatch-wrapper)

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set default output directory
BASE_OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# Get effective job ID (SLURM_ARRAY_JOB_ID if in array job, otherwise SLURM_JOB_ID, otherwise "local")
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"

# Create subdirectory under OUTPUT_DIR with job ID
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

# Export OUTPUT_DIR for the Python script
export OUTPUT_DIR

# Call the Python payload script
"${SCRIPT_DIR}/test_job_task_identifiers.py"
