#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

JOB_COUNT="${1:-100}"
CPUS_PER_TASK="${2:-1}"

if ! [[ "$JOB_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [job_count] [cpus_per_task]"
    echo "job_count must be a positive integer"
    exit 1
fi

if ! [[ "$CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [job_count] [cpus_per_task]"
    echo "cpus_per_task must be a positive integer"
    exit 1
fi

BATCH_ID="batch_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="submit_slurm/logs/${BATCH_ID}"
DATA_DIR="results/cardiac_spect/slurm/${BATCH_ID}"
SBATCH_FILE="${LOG_DIR}/cardiac_spect_sim_slurm.sbatch"

mkdir -p "$LOG_DIR" "$DATA_DIR"

cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=cardiac-spect
#SBATCH --array=0-$((JOB_COUNT - 1))
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --output=${LOG_DIR}/job_%A_%a.out
#SBATCH --error=${LOG_DIR}/job_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=8G

export PYTHONPATH="$REPO_ROOT/qmirt/src${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_DIR="$DATA_DIR"

"$SCRIPT_DIR/wrapper_cardiac_spect_sim_slurm.sh"
EOF

echo "Created output folder: $DATA_DIR"
echo "Created log folder:    $LOG_DIR"
echo "Created sbatch file:   $SBATCH_FILE"

sbatch "$SBATCH_FILE"
