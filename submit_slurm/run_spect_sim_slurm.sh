#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SIM_WRAPPER="${SCRIPT_DIR}/wrapper_brain_spect_sim_slurm.sh"
SIM_LABEL="brain_spect"
SIM_TYPE="brain"
SIM_PYTHON_SCRIPT="payload/python/gate_sim_brain_spect_boolean.py"
OUTPUT_SUBDIR="brain_spect_sim"
JOB_COUNT="100"
CPUS_PER_TASK="1"
TIME_LIMIT="12:00:00"
MEM_GB="4"
PARTITION="normal"
DRY_RUN=0
SOURCE_ACTIVITY_BQ="${SOURCE_ACTIVITY_BQ:-3.7e5}"
CHUNK_DURATION_S="${CHUNK_DURATION_S:-1.0}"
NUM_CHUNKS="${NUM_CHUNKS:-1}"
VALID_PARTITIONS="normal long bigmem interactive debug"

usage() {
    echo "Usage: $0 [brain|cardiac|/path/to/wrapper.sh] [job_count] [cpus_per_task] [time_limit] [mem_gb]"
    echo "  or:    $0 [--wrapper /path/to/wrapper.sh] [--job-count N] [--cpus-per-task N] [--time-limit HH:MM:SS] [--mem-gb N] [--partition PART] [--source-activity-bq VALUE] [--chunk-duration-s VALUE] [--num-chunks N] [--dry-run]"
    echo "Supported simulation types: brain, cardiac"
    echo "Supported ERIS partitions: ${VALID_PARTITIONS}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        brain)
            SIM_WRAPPER="${SCRIPT_DIR}/wrapper_brain_spect_sim_slurm.sh"
            SIM_LABEL="brain_spect"
            SIM_TYPE="brain"
            SIM_PYTHON_SCRIPT="payload/python/gate_sim_brain_spect_boolean.py"
            OUTPUT_SUBDIR="brain_spect_sim"
            shift
            ;;
        cardiac)
            SIM_WRAPPER="${SCRIPT_DIR}/wrapper_cardiac_spect_sim_slurm.sh"
            SIM_LABEL="cardiac_spect"
            SIM_TYPE="cardiac"
            SIM_PYTHON_SCRIPT="payload/python/gate_sim_cardiac_spect_no_boolean.py"
            OUTPUT_SUBDIR="cardiac_spect_sim"
            shift
            ;;
        --wrapper)
            if [[ $# -lt 2 ]]; then
                usage
                exit 1
            fi
            SIM_WRAPPER="$2"
            if [[ "$SIM_WRAPPER" != /* ]]; then
                SIM_WRAPPER="${SCRIPT_DIR}/$SIM_WRAPPER"
            fi
            SIM_LABEL="$(basename "${SIM_WRAPPER%.*}")"
            shift 2
            ;;
        --job-count)
            if [[ $# -lt 2 ]]; then usage; exit 1; fi
            JOB_COUNT="$2"
            shift 2
            ;;
        --cpus-per-task)
            if [[ $# -lt 2 ]]; then usage; exit 1; fi
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        --time-limit)
            if [[ $# -lt 2 ]]; then usage; exit 1; fi
            TIME_LIMIT="$2"
            shift 2
            ;;
        --mem-gb)
            if [[ $# -lt 2 ]]; then usage; exit 1; fi
            MEM_GB="$2"
            shift 2
            ;;
        --partition)
            if [[ $# -lt 2 ]]; then usage; exit 1; fi
            PARTITION="$2"
            case "$PARTITION" in
                normal|long|bigmem|interactive|debug)
                    ;;
                *)
                    echo "Error: unsupported ERIS partition '$PARTITION'"
                    echo "Supported ERIS partitions: ${VALID_PARTITIONS}"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --source-activity-bq)
            if [[ $# -lt 2 ]]; then usage; exit 1; fi
            SOURCE_ACTIVITY_BQ="$2"
            shift 2
            ;;
        --chunk-duration-s)
            if [[ $# -lt 2 ]]; then usage; exit 1; fi
            CHUNK_DURATION_S="$2"
            shift 2
            ;;
        --num-chunks)
            if [[ $# -lt 2 ]]; then usage; exit 1; fi
            NUM_CHUNKS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            if [[ -f "$1" ]]; then
                SIM_WRAPPER="$1"
                if [[ "$SIM_WRAPPER" != /* ]]; then
                    SIM_WRAPPER="${SCRIPT_DIR}/$SIM_WRAPPER"
                fi
                SIM_LABEL="$(basename "${SIM_WRAPPER%.*}")"
                shift
            elif [[ -z "${JOB_COUNT_SET:-}" ]]; then
                JOB_COUNT="$1"
                JOB_COUNT_SET=1
                shift
            elif [[ -z "${CPUS_PER_TASK_SET:-}" ]]; then
                CPUS_PER_TASK="$1"
                CPUS_PER_TASK_SET=1
                shift
            elif [[ -z "${TIME_LIMIT_SET:-}" ]]; then
                TIME_LIMIT="$1"
                TIME_LIMIT_SET=1
                shift
            elif [[ -z "${MEM_GB_SET:-}" ]]; then
                MEM_GB="$1"
                MEM_GB_SET=1
                shift
            else
                usage
                echo "Unexpected argument: $1"
                exit 1
            fi
            ;;
    esac
done

case "$PARTITION" in
    normal|long|bigmem|interactive|debug)
        ;;
    *)
        echo "Error: unsupported ERIS partition '$PARTITION'"
        echo "Supported ERIS partitions: ${VALID_PARTITIONS}"
        exit 1
        ;;
 esac

if ! [[ "$JOB_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [brain|cardiac|/path/to/wrapper.sh] [job_count] [cpus_per_task] [time_limit] [mem_gb]"
    echo "job_count must be a positive integer"
    exit 1
fi

if ! [[ "$CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [brain|cardiac|/path/to/wrapper.sh] [job_count] [cpus_per_task] [time_limit] [mem_gb]"
    echo "cpus_per_task must be a positive integer"
    exit 1
fi

if ! [[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [brain|cardiac|/path/to/wrapper.sh] [job_count] [cpus_per_task] [time_limit] [mem_gb]"
    echo "mem_gb must be a positive integer"
    exit 1
fi

if [[ ! -f "$SIM_WRAPPER" ]]; then
    echo "Error: wrapper script not found: $SIM_WRAPPER"
    exit 1
fi

BATCH_ID="batch_$(date +%Y%m%d_%H%M%S)"
SCRATCH_ROOT="/scratch/f/fh890"
LOG_DIR="${REPO_ROOT}/submit_slurm/logs/${BATCH_ID}"
DATA_DIR="${SCRATCH_ROOT}/${OUTPUT_SUBDIR}/${BATCH_ID}"
CONTAINER_SIF="${REPO_ROOT}/submit_slurm/qmirt-gate-10-sim-sif_v1.0.0.sif"
SBATCH_FILE="${LOG_DIR}/${SIM_LABEL}_sim_slurm.sbatch"

if [[ ! -f "$CONTAINER_SIF" ]]; then
    echo "Error: Apptainer SIF not found at $CONTAINER_SIF"
    echo "Pull it into submit_slurm/ first, for example:"
    echo "  cd ${REPO_ROOT}/submit_slurm"
    echo "  apptainer pull oras://ghcr.io/fanghandev/qmirt-gate-10-sim-sif:v1.0.0"
    exit 1
fi

mkdir -p "$LOG_DIR" "$DATA_DIR"

cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=${SIM_LABEL}
#SBATCH --array=0-$((JOB_COUNT - 1))
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM_GB}G
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/job_%A_%a.out
#SBATCH --error=${LOG_DIR}/job_%A_%a.err

export REPO_ROOT="${REPO_ROOT}"
export OUTPUT_DIR="${DATA_DIR}"
export CONTAINER_SIF="${CONTAINER_SIF}"
export PYTHONPATH="${REPO_ROOT}/qmirt/src${PYTHONPATH:+:$PYTHONPATH}"
export SOURCE_ACTIVITY_BQ="${SOURCE_ACTIVITY_BQ}"
export CHUNK_DURATION_S="${CHUNK_DURATION_S}"
export NUM_CHUNKS="${NUM_CHUNKS}"
export MAX_TASK_SECONDS="${MAX_TASK_SECONDS:-0}"
export SIM_TYPE="${SIM_TYPE}"
export SIM_PYTHON_SCRIPT="${SIM_PYTHON_SCRIPT}"

bash "${SIM_WRAPPER}"
EOF

echo "Simulation wrapper: ${SIM_WRAPPER}"
echo "Simulation type: ${SIM_TYPE}"
echo "Job count: ${JOB_COUNT}"
echo "CPUs per task: ${CPUS_PER_TASK}"
echo "Time limit: ${TIME_LIMIT}"
echo "Memory: ${MEM_GB}G"
echo "Partition: ${PARTITION} (ERIS default available partition set)"
echo "Source activity: ${SOURCE_ACTIVITY_BQ} Bq"
echo "Chunk duration: ${CHUNK_DURATION_S} s"
echo "Num chunks: ${NUM_CHUNKS}"
echo "Created output folder: $DATA_DIR"
echo "Created log folder:    $LOG_DIR"
echo "Created sbatch file:   $SBATCH_FILE"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry-run mode enabled: showing generated sbatch configuration without submitting"
    echo "--- sbatch file preview ---"
    cat "$SBATCH_FILE"
    echo "--- end preview ---"
    exit 0
fi

echo "Submitting array job with ${JOB_COUNT} tasks..."
sbatch "$SBATCH_FILE"
