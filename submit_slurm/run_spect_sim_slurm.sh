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
DRY_RUN=0
SOURCE_ACTIVITY_BQ="${SOURCE_ACTIVITY_BQ:-3.7e5}"
CHUNK_DURATION_S="${CHUNK_DURATION_S:-1.0}"
NUM_CHUNKS="${NUM_CHUNKS:-1}"

# Initialize cluster-specific variables
CLUSTER=""
ACCOUNT=""
PARTITION=""
CONCURRENT_LIMIT=""

usage() {
    echo "Usage: $0 [brain|cardiac|/path/to/wrapper.sh] [job_count] [cpus_per_task] [time_limit] [mem_gb]"
    echo "  or:    $0 [--wrapper /path/to/wrapper.sh] [--job-count N] [--cpus-per-task N] [--time-limit HH:MM:SS] [--mem-gb N]"
    echo "            [--partition PART] [--account ALLOCATION_ID] [--cluster eris|expanse] [--concurrent-limit LIMIT]"
    echo "            [--source-activity-bq VALUE] [--chunk-duration-s VALUE] [--num-chunks N] [--dry-run]"
    echo "Supported simulation types: brain, cardiac"
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
            SIM_PYTHON_SCRIPT="payload/python/gate_sim_cardiac_spect_boolean.py"
            OUTPUT_SUBDIR="cardiac_spect_sim"
            shift
            ;;
        --wrapper)
            if [[ $# -lt 2 ]]; then usage; exit 1; fi
            SIM_WRAPPER="$2"
            if [[ "$SIM_WRAPPER" != /* ]]; then SIM_WRAPPER="${SCRIPT_DIR}/$SIM_WRAPPER"; fi
            SIM_LABEL="$(basename "${SIM_WRAPPER%.*}")"
            shift 2
            ;;
        --job-count) JOB_COUNT="$2"; shift 2 ;;
        --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
        --time-limit) TIME_LIMIT="$2"; shift 2 ;;
        --mem-gb) MEM_GB="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --account|-A) ACCOUNT="$2"; shift 2 ;;
        --cluster) CLUSTER="$2"; shift 2 ;;
        --concurrent-limit) CONCURRENT_LIMIT="$2"; shift 2 ;;
        --source-activity-bq) SOURCE_ACTIVITY_BQ="$2"; shift 2 ;;
        --chunk-duration-s) CHUNK_DURATION_S="$2"; shift 2 ;;
        --num-chunks) NUM_CHUNKS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *)
            if [[ -f "$1" ]]; then
                SIM_WRAPPER="$1"
                if [[ "$SIM_WRAPPER" != /* ]]; then SIM_WRAPPER="${SCRIPT_DIR}/$SIM_WRAPPER"; fi
                SIM_LABEL="$(basename "${SIM_WRAPPER%.*}")"
                shift
            elif [[ -z "${JOB_COUNT_SET:-}" ]]; then JOB_COUNT="$1"; JOB_COUNT_SET=1; shift
            elif [[ -z "${CPUS_PER_TASK_SET:-}" ]]; then CPUS_PER_TASK="$1"; CPUS_PER_TASK_SET=1; shift
            elif [[ -z "${TIME_LIMIT_SET:-}" ]]; then TIME_LIMIT="$1"; TIME_LIMIT_SET=1; shift
            elif [[ -z "${MEM_GB_SET:-}" ]]; then MEM_GB="$1"; MEM_GB_SET=1; shift
            else
                usage
                echo "Unexpected argument: $1"
                exit 1
            fi
            ;;
    esac
done

# --- Cluster Detection & Configuration ---
if [[ -z "$CLUSTER" ]]; then
    if [[ "$(hostname)" == *"expanse"* ]]; then
        CLUSTER="expanse"
    else
        CLUSTER="eris"
    fi
fi

if [[ "$CLUSTER" == "expanse" ]]; then
    VALID_PARTITIONS="compute shared debug"
    PARTITION="${PARTITION:-shared}"
    
    if [[ -z "$ACCOUNT" ]]; then
        echo "Error: --account (-A) is required on ACCESS Expanse (e.g., -A med123456)"
        exit 1
    fi
    
    # Use Expanse's Lustre scratch file system based on the allocation account
    SCRATCH_ROOT="/expanse/lustre/projects/${ACCOUNT}/${USER}"

elif [[ "$CLUSTER" == "eris" ]]; then
    VALID_PARTITIONS="normal long bigmem interactive debug"
    PARTITION="${PARTITION:-normal}"
    
    # Default ERIS scratch root
    SCRATCH_ROOT="/scratch/f/fh890"
else
    echo "Error: Unknown cluster '$CLUSTER'. Use 'eris' or 'expanse'."
    exit 1
fi

case "$PARTITION" in
    compute|shared|normal|long|bigmem|interactive|debug)
        ;;
    *)
        echo "Error: unsupported partition '$PARTITION' for cluster '$CLUSTER'"
        echo "Supported partitions on $CLUSTER: ${VALID_PARTITIONS}"
        exit 1
        ;;
esac

if ! [[ "$JOB_COUNT" =~ ^[1-9][0-9]*$ ]]; then echo "job_count must be a positive integer"; exit 1; fi
if ! [[ "$CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then echo "cpus_per_task must be a positive integer"; exit 1; fi
if ! [[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]]; then echo "mem_gb must be a positive integer"; exit 1; fi

if [[ ! -f "$SIM_WRAPPER" ]]; then
    echo "Error: wrapper script not found: $SIM_WRAPPER"
    exit 1
fi

BATCH_ID="batch_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/submit_slurm/logs/${BATCH_ID}"
DATA_DIR="${SCRATCH_ROOT}/${OUTPUT_SUBDIR}/${BATCH_ID}"
CONTAINER_SIF="${REPO_ROOT}/submit_slurm/qmirt-gate-10-sim-sif_v1.0.0.sif"
SBATCH_FILE="${LOG_DIR}/${SIM_LABEL}_sim_slurm.sbatch"

if [[ ! -f "$CONTAINER_SIF" ]]; then
    echo "Error: Apptainer SIF not found at $CONTAINER_SIF"
    exit 1
fi

mkdir -p "$LOG_DIR" "$DATA_DIR"

# Generate the sbatch file dynamically
cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=${SIM_LABEL}
EOF

# Inject Array with Concurrency Limit if provided
if [[ -n "$CONCURRENT_LIMIT" ]]; then
    echo "#SBATCH --array=0-$((JOB_COUNT - 1))%${CONCURRENT_LIMIT}" >> "$SBATCH_FILE"
else
    echo "#SBATCH --array=0-$((JOB_COUNT - 1))" >> "$SBATCH_FILE"
fi

cat >> "$SBATCH_FILE" <<EOF
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM_GB}G
#SBATCH --partition=${PARTITION}
EOF

# Inject account if provided (Required for ACCESS)
if [[ -n "$ACCOUNT" ]]; then
    echo "#SBATCH --account=${ACCOUNT}" >> "$SBATCH_FILE"
fi

# Append the rest of the file
cat >> "$SBATCH_FILE" <<EOF
#SBATCH --output=${LOG_DIR}/job_%A_%a.out
#SBATCH --error=${LOG_DIR}/job_%A_%a.err

export REPO_ROOT="${REPO_ROOT}"
export OUTPUT_DIR="${DATA_DIR}"
export SCRATCH_ROOT="${SCRATCH_ROOT}"
export CONTAINER_SIF="${CONTAINER_SIF}"
export PYTHONPATH="${REPO_ROOT}/qmirt/src\${PYTHONPATH:+:\$PYTHONPATH}"
export SOURCE_ACTIVITY_BQ="${SOURCE_ACTIVITY_BQ}"
export CHUNK_DURATION_S="${CHUNK_DURATION_S}"
export NUM_CHUNKS="${NUM_CHUNKS}"
export MAX_TASK_SECONDS="${MAX_TASK_SECONDS:-0}"
export SIM_TYPE="${SIM_TYPE}"
export SIM_PYTHON_SCRIPT="${SIM_PYTHON_SCRIPT}"
export SCRATCH_ROOT="${SCRATCH_ROOT}"

bash "${SIM_WRAPPER}"
EOF

echo "Simulation wrapper: ${SIM_WRAPPER}"
echo "Cluster mode: ${CLUSTER}"
echo "Simulation type: ${SIM_TYPE}"
echo "Job count: ${JOB_COUNT}"
if [[ -n "$CONCURRENT_LIMIT" ]]; then echo "Concurrent limit: ${CONCURRENT_LIMIT}"; fi
echo "CPUs per task: ${CPUS_PER_TASK}"
echo "Time limit: ${TIME_LIMIT}"
echo "Memory: ${MEM_GB}G"
echo "Partition: ${PARTITION}"
if [[ -n "$ACCOUNT" ]]; then echo "Account: ${ACCOUNT}"; fi
echo "Source activity: ${SOURCE_ACTIVITY_BQ} Bq"
echo "Created output folder: $DATA_DIR"
echo "Created log folder:    $LOG_DIR"
echo "Created sbatch file:   $SBATCH_FILE"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "--- sbatch file preview ---"
    cat "$SBATCH_FILE"
    exit 0
fi

sbatch "$SBATCH_FILE"