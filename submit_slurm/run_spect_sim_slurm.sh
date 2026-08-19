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
CPUS_PER_TASK="128"
TIME_LIMIT="12:00:00"
MEM_GB="4"
DRY_RUN=0
TEST_MODE=0
SOURCE_ACTIVITY_BQ="${SOURCE_ACTIVITY_BQ:-6.25e6}"
CHUNK_DURATION_S="${CHUNK_DURATION_S:-1.0}"
NUM_CHUNKS="${NUM_CHUNKS:-1}"
NUM_LOOPS="${NUM_LOOPS:-1}"
SPARSE_SRM="${SPARSE_SRM:-0}"
SRM_FOV_SIZE_MM="${SRM_FOV_SIZE_MM:-210}"
PROFILE_RESOURCES="${PROFILE_RESOURCES:-1}"
PROFILE_INTERVAL_S="${PROFILE_INTERVAL_S:-5}"

# Initialize cluster-specific variables
CLUSTER=""
ACCOUNT=""
PARTITION=""
CONCURRENT_LIMIT=""

usage() {
    echo "Usage: $0 [brain|cardiac|/path/to/wrapper.sh] [job_count] [cpus_per_task] [time_limit] [mem_gb]"
    echo "  or:    $0 [--wrapper /path/to/wrapper.sh] [--job-count N] [--cpus-per-task N] [--time-limit HH:MM:SS] [--mem-gb N]"
    echo "            [--partition PART] [--account ALLOCATION_ID] [--cluster eris|expanse|bridges2] [--concurrent-limit LIMIT]"
    echo "            [--source-activity-bq VALUE] [--chunk-duration-s VALUE] [--num-chunks N]"
    echo "            [--sparse-srm] [--num-loops N] [--srm-fov-size-mm VALUE]"
    echo "            [--profile-resources|--no-profile-resources] [--profile-interval-s SECONDS]"
    echo "            [--test-mode] [--dry-run]"
    echo "Supported simulation types: brain, cardiac"
    echo "Supported clusters: eris, expanse, bridges2"
    echo ""
    echo "Array allocation (always one node/task per simulation):"
    echo "  --job-count N          Number of array jobs (independent simulations)"
    echo "  --cpus-per-task N      Threads per simulation (e.g., 128 for MT, 1 for ST)"
    echo "  --concurrent-limit N   Max simultaneous array jobs"
    echo "  Internal scheduler shape is fixed: --nodes=1 and --ntasks=1 per array job"
    echo ""
    echo "Test mode: --test-mode sets job_count=2, time_limit=0:30:00, cpus_per_task=4, activity=1e4, chunks=1"
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
        --job-count) JOB_COUNT="$2"; JOB_COUNT_SET=1; shift 2 ;;
        --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
        --time-limit) TIME_LIMIT="$2"; shift 2 ;;
        --mem-gb) MEM_GB="$2"; MEM_GB_SET=1; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --account|-A) ACCOUNT="$2"; shift 2 ;;
        --cluster) CLUSTER="$2"; shift 2 ;;
        --concurrent-limit) CONCURRENT_LIMIT="$2"; shift 2 ;;
        --source-activity-bq) SOURCE_ACTIVITY_BQ="$2"; shift 2 ;;
        --chunk-duration-s) CHUNK_DURATION_S="$2"; shift 2 ;;
        --num-chunks) NUM_CHUNKS="$2"; shift 2 ;;
        --sparse-srm) SPARSE_SRM=1; shift ;;
        --num-loops) NUM_LOOPS="$2"; shift 2 ;;
        --srm-fov-size-mm) SRM_FOV_SIZE_MM="$2"; shift 2 ;;
        --profile-resources) PROFILE_RESOURCES=1; shift ;;
        --no-profile-resources) PROFILE_RESOURCES=0; shift ;;
        --profile-interval-s) PROFILE_INTERVAL_S="$2"; shift 2 ;;
        --nodes)
            echo "Error: --nodes is not a user-facing option in array mode."
            echo "       Control throughput with --job-count and per-job threads with --cpus-per-task."
            exit 1
            ;;
        --test-mode) TEST_MODE=1; shift ;;
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

# --- Test Mode Configuration ---
if [[ "$TEST_MODE" -eq 1 ]]; then
    echo "Test mode enabled: using reduced parameters"
    JOB_COUNT=2
    CPUS_PER_TASK=4
    TIME_LIMIT="0:30:00"
    MEM_GB=8
    CHUNK_DURATION_S=0.1
    NUM_CHUNKS=1
    SOURCE_ACTIVITY_BQ=1e4
fi

if [[ "$SPARSE_SRM" == "1" ]] && [[ "$SIM_TYPE" != "brain" ]]; then
    echo "Error: --sparse-srm is currently supported only for brain simulations."
    exit 1
fi

# --- Cluster Detection & Configuration ---
if [[ -z "$CLUSTER" ]]; then
    HOSTNAME=$(hostname)
    if [[ "$HOSTNAME" == *"expanse"* ]]; then
        CLUSTER="expanse"
    elif [[ "$HOSTNAME" == *"bridges"* ]]; then
        CLUSTER="bridges2"
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

elif [[ "$CLUSTER" == "bridges2" ]]; then
    VALID_PARTITIONS="RM RM-512 RM-shared RM-small GPU GPU-shared GPU-small EM ROBO ROBO-8 HACC GPU-dev applications"
    PARTITION="${PARTITION:-RM}"

    if [[ -n "${PROJECT:-}" ]]; then
        # On Bridges2, PROJECT is already the project root path provided by the system.
        SCRATCH_ROOT="${PROJECT}"
    elif [[ -n "${ACCOUNT:-}" ]]; then
        # Local fallback for dry runs or non-allocated shells.
        SCRATCH_ROOT="/ocean/projects/${ACCOUNT}/${USER}"
    else
        if [[ "$DRY_RUN" -eq 1 ]]; then
            SCRATCH_ROOT="${HOME}/scratch/qmirt-bridges2"
            echo "Warning: PROJECT is not set in this shell; using local preview scratch path ${SCRATCH_ROOT} for dry-run only."
        else
            echo "Error: no project allocation is available for PSC Bridges2. The system normally sets PROJECT automatically; for local testing, pass --account (-A)."
            exit 1
        fi
    fi

elif [[ "$CLUSTER" == "eris" ]]; then
    VALID_PARTITIONS="normal long bigmem interactive debug"
    PARTITION="${PARTITION:-normal}"
    
    # Default ERIS scratch root
    SCRATCH_ROOT="/scratch/f/fh890"
else
    echo "Error: Unknown cluster '$CLUSTER'. Use 'eris', 'expanse', or 'bridges2'."
    exit 1
fi

# Validate partition against cluster-specific valid partitions
if [[ ! " $VALID_PARTITIONS " =~ " $PARTITION " ]]; then
    echo "Error: unsupported partition '$PARTITION' for cluster '$CLUSTER'"
    echo "Supported partitions on $CLUSTER: ${VALID_PARTITIONS}"
    exit 1
fi

# Auto-tune memory for high-thread jobs only when user did not pass --mem-gb.
if [[ -z "${MEM_GB_SET:-}" ]] && [[ "$CPUS_PER_TASK" -ge 64 ]] && [[ "$TEST_MODE" -eq 0 ]]; then
    case "$CLUSTER" in
        bridges2)
            if [[ "$PARTITION" == "RM-512" ]]; then
                MEM_GB=460
                echo "Auto-adjusted memory to ${MEM_GB}GB for high-thread jobs on bridges2/${PARTITION}"
            else
                MEM_GB=220
                echo "Auto-adjusted memory to ${MEM_GB}GB for high-thread jobs on bridges2/${PARTITION}"
            fi
            ;;
        expanse|eris)
            MEM_GB=220
            echo "Auto-adjusted memory to ${MEM_GB}GB for high-thread jobs on ${CLUSTER}"
            ;;
    esac
fi

if ! [[ "$JOB_COUNT" =~ ^[1-9][0-9]*$ ]]; then echo "job_count must be a positive integer"; exit 1; fi
if ! [[ "$CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then echo "cpus_per_task must be a positive integer"; exit 1; fi
if ! [[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]]; then echo "mem_gb must be a positive integer"; exit 1; fi
if ! [[ "$NUM_LOOPS" =~ ^[1-9][0-9]*$ ]]; then echo "num_loops must be a positive integer"; exit 1; fi
if ! [[ "$SRM_FOV_SIZE_MM" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$(awk -v size="$SRM_FOV_SIZE_MM" 'BEGIN { print (size > 0) ? 1 : 0 }')" != "1" ]]; then
    echo "srm_fov_size_mm must be a positive number"
    exit 1
fi
if [[ "$PROFILE_RESOURCES" != "0" && "$PROFILE_RESOURCES" != "1" ]]; then
    echo "profile_resources must be 0 or 1"
    exit 1
fi
if ! [[ "$PROFILE_INTERVAL_S" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$(awk -v interval="$PROFILE_INTERVAL_S" 'BEGIN { print (interval > 0) ? 1 : 0 }')" != "1" ]]; then
    echo "profile_interval_s must be a positive number"
    exit 1
fi
if [[ -n "$CONCURRENT_LIMIT" ]]; then
    if ! [[ "$CONCURRENT_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
        echo "concurrent_limit must be a positive integer"
        exit 1
    fi
    if [[ "$CONCURRENT_LIMIT" -gt "$JOB_COUNT" ]]; then
        echo "concurrent_limit (${CONCURRENT_LIMIT}) cannot exceed job_count (${JOB_COUNT})"
        exit 1
    fi
fi

if [[ ! -f "$SIM_WRAPPER" ]]; then
    echo "Error: wrapper script not found: $SIM_WRAPPER"
    exit 1
fi

EXPECTED_EVENTS_PER_JOB="$(awk -v a="$SOURCE_ACTIVITY_BQ" -v d="$CHUNK_DURATION_S" -v c="$NUM_CHUNKS" -v t="$CPUS_PER_TASK" 'BEGIN { printf "%.0f", a * d * c * t }')"
EXPECTED_EVENTS_TOTAL="$(awk -v a="$SOURCE_ACTIVITY_BQ" -v d="$CHUNK_DURATION_S" -v c="$NUM_CHUNKS" -v t="$CPUS_PER_TASK" -v j="$JOB_COUNT" 'BEGIN { printf "%.0f", a * d * c * t * j }')"
EXPECTED_EVENTS_PER_JOB_FMT="$(awk -v n="$EXPECTED_EVENTS_PER_JOB" 'function comma(x, s, r) { s = x ""; while (length(s) > 3) { r = "," substr(s, length(s)-2, 3) r; s = substr(s, 1, length(s)-3) } return s r } BEGIN { print comma(n) }')"
EXPECTED_EVENTS_TOTAL_FMT="$(awk -v n="$EXPECTED_EVENTS_TOTAL" 'function comma(x, s, r) { s = x ""; while (length(s) > 3) { r = "," substr(s, length(s)-2, 3) r; s = substr(s, 1, length(s)-3) } return s r } BEGIN { print comma(n) }')"

BATCH_ID="batch_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/submit_slurm/logs/${BATCH_ID}"
DATA_DIR="${SCRATCH_ROOT}/${OUTPUT_SUBDIR}/${BATCH_ID}"
CONTAINER_SIF="${CONTAINER_SIF:-${REPO_ROOT}/submit_slurm/qmirt-gate-10-sim-sif_v1.0.0.sif}"
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

# Always job array mode (one node/task per array element)
if [[ -n "$CONCURRENT_LIMIT" ]]; then
    echo "#SBATCH --array=0-$((JOB_COUNT - 1))%${CONCURRENT_LIMIT}" >> "$SBATCH_FILE"
else
    echo "#SBATCH --array=0-$((JOB_COUNT - 1))" >> "$SBATCH_FILE"
fi
echo "#SBATCH --nodes=1" >> "$SBATCH_FILE"
echo "#SBATCH --ntasks=1" >> "$SBATCH_FILE"
echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}" >> "$SBATCH_FILE"

cat >> "$SBATCH_FILE" <<EOF
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
export NUM_LOOPS="${NUM_LOOPS}"
export SPARSE_SRM="${SPARSE_SRM}"
export SRM_FOV_SIZE_MM="${SRM_FOV_SIZE_MM}"
export MAX_TASK_SECONDS="${MAX_TASK_SECONDS:-0}"
export PROFILE_RESOURCES="${PROFILE_RESOURCES}"
export PROFILE_INTERVAL_S="${PROFILE_INTERVAL_S}"
export SIM_TYPE="${SIM_TYPE}"
export SIM_PYTHON_SCRIPT="${SIM_PYTHON_SCRIPT}"

bash "${SIM_WRAPPER}"
EOF

echo "Simulation wrapper: ${SIM_WRAPPER}"
echo "Cluster mode: ${CLUSTER}"
echo "Simulation type: ${SIM_TYPE}"
echo "Allocation mode: Job array"
echo "Job count: ${JOB_COUNT}"
if [[ -n "$CONCURRENT_LIMIT" ]]; then echo "Concurrent limit: ${CONCURRENT_LIMIT}"; fi
echo "Nodes per job: 1"
echo "Tasks per job: 1"
echo "CPUs per task: ${CPUS_PER_TASK} (threads per independent simulation)"
echo "Time limit: ${TIME_LIMIT}"
echo "Memory: ${MEM_GB}G"
echo "Partition: ${PARTITION}"
if [[ -n "$ACCOUNT" ]]; then echo "Account: ${ACCOUNT}"; fi
echo "Source activity: ${SOURCE_ACTIVITY_BQ} Bq"
echo "Chunk duration: ${CHUNK_DURATION_S} s"
echo "Num chunks: ${NUM_CHUNKS}"
echo "Num loops: ${NUM_LOOPS}"
echo "Sparse SRM: ${SPARSE_SRM}"
if [[ "$SPARSE_SRM" == "1" ]]; then echo "SRM FOV extent: ${SRM_FOV_SIZE_MM} mm"; fi
echo "Expected events per job (approx): ${EXPECTED_EVENTS_PER_JOB_FMT}"
echo "Expected events across all jobs (approx): ${EXPECTED_EVENTS_TOTAL_FMT}"
if [[ "$TEST_MODE" -eq 1 ]]; then echo "*** TEST MODE ENABLED ***"; fi
echo "Created output folder: $DATA_DIR"
echo "Created log folder:    $LOG_DIR"
echo "Created sbatch file:   $SBATCH_FILE"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "--- sbatch file preview ---"
    cat "$SBATCH_FILE"
    exit 0
fi

sbatch "$SBATCH_FILE"