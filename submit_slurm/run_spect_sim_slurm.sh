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
COMBINE_AFTER=0
COMBINE_MEM_GB="64"
COMBINE_TIME_LIMIT="2:00:00"
COMBINE_CPUS="4"
COMBINE_PARTITION=""
COMBINE_GROUPS="0"
AUTO_REPORT=0
REPORT_INTERVAL_S="60"
REPORT_CPUS="1"
REPORT_MEM_GB="2"
REPORT_TIME_LIMIT="24:00:00"
REPORT_PARTITION=""
ARRAY_DEPENDENCY="${ARRAY_DEPENDENCY:-}"
CAMPAIGN_GROUP_ID="${CAMPAIGN_GROUP_ID:-}"
CAMPAIGN_PART_INDEX="${CAMPAIGN_PART_INDEX:-}"
CAMPAIGN_PART_COUNT="${CAMPAIGN_PART_COUNT:-}"

# Initialize cluster-specific variables
CLUSTER=""
ACCOUNT=""
PARTITION=""
CONCURRENT_LIMIT=""
PROJECT_DIR=""
# Node-local scratch template, expanded inside the generated sbatch (per array element).
LOCAL_SCRATCH_TEMPLATE=""
BATCH_ID="${BATCH_ID:-}"

usage() {
    echo "Usage: $0 [brain|cardiac|/path/to/wrapper.sh] [job_count] [cpus_per_task] [time_limit] [mem_gb]"
    echo "  or:    $0 [--wrapper /path/to/wrapper.sh] [--job-count N] [--cpus-per-task N] [--time-limit HH:MM:SS] [--mem-gb N]"
    echo "            [--partition PART] [--account ALLOCATION_ID] [--cluster expanse|eris|bridges2] [--concurrent-limit LIMIT]"
    echo "            [--project-dir PATH]  Shared campaign root; on Expanse defaults to /expanse/lustre/projects/<group>/\$USER"
    echo "            [--source-activity-bq VALUE] [--chunk-duration-s VALUE] [--num-chunks N]"
    echo "            [--sparse-srm|--no-sparse-srm] [--num-loops N] [--srm-fov-size-mm VALUE]"
    echo "              Sparse SRM is on by default for brain simulations."
    echo "            [--profile-resources|--no-profile-resources] [--profile-interval-s SECONDS]"
    echo "            [--combine-after] [--combine-partition PART] [--combine-cpus N] [--combine-mem-gb N] [--combine-time-limit HH:MM:SS]"
    echo "            [--combine-groups N]  Tree reduction: merge tasks in N parallel groups before the final merge"
    echo "            [--auto-report] [--report-interval-s SECONDS] [--report-partition PART]"
    echo "            [--report-cpus N] [--report-mem-gb N] [--report-time-limit HH:MM:SS]"
    echo "            [--array-dependency DEPENDENCY]  Slurm dependency for the simulation array"
    echo "              Submits a small Slurm job (not a login-node process) that refreshes progress.json"
    echo "            [--test-mode] [--dry-run]"
    echo "Supported simulation types: brain, cardiac"
    echo "Supported clusters: expanse (default), eris, bridges2"
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
        --partition) PARTITION="$2"; PARTITION_SET=1; shift 2 ;;
        --account|-A) ACCOUNT="$2"; shift 2 ;;
        --cluster) CLUSTER="$2"; shift 2 ;;
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        --concurrent-limit) CONCURRENT_LIMIT="$2"; shift 2 ;;
        --source-activity-bq) SOURCE_ACTIVITY_BQ="$2"; shift 2 ;;
        --chunk-duration-s) CHUNK_DURATION_S="$2"; shift 2 ;;
        --num-chunks) NUM_CHUNKS="$2"; shift 2 ;;
        --sparse-srm) SPARSE_SRM=1; SPARSE_SRM_SET=1; shift ;;
        --no-sparse-srm) SPARSE_SRM=0; SPARSE_SRM_SET=1; shift ;;
        --num-loops) NUM_LOOPS="$2"; shift 2 ;;
        --srm-fov-size-mm) SRM_FOV_SIZE_MM="$2"; shift 2 ;;
        --profile-resources) PROFILE_RESOURCES=1; shift ;;
        --no-profile-resources) PROFILE_RESOURCES=0; shift ;;
        --profile-interval-s) PROFILE_INTERVAL_S="$2"; shift 2 ;;
        --combine-after) COMBINE_AFTER=1; shift ;;
        --combine-mem-gb) COMBINE_MEM_GB="$2"; shift 2 ;;
        --combine-time-limit) COMBINE_TIME_LIMIT="$2"; shift 2 ;;
        --combine-cpus) COMBINE_CPUS="$2"; shift 2 ;;
        --combine-partition) COMBINE_PARTITION="$2"; shift 2 ;;
        --combine-groups) COMBINE_GROUPS="$2"; shift 2 ;;
        --auto-report) AUTO_REPORT=1; shift ;;
        --report-interval-s) REPORT_INTERVAL_S="$2"; shift 2 ;;
        --report-cpus) REPORT_CPUS="$2"; shift 2 ;;
        --report-mem-gb) REPORT_MEM_GB="$2"; shift 2 ;;
        --report-time-limit) REPORT_TIME_LIMIT="$2"; shift 2 ;;
        --report-partition) REPORT_PARTITION="$2"; shift 2 ;;
        --array-dependency) ARRAY_DEPENDENCY="$2"; shift 2 ;;
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

# Brain campaigns are sparse-SRM by default; every task must emit a partial SRM.
if [[ -z "${SPARSE_SRM_SET:-}" ]] && [[ "$SIM_TYPE" == "brain" ]]; then
    SPARSE_SRM=1
fi

if [[ "$SPARSE_SRM" == "1" ]] && [[ "$SIM_TYPE" != "brain" ]]; then
    echo "Error: --sparse-srm is currently supported only for brain simulations."
    exit 1
fi

if [[ "$COMBINE_AFTER" -eq 1 ]] && [[ "$SPARSE_SRM" != "1" ]]; then
    echo "Error: --combine-after requires --sparse-srm."
    exit 1
fi

# --- Cluster Detection & Configuration ---
if [[ -z "$CLUSTER" ]]; then
    HOSTNAME=$(hostname)
    if [[ "$HOSTNAME" == *"expanse"* ]]; then
        CLUSTER="expanse"
    elif [[ "$HOSTNAME" == *"bridges"* ]]; then
        CLUSTER="bridges2"
    elif [[ "$HOSTNAME" == *"eris"* ]]; then
        CLUSTER="eris"
    else
        CLUSTER="expanse"
    fi
fi

if [[ "$CLUSTER" == "expanse" ]]; then
    VALID_PARTITIONS="compute shared large-shared debug preempt ind-compute ind-shared"
    PARTITION="${PARTITION:-shared}"
    # Expanse does not set SLURM_TMPDIR; this is its per-job node-local NVMe path.
    LOCAL_SCRATCH_TEMPLATE='/scratch/${USER}/job_${SLURM_JOB_ID}'

    if [[ -z "$ACCOUNT" ]]; then
        echo "Error: --account (-A) is required on ACCESS Expanse (e.g., -A mde260019)"
        exit 1
    fi

    # The Lustre projects directory is named after the SDSC unix group, which is a
    # different string from the Slurm account (e.g. account mde260019 -> group mgh102).
    if [[ -z "$PROJECT_DIR" ]]; then
        PROJECT_DIR="$(ls -d /expanse/lustre/projects/*/"${USER}" 2>/dev/null | head -n 1 || true)"
    fi
    if [[ -n "$PROJECT_DIR" ]]; then
        SCRATCH_ROOT="$PROJECT_DIR"
    elif [[ "$DRY_RUN" -eq 1 ]]; then
        SCRATCH_ROOT="${HOME}/scratch/qmirt-expanse"
        echo "Warning: Expanse projects directory not visible from this host; using local preview path ${SCRATCH_ROOT} for dry-run only."
    else
        echo "Error: could not resolve the Expanse projects directory for ${USER}."
        echo "       Pass --project-dir /expanse/lustre/projects/<group>/${USER} (find <group> with 'id -Gn')."
        exit 1
    fi

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

# Expanse's shared-normal QOS allows cpu=127, so a 128-thread job on shared is
# rejected with QOSMaxCpuPerJobLimit. That thread count is a whole node anyway.
if [[ "$CLUSTER" == "expanse" ]] && [[ "$PARTITION" == "shared" || "$PARTITION" == "ind-shared" ]]; then
    SHARED_CPU_CAP=127
    if [[ "$CPUS_PER_TASK" -gt "$SHARED_CPU_CAP" ]]; then
        if [[ -n "${PARTITION_SET:-}" ]]; then
            echo "Error: --cpus-per-task ${CPUS_PER_TASK} exceeds the Expanse ${PARTITION} QOS limit of ${SHARED_CPU_CAP} CPUs."
            echo "       Slurm rejects this with QOSMaxCpuPerJobLimit."
            echo "       Use --cpus-per-task ${SHARED_CPU_CAP}, or --partition compute for a whole node."
            exit 1
        fi
        PARTITION="compute"
        echo "Switched to partition compute: ${CPUS_PER_TASK} CPUs exceeds the ${SHARED_CPU_CAP}-CPU limit of the shared QOS"
    fi
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

# Expanse shared partitions allocate roughly 2G of memory per requested core.
if [[ "$CLUSTER" == "expanse" ]] && [[ "$PARTITION" == "shared" || "$PARTITION" == "ind-shared" || "$PARTITION" == "large-shared" ]]; then
    if [[ "$PARTITION" == "large-shared" ]]; then
        MEM_PER_CPU_CAP_GB=15
    else
        MEM_PER_CPU_CAP_GB=2
    fi
    SHARED_MEM_CAP_GB=$(( CPUS_PER_TASK * MEM_PER_CPU_CAP_GB ))
    if [[ "$MEM_GB" -gt "$SHARED_MEM_CAP_GB" ]]; then
        if [[ -n "${MEM_GB_SET:-}" ]]; then
            echo "Error: --mem-gb ${MEM_GB} exceeds the Expanse ${PARTITION} limit of ~${MEM_PER_CPU_CAP_GB}G per core (${SHARED_MEM_CAP_GB}G for ${CPUS_PER_TASK} CPUs)."
            echo "       Raise --cpus-per-task, lower --mem-gb, or use --partition compute."
            exit 1
        fi
        MEM_GB="$SHARED_MEM_CAP_GB"
        echo "Capped memory to ${MEM_GB}GB for Expanse ${PARTITION} (~${MEM_PER_CPU_CAP_GB}G per core)"
    fi
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
if ! [[ "$REPORT_INTERVAL_S" =~ ^[1-9][0-9]*$ ]]; then
    echo "report_interval_s must be a positive integer"
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

# The combine and reporter are small; on Expanse the compute partition bills a whole
# node per job, so never inherit it from the simulation.
if [[ "$CLUSTER" == "expanse" ]] && [[ "$PARTITION" == "compute" ]]; then
    HELPER_PARTITION="shared"
else
    HELPER_PARTITION="$PARTITION"
fi

COMBINE_PARTITION="${COMBINE_PARTITION:-$HELPER_PARTITION}"
if [[ "$COMBINE_AFTER" -eq 1 ]]; then
    if ! [[ "$COMBINE_CPUS" =~ ^[1-9][0-9]*$ ]]; then echo "combine_cpus must be a positive integer"; exit 1; fi
    if ! [[ "$COMBINE_MEM_GB" =~ ^[1-9][0-9]*$ ]]; then echo "combine_mem_gb must be a positive integer"; exit 1; fi
    if ! [[ "$COMBINE_GROUPS" =~ ^[0-9]+$ ]]; then echo "combine_groups must be a non-negative integer"; exit 1; fi
    if [[ "$COMBINE_GROUPS" -gt "$JOB_COUNT" ]]; then
        echo "combine_groups (${COMBINE_GROUPS}) cannot exceed job_count (${JOB_COUNT})"
        exit 1
    fi
    if [[ ! " $VALID_PARTITIONS " =~ " $COMBINE_PARTITION " ]]; then
        echo "Error: unsupported combine partition '$COMBINE_PARTITION' for cluster '$CLUSTER'"
        exit 1
    fi
fi

REPORT_PARTITION="${REPORT_PARTITION:-$HELPER_PARTITION}"
if [[ "$AUTO_REPORT" -eq 1 ]]; then
    if ! [[ "$REPORT_CPUS" =~ ^[1-9][0-9]*$ ]]; then echo "report_cpus must be a positive integer"; exit 1; fi
    if ! [[ "$REPORT_MEM_GB" =~ ^[1-9][0-9]*$ ]]; then echo "report_mem_gb must be a positive integer"; exit 1; fi
    if [[ ! " $VALID_PARTITIONS " =~ " $REPORT_PARTITION " ]]; then
        echo "Error: unsupported report partition '$REPORT_PARTITION' for cluster '$CLUSTER'"
        exit 1
    fi
fi

if [[ ! -f "$SIM_WRAPPER" ]]; then
    echo "Error: wrapper script not found: $SIM_WRAPPER"
    exit 1
fi

EXPECTED_EVENTS_PER_LOOP="$(awk -v a="$SOURCE_ACTIVITY_BQ" -v d="$CHUNK_DURATION_S" -v c="$NUM_CHUNKS" -v t="$CPUS_PER_TASK" 'BEGIN { printf "%.0f", a * d * c * t }')"
EXPECTED_EVENTS_PER_JOB="$(awk -v e="$EXPECTED_EVENTS_PER_LOOP" -v l="$NUM_LOOPS" 'BEGIN { printf "%.0f", e * l }')"
EXPECTED_EVENTS_TOTAL="$(awk -v e="$EXPECTED_EVENTS_PER_JOB" -v j="$JOB_COUNT" 'BEGIN { printf "%.0f", e * j }')"
EXPECTED_EVENTS_PER_LOOP_FMT="$(awk -v n="$EXPECTED_EVENTS_PER_LOOP" 'function comma(x, s, r) { s = x ""; while (length(s) > 3) { r = "," substr(s, length(s)-2, 3) r; s = substr(s, 1, length(s)-3) } return s r } BEGIN { print comma(n) }')"
EXPECTED_EVENTS_PER_JOB_FMT="$(awk -v n="$EXPECTED_EVENTS_PER_JOB" 'function comma(x, s, r) { s = x ""; while (length(s) > 3) { r = "," substr(s, length(s)-2, 3) r; s = substr(s, 1, length(s)-3) } return s r } BEGIN { print comma(n) }')"
EXPECTED_EVENTS_TOTAL_FMT="$(awk -v n="$EXPECTED_EVENTS_TOTAL" 'function comma(x, s, r) { s = x ""; while (length(s) > 3) { r = "," substr(s, length(s)-2, 3) r; s = substr(s, 1, length(s)-3) } return s r } BEGIN { print comma(n) }')"

if [[ -z "$BATCH_ID" ]]; then
    BATCH_ID="batch_$(date +%Y%m%d_%H%M%S)"
elif ! [[ "$BATCH_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "batch_id may contain only letters, numbers, dots, underscores, and hyphens" >&2
    exit 1
fi
LOG_DIR="${REPO_ROOT}/submit_slurm/logs/${BATCH_ID}"
DATA_DIR="${SCRATCH_ROOT}/${OUTPUT_SUBDIR}/${BATCH_ID}"
CONTAINER_SIF="${CONTAINER_SIF:-${REPO_ROOT}/submit_slurm/qmirt-gate-10-sim-sif_v1.0.0.sif}"
SBATCH_FILE="${LOG_DIR}/${SIM_LABEL}_sim_slurm.sbatch"

if [[ ! -f "$CONTAINER_SIF" ]]; then
    echo "Error: Apptainer SIF not found at $CONTAINER_SIF"
    exit 1
fi

mkdir -p "$LOG_DIR"
if [[ "$DRY_RUN" -eq 1 ]]; then
    # SCRATCH_ROOT may be unreachable off-cluster, so only previews skip creating it.
    mkdir -p "$DATA_DIR" 2>/dev/null || echo "Dry run: skipping creation of $DATA_DIR"
else
    mkdir -p "$DATA_DIR"
fi

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
$(if [[ -n "$ARRAY_DEPENDENCY" ]]; then printf '#SBATCH --dependency=%s\n' "$ARRAY_DEPENDENCY"; fi)

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
EOF

if [[ -n "$LOCAL_SCRATCH_TEMPLATE" ]]; then
    # Written unexpanded on purpose: resolved per array element at run time.
    echo "export LOCAL_SCRATCH_ROOT=\"${LOCAL_SCRATCH_TEMPLATE}\"" >> "$SBATCH_FILE"
fi

cat >> "$SBATCH_FILE" <<EOF

bash "${SIM_WRAPPER}"
EOF

COMBINE_SBATCH_FILE="${LOG_DIR}/${SIM_LABEL}_campaign_combine.sbatch"
GROUP_SBATCH_FILE="${LOG_DIR}/${SIM_LABEL}_group_combine.sbatch"
if [[ "$COMBINE_AFTER" -eq 1 ]]; then
    if [[ "$COMBINE_GROUPS" -gt 1 ]]; then
        FINAL_STAGE="groups"
        FINAL_EXPECTED="$COMBINE_GROUPS"
    else
        FINAL_STAGE="tasks"
        FINAL_EXPECTED="$JOB_COUNT"
    fi

    if [[ "$COMBINE_GROUPS" -gt 1 ]]; then
        cat > "$GROUP_SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=${SIM_LABEL}_group_combine
#SBATCH --array=0-$((COMBINE_GROUPS - 1))
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${COMBINE_CPUS}
#SBATCH --time=${COMBINE_TIME_LIMIT}
#SBATCH --mem=${COMBINE_MEM_GB}G
#SBATCH --partition=${COMBINE_PARTITION}
EOF
        if [[ -n "$ACCOUNT" ]]; then
            echo "#SBATCH --account=${ACCOUNT}" >> "$GROUP_SBATCH_FILE"
        fi
        cat >> "$GROUP_SBATCH_FILE" <<EOF
#SBATCH --output=${LOG_DIR}/group_combine_%A_%a.out
#SBATCH --error=${LOG_DIR}/group_combine_%A_%a.err

export CONTAINER_SIF="${CONTAINER_SIF}"

bash "${SCRIPT_DIR}/wrapper_campaign_combine.sh" \\
    --campaign-dir "${DATA_DIR}" \\
    --input-stage tasks \\
    --shard-index \${SLURM_ARRAY_TASK_ID} \\
    --shard-count ${COMBINE_GROUPS} \\
    --output-dir "${DATA_DIR}/group_\${SLURM_ARRAY_TASK_ID}"
EOF
    fi

    cat > "$COMBINE_SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=${SIM_LABEL}_combine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${COMBINE_CPUS}
#SBATCH --time=${COMBINE_TIME_LIMIT}
#SBATCH --mem=${COMBINE_MEM_GB}G
#SBATCH --partition=${COMBINE_PARTITION}
EOF
    if [[ -n "$ACCOUNT" ]]; then
        echo "#SBATCH --account=${ACCOUNT}" >> "$COMBINE_SBATCH_FILE"
    fi
    cat >> "$COMBINE_SBATCH_FILE" <<EOF
#SBATCH --output=${LOG_DIR}/combine_%j.out
#SBATCH --error=${LOG_DIR}/combine_%j.err

export CONTAINER_SIF="${CONTAINER_SIF}"

bash "${SCRIPT_DIR}/wrapper_campaign_combine.sh" \\
    --campaign-dir "${DATA_DIR}" \\
    --input-stage ${FINAL_STAGE} \\
    --expected-tasks ${FINAL_EXPECTED}
EOF
fi

# Record exact provenance for this campaign: repo state, container image, and
# every resolved simulation/scheduler parameter, so a run can be reproduced later.
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]]; then
    GIT_COMMIT="${GIT_COMMIT}-dirty"
fi
CONTAINER_SHA256="$(sha256sum "$CONTAINER_SIF" 2>/dev/null | cut -d' ' -f1)"
MANIFEST_FILE="${LOG_DIR}/campaign_manifest.json"
cat > "$MANIFEST_FILE" <<EOF
{
  "batch_id": "${BATCH_ID}",
    "campaign_group_id": "${CAMPAIGN_GROUP_ID}",
    "campaign_part_index": ${CAMPAIGN_PART_INDEX:-null},
    "campaign_part_count": ${CAMPAIGN_PART_COUNT:-null},
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "repo_git_commit": "${GIT_COMMIT}",
  "container_sif": "${CONTAINER_SIF}",
  "container_sha256": "${CONTAINER_SHA256}",
  "sim_wrapper": "${SIM_WRAPPER}",
  "sim_label": "${SIM_LABEL}",
  "sim_type": "${SIM_TYPE}",
  "sim_python_script": "${SIM_PYTHON_SCRIPT}",
  "cluster": "${CLUSTER}",
  "partition": "${PARTITION}",
  "account": "${ACCOUNT}",
  "job_count": ${JOB_COUNT},
  "cpus_per_task": ${CPUS_PER_TASK},
  "time_limit": "${TIME_LIMIT}",
  "mem_gb": ${MEM_GB},
  "concurrent_limit": "${CONCURRENT_LIMIT}",
  "source_activity_bq": ${SOURCE_ACTIVITY_BQ},
  "chunk_duration_s": ${CHUNK_DURATION_S},
  "num_chunks": ${NUM_CHUNKS},
  "num_loops": ${NUM_LOOPS},
    "expected_events_per_loop": ${EXPECTED_EVENTS_PER_LOOP},
    "expected_events_per_job": ${EXPECTED_EVENTS_PER_JOB},
    "expected_events_total": ${EXPECTED_EVENTS_TOTAL},
    "geometry_provenance_file": "geometry_provenance.json",
  "sparse_srm": ${SPARSE_SRM},
  "srm_fov_size_mm": ${SRM_FOV_SIZE_MM},
  "profile_resources": ${PROFILE_RESOURCES},
  "profile_interval_s": ${PROFILE_INTERVAL_S},
  "combine_after": ${COMBINE_AFTER},
  "combine_groups": ${COMBINE_GROUPS},
  "auto_report": ${AUTO_REPORT},
  "report_interval_s": ${REPORT_INTERVAL_S},
  "report_cpus": ${REPORT_CPUS},
  "report_mem_gb": ${REPORT_MEM_GB},
  "report_time_limit": "${REPORT_TIME_LIMIT}",
  "report_partition": "${REPORT_PARTITION}",
  "test_mode": ${TEST_MODE},
  "scratch_root": "${SCRATCH_ROOT}",
  "local_scratch_template": "${LOCAL_SCRATCH_TEMPLATE}",
  "data_dir": "${DATA_DIR}",
  "log_dir": "${LOG_DIR}"
}
EOF
if [[ "$DRY_RUN" -eq 0 ]]; then
    cp "$MANIFEST_FILE" "${DATA_DIR}/campaign_manifest.json"
fi

update_submission_manifests() {
    local array_job_id="$1"
    local report_job_id="${2:-}"
    local group_job_id="${3:-}"
    local combine_job_id="${4:-}"
    local manifest
    for manifest in "$MANIFEST_FILE" "${DATA_DIR}/campaign_manifest.json"; do
        [[ -f "$manifest" ]] || continue
        python3 - "$manifest" "$array_job_id" "$report_job_id" "$group_job_id" "$combine_job_id" <<'PY'
import json
import os
import sys

path, array_job_id, report_job_id, group_job_id, combine_job_id = sys.argv[1:]
with open(path) as handle:
    manifest = json.load(handle)
manifest["slurm_jobs"] = {
    "array": array_job_id,
    "progress_reporter": report_job_id or None,
    "group_combine": group_job_id or None,
    "campaign_combine": combine_job_id or None,
}
temporary_path = f"{path}.tmp"
with open(temporary_path, "w") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")
os.replace(temporary_path, path)
PY
    done
}

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
echo "Shared campaign root: ${SCRATCH_ROOT}"
if [[ -n "$LOCAL_SCRATCH_TEMPLATE" ]]; then echo "Node-local scratch: ${LOCAL_SCRATCH_TEMPLATE}"; fi
echo "Source activity: ${SOURCE_ACTIVITY_BQ} Bq"
echo "Chunk duration: ${CHUNK_DURATION_S} s"
echo "Num chunks: ${NUM_CHUNKS}"
echo "Num loops: ${NUM_LOOPS}"
echo "Sparse SRM: ${SPARSE_SRM}"
if [[ "$SPARSE_SRM" == "1" ]]; then echo "SRM FOV extent: ${SRM_FOV_SIZE_MM} mm"; fi
echo "Expected events per loop (approx): ${EXPECTED_EVENTS_PER_LOOP_FMT}"
echo "Expected events per job (approx): ${EXPECTED_EVENTS_PER_JOB_FMT}"
echo "Expected events across all jobs (approx): ${EXPECTED_EVENTS_TOTAL_FMT}"
if [[ "$TEST_MODE" -eq 1 ]]; then echo "*** TEST MODE ENABLED ***"; fi
echo "Created output folder: $DATA_DIR"
echo "Created log folder:    $LOG_DIR"
echo "Created sbatch file:   $SBATCH_FILE"
echo "Created manifest file: $MANIFEST_FILE"
if [[ "$COMBINE_AFTER" -eq 1 ]]; then
    echo "Campaign combine:      enabled (${COMBINE_PARTITION}, ${COMBINE_CPUS} CPUs, ${COMBINE_MEM_GB}G, ${COMBINE_TIME_LIMIT})"
    if [[ "$COMBINE_GROUPS" -gt 1 ]]; then
        echo "Tree reduction:        ${COMBINE_GROUPS} parallel groups, then one final merge"
        echo "Created group file:    $GROUP_SBATCH_FILE"
    fi
    echo "Created combine file:  $COMBINE_SBATCH_FILE"
fi
if [[ "$AUTO_REPORT" -eq 1 ]]; then
    echo "Auto report:           enabled (small Slurm job, ${REPORT_CPUS} CPU/${REPORT_MEM_GB}G/${REPORT_TIME_LIMIT} on ${REPORT_PARTITION}, refresh every ${REPORT_INTERVAL_S}s)"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "--- sbatch file preview ---"
    cat "$SBATCH_FILE"
    if [[ "$COMBINE_AFTER" -eq 1 ]]; then
        if [[ "$COMBINE_GROUPS" -gt 1 ]]; then
            echo "--- group combine sbatch preview ---"
            cat "$GROUP_SBATCH_FILE"
        fi
        echo "--- campaign combine sbatch preview ---"
        cat "$COMBINE_SBATCH_FILE"
    fi
    if [[ "$AUTO_REPORT" -eq 1 ]]; then
        echo "--- progress reporter sbatch preview (job IDs are placeholders until submission) ---"
        echo "bash ${SCRIPT_DIR}/wrapper_generate_progress_report.sh --campaign-dir ${DATA_DIR} --expected-tasks ${JOB_COUNT} --job-id <ARRAY_JOB_ID> --watch-job-id <LAST_STAGE_JOB_ID> --output ${DATA_DIR}/progress.json --interval-s ${REPORT_INTERVAL_S}"
    fi
    exit 0
fi

ARRAY_JOB_ID="$(sbatch --parsable "$SBATCH_FILE")"
echo "Submitted array job ${ARRAY_JOB_ID}"

if [[ "$COMBINE_AFTER" -eq 1 ]]; then
    # afterany: aggregate whatever succeeded rather than requiring every element to pass.
    COMBINE_DEPENDENCY="$ARRAY_JOB_ID"
    if [[ "$COMBINE_GROUPS" -gt 1 ]]; then
        GROUP_JOB_ID="$(sbatch --parsable \
            --dependency=afterany:"${ARRAY_JOB_ID}" \
            --kill-on-invalid-dep=yes \
            "$GROUP_SBATCH_FILE")"
        echo "Submitted group combine array ${GROUP_JOB_ID} (afterany:${ARRAY_JOB_ID})"
        COMBINE_DEPENDENCY="$GROUP_JOB_ID"
    fi
    COMBINE_JOB_ID="$(sbatch --parsable \
        --dependency=afterany:"${COMBINE_DEPENDENCY}" \
        --kill-on-invalid-dep=yes \
        "$COMBINE_SBATCH_FILE")"
    echo "Submitted campaign combine job ${COMBINE_JOB_ID} (afterany:${COMBINE_DEPENDENCY})"
fi

if [[ "$AUTO_REPORT" -eq 1 ]]; then
    # Real job IDs are only known now, so the reporter sbatch is generated post-submission
    # (unlike the combine sbatch files, its body must embed --job-id/--watch-job-id literally).
    WATCH_JOB_ID="${COMBINE_JOB_ID:-$ARRAY_JOB_ID}"
    REPORT_SBATCH_FILE="${LOG_DIR}/${SIM_LABEL}_progress_report.sbatch"
    cat > "$REPORT_SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=${SIM_LABEL}_progress_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${REPORT_CPUS}
#SBATCH --time=${REPORT_TIME_LIMIT}
#SBATCH --mem=${REPORT_MEM_GB}G
#SBATCH --partition=${REPORT_PARTITION}
EOF
    if [[ -n "$ACCOUNT" ]]; then
        echo "#SBATCH --account=${ACCOUNT}" >> "$REPORT_SBATCH_FILE"
    fi
    cat >> "$REPORT_SBATCH_FILE" <<EOF
#SBATCH --output=${LOG_DIR}/progress_report_%j.out
#SBATCH --error=${LOG_DIR}/progress_report_%j.err

export CONTAINER_SIF="${CONTAINER_SIF}"

bash "${SCRIPT_DIR}/wrapper_generate_progress_report.sh" \\
    --campaign-dir "${DATA_DIR}" \\
    --expected-tasks ${JOB_COUNT} \\
    --job-id ${ARRAY_JOB_ID} \\
    --watch-job-id ${WATCH_JOB_ID} \\
    --output "${DATA_DIR}/progress.json" \\
    --interval-s ${REPORT_INTERVAL_S}
EOF
    REPORT_JOB_ID="$(sbatch --parsable \
        --dependency=after:"${ARRAY_JOB_ID}" \
        "$REPORT_SBATCH_FILE")"
    echo "Submitted progress reporter job ${REPORT_JOB_ID} (${REPORT_CPUS} CPU, ${REPORT_MEM_GB}G, ${REPORT_TIME_LIMIT}, partition ${REPORT_PARTITION}; watching job ${WATCH_JOB_ID})"
    echo "Progress file: ${DATA_DIR}/progress.json"
    echo "Reporter sbatch: ${REPORT_SBATCH_FILE}"
fi

update_submission_manifests "$ARRAY_JOB_ID" "${REPORT_JOB_ID:-}" "${GROUP_JOB_ID:-}" "${COMBINE_JOB_ID:-}"
echo "Updated campaign manifests with submitted Slurm job IDs"