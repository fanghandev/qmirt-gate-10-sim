#!/bin/bash
# Submit a cardiac-SPECT sparse-SRM campaign to OSPool.
#
# Every job simulates independently and returns only a partial sparse SRM, so the
# ROOT files never leave the execute node. The workstation aggregates the partials
# with run_cardiac_campaign_combine.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

JOB_COUNT="1000"
NUM_LOOPS="1"
SOURCE_ACTIVITY_BQ="5e6"
NUM_CHUNKS="100"
CHUNK_DURATION_S="1.0"
FOV_SIZE_MM="150.0"
RESOLUTIONS_MM="1,1.5,2"
REQUEST_MEMORY="4GB"
REQUEST_DISK="10GB"
PROJECT_NAME="MGH_Sabet"
DATA_ROOT="/ospool/ap40/data/fang.han"
CONTAINER_IMAGE="osdf:///ospool/ap40/data/fang.han/qmirt-gate-10-sim.sif"
USE_OSDF=1
PAYLOAD_URL=""
DRY_RUN=0

usage() {
    echo "Usage: $0 [--job-count N] [--num-loops N] [--source-activity-bq VALUE]"
    echo "          [--num-chunks N] [--chunk-duration-s VALUE] [--fov-size-mm VALUE]"
    echo "          [--resolutions-mm 1,1.5,2] [--request-memory 4GB] [--request-disk 10GB]"
    echo "          [--data-root PATH] [--project-name NAME] [--container IMAGE] [--dry-run]"
    echo "          [--payload-url osdf:///...] [--no-osdf]"
    echo ""
    echo "Each job runs --num-loops independent simulations, converts each to a sparse"
    echo "SRM, deletes the ROOT files, and returns one tarball of partial SRMs."
    echo "--fov-size-mm is a sphere diameter; the SRM grid is the cube that contains it."
    echo ""
    echo "By default the payload is published to OSDF and pulled through the site cache,"
    echo "so the access point does not re-send it for every job. --no-osdf falls back to"
    echo "staging the files directly from the access point."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-count) JOB_COUNT="$2"; shift 2 ;;
        --num-loops) NUM_LOOPS="$2"; shift 2 ;;
        --source-activity-bq) SOURCE_ACTIVITY_BQ="$2"; shift 2 ;;
        --num-chunks) NUM_CHUNKS="$2"; shift 2 ;;
        --chunk-duration-s) CHUNK_DURATION_S="$2"; shift 2 ;;
        --fov-size-mm) FOV_SIZE_MM="$2"; shift 2 ;;
        --resolutions-mm) RESOLUTIONS_MM="$2"; shift 2 ;;
        --request-memory) REQUEST_MEMORY="$2"; shift 2 ;;
        --request-disk) REQUEST_DISK="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --project-name) PROJECT_NAME="$2"; shift 2 ;;
        --container) CONTAINER_IMAGE="$2"; shift 2 ;;
        --payload-url) PAYLOAD_URL="$2"; USE_OSDF=1; shift 2 ;;
        --no-osdf) USE_OSDF=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unexpected argument: $1" >&2; usage; exit 2 ;;
    esac
done

if ! [[ "$JOB_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "job_count must be a positive integer" >&2; exit 2
fi
if ! [[ "$NUM_LOOPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "num_loops must be a positive integer" >&2; exit 2
fi
if ! [[ "$NUM_CHUNKS" =~ ^[1-9][0-9]*$ ]]; then
    echo "num_chunks must be a positive integer" >&2; exit 2
fi

# Each resolution must divide the FOV evenly or the worker fails after simulating.
IFS=',' read -r -a resolution_list <<< "$RESOLUTIONS_MM"
for resolution in "${resolution_list[@]}"; do
    if [[ "$(awk -v f="$FOV_SIZE_MM" -v r="$resolution" \
        'BEGIN { n = f / r; print (r > 0 && n == int(n)) ? 1 : 0 }')" != "1" ]]; then
        echo "fov_size_mm ${FOV_SIZE_MM} is not divisible by resolution ${resolution}" >&2
        exit 2
    fi
done

BATCH_ID="batch_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="${DATA_ROOT}/cardiac_spect_srm/${BATCH_ID}"
LOG_DIR="logs/${BATCH_ID}"
SUB_FILE="${LOG_DIR}/cardiac_sparse_srm.sub"
STAGE_DIR="${LOG_DIR}/stage"

mkdir -p "$DATA_DIR" "$LOG_DIR"

# transfer_input_files is re-sent by the access point for every job, so keep it
# minimal: stage only what a cardiac job opens, or hand the whole payload to OSDF
# where the site cache serves it once per site instead of once per job.
if [[ "$USE_OSDF" -eq 1 ]]; then
    if [[ -z "$PAYLOAD_URL" ]]; then
        PAYLOAD_URL="$(bash "$SCRIPT_DIR/publish_payload_osdf.sh" | tail -1)"
    fi
    TRANSFER_INPUT="$PAYLOAD_URL"
    INPUT_DESCRIPTION="OSDF: $PAYLOAD_URL"
else
    STAGE_DIR="${LOG_DIR}/stage"
    rm -rf "$STAGE_DIR"
    mkdir -p "$STAGE_DIR/payload/python" "$STAGE_DIR/persistent_data" "$STAGE_DIR/qmirt"
    cp "$REPO_ROOT"/payload/python/*.py "$STAGE_DIR/payload/python/"
    cp -r "$REPO_ROOT/persistent_data/cardiac_spect" "$STAGE_DIR/persistent_data/"
    cp "$REPO_ROOT/persistent_data/GateMaterials.db" "$STAGE_DIR/persistent_data/"
    # search_dir_up() walks up from the payload script, so the directory must keep
    # the name "persistent_data" in the sandbox root.
    cp -r "$REPO_ROOT/qmirt/src" "$STAGE_DIR/qmirt/"
    STAGE_DIR_ABS="$(cd "$STAGE_DIR" && pwd)"
    STAGE_MB="$(du -sm "$STAGE_DIR" | cut -f1)"
    TRANSFER_INPUT="${STAGE_DIR_ABS}/payload, ${STAGE_DIR_ABS}/persistent_data, ${STAGE_DIR_ABS}/qmirt"
    INPUT_DESCRIPTION="access point: ${STAGE_MB} MB per job from ${STAGE_DIR_ABS}"
fi

cat > "$SUB_FILE" <<EOF
executable = ${SCRIPT_DIR}/wrapper_cardiac_sparse_srm.sh
arguments = \$(ClusterId) \$(ProcId) ${NUM_LOOPS} ${SOURCE_ACTIVITY_BQ} ${NUM_CHUNKS} ${CHUNK_DURATION_S} ${FOV_SIZE_MM} ${RESOLUTIONS_MM}

+SingularityImage = "${CONTAINER_IMAGE}"
+ProjectName = "${PROJECT_NAME}"
transfer_input_files = ${TRANSFER_INPUT}

# Only the partial SRMs come back; ROOT files are deleted on the execute node.
transfer_output_files = srm_c_\$(ClusterId)_p_\$(ProcId).tar.gz
transfer_output_remaps = "srm_c_\$(ClusterId)_p_\$(ProcId).tar.gz = \$(out_dir)/srm_c_\$(ClusterId)_p_\$(ProcId).tar.gz"

log = \$(log_dir)/job_\$(ClusterId)_\$(ProcId).log
output = \$(log_dir)/job_\$(ClusterId)_\$(ProcId).out
error = \$(log_dir)/job_\$(ClusterId)_\$(ProcId).err

request_cpus = 1
request_memory = ${REQUEST_MEMORY}
request_disk = ${REQUEST_DISK}

# OSPool preemption is routine; retry rather than leaving holes in the campaign.
max_retries = 5
requirements = (HAS_SINGULARITY == True)

queue ${JOB_COUNT}
EOF

cat > "${DATA_DIR}/campaign_manifest.json" <<EOF
{
  "batch_id": "${BATCH_ID}",
  "scanner": "cardiac_spect",
  "job_count": ${JOB_COUNT},
  "num_loops": ${NUM_LOOPS},
  "source_activity_bq": "${SOURCE_ACTIVITY_BQ}",
  "num_chunks": ${NUM_CHUNKS},
  "chunk_duration_s": ${CHUNK_DURATION_S},
  "fov_size_mm": ${FOV_SIZE_MM},
  "resolutions_mm": "${RESOLUTIONS_MM}",
  "expected_partials": $((JOB_COUNT * NUM_LOOPS)),
  "payload_input": "${TRANSFER_INPUT}",
  "git_commit": "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)",
  "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "Batch id:        $BATCH_ID"
echo "Data folder:     $DATA_DIR"
echo "Log folder:      $LOG_DIR"
echo "Submit file:     $SUB_FILE"
echo "Job input:       $INPUT_DESCRIPTION"
echo "Jobs:            $JOB_COUNT x $NUM_LOOPS loop(s) = $((JOB_COUNT * NUM_LOOPS)) partial SRMs"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ""
    echo "--- $SUB_FILE ---"
    cat "$SUB_FILE"
    echo "Dry run: not submitting."
    exit 0
fi

condor_submit "$SUB_FILE" out_dir="$DATA_DIR" log_dir="$LOG_DIR"
