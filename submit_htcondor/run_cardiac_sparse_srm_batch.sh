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

# Shown in --help; the real check happens after parsing, once condor_config_val
# can be consulted on an access point.
MAX_JOBS_PER_SUBMISSION_HINT="$(condor_config_val -schedd MAX_JOBS_PER_SUBMISSION 2>/dev/null || echo 20000)"

JOB_COUNT="1000"
NUM_LOOPS="1"
SOURCE_ACTIVITY_BQ="5e6"
NUM_CHUNKS="100"
CHUNK_DURATION_S="1.0"
FOV_SIZE_MM="210.0"
RESOLUTIONS_MM="1,1.5,2"
REQUEST_MEMORY="4GB"
REQUEST_DISK="10GB"
PROJECT_NAME="MGH_Sabet"
DATA_ROOT="/ospool/ap40/data/fang.han"
CONTAINER_IMAGE="osdf:///ospool/ap40/data/fang.han/qmirt-gate-10-sim.sif"
USE_OSDF=1
PAYLOAD_URL=""
# HTCondor has no Slurm-style "array%limit"; throttling is done by materializing
# only some of the cluster at a time.
MAX_IDLE="2000"
MAX_MATERIALIZE=""
DRY_RUN=0

usage() {
    echo "Usage: $0 [--job-count N] [--num-loops N] [--source-activity-bq VALUE]"
    echo "          [--num-chunks N] [--chunk-duration-s VALUE] [--fov-size-mm VALUE]"
    echo "          [--resolutions-mm 1,1.5,2] [--request-memory 4GB] [--request-disk 10GB]"
    echo "          [--data-root PATH] [--project-name NAME] [--container IMAGE] [--dry-run]"
    echo "          [--payload-url osdf:///...] [--no-osdf]"
    echo "          [--max-idle N] [--max-materialize N]"
    echo ""
    echo "Each job runs --num-loops independent simulations, converts each to a sparse"
    echo "SRM, deletes the ROOT files, and returns one tarball of partial SRMs."
    echo "--fov-size-mm is a sphere diameter; the SRM grid is the cube that contains it."
    echo ""
    echo "--max-idle throttles how many jobs sit idle at once (HTCondor's answer to"
    echo "Slurm's 'array%limit'); --max-materialize caps how many exist in the queue."
    echo "This access point allows ${MAX_JOBS_PER_SUBMISSION_HINT} jobs per submission, so larger campaigns"
    echo "must be split across several submissions."
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
        --max-idle) MAX_IDLE="$2"; shift 2 ;;
        --max-materialize) MAX_MATERIALIZE="$2"; shift 2 ;;
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

# A submission over the schedd limit is rejected outright, which is a slow way to
# find out after publishing a payload.
SUBMISSION_LIMIT="$(condor_config_val -schedd MAX_JOBS_PER_SUBMISSION 2>/dev/null || echo '')"
if [[ "$SUBMISSION_LIMIT" =~ ^[0-9]+$ ]] && [[ "$JOB_COUNT" -gt "$SUBMISSION_LIMIT" ]]; then
    echo "Error: --job-count ${JOB_COUNT} exceeds this access point's MAX_JOBS_PER_SUBMISSION (${SUBMISSION_LIMIT})." >&2
    echo "Split the campaign across several submissions, for example:" >&2
    echo "  for i in \$(seq 1 $(( (JOB_COUNT + SUBMISSION_LIMIT - 1) / SUBMISSION_LIMIT ))); do" >&2
    echo "      $0 --job-count ${SUBMISSION_LIMIT} ...   # same --payload-url for all of them" >&2
    echo "  done" >&2
    exit 2
fi
OWNER_LIMIT="$(condor_config_val -schedd MAX_JOBS_PER_OWNER 2>/dev/null || echo '')"
if [[ "$OWNER_LIMIT" =~ ^[0-9]+$ ]] && [[ "$JOB_COUNT" -gt "$OWNER_LIMIT" ]]; then
    echo "Error: --job-count ${JOB_COUNT} exceeds MAX_JOBS_PER_OWNER (${OWNER_LIMIT})." >&2
    exit 2
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

SRM_CONVERTER="$REPO_ROOT/payload/python/create_spect_sparse_srm_from_batch_root.py"
if [[ ! -f "$SRM_CONVERTER" ]]; then
    echo "Error: required SRM converter not found: $SRM_CONVERTER" >&2
    exit 1
fi

# Capture the resolved scanner geometry once in the durable batch directory.
# The OSPool access point has the campaign SIF available locally, while its host
# Python intentionally does not carry the simulation dependencies.
if [[ "$CONTAINER_IMAGE" == osdf:///* ]]; then
    LOCAL_CONTAINER_IMAGE="/${CONTAINER_IMAGE#osdf:///}"
else
    LOCAL_CONTAINER_IMAGE="$CONTAINER_IMAGE"
fi
if ! command -v apptainer >/dev/null 2>&1 || [[ ! -f "$LOCAL_CONTAINER_IMAGE" ]]; then
    echo "Error: cannot generate cardiac geometry provenance with $LOCAL_CONTAINER_IMAGE" >&2
    exit 1
fi
apptainer exec \
    --bind "$REPO_ROOT:$REPO_ROOT" \
    --bind "$DATA_DIR:$DATA_DIR" \
    --env "PYTHONPATH=$REPO_ROOT/qmirt/src" \
    "$LOCAL_CONTAINER_IMAGE" \
    python3 "$REPO_ROOT/payload/python/write_cardiac_spect_geometry_provenance.py" \
    --output "$DATA_DIR/geometry_provenance.json" \
    --fov-size-mm "$FOV_SIZE_MM"

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

# Throttle: keep the schedd and the pool from seeing the whole campaign at once.
max_idle = ${MAX_IDLE}
EOF

if [[ -n "$MAX_MATERIALIZE" ]]; then
    echo "max_materialize = ${MAX_MATERIALIZE}" >> "$SUB_FILE"
fi

cat >> "$SUB_FILE" <<EOF

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
    "geometry_provenance_file": "geometry_provenance.json",
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

submission_output="$(condor_submit "$SUB_FILE" out_dir="$DATA_DIR" log_dir="$LOG_DIR")"
printf '%s\n' "$submission_output"
if [[ "$submission_output" =~ [Cc]luster[[:space:]]+([0-9]+) ]]; then
    CONDOR_CLUSTER_ID="${BASH_REMATCH[1]}"
    python3 - "$DATA_DIR/campaign_manifest.json" "$CONDOR_CLUSTER_ID" <<'PY'
import json
import os
import sys

path = sys.argv[1]
with open(path) as handle:
    manifest = json.load(handle)
manifest["condor_cluster_id"] = sys.argv[2]
temporary_path = f"{path}.tmp"
with open(temporary_path, "w") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")
os.replace(temporary_path, path)
PY
    echo "Recorded HTCondor cluster ID ${CONDOR_CLUSTER_ID} in campaign manifest."
else
    echo "Warning: unable to parse HTCondor cluster ID from submission output." >&2
fi
