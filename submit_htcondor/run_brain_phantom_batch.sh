#!/bin/bash
# Submit a brain SPECT phantom acquisition to OSPool: one job per time slice
# (wrapper_brain_phantom_sim.sh). Run on the access point.
#
#   run_brain_phantom_batch.sh --label NAME --payload-url osdf:///.../qmirt-brain-phantom-payload-<hash>.tar.gz \
#       --phantom-spec small_jaszczak --activity-bq 5.55e8 [--activity-region all|brain] \
#       --slice-s 0.9 --job-count 1000 [--slice-offset 0] [--chunk-s SLICE] [--memory 3GB] [--keep-root] [--dry-run]
#
# Outputs: /ospool/ap40/data/fang.han/brain_phantom/<label>_<UTC>/ (campaign.env,
# campaign_manifest.json, phantom_c_<cluster>_p_<proc>.tar.gz); logs under submit_htcondor/logs/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="/ospool/ap40/data/fang.han/brain_phantom"
CONTAINER_IMAGE="osdf:///ospool/ap40/data/fang.han/qmirt-gate-10-sim.sif"
LABEL="" PAYLOAD_URL="" SPEC="" ACTIVITY="" REGION="all" SLICE_S="" JOBS="" OFFSET=0 CHUNK_S="" MEMORY="3GB" KEEP_ROOT=0 DRY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label) LABEL="$2"; shift 2 ;;
        --payload-url) PAYLOAD_URL="$2"; shift 2 ;;
        --phantom-spec) SPEC="$2"; shift 2 ;;
        --activity-bq) ACTIVITY="$2"; shift 2 ;;
        --activity-region) REGION="$2"; shift 2 ;;
        --slice-s) SLICE_S="$2"; shift 2 ;;
        --job-count) JOBS="$2"; shift 2 ;;
        --slice-offset) OFFSET="$2"; shift 2 ;;
        --chunk-s) CHUNK_S="$2"; shift 2 ;;
        --memory) MEMORY="$2"; shift 2 ;;
        --keep-root) KEEP_ROOT=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n 2,11p "$0"; exit 0 ;;
        *) echo "Unexpected argument: $1" >&2; exit 2 ;;
    esac
done
for v in LABEL PAYLOAD_URL SPEC ACTIVITY SLICE_S JOBS; do
    [[ -n "${!v}" ]] || { echo "Missing --$(echo "$v" | tr 'A-Z_' 'a-z-')" >&2; exit 2; }
done
CHUNK_S="${CHUNK_S:-$SLICE_S}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CAMPAIGN_DIR="${DATA_ROOT}/${LABEL}_${STAMP}"
LOG_DIR="${SCRIPT_DIR}/logs/brain_phantom/${LABEL}_${STAMP}"
mkdir -p "$CAMPAIGN_DIR" "$LOG_DIR"
ENV_FILE="$CAMPAIGN_DIR/campaign.env"
cat > "$ENV_FILE" <<EOF
PHANTOM_SPEC=${SPEC}
ACTIVITY_BQ=${ACTIVITY}
ACTIVITY_REGION=${REGION}
SLICE_S=${SLICE_S}
SLICE_OFFSET=${OFFSET}
CHUNK_S=${CHUNK_S}
PHYSICS_LIST=G4EmStandardPhysics_option4
KEEP_ROOT=${KEEP_ROOT}
EOF

SUB_FILE="$LOG_DIR/brain_phantom.sub"
cat > "$SUB_FILE" <<EOF
executable = ${SCRIPT_DIR}/wrapper_brain_phantom_sim.sh
arguments = \$(ClusterId) \$(ProcId) campaign.env
+SingularityImage = "${CONTAINER_IMAGE}"
transfer_input_files = ${PAYLOAD_URL}, ${ENV_FILE}
transfer_output_files = phantom_c_\$(ClusterId)_p_\$(ProcId).tar.gz
transfer_output_remaps = "phantom_c_\$(ClusterId)_p_\$(ProcId).tar.gz = ${CAMPAIGN_DIR}/phantom_c_\$(ClusterId)_p_\$(ProcId).tar.gz"
log = ${LOG_DIR}/job_\$(ClusterId)_\$(ProcId).log
output = ${LOG_DIR}/job_\$(ClusterId)_\$(ProcId).out
error = ${LOG_DIR}/job_\$(ClusterId)_\$(ProcId).err
request_cpus = 1
request_memory = ${MEMORY}
request_disk = 6GB
queue ${JOBS}
EOF

cat > "$CAMPAIGN_DIR/campaign_manifest.json" <<EOF
{
  "label": "${LABEL}", "created_utc": "${STAMP}", "git_commit": "$(git -C "$REPO_ROOT" rev-parse --short HEAD)",
  "payload_input": "${PAYLOAD_URL}", "container": "${CONTAINER_IMAGE}",
  "phantom_spec": "${SPEC}", "activity_bq": "${ACTIVITY}", "activity_region": "${REGION}",
  "slice_s": "${SLICE_S}", "slice_offset": ${OFFSET}, "chunk_s": "${CHUNK_S}", "job_count": ${JOBS},
  "acquisition_covered_s": [$(python3 -c "print(${OFFSET} * ${SLICE_S})"), $(python3 -c "print((${OFFSET} + ${JOBS}) * ${SLICE_S})")],
  "submit_file": "${SUB_FILE}"
}
EOF
echo "Campaign: $CAMPAIGN_DIR"
if [[ "$DRY" -eq 1 ]]; then
    echo "Dry run; submit file: $SUB_FILE"; cat "$SUB_FILE"; exit 0
fi
condor_submit "$SUB_FILE" | tee "$CAMPAIGN_DIR/condor_submit.txt"
