#!/bin/bash
# Submit a brain SPECT phantom acquisition to OSPool: one job per time slice
# (wrapper_brain_phantom_sim.sh). Run on the access point.
#
#   run_brain_phantom_batch.sh --label NAME --payload-url osdf:///.../qmirt-brain-phantom-payload-<hash>.tar.gz \
#       --phantom-spec small_jaszczak --activity-bq 5.55e8 [--activity-region all|brain] \
#       --slice-s 0.9 --job-count 1000 [--slice-offset 0] [--chunk-s SLICE] [--memory 3GB] [--keep-root] [--dry-run]
#   run_brain_phantom_batch.sh --into CAMPAIGN_DIR --slices 3,17,250 [--memory 3GB]   # resubmit slices
#
# Jobs need x86_64-v3/v4 nodes (the container's polars crashes with SIGILL on v2) and go back to
# the queue when they exit non-zero (up to 5 starts).
# Outputs: /ospool/ap40/data/fang.han/brain_phantom/<label>_<UTC>/ (campaign.env,
# campaign_manifest.json, phantom_c_<cluster>_p_<proc>.tar.gz); logs under submit_htcondor/logs/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="/ospool/ap40/data/fang.han/brain_phantom"
CONTAINER_IMAGE="osdf:///ospool/ap40/data/fang.han/qmirt-gate-10-sim.sif"
LABEL="" PAYLOAD_URL="" SPEC="" ACTIVITY="" REGION="all" SLICE_S="" JOBS="" OFFSET=0 CHUNK_S="" MEMORY="3GB" KEEP_ROOT=0 DRY=0
INTO="" SLICE_LIST=""

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
        --into) INTO="$2"; shift 2 ;;
        --slices) SLICE_LIST="$2"; shift 2 ;;
        -h|--help) sed -n 2,11p "$0"; exit 0 ;;
        *) echo "Unexpected argument: $1" >&2; exit 2 ;;
    esac
done
REQUIREMENTS='((TARGET.Microarch == "x86_64-v3") || (TARGET.Microarch == "x86_64-v4"))'
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ -n "$INTO" ]]; then  # resubmit given slices into an existing campaign
    [[ -n "$SLICE_LIST" && -f "$INTO/campaign.env" ]] || { echo "--into needs an existing campaign dir and --slices" >&2; exit 2; }
    CAMPAIGN_DIR="$INTO"
    LOG_DIR="${SCRIPT_DIR}/logs/brain_phantom/$(basename "$INTO")_resubmit_${STAMP}"
    mkdir -p "$LOG_DIR"
    PAYLOAD_URL="$(python3 -c "import json; print(json.load(open('$INTO/campaign_manifest.json'))['payload_input'])")"
    tr ',' '\n' <<<"$SLICE_LIST" | grep -v '^$' > "$LOG_DIR/slices.txt"
    SUB_FILE="$LOG_DIR/brain_phantom_resubmit.sub"
    cat > "$SUB_FILE" <<EOF
executable = ${SCRIPT_DIR}/wrapper_brain_phantom_sim.sh
arguments = \$(ClusterId) \$(ProcId) campaign.env \$(SLICE)
+SingularityImage = "${CONTAINER_IMAGE}"
requirements = ${REQUIREMENTS}
on_exit_remove = (ExitCode == 0) || (NumJobStarts >= 5)
periodic_release = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 1800)
transfer_input_files = ${PAYLOAD_URL}, ${INTO}/campaign.env
transfer_output_files = phantom_c_\$(ClusterId)_p_\$(ProcId).tar.gz
transfer_output_remaps = "phantom_c_\$(ClusterId)_p_\$(ProcId).tar.gz = ${INTO}/phantom_c_\$(ClusterId)_p_\$(ProcId).tar.gz"
log = ${LOG_DIR}/job_\$(ClusterId)_\$(ProcId).log
output = ${LOG_DIR}/job_\$(ClusterId)_\$(ProcId).out
error = ${LOG_DIR}/job_\$(ClusterId)_\$(ProcId).err
request_cpus = 1
request_memory = ${MEMORY}
request_disk = 6GB
queue SLICE from ${LOG_DIR}/slices.txt
EOF
    echo "Resubmitting $(wc -l < "$LOG_DIR/slices.txt") slices into $INTO"
    if [[ "$DRY" -eq 1 ]]; then cat "$SUB_FILE"; exit 0; fi
    condor_submit "$SUB_FILE" | tee -a "$INTO/resubmissions.txt"
    exit 0
fi

for v in LABEL PAYLOAD_URL SPEC ACTIVITY SLICE_S JOBS; do
    [[ -n "${!v}" ]] || { echo "Missing --$(echo "$v" | tr 'A-Z_' 'a-z-')" >&2; exit 2; }
done
CHUNK_S="${CHUNK_S:-$SLICE_S}"

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
requirements = ${REQUIREMENTS}
on_exit_remove = (ExitCode == 0) || (NumJobStarts >= 5)
periodic_release = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 1800)
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
