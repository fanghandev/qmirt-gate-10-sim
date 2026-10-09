#!/bin/bash
# Submit a brain SPECT phantom acquisition on MGB ERIS (Slurm + Apptainer). Each array task
# runs --procs-per-task single-threaded simulations side by side, one time slice each
# (gate_sim_brain_spect_boolean.py --mode phantom), each reduced to list-mode (the ROOT file
# is deleted). Single-threaded because the container's opengate 10.1.0 scatter counter
# (ProcessDefinedStepInVolumeAttribute) refuses multi-threaded runs. Run on an ERIS login node.
#
#   run_brain_phantom_eris.sh --label NAME --payload qmirt-brain-phantom-payload-<hash>.tar.gz \
#       --phantom-spec small_jaszczak --activity-bq 5.55e8 [--activity-region all|brain] \
#       --slice-s 0.9 --slice-count 1000 --procs-per-task 16 --partition PART [--time 12:00:00] \
#       [--mem-per-proc 3G] [--account hsabet] [--slice-offset 0] [--chunk-s S] [--root DIR] [--dry-run]
#
# A slice runs as equal GATE runs (chunks) of at most 1 s unless --chunk-s is given.
# The payload tarball is the OSPool one (submit_htcondor/publish_brain_phantom_payload.sh);
# it is unpacked once into <root>/<label>_<UTC>/payload_root. Outputs per slice:
# <campaign>/slices/slice_<s>/ (listmode.npz, stats, manifest, job.log, job_info.json, exit_code.txt).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SIF="${SIF:-$SCRIPT_DIR/qmirt-gate-10-sim-sif_v1.0.0.sif}"
ROOT="${BRAIN_PHANTOM_ROOT:-$HOME/brain_phantom}"
LABEL="" PAYLOAD="" SPEC="" ACTIVITY="" REGION="all" SLICE_S="" SLICES="" PROCS=16 PARTITION="" TIME="12:00:00"
MEM_PER_PROC="3G" ACCOUNT="hsabet" OFFSET=0 CHUNK_S="" DRY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label) LABEL="$2"; shift 2 ;;
        --payload) PAYLOAD="$2"; shift 2 ;;
        --phantom-spec) SPEC="$2"; shift 2 ;;
        --activity-bq) ACTIVITY="$2"; shift 2 ;;
        --activity-region) REGION="$2"; shift 2 ;;
        --slice-s) SLICE_S="$2"; shift 2 ;;
        --slice-count) SLICES="$2"; shift 2 ;;
        --procs-per-task) PROCS="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --mem-per-proc) MEM_PER_PROC="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --slice-offset) OFFSET="$2"; shift 2 ;;
        --chunk-s) CHUNK_S="$2"; shift 2 ;;
        --root) ROOT="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n 2,16p "$0"; exit 0 ;;
        *) echo "Unexpected argument: $1" >&2; exit 2 ;;
    esac
done
for v in LABEL PAYLOAD SPEC ACTIVITY SLICE_S SLICES PARTITION; do
    [[ -n "${!v}" ]] || { echo "Missing --$(echo "$v" | tr 'A-Z_' 'a-z-')" >&2; exit 2; }
done
[[ -f "$SIF" ]] || { echo "Container not found: $SIF" >&2; exit 1; }

if [[ -z "$CHUNK_S" ]]; then
    NUM_CHUNKS=$(python3 -c "import math; print(max(1, math.ceil(${SLICE_S} - 1e-9)))")
    CHUNK_S=$(python3 -c "print(${SLICE_S} / ${NUM_CHUNKS})")
else
    NUM_CHUNKS=$(python3 -c "print(max(1, round(${SLICE_S} / ${CHUNK_S})))")
fi
TASKS=$(( (SLICES + PROCS - 1) / PROCS ))
MEM_MB=$(python3 -c "u='${MEM_PER_PROC}'; print(int(float(u[:-1]) * (1024 if u[-1] in 'Gg' else 1) * ${PROCS}))")

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CAMPAIGN="${ROOT}/${LABEL}_${STAMP}"
mkdir -p "$CAMPAIGN/payload_root" "$CAMPAIGN/slices" "$CAMPAIGN/logs"
tar -xzf "$PAYLOAD" -C "$CAMPAIGN/payload_root"

# task.sh: the slice runner, parameters baked in as variables (no nested quoting)
TASK_SCRIPT="$CAMPAIGN/task.sh"
{
    echo '#!/bin/bash'
    echo "# array task t runs slices t * PROCS + j + OFFSET (j < PROCS), side by side"
    printf 'CAMPAIGN=%q\nSIF=%q\nPROCS=%q\nOFFSET=%q\nSLICES=%q\nSLICE_S=%q\nCHUNK_S=%q\nNUM_CHUNKS=%q\n' \
        "$CAMPAIGN" "$SIF" "$PROCS" "$OFFSET" "$SLICES" "$SLICE_S" "$CHUNK_S" "$NUM_CHUNKS"
    printf 'SPEC=%q\nACTIVITY=%q\nREGION=%q\n' "$SPEC" "$ACTIVITY" "$REGION"
    cat <<'TASK'
set -uo pipefail
run_slice() {
    local s="$1" out="$CAMPAIGN/slices/slice_$1" start t0 code
    start=$(date +%s)
    mkdir -p "$out"
    t0=$(python3 -c "print($s * $SLICE_S)")
    apptainer exec --bind "$CAMPAIGN" --pwd "$CAMPAIGN/payload_root" \
        --env PYTHONPATH="$CAMPAIGN/payload_root/qmirt/src" \
        --env QMIRT_PHANTOM_DATA="$CAMPAIGN/payload_root/payload/phantom_sources" \
        --env POLARS_MAX_THREADS=1 "$SIF" \
        python3 payload/python/gate_sim_brain_spect_boolean.py -o "$out" -j "${SLURM_ARRAY_JOB_ID}" -k "$s" \
            --execution-environment slurm -n 1 -d "$CHUNK_S" -c "$NUM_CHUNKS" --time-start-s "$t0" \
            --fov-shape sphere --fov-size-mm 288 --actor-layout merged --shield-model csg \
            --physics-list G4EmStandardPhysics_option4 --mode phantom \
            --phantom-spec "payload/phantom_specs/${SPEC}.json" \
            --phantom-activity-bq "$ACTIVITY" --phantom-activity-region "$REGION" --energy-resolution 0 \
        > "$out/job.log" 2>&1 \
    && apptainer exec --bind "$CAMPAIGN" --pwd "$CAMPAIGN/payload_root" \
        --env PYTHONPATH="$CAMPAIGN/payload_root/qmirt/src" \
        --env QMIRT_PHANTOM_DATA="$CAMPAIGN/payload_root/payload/phantom_sources" "$SIF" \
        python3 payload/python/reduce_phantom_singles.py "$out" >> "$out/job.log" 2>&1
    code=$?
    echo "$code" > "$out/exit_code.txt"
    [[ -f "$out/listmode.npz" ]] && rm -f "$out"/pixel_singles_*.root
    rm -rf "$out/phantom_image"
    printf '{"wall_s": %d, "host": "%s", "cpu": "%s"}\n' "$(( $(date +%s) - start ))" "$(hostname)" \
        "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)" > "$out/job_info.json"
}
for ((j = 0; j < PROCS; j++)); do
    s=$(( SLURM_ARRAY_TASK_ID * PROCS + j + OFFSET ))
    (( s < SLICES + OFFSET )) || break
    run_slice "$s" &
done
wait
TASK
} > "$TASK_SCRIPT"
chmod +x "$TASK_SCRIPT"

cat > "$CAMPAIGN/campaign_manifest.json" <<JSON
{
  "label": "${LABEL}", "created_utc": "${STAMP}", "cluster": "ERIS",
  "git_commit": "$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "payload": "$(basename "$PAYLOAD")", "container": "${SIF}",
  "phantom_spec": "${SPEC}", "activity_bq": "${ACTIVITY}", "activity_region": "${REGION}",
  "slice_s": "${SLICE_S}", "slice_offset": ${OFFSET}, "chunk_s": "${CHUNK_S}", "chunks_per_slice": ${NUM_CHUNKS},
  "slice_count": ${SLICES}, "procs_per_task": ${PROCS}, "array_tasks": ${TASKS},
  "acquisition_covered_s": [$(python3 -c "print(${OFFSET} * ${SLICE_S})"), $(python3 -c "print((${OFFSET} + ${SLICES}) * ${SLICE_S})")]
}
JSON

SB=(sbatch --parsable --job-name "bp_${LABEL}" --account "$ACCOUNT" --partition "$PARTITION" --array "0-$((TASKS - 1))"
    --cpus-per-task "$PROCS" --mem "${MEM_MB}M" --time "$TIME"
    --output "$CAMPAIGN/logs/task_%a.out" --error "$CAMPAIGN/logs/task_%a.err"
    --wrap "module load Apptainer/1.4.2 2>/dev/null || true; bash $TASK_SCRIPT")
echo "Campaign: $CAMPAIGN"
if [[ "$DRY" -eq 1 ]]; then printf '%q ' "${SB[@]}"; echo; exit 0; fi
JOB_ID="$("${SB[@]}")"
echo "$JOB_ID" > "$CAMPAIGN/slurm_job_id.txt"
echo "Submitted array job $JOB_ID: ${TASKS} tasks x ${PROCS} slices"
