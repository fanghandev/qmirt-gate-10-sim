#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

container_uri="${GATE10MC_IMAGE_URI:-docker://gitlab.partners.org:5050/rpil/qmirt/simulation/gate10mc:f2de53ce}"
container_image="${GATE10MC_SIF:-}"
if [[ -n "$container_image" && -f "$container_image" ]]; then
  apptainer_exec=(apptainer exec "$container_image")
else
  apptainer_exec=(apptainer exec "$container_uri")
fi

task_id="${SLURM_PROCID:-0}"
run_group_id="${RUN_GROUP_ID:-$task_id}"
source_activity_bq="${SOURCE_ACTIVITY_BQ:-1e6}"
chunk_duration_s="${CHUNK_DURATION_S:-0.1}"
num_chunks="${NUM_CHUNKS:-1}"
num_threads="${NUM_THREADS:-1}"
output_parent="${OUTPUT_PARENT:-sim_data/dc_spect/interactive_debug}"
job_output_root="${JOB_OUTPUT_ROOT:-${output_parent}/job_${SLURM_JOB_ID:-interactive}}"
outdir="${JOB_OUTPUT_ROOT:-${job_output_root}}/task_${task_id}"

if ! [[ "$run_group_id" =~ ^[0-9]+$ ]]; then
  echo "Error: RUN_GROUP_ID must be an integer, got '$run_group_id'." >&2
  exit 2
fi

mkdir -p "$outdir"

cat > "$outdir/run_plan.txt" <<EOF
job_id: ${SLURM_JOB_ID:-interactive}
task_id: $task_id
run_group_id: $run_group_id
num_threads: $num_threads
source_activity_bq: $source_activity_bq
chunk_duration_s: $chunk_duration_s
num_chunks: $num_chunks
output_dir: $outdir
container_image: $container_image
container_uri: $container_uri
EOF

echo "Interactive debug launch plan:"
echo "  REPO_ROOT: $repo_root"
echo "  OUTDIR: $outdir"
echo "  SOURCE_ACTIVITY_BQ: $source_activity_bq"
echo "  CHUNK_DURATION_S: $chunk_duration_s"
echo "  NUM_CHUNKS: $num_chunks"
echo "  NUM_THREADS: $num_threads"
echo "  RUN_GROUP_ID: $run_group_id"

cd "$repo_root"
"${apptainer_exec[@]}" python3 slurm/dc_spect_run_sim_singles_batch_slurm.py \
  --source-activity-bq "$source_activity_bq" \
  --chunk-duration-s "$chunk_duration_s" \
  --num-chunks "$num_chunks" \
  --run-group-id "$run_group_id" \
  --num-threads "$num_threads" \
  --output-dir "$outdir"
