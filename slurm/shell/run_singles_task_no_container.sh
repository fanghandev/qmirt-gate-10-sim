#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "$script_dir/bootstrap_cluster_env.sh"
bash "$script_dir/check_cluster_env.sh" "$PYTHON_BIN"

task_id="${SLURM_PROCID:-${SLURM_ARRAY_TASK_ID:-0}}"
task_count="${SLURM_NTASKS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
if [[ -z "$task_count" || "$task_count" -lt 1 ]]; then
  task_count=1
fi

num_threads="${NUM_THREADS:-1}"
if [[ "$num_threads" != "1" ]]; then
  echo "Warning: forcing NUM_THREADS=1 for single-thread jobs (got $num_threads)."
  num_threads=1
fi

run_group_id="${RUN_GROUP_ID:-$task_id}"
source_activity_bq="${SOURCE_ACTIVITY_BQ:-12500000}"
chunk_duration_s="${CHUNK_DURATION_S:-0.125}"
num_chunks="${NUM_CHUNKS:-10}"
output_parent="${OUTPUT_PARENT:-/data/fanghan/opengate_sim/data/dc_spect/singles_batch_slurm}"
job_output_root="${JOB_OUTPUT_ROOT:-${output_parent}/campaign_${SLURM_JOB_ID:-nojid}}"
max_task_seconds="${MAX_TASK_SECONDS:-0}"

if [[ -n "${TOTAL_EXPECTED_EVENTS:-}" ]]; then
  task_target_events="$(python3 - <<PY
import math
total = float("${TOTAL_EXPECTED_EVENTS}")
tasks = max(1, int("${task_count}"))
print(math.ceil(total / tasks))
PY
)"
  events_per_chunk="$(python3 - <<PY
source_activity_bq = float("${source_activity_bq}")
chunk_duration_s = float("${chunk_duration_s}")
print(source_activity_bq * chunk_duration_s)
PY
)"
  num_chunks="$(python3 - <<PY
import math
task_target_events = float("${task_target_events}")
events_per_chunk = float("${events_per_chunk}")
print(max(1, math.ceil(task_target_events / events_per_chunk)))
PY
)"
fi

mkdir -p "$job_output_root"

timestamp="$(date +"%Y%m%d_%H%M%S")"
job_id="${SLURM_JOB_ID:-nojid}"
outdir="${job_output_root}/task_${task_id}"
mkdir -p "$outdir"

exec > >(tee -a "$outdir/slurm_stdout.txt") 2> >(tee -a "$outdir/slurm_stderr.txt" >&2)

cat > "$outdir/run_plan.txt" <<EOF
job_id: $job_id
task_id: $task_id
task_count: $task_count
run_group_id: $run_group_id
num_threads: $num_threads
source_activity_bq: $source_activity_bq
chunk_duration_s: $chunk_duration_s
num_chunks: $num_chunks
total_expected_events: ${TOTAL_EXPECTED_EVENTS:-}
timestamp: $timestamp
EOF

echo "Running host-based SLURM task with:"
echo "  JOB_OUTPUT_ROOT: $job_output_root"
echo "  TASK_ID: $task_id"
echo "  TASK_COUNT: $task_count"
echo "  RUN_GROUP_ID: $run_group_id"
echo "  SOURCE_ACTIVITY_BQ: $source_activity_bq"
echo "  CHUNK_DURATION_S: $chunk_duration_s"
echo "  NUM_CHUNKS: $num_chunks"
echo "  NUM_THREADS: $num_threads"
echo "  OUTDIR: $outdir"
echo "  MAX_TASK_SECONDS: $max_task_seconds"

python_script_dir="$(cd -- "$script_dir/../python" && pwd)"
sim_cmd=(
  "$PYTHON_BIN" "$python_script_dir/dc_spect_run_sim_singles_batch_slurm.py"
  --source-activity-bq "$source_activity_bq"
  --chunk-duration-s "$chunk_duration_s"
  --num-chunks "$num_chunks"
  --run-group-id "$run_group_id"
  --num-threads "$num_threads"
  --output-dir "$outdir"
)

if [[ "$max_task_seconds" =~ ^[0-9]+$ ]] && [[ "$max_task_seconds" -gt 0 ]]; then
  timeout --signal=TERM --kill-after=30 "$max_task_seconds" "${sim_cmd[@]}"
else
  "${sim_cmd[@]}"
fi