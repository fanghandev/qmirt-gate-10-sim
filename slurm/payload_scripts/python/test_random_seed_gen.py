import hashlib
import argparse
import os


def generate_unique_seed(job_id: str, task_id: str):
    seed_string = f"gate_sim_{job_id}_{task_id}"
    unique_seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
    return job_id, task_id, unique_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print a deterministic seed for Slurm jobs")
    parser.add_argument("--job-id", default=None, help="Slurm job ID to include in the seed")
    parser.add_argument("--task-id", default=None, help="Slurm task ID to include in the seed")
    args = parser.parse_args()

    job_id = args.job_id or os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID", "0")
    task_id = args.task_id or os.environ.get("SLURM_ARRAY_TASK_ID") or os.environ.get("SLURM_PROCID") or "0"

    job_id, task_id, unique_seed = generate_unique_seed(str(job_id), str(task_id))
    print(f"JOB_ID: {job_id}\nTASK_ID: {task_id}\nUnique Seed: {unique_seed}")
