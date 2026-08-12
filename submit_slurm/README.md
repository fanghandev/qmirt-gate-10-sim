# How to Use

1. From the repository root, run one of the `run_*.sh` helpers in `submit_slurm/`.
2. The helper creates a dated log folder and a per-job output folder, then submits a SLURM array job.
3. Each SLURM task runs the matching wrapper, which forwards the SLURM job and task IDs to the Python entrypoint.
4. Adjust `--cpus-per-task`, `--time`, and `--mem` in the generated `.sbatch` file if your cluster policy requires different defaults.
