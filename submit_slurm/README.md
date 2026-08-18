# How to Use

## Use the `apptainer` image for `OpenGATE 10`

### Pull the image

#### 1. Request a interactive session on the SLRUM-managed cluster

```bash
srun --pty -p interactive /bin/bash
```

#### 2. Load the `Apptainer` module

```bash
module load Apptainer
```

#### 3. Pull the image from the GitHub Container Registry

```bash
apptainer pull oras://ghcr.io/fanghandev/qmirt-gate-10-sim-sif:v1.0.0
```

### Testing the image

```bash
apptainer exec --bind /scratch qmirt-gate-10-sim-sif_v1.0.0.sif python3 --version;python3 -c "from importlib.metadata import version;print(version('opengate'))"
```

you should see the output similar to:

```bash
Python 3.12.3
10.1.0
```

## Running jobs on a SLURM-managed cluster

Use the main submission helper in `submit_slurm/run_spect_sim_slurm.sh` from the repository root.

1. Pick the simulation type: `brain` or `cardiac`.
2. Set the cluster and account as needed:
   - ERIS: `--cluster eris`
   - Expanse: `--cluster expanse --account <project_id>`
   - Bridges2: `--cluster bridges2` and rely on the system-provided `PROJECT` variable when available; `PROJECT` is already the project root path (for example `/ocean/projects/med260005p/fhan1`), so do not append `${USER}` again.
3. Use `--dry-run` first to inspect the generated `.sbatch` file before submitting.
4. The helper creates a dated log folder and a scratch/output directory, then submits the job with `sbatch`.
5. Default behavior is a SLURM array job; use `--nodes N` for one-task-per-node whole-node execution with full-node multithreading.
6. Use `--test-mode` for a fast pilot run.

### Examples

```bash
./submit_slurm/run_spect_sim_slurm.sh brain --cluster bridges2 --account <project_id> --dry-run
./submit_slurm/run_spect_sim_slurm.sh brain --cluster bridges2 --test-mode --nodes 1 --dry-run
./submit_slurm/run_spect_sim_slurm.sh brain --job-count 20 --cpus-per-task 4 --time-limit 04:00:00 --mem-gb 16 --dry-run
```

The wrapper script forwards SLURM job metadata to the Python entrypoint and passes `-n ${SLURM_CPUS_PER_TASK:-1}` so GATE can use multithreading when the request includes more than one CPU per task.
