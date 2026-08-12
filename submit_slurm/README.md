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

1. From the repository root, run one of the `run_*.sh` helpers in `submit_slurm/`.
2. The helper creates a dated log folder and a per-job output folder, then submits a SLURM array job.
3. Each SLURM task runs the matching wrapper, which forwards the SLURM job and task IDs to the Python entrypoint.
4. Adjust `--cpus-per-task`, `--time`, and `--mem` in the generated `.sbatch` file if your cluster policy requires different defaults.
