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

### Resource profiling

The wrappers also collect resource samples by default. Each task writes these files into its output directory:

- `resource_profile.tsv`: raw samples at `PROFILE_INTERVAL_S` second intervals.
- `resource_profile_summary.txt`: average and peak RSS memory, total CPU percentage, CPU percentage relative to the requested allocation, observed threads, cumulative process read/write volume, and filesystem occupancy.

This works on a local non-SLURM server as well as inside SLURM. On a local server, run the wrapper directly after setting `OUTPUT_DIR`, `SCRATCH_ROOT`, and `CONTAINER_SIF`; `SLURM_CPUS_PER_TASK` is optional and defaults to one. For a shorter interval:

```bash
submit_slurm/wrapper_brain_spect_sim_slurm.sh --profile-interval-s 1
```

For the main launcher, use `--no-profile-resources` or `--profile-resources`; `PROFILE_RESOURCES=0` and `PROFILE_INTERVAL_S=1` remain supported as defaults. The process I/O values are cumulative bytes attributed to the simulation process tree, not physical device throughput or disk queue utilization. On SLURM, compare the saved profile with scheduler accounting after completion, for example `sacct -j JOB_ID --format=JobID,Elapsed,AllocCPUS,TotalCPU,MaxRSS,MaxDiskRead,MaxDiskWrite,State` and `seff JOB_ID` where available.

### Sparse brain-SPECT workflow

Sparse mode leaves `gate_sim_brain_spect_boolean.py` unchanged. The wrapper runs the existing simulation repeatedly, retaining `NUM_CHUNKS` inside every Gate invocation for Geant4 event-number protection. After each invocation it converts the local ROOT files into three sparse SRMs, copies only the NPZ files to shared storage, and deletes that loop's ROOT files.

`NUM_LOOPS` and `NUM_CHUNKS` have different meanings:

- `--num-loops N`: number of independent Gate invocations.
- `--num-chunks N`: timing intervals inside each invocation.

Example local test on a non-SLURM server:

```bash
mkdir -p /tmp/qmirt-scratch results/local_sparse_test
SCRATCH_ROOT=/tmp/qmirt-scratch \
OUTPUT_DIR="$PWD/results/local_sparse_test" \
CONTAINER_SIF="$PWD/submit_slurm/qmirt-gate-10-sim-sif_v1.0.0.sif" \
SLURM_CPUS_PER_TASK=4 \
SOURCE_ACTIVITY_BQ=1e4 \
CHUNK_DURATION_S=0.1 \
NUM_CHUNKS=1 \
NUM_LOOPS=2 \
bash submit_slurm/wrapper_brain_spect_sim_slurm.sh --sparse-srm --profile-interval-s 1
```

The production launcher equivalent is:

```bash
bash submit_slurm/run_spect_sim_slurm.sh brain \
   --sparse-srm \
   --num-loops 100 \
   --num-chunks 10 \
   --chunk-duration-s 1 \
   --srm-fov-size-mm 210 \
   --dry-run
```

The default sparse outputs are `final_srm_1mm.npz`, `final_srm_1p5mm.npz`, and `final_srm_2mm.npz`. Intermediate ROOT files are placed under `${SLURM_TMPDIR}`, `${TMPDIR}`, or `SCRATCH_ROOT` and are removed only after successful conversion. Use `LOCAL_SCRATCH_ROOT` to select another node-local directory.

The sparse worker currently reconstructs a Cartesian extent of `[-fov_size_mm/2, fov_size_mm/2)` while the existing simulation uses the current FOV arguments unchanged. Confirm the intended spherical source radius before production runs because the existing sphere volume and source-radius expressions do not currently use the same interpretation of `fov-size-mm`.
