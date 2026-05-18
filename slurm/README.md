# SLURM Singles Workflows (Single-thread)

This folder contains cluster-ready, single-thread workflows for the DC-SPECT singles simulation.

## Eris2 Login and Interactive Shell

Use the current Eris2 nucleus login hosts for interactive work:

```bash
ssh <userid>@eris2n7.research.partners.org
ssh <userid>@eris2n8.research.partners.org
```

After logging in, request an interactive Slurm shell on the interactive partition:

```bash
srun --pty -p interactive /bin/bash
module avail apptainer
module load apptainer/<version>
```

This is the supported setup for running the container-based debug launcher and for checking the image before submitting larger jobs.

## Files

- `submit_singles_no_container.slurm`: host-only SLURM submission script.
- `run_singles_task_no_container.sh`: host-only per-task runner.
- `submit_production_simulate_singles_apptainers.slurm`: container-based SLURM submission script run through Apptainer.
- `submit_merge_singles_stats.slurm`: post-processing merge job for array task stats.
- `run_singles_task_apptainer.sh`: container-based per-task runner run through Apptainer.
- `test_job_task_identifiers.slurm`: prints `SLURM_JOB_ID`, `SLURM_ARRAY_JOB_ID`, `SLURM_ARRAY_TASK_ID`, and step/task IDs.
- `run_interactive_debug_apptainer.sh`: single-node interactive debug launcher.
- `dc_spect_run_sim_singles_batch_slurm.py`: isolated OpenGATE simulation entrypoint for SLURM runs.
- `merge_singles_stats.py`: merges per-task simulation stats into one job-level `stats.txt`.

## Query Available Resources

```bash
sinfo -o "%16P %5a %12l %8D %30C"
```

Example output:

```bash
PARTITION        AVAIL TIMELIMIT    NODES    CPUS(A/I/O/T)
bigmem           up    2-00:00:00   13       68/1156/0/1224
normal*          up    1-00:00:00   7        342/330/0/672
long             up    7-00:00:00   13       116/444/480/1040
debug            up    30:00        1        0/80/0/80
interactive      up    12:00:00     3        100/140/0/240
devel            up    1-00:00:00   1        0/80/0/80
devel-filemove   up    1-00:00:00   1        2/94/0/96
```

## Submit

The submission wrapper script `sbatch-wrapper.sh` allows you to submit containerized DC-SPECT jobs with per-array-job output organization using the `run_gate_sim_dc_spect_slurm.sh` payload.

### Basic usage

Submit a simple test run (single task):

```bash
sbatch-wrapper.sh \
  -N "dc-spect-test" \
  -p debug \
  -n 1 \
  -o /scratch/f/fh890/test-sim \
  -e "ALL,GATE_SIM_PREFIX=/PHShome/fh890/gate10mc,CONTAINER_SIF=/PHShome/fh890/gate10mc/gate10mc.sif,SOURCE_ACTIVITY_BQ=1e5,CHUNK_DURATION_S=1.0,NUM_CHUNKS=10" \
  -- payload_scripts/run_gate_sim_dc_spect_slurm.sh
```

Submit an array job (10 tasks, 0-9):

```bash
sbatch-wrapper.sh \
  -N "dc-spect-array" \
  -p debug \
  -a "0-9" \
  -n 1 \
  -o /scratch/f/fh890/dc-spect-run \
  -e "ALL,GATE_SIM_PREFIX=/PHShome/fh890/gate10mc,CONTAINER_SIF=/PHShome/fh890/gate10mc/gate10mc.sif,SOURCE_ACTIVITY_BQ=3.7e5,CHUNK_DURATION_S=1.0,NUM_CHUNKS=50" \
  -- payload_scripts/run_gate_sim_dc_spect_slurm.sh
```

### Test with 2 Million events

```bash
./sbatch-wrapper.sh  -N "two-million" -p debug -a "0-19" -n 1   -o /scratch/f/fh890/dc-spect-test    -e "ALL,GATE_SIM_PREFIX=/PHShome/fh890/gate10mc,CONTAINER_SIF=/PHShome/fh890/gate10mc/gate10mc.sif,SOURCE_ACTIVITY_BQ=1e4,CHUNK_DURATION_S=1.0,NUM_CHUNKS=10"   -- payload_scripts/run_gate_sim_dc_spect_slurm.sh
```

Example output directory structure after completion:

```bash
 /PHShome/fh890/scratch/dc-spect-test
└──  two-million
    ├──  1370484
    │   ├──  a_1370484_j_0.root
    │   ├──  a_1370484_j_1.root
    │   ├──  a_1370484_j_2.root
    │   ├──  a_1370484_j_3.root
    │   ├──  a_1370484_j_4.root
    │   ├── ...
    │   ├──  a_1370484_j_19.root
    │   ├──  a_1370484_j_0_sim_stats.txt
    │   ├──  a_1370484_j_1_sim_stats.txt
    │   ├──  a_1370484_j_2_sim_stats.txt
    │   ├──  a_1370484_j_3_sim_stats.txt
    │   ├──  a_1370484_j_4_sim_stats.txt
    │   ├── ...
    │   ├──  a_1370484_j_19_sim_stats.txt
    │   ├──  task_0_wall_time.txt
    │   ├──  task_1_wall_time.txt
    │   ├──  task_2_wall_time.txt
    │   ├──  task_3_wall_time.txt
    │   ├──  task_4_wall_time.txt
    │   ├── ...
    │   └──  task_19_wall_time.txt
    └──  slurm_logs
        ├──  1370484_0.err
        ├──  1370484_1.err
        ├──  1370484_2.err
        ├──  1370484_3.err
        ├──  1370484_4.err
        ├── ...
        ├──  1370484_19.err
        ├──  1370484_0.out
        ├──  1370484_1.out
        ├──  1370484_2.out
        ├──  1370484_3.out
        ├──  1370484_4.out
        ├── ...
        └──  1370484_19.out
```

### Example: Production run with 400 jobs

#### Dry-run mode

```bash
sbatch-wrapper.sh --dry-run -N "two-trillion" -p bigmem -a "0-399" -n 1 \
  -o /scratch/f/fh890/dc-spect-production  \
  -e "ALL,GATE_SIM_PREFIX=/PHShome/fh890/gate10mc,CONTAINER_SIF=/PHShome/fh890/gate10mc/gate10mc.sif,SOURCE_ACTIVITY_BQ=1e8,CHUNK_DURATION_S=1.0,NUM_CHUNKS=50" \
  -- payload_scripts/run_gate_sim_dc_spect_slurm.sh
```

#### Actual submission by removing the `--dry-run` flag:

```bash
sbatch-wrapper.sh -N "two-trillion" -p bigmem -a "0-399" -n 1 \
  -o /scratch/f/fh890/dc-spect-production  \
  -e "ALL,GATE_SIM_PREFIX=/PHShome/fh890/gate10mc,CONTAINER_SIF=/PHShome/fh890/gate10mc/gate10mc.sif,SOURCE_ACTIVITY_BQ=1e8,CHUNK_DURATION_S=1.0,NUM_CHUNKS=50" \
  -- payload_scripts/run_gate_sim_dc_spect_slurm.sh
```

This submits 400 tasks (max 20 running concurrently) with 1 task per job, 2-hour time limit, and output under `/scratch/f/fh890/dc-spect-production/<array_job_id>/`.

## Existing Payload Scripts

- `run_gate_sim_dc_spect_slurm.sh`: SLURM payload script that runs the OpenGATE simulation with environment variable configuration.

  Example usage:

  ```bash
  sbatch-wrapper.sh -N "short-run" -p debug -a "0-9" -n 1 \
    -o /scratch/f/fh890/dc-spect-sim \
    -e "ALL,GATE_SIM_PREFIX=/PHShome/fh890/gate10mc,CONTAINER_SIF=/PHShome/fh890/gate10mc/gate10mc.sif, SOURCE_ACTIVITY_BQ=3.7e5,CHUNK_DURATION_S=1.0,NUM_CHUNKS=5" \
    -- payload_scripts/run_gate_sim_dc_spect_slurm.sh
  ```

- `run_check_sim_root_files.sh`: SLURM payload script that checks for the existence of the expected ROOT files after a simulation run.

  Example usage:

  ```bash
  sbatch-wrapper.sh -N "check-root" -p debug -a "0-10" \
    -o /scratch/f/fh890/dc-spect-sim \
    -- payload_scripts/run_check_sim_root_files.sh /scratch/f/fh890/dc-spect-sim/short-run/1360521/
  ```
