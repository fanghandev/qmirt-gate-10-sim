# Expanse + workstation full-cycle test plan

End-to-end validation of the brain-SPECT sparse-SRM workflow: simulate on SDSC Expanse, pull
finished partial SRMs to the workstation over Globus, combine locally, and serve the dashboard.

Run the phases in order. Phases 1–3 are the least-exercised ground; phases 4–8 have already been
validated against these exact endpoints with synthetic fixtures.

## Environment facts this plan assumes

| Item                               | Value                                                                                |
| :--------------------------------- | :----------------------------------------------------------------------------------- |
| Expanse user / Slurm account       | `fhan1` / `mgh102`                                                                   |
| Expanse projects dir               | `/expanse/lustre/projects/mgh102/fhan1`                                              |
| Container runtime on Expanse       | `module load singularitypro` (there is **no** `apptainer`)                           |
| Default partition                  | `shared` (valid: `compute shared large-shared debug preempt ind-compute ind-shared`) |
| Node-local scratch                 | `/scratch/$USER/job_$SLURM_JOB_ID`                                                   |
| sshfs mount on workstation         | `expanse:/expanse/lustre/projects/mgh102/fhan1` → `~/sdsc-expanse`                   |
| Globus collection (Expanse Lustre) | `8735b734-00dc-4659-be0d-ff96beaff17b`, rooted at `/expanse/lustre`                  |
| Globus collection (rpil64c GCP)    | `c88668d2-ab9f-11f1-b472-0afff7074b21`                                               |
| Local landing root                 | `/data/fanghan/opengate_sim/data/brain_spect`                                        |
| `globus` CLI                       | `~/.local/share/micromamba/envs/opengate/bin/globus` (`ma opengate`)                 |

---

## Phase 1 — Stage the repo and container on Expanse

```bash
ssh expanse
git clone https://github.com/fanghandev/qmirt-gate-10-sim.git ~/qmirt-gate-10-sim
cd ~/qmirt-gate-10-sim
```

Pull the container (takes a few minutes):

```bash
module load singularitypro
mkdir -p ~/apptainer_cache && export SINGULARITY_CACHEDIR=~/apptainer_cache
singularity pull qmirt-gate-10-sim-sif_v1.0.0.sif \
  oras://ghcr.io/fanghandev/qmirt-gate-10-sim-sif:v1.0.0
mv qmirt-gate-10-sim-sif_v1.0.0.sif submit_slurm/
ls -lh submit_slurm/*.sif
```

**Gate check** — nothing downstream works if this fails:

```bash
singularity exec submit_slurm/qmirt-gate-10-sim-sif_v1.0.0.sif python3 -c "from importlib.metadata import version;print('OPEN GATE version',version('opengate'))"
```

## Phase 2 — Dry run, then submit the pilot

```bash
cd ~/qmirt-gate-10-sim
./submit_slurm/run_spect_sim_slurm.sh brain --account mgh102 --test-mode --num-loops 2 --dry-run
```

Confirm in the output:

- `Cluster mode: expanse`, `Partition: shared`, `Sparse SRM: 1`
- `Shared campaign root: /expanse/lustre/projects/mgh102/fhan1`
- the sbatch body contains `export LOCAL_SCRATCH_ROOT="/scratch/${USER}/job_${SLURM_JOB_ID}"`

Submit for real:

```bash
./submit_slurm/run_spect_sim_slurm.sh brain --account mgh102 \
  --test-mode --num-loops 2 --job-count 2 --auto-report --report-interval-s 30
```

Record the printed `DATA_DIR` and array job ID, then watch:

```bash
squeue -u $USER
sacct -j <ARRAY_JOB_ID> -X --format=JobID,State,Elapsed
```

## Phase 3 — Verify the cluster side

```bash
DATA_DIR=/expanse/lustre/projects/mgh102/fhan1/brain_spect_sim/<batch_id>
ls "$DATA_DIR"                        # task_0/ task_1/ campaign_manifest.json progress.json
ls "$DATA_DIR/task_0"                 # final_srm_{1mm,1p5mm,2mm}.npz + TASK_COMPLETE.json
cat "$DATA_DIR/task_0/TASK_COMPLETE.json"
```

The marker must list all three NPZs with sha256 values. A task directory that exists without a
marker means the run failed after simulating — check `submit_slurm/logs/<batch_id>/job_*.err`.
That is exactly the case the marker is designed to keep out of the transfer.

## Phase 4 — Configure the workstation

On `rpil64c`:

```bash
cd ~/Work/RPIL/QMIRT/qmirt-gate-10-sim
export BATCH=<batch_id>
cp submit_slurm/globus.env.example submit_slurm/globus.env
```

Edit `submit_slurm/globus.env` so every path points at the same batch:

```bash
QMIRT_SRC_ENDPOINT=8735b734-00dc-4659-be0d-ff96beaff17b
QMIRT_SRC_PATH=/projects/mgh102/fhan1/brain_spect_sim/<batch_id>
QMIRT_DST_ENDPOINT=c88668d2-ab9f-11f1-b472-0afff7074b21
QMIRT_DST_PATH=/data/fanghan/opengate_sim/data/brain_spect/<batch_id>
QMIRT_LOCAL_PATH=/data/fanghan/opengate_sim/data/brain_spect/<batch_id>
QMIRT_MOUNT_PATH=/home/fanghan/sdsc-expanse/brain_spect_sim/<batch_id>
QMIRT_GLOBUS_CLI=/home/fanghan/.local/share/micromamba/envs/opengate/bin/globus
QMIRT_EXPECTED_TASKS=2
```

Note the three different spellings of the same directory: Globus is collection-relative
(`/projects/...`), Slurm is POSIX (`/expanse/lustre/projects/...`), and the mount is
`~/sdsc-expanse/...`.

Preflight — all three must succeed:

```bash
findmnt -T ~/sdsc-expanse                     # mount alive
ls ~/sdsc-expanse/brain_spect_sim/$BATCH      # campaign visible via mount
~/Downloads/globusconnectpersonal-3.3.0/globusconnectpersonal -status
```

## Phase 5 — Exercise the two paths separately

Progress tracking uses the mount only. This must finish in well under a second and must **not**
create an entry in the Globus Activity tab:

```bash
ma opengate
time python3 payload/python/globus_pull_campaign.py --config submit_slurm/globus.env --progress-only
```

Dry-run the pull. The batch must contain only `--recursive task_N task_N` lines — no
`progress.json`, since campaign files come off the mount:

```bash
python3 payload/python/globus_pull_campaign.py --config submit_slurm/globus.env --dry-run
```

Run the real cycle:

```bash
python3 payload/python/globus_pull_campaign.py --config submit_slurm/globus.env \
  --combine-after-pull --report-after-pull
```

Expected sequence: `Verified 2 task(s); 0 failed verification` → `Combining campaign locally...` →
`Refreshing local progress report...`.

## Phase 6 — Verify local results

```bash
CAMPAIGN=/data/fanghan/opengate_sim/data/brain_spect/$BATCH
ls "$CAMPAIGN"    # task_0/ task_1/ final_srm_*.npz combined_srm_metadata.json
                  # progress.json cluster_progress.json pull_ledger.json
python3 -c "
import json; d=json.load(open('$CAMPAIGN/progress.json'))
print('tasks:', d['tasks'])
print('srm:', {k: v.get('total_counts') for k,v in d['srm'].items()})
print('cluster keys:', list(d.get('cluster', {})))
"
```

Expect `tasks.complete == 2`, non-zero `total_counts` for each SRM label, and a `cluster` section
carrying the queue state that only `sacct` on Expanse can produce.

Idempotency check — a second run must submit no transfer:

```bash
python3 payload/python/globus_pull_campaign.py --config submit_slurm/globus.env --report-after-pull
# expect: "No new tasks to transfer."
```

## Phase 7 — Dashboard and timers

```bash
python3 monitor/serve_monitor.py --host 127.0.0.1 --port 8765 \
  --local-json "$CAMPAIGN/progress.json" &
curl -s http://127.0.0.1:8765/api/progress | python3 -m json.tool | head -20
```

Install the timers (edit the paths inside the unit files if the repo is not at
`~/Work/RPIL/QMIRT/qmirt-gate-10-sim`):

```bash
mkdir -p ~/.config/systemd/user
cp submit_slurm/systemd/qmirt-*.service submit_slurm/systemd/qmirt-*.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now qmirt-progress-refresh.timer qmirt-globus-pull.timer
loginctl enable-linger "$USER"
systemctl --user list-timers 'qmirt-*'
journalctl --user -u qmirt-progress-refresh.service -n 20
```

`qmirt-progress-refresh` runs every minute over the mount; `qmirt-globus-pull` runs every 30 minutes
and is the only thing that creates Globus transfer tasks.

## Phase 8 — Clean up the pilot

```bash
systemctl --user disable --now qmirt-progress-refresh.timer qmirt-globus-pull.timer
rm -rf "$CAMPAIGN"
ssh expanse "rm -rf /expanse/lustre/projects/mgh102/fhan1/brain_spect_sim/$BATCH"
```

---

## Scaling up after the pilot passes

```bash
./submit_slurm/run_spect_sim_slurm.sh brain --account mgh102 \
  --job-count 100 --cpus-per-task 8 --time-limit 12:00:00 \
  --num-loops 50 --num-chunks 10 --chunk-duration-s 1.0 \
  --srm-fov-size-mm 210 --auto-report --report-interval-s 60
```

Then update `QMIRT_SRC_PATH`, `QMIRT_DST_PATH`, `QMIRT_LOCAL_PATH`, `QMIRT_MOUNT_PATH`, and
`QMIRT_EXPECTED_TASKS` in `globus.env` for the new batch. Once a few cycles have completed cleanly,
add `--purge-after-pull` to the `qmirt-globus-pull` service so the 2 TB Projects allocation behaves
as a rolling buffer rather than an archive.

## Troubleshooting

| Symptom                                                | Cause                                     | Fix                                                                      |
| :----------------------------------------------------- | :---------------------------------------- | :----------------------------------------------------------------------- |
| Jobs fail immediately, `ModuleNotFoundError: opengate` | container not found or runtime missing    | re-run the Phase 1 gate check; confirm `submit_slurm/*.sif` exists       |
| `could not resolve the Expanse projects directory`     | running off-cluster or unusual group      | pass `--project-dir /expanse/lustre/projects/mgh102/$USER`               |
| `unsupported partition`                                | partition not in the Expanse list         | use `shared`, `compute`, `large-shared`, `debug`, or `preempt`           |
| `--mem-gb exceeds the Expanse shared limit`            | shared partitions give ~2 G per core      | raise `--cpus-per-task`, lower `--mem-gb`, or use `--partition compute`  |
| `Mount path is not readable (stale mount?)`            | sshfs dropped                             | `fusermount -u ~/sdsc-expanse` then remount                              |
| `ConsentRequired` from `globus ls`                     | missing collection consent                | re-run the `globus session consent` command from the README              |
| Endpoint `EndpointPermissionDenied`                    | GCP not sharing the path                  | add the path to `~/.globusonline/lta/config-paths`, restart GCP          |
| Task pulled but never verified                         | truncated transfer or mismatched checksum | it stays out of the ledger and retries automatically; check `journalctl` |
| Dashboard stale                                        | progress timer stopped or mount down      | `systemctl --user status qmirt-progress-refresh.service`                 |
