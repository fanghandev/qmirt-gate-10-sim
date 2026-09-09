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

### SRM output layout

The combine step changes format depending on where it runs, so check for the right files at each
stage:

| Stage                            | Runs on     | Split | Output                                                    |
| :------------------------------- | :---------- | :---- | :-------------------------------------------------------- |
| per-task combine                 | Expanse     | no    | `task_N/final_srm_<label>.npz` (coords/counts)            |
| group/shard combine (`--shard-count > 1`) | Expanse | no  | `group_N/final_srm_<label>.npz` (coords/counts)           |
| final campaign combine           | workstation | yes   | `final_srm_<label>_head_01..73.npz` (scipy CSR)           |

Per-head splitting is the default for a final merge and is disabled automatically while sharding,
because a CSR file cannot be fed back into another combine. Each per-head file is a
$(625 \text{ pixels}, \text{grid}^3 \text{ voxels})$ CSR matrix with `int64` counts; the voxel column index is
$x\cdot g^2 + y\cdot g + z$. Grid, voxel size, extent and energy window live in `combined_srm_metadata.json`,
not inside the NPZs, since `save_npz` cannot carry extra arrays.

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
singularity exec submit_slurm/qmirt-gate-10-sim-sif_v1.0.0.sif python3 -c \
  "from importlib.metadata import version;print('OPEN GATE version',version('opengate'))"
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

Note that `--srm-fov-size-mm 210` is a sphere **diameter**: the source fills a 105 mm-radius sphere
and the reconstruction grid is the ±105 mm cube that circumscribes it.

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

Task output must **not** be split per head — exactly three NPZs, no `_head_NN` files:

```bash
ls "$DATA_DIR/task_0" | grep -c '_head_'          # expect 0
singularity exec submit_slurm/qmirt-gate-10-sim-sif_v1.0.0.sif python3 -c "
import numpy as np
d = np.load('$DATA_DIR/task_0/final_srm_2mm.npz')
print('keys:', sorted(d.files)[:4], '...')
print('entries:', d['coords'].shape, 'counts:', int(d['counts'].sum()))
print('energy window keV:', float(d['energy_min_kev'][0]), float(d['energy_max_kev'][0]))
"
```

The window must read `126.0 154.0` (20% at 140 keV). If it still shows `139.3 140.7`, the payload
on Expanse predates the digitizer-blurring change and the campaign should be re-submitted.

## Phase 4 — Configure the workstation

Set the endpoints and path roots once:

```bash
cd ~/Work/RPIL/QMIRT/qmirt-gate-10-sim
cp submit_slurm/globus.env.example submit_slurm/globus.env
```

Edit `submit_slurm/globus.env` and confirm the stable settings (the batch component of each path is
rewritten automatically in the next step):

```bash
QMIRT_SRC_ENDPOINT=8735b734-00dc-4659-be0d-ff96beaff17b
QMIRT_SRC_PATH=/projects/mgh102/fhan1/brain_spect_sim/<any_batch>
QMIRT_DST_ENDPOINT=c88668d2-ab9f-11f1-b472-0afff7074b21
QMIRT_DST_PATH=/data/fanghan/opengate_sim/data/brain_spect/<any_batch>
QMIRT_LOCAL_PATH=/data/fanghan/opengate_sim/data/brain_spect/<any_batch>
QMIRT_MOUNT_PATH=/home/fanghan/sdsc-expanse/brain_spect_sim/<any_batch>
QMIRT_GLOBUS_CLI=/home/fanghan/.local/share/micromamba/envs/opengate/bin/globus
```

Note the three different spellings of the same directory: Globus is collection-relative
(`/projects/...`), Slurm is POSIX (`/expanse/lustre/projects/...`), and the mount is
`~/sdsc-expanse/...`. Only the roots matter here.

Now select the pilot campaign. This rewrites the batch id in all four path keys and sets
`QMIRT_EXPECTED_TASKS` from the campaign manifest:

```bash
./submit_slurm/select_campaign.sh --list
./submit_slurm/select_campaign.sh --latest        # or --batch batch_20260908_111936
```

Expected output:

```plain
Campaign:       batch_20260908_111936
Mount path:     /home/fanghan/sdsc-expanse/brain_spect_sim/batch_20260908_111936
Globus source:  /projects/mgh102/fhan1/brain_spect_sim/batch_20260908_111936
Local landing:  /data/fanghan/opengate_sim/data/brain_spect/batch_20260908_111936
Expected tasks: 2
Tasks ready:    2
```

`Tasks ready` counts `TASK_COMPLETE.json` markers, so it tells you immediately how much the puller
will move. Preflight — all three must succeed:

```bash
findmnt -T ~/sdsc-expanse                     # mount alive
~/Downloads/globusconnectpersonal-3.3.0/globusconnectpersonal -status
globus whoami
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
`Split per head: 1` → `Wrote 219 per-head SRMs to ...` → `Refreshing local progress report...`.

## Phase 6 — Verify local results

```bash
CAMPAIGN=/data/fanghan/opengate_sim/data/brain_spect/$BATCH
ls "$CAMPAIGN"    # task_0/ task_1/ final_srm_<label>_head_NN.npz combined_srm_metadata.json
                  # progress.json cluster_progress.json pull_ledger.json
ls "$CAMPAIGN" | grep -c '_head_'                     # expect 219 = 73 heads x 3 resolutions
ls "$CAMPAIGN"/final_srm_*.npz | grep -vc '_head_'    # expect 0; no single combined file remains
```

Confirm the split is lossless against the task inputs it was built from, and that the matrices have
the expected shape:

```bash
ma opengate
python3 -c "
import glob, numpy as np
from scipy.sparse import load_npz
c = '$CAMPAIGN'
for label in ('1mm', '1p5mm', '2mm'):
    tasks = sorted(glob.glob(f'{c}/task_*/final_srm_{label}.npz'))
    ref = sum(int(np.load(p)['counts'].sum()) for p in tasks)
    heads = sorted(glob.glob(f'{c}/final_srm_{label}_head_*.npz'))
    got = sum(int(load_npz(p).sum()) for p in heads)
    m = load_npz(heads[0])
    print(f'{label}: {len(tasks)} tasks -> {len(heads)} heads | counts {ref} vs {got} '
          f'| lossless {ref == got} | shape {m.shape} dtype {m.dtype}')
"
```

Every line must report `lossless True`, 73 heads and a row count of 625. Column counts should be
210³ / 140³ / 105³ for 1 mm / 1.5 mm / 2 mm.

The metadata sidecar carries everything the CSR files cannot:

```bash
python3 -c "
import json
d = json.load(open('$CAMPAIGN/combined_srm_metadata.json'))['resolutions']['2mm']
print({k: d[k] for k in ('layout', 'num_heads', 'pixels_per_head', 'grid_size',
                         'voxel_size_mm', 'energy_min_kev', 'energy_max_kev',
                         'input_count', 'complete', 'accumulated_counts')})
print('head files listed:', len(d['outputs']))
"
```

Expect `layout: per_head_csr`, `num_heads: 73`, `pixels_per_head: 625`, the 126–154 keV window and
`input_count` equal to the number of pulled tasks.

Then check the report the dashboard consumes. There is no combined NPZ to read, so this also
exercises the per-head fallback loader in `report_campaign_progress.py`:

```bash
python3 -c "
import json; d=json.load(open('$CAMPAIGN/progress.json'))
print('tasks:', d['tasks'])
print('srm available:', {k: v.get('available') for k, v in d['srm'].items()})
print('srm counts:', {k: v.get('total_counts') for k, v in d['srm'].items()})
print('distinct crystals:', {k: v.get('distinct_crystals') for k, v in d['srm'].items()})
print('cluster keys:', list(d.get('cluster', {})))
"
```

Expect `tasks.complete == 2`, `available` true for every label, `total_counts` matching the sums
above, `distinct_crystals` up to 73, and a `cluster` section carrying the queue state that only
`sacct` on Expanse can produce. If `available` is false the fallback failed — usually a missing or
stale `combined_srm_metadata.json`, which the loader needs for `grid_size`.

Spot-check one detector element through the same path, choosing a head that has data from the
`heads` list in the metadata:

```bash
python3 payload/python/report_campaign_progress.py --srm-dir "$CAMPAIGN" \
  --srm-labels 2mm --pixel-query --crystal 1 --pixel 0 | python3 -m json.tool | head -20
```

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

Open <http://127.0.0.1:8765> and confirm the SRM panels render for every label and that the
crystal/pixel selectors are populated. The dashboard never opens NPZs itself — it only consumes
`progress.json` — so per-head splitting is invisible to it provided Phase 6 reported
`available: true`.

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

Then point the workstation at the new batch and restart the timers:

```bash
./submit_slurm/select_campaign.sh --latest
systemctl --user restart qmirt-progress-refresh.timer qmirt-globus-pull.timer
```

Once a few cycles have completed cleanly,
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
| `No 1mm input SRMs found` at the final combine         | a stage wrote per-head files as its input | intermediate combines need `--no-split-per-head`; shard combines set it automatically |
| `srm.available: false` in `progress.json`              | per-head files without the metadata sidecar | restore `combined_srm_metadata.json`, or re-run the campaign combine     |
| `Head IDs out of range for --num-heads 73`             | geometry with a different head count      | pass `--num-heads`/`--pixels-per-head` to `combine_spect_sparse_srm.py`  |
| Dashboard stale                                        | progress timer stopped or mount down      | `systemctl --user status qmirt-progress-refresh.service`                 |
