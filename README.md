# qmirt-gate-10-sim

GATE 10 Simulation for QMIRT project

## Docker CI/CD

GitHub Actions builds the Docker image on pull requests to `main`.
On pushes to `main` and version tags (`v*`), it also publishes the image to:

`ghcr.io/fanghandev/qmirt-gate-10-sim`

## How to use the `sif ` image on OSPool

1. Create `apptainer_cache` directory in your home directory if it does not exist:

```bash
mkdir -p ~/apptainer_cache
```

2. Pull the image from GitHub Container Registry:

```bash
apptainer pull oras://ghcr.io/fanghandev/qmirt-gate-10-sim-sif:v1.0.0
```

3. Move the image to /ospool/ap40/data/username/qmirt-gate-10-sim-sif:

```bash
mv qmirt-gate-10-sim-sif_v1.0.0.sif /ospool/ap40/data/$USER/qmirt-gate-10-sim.sif
```

## SLURM submission helper

Use the helper in [submit_slurm/run_spect_sim_slurm.sh](submit_slurm/run_spect_sim_slurm.sh) to submit brain or cardiac simulations to a SLURM-managed cluster.

### Basic usage

From the repository root:

```bash
./submit_slurm/run_spect_sim_slurm.sh brain --account <slurm_account> --dry-run
./submit_slurm/run_spect_sim_slurm.sh cardiac --cluster expanse --account <slurm_account> --dry-run
./submit_slurm/run_spect_sim_slurm.sh brain --cluster eris --dry-run
```

The script:

- auto-detects the cluster from `hostname` when `--cluster` is omitted, defaulting to `expanse`
- validates supported partitions for the selected cluster
- creates a dated log directory and a per-batch scratch/output directory
- generates a `.sbatch` file and submits it with `sbatch`
- uses the Apptainer image in `submit_slurm/`

### Cluster-specific notes

- `expanse` (default): requires `--account`. The Lustre projects directory is named after your
  SDSC unix group, which is **not** the same string as the Slurm account (for example, account
  `mde260019` maps to group `mgh102`). The script resolves it automatically with
  `ls -d /expanse/lustre/projects/*/$USER`; override with `--project-dir` when running off-cluster.
  ROOT files go to the node-local NVMe at `/scratch/$USER/job_$SLURM_JOB_ID` (Expanse does not set
  `SLURM_TMPDIR`). On `shared`/`shared-preempt` the requested memory is capped at ~2 GB per core.
- `eris`: uses `/scratch/f/fh890` and default partition `normal`
- `bridges2`: uses the system-provided `PROJECT` environment variable when available; `PROJECT` is already the project root path (for example `/ocean/projects/med260005p/fhan1`), so the script uses it directly and does not append `${USER}` again

`eris` and `bridges2` remain supported but are no longer the clusters this workflow is exercised on.

### Job-array mode (default)

```bash
./submit_slurm/run_spect_sim_slurm.sh brain \
  --job-count 20 \
  --cpus-per-task 4 \
  --time-limit 04:00:00 \
  --mem-gb 8 \
  --account <slurm_account> \
  --partition shared
```

This submits a SLURM array job and each task runs with its own task ID and output directory.

### Whole-node mode

If you want one task per node with multithreading enabled for the full node, use `--nodes`:

```bash
./submit_slurm/run_spect_sim_slurm.sh brain \
  --account <slurm_account> \
  --partition compute \
  --nodes 2
```

This requests whole-node allocation with one task per node and sets `--cpus-per-task` to the node size for GATE multithreading.

### Test mode

```bash
./submit_slurm/run_spect_sim_slurm.sh brain --account <slurm_account> --test-mode --dry-run
```

This reduces the run to a small pilot configuration suitable for quick validation.

### Sparse brain-SPECT SRM mode

Sparse mode is **on by default for brain simulations** (pass `--no-sparse-srm` to opt out). It keeps the existing Gate simulation script unchanged. The brain wrapper runs multiple independent simulations, preserving `--num-chunks` inside each invocation to limit Geant4 event-number growth. Each loop writes ROOT files to local scratch, converts them to 1 mm, 1.5 mm, and 2 mm sparse NPZ matrices, copies the NPZ files to shared output, and deletes the intermediate ROOT files.

```bash
bash submit_slurm/run_spect_sim_slurm.sh brain \
  --num-loops 100 \
  --num-chunks 10 \
  --chunk-duration-s 1 \
  --srm-fov-size-mm 210 \
  --dry-run
```

`--num-loops` controls independent Gate invocations; `--num-chunks` controls timing intervals within each invocation. Final files are `final_srm_1mm.npz`, `final_srm_1p5mm.npz`, and `final_srm_2mm.npz`.

Once the per-task combine succeeds, the wrapper writes `task_<id>/TASK_COMPLETE.json` containing the SHA-256 and byte size of every `final_srm_*.npz`. That marker is the only signal the workstation puller uses to decide a task is safe to transfer, so it is written atomically (temp file plus rename) and is never created for a task that produced no SRMs.

For a local non-SLURM test, set `SCRATCH_ROOT`, `OUTPUT_DIR`, `CONTAINER_SIF`, and `SLURM_CPUS_PER_TASK`, then run `bash submit_slurm/wrapper_brain_spect_sim_slurm.sh --sparse-srm`. Detailed setup and cleanup behavior is documented in [submit_slurm/README.md](submit_slurm/README.md).

### Dry-run / inspect generated script

```bash
./submit_slurm/run_spect_sim_slurm.sh brain \
  --account <slurm_account> \
  --test-mode \
  --dry-run
```

This prints the generated `.sbatch` script without submitting it, so you can confirm the job directives, partitions, and output paths before submitting.

## Running an Expanse production campaign with Globus pull and the local monitor

This walks through submitting a sparse brain-SPECT campaign on SDSC Expanse, pulling completed
partial SRMs down to the workstation with `globus-cli`, combining them locally, and watching
progress from a dashboard served on your own machine.

The division of labor is:

| Stage | Runs on | Output |
| --- | --- | --- |
| Gate simulation + per-loop sparse SRM | Expanse compute node (node-local NVMe) | per-loop `srm_*.npz` |
| Per-task combine | Expanse compute node | `task_N/final_srm_*.npz` + `TASK_COMPLETE.json` |
| Progress report | Expanse (small Slurm job) | `cluster_progress.json` (queue state) |
| Progress polling + listing | workstation, sshfs mount (`~/sdsc-expanse`) | no Globus task |
| Bulk transfer | workstation, `globus-cli` | verified `task_N/` under `/data/fanghan/...` |
| Campaign combine | workstation (automatic after each pull) | campaign `final_srm_*.npz` |
| Dashboard report | workstation (automatic, every minute) | `progress.json` |

Only the small per-task SRMs and stats ever land on Lustre; the 2 TB Projects allocation acts as a
rolling buffer rather than an archive, so a campaign can be far larger than the allocation.

For a copy-pasteable validation run covering every stage, including staging the container on Expanse
and installing the workstation timers, see
[submit_slurm/expanse_globus_test_plan.md](submit_slurm/expanse_globus_test_plan.md).

### 0. Test the workflow end-to-end with a small pilot first

Before trusting the pipeline for a full production campaign, validate every stage (submission →
sparse SRM → per-task combine → progress report → Globus pull → local combine → dashboard) with a
cheap, fast pilot:

```bash
./submit_slurm/run_spect_sim_slurm.sh brain \
  --account <slurm_account> \
  --test-mode \
  --num-loops 2 \
  --job-count 2 \
  --auto-report
```

`--test-mode` shrinks the run to `job-count=2`, `cpus-per-task=4`, `time-limit=0:30:00`, and a low
source activity, so it finishes in minutes instead of hours. Sparse SRM mode is already the default
for `brain`. Once the array job shows `COMPLETED` in `sacct`, confirm each stage produced what's
expected:

```bash
# on Expanse, after the jobs finish
ls "$DATA_DIR"                                  # expect task_0/, task_1/, campaign_manifest.json
cat "$DATA_DIR/task_0/TASK_COMPLETE.json"       # per-task checksums; the puller keys on this
ls "$DATA_DIR/task_0/srm_chunks"/*_run_manifest.json   # per-loop resolved simulation parameters
```

Every task directory must contain `TASK_COMPLETE.json` and three `final_srm_*.npz` files before the
workstation will pull it.

### 1. Submit the campaign on Expanse

```bash
./submit_slurm/run_spect_sim_slurm.sh brain \
  --account <slurm_account> \
  --job-count 100 \
  --cpus-per-task 8 \
  --time-limit 12:00:00 \
  --num-loops 50 \
  --num-chunks 10 \
  --chunk-duration-s 1.0 \
  --srm-fov-size-mm 210 \
  --auto-report \
  --report-interval-s 60
```

Drop `--dry-run` only after you've inspected the generated `.sbatch` file. Note the printed
`DATA_DIR` (campaign output directory under `/expanse/lustre/projects/<group>/$USER`) and the
`Submitted array job <ARRAY_JOB_ID>` line — you'll need both for the next steps.

There is deliberately no `--combine-after` here: on Expanse the campaign-wide reduction happens on
the workstation after transfer (step 4), which keeps the heavy merge off your allocation. The
per-task combine still runs on the cluster because it collapses `num_loops × 3` files into 3.

Every submission also writes a `campaign_manifest.json` (to both the log directory and `DATA_DIR`) recording the exact repo git commit, container SIF path/SHA-256, and every resolved scheduler/simulation parameter for that batch — this is the record to keep for reproducing or auditing a production run later. Each individual Gate invocation additionally writes its own `a_<job_id>_j_<task_id>[_loop_<loop_id>]_run_manifest.json` next to its stats file, capturing the fully resolved Python simulation parameters (including the random seed) actually used for that invocation; in sparse-SRM mode these end up under `task_<id>/srm_chunks/` alongside the per-loop stats and SRM chunks.

### 2. Generate a progress JSON on Expanse

`--auto-report` above submits its own small, independent Slurm job (not a login-node process — login nodes typically kill or discourage long-running background processes) that regenerates `"$DATA_DIR/progress.json"` every `--report-interval-s` seconds. By default it requests `--report-cpus 1 --report-mem-gb 2 --report-time-limit 24:00:00` on the same partition as the main job (override with `--report-partition`); increase `--report-time-limit` for campaigns expected to run longer than 24 hours. It uses `sacct`/`squeue` on the array job for queue/run timing, and stops itself (after one final refresh) once the array job leaves the queue — so its own time limit only needs to be a safety cap, not an exact estimate. Because the wrapper copies each loop's stats file out to `srm_chunks/` as soon as that loop finishes, `progress.json` advances at per-loop granularity rather than only when a whole task completes. Its sbatch file and `%j.out`/`%j.err` logs are under `<LOG_DIR>`, printed in the submission summary as `Submitted progress reporter job <REPORT_JOB_ID>`.

If you didn't pass `--auto-report`, or want an ad hoc refresh, run the same report generator directly on the login node instead:

```bash
python3 payload/python/report_campaign_progress.py \
  --campaign-dir "$DATA_DIR" \
  --expected-tasks 100 \
  --job-id <ARRAY_JOB_ID> \
  --output "$DATA_DIR/progress.json"
```

You can also (re-)submit the same reporter job manually (e.g. to re-attach monitoring to an already-running campaign) with `sbatch --wrap='bash submit_slurm/wrapper_generate_progress_report.sh --campaign-dir "$DATA_DIR" --expected-tasks 100 --job-id <ARRAY_JOB_ID> --watch-job-id <ARRAY_JOB_ID> --output "$DATA_DIR/progress.json" --interval-s 60' --cpus-per-task=1 --mem=2G --time=24:00:00 --partition=<partition>`.

**How concurrent writes to `progress.json` are avoided:** `report_campaign_progress.py` always writes to `progress.json.tmp` and then atomically renames it into place, so anything reading the file (the monitor, `cat`, `scp`) only ever sees a complete, valid JSON document — never a half-written one. That alone doesn't stop two _writer_ processes from racing each other, though, so `wrapper_generate_progress_report.sh` additionally takes an `flock` lock on `<output>.reporter.lock` for as long as it runs: if a second reporter job is accidentally submitted for the same `--output` path, it prints an error and exits immediately instead of corrupting the file. A one-off manual `report_campaign_progress.py` run (not through the wrapper) isn't locked, so avoid running that by hand against the same output path while a reporter job is also active. Since every campaign gets its own `progress.json`/lock file under its own `DATA_DIR`, running several campaigns at once is safe — their reporter jobs never contend with each other.

### 3. Pull completed tasks to the workstation with Globus

Two different jobs are deliberately split by cost:

- **Progress tracking** reads an sshfs mount of the Expanse project directory. It is a plain file
  read, so it can poll every minute without touching Globus.
- **Bulk transfer** of finished `task_*/` payloads uses Globus, which handles checksums, retries and
  large files properly. Each transfer shows up as one task in the Globus Activity tab.

Doing progress polling over Globus would create a transfer task per refresh and flood Activity, so
the mount handles all the light-weight listing and metadata reads.

Mount the project directory once (add it to `/etc/fstab` or a user unit to make it persistent):

```bash
mkdir -p ~/sdsc-expanse
sshfs expanse:/expanse/lustre/projects/mgh102/fhan1 ~/sdsc-expanse
findmnt -T ~/sdsc-expanse       # confirm it is mounted
```

One-time Globus setup on `rpil64c`:

```bash
# Globus Connect Personal must be running and must share the landing directory.
~/Downloads/globusconnectpersonal-3.3.0/globusconnectpersonal -status
grep /data/fanghan ~/.globusonline/lta/config-paths   # add "/data/fanghan/opengate_sim/,0,1" if absent

# Mapped collections need a one-time consent (single-quote it; zsh globs the brackets).
globus session consent 'urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/8735b734-00dc-4659-be0d-ff96beaff17b/data_access]'
```

Copy `submit_slurm/globus.env.example` to `submit_slurm/globus.env` (gitignored) and set the campaign
paths, including `QMIRT_MOUNT_PATH`. Note that the Expanse Lustre collection is rooted at
`/expanse/lustre`, so the Globus path is `/projects/<group>/$USER/...` while Slurm sees
`/expanse/lustre/projects/<group>/$USER/...` and the mount sees `~/sdsc-expanse/...`.

```bash
# Cheap: mount only, no Globus task at all.
python3 payload/python/globus_pull_campaign.py --config submit_slurm/globus.env --progress-only

# Full cycle: mount for listing, Globus for the bulk task payloads.
python3 payload/python/globus_pull_campaign.py --config submit_slurm/globus.env --dry-run
python3 payload/python/globus_pull_campaign.py --config submit_slurm/globus.env \
  --combine-after-pull --report-after-pull
```

When `QMIRT_MOUNT_PATH` is set, the run scans `task_*/` directories on the mount, selects only those
containing `TASK_COMPLETE.json`, and puts **only those task payloads** in a single batched
`globus transfer --sync-level checksum`. Campaign-level files (`progress.json`,
`campaign_manifest.json`) are copied straight off the mount and never enter the transfer. After the
transfer it re-computes SHA-256 locally and records verified tasks in `pull_ledger.json`. Tasks that
fail verification are left out of the ledger and retried, so the pull is idempotent and resumable.
A cycle with no new tasks submits no Globus task at all.

With `--combine-after-pull --report-after-pull` the same run then closes the loop automatically:

1. **Combine** — re-runs the campaign reduction over every task pulled so far, but only when new
   tasks were actually verified, so an idle cycle costs nothing.
2. **Report** — regenerates `progress.json` from the local campaign directory, including SRM
   statistics read from the freshly combined `final_srm_*.npz`.

The cluster's own report is stored as `cluster_progress.json` so it cannot clobber the local one;
its `slurm`/`rates`/`time` sections are grafted into the local `progress.json` under a `cluster`
key, since `sacct` is only reachable from Expanse. The dashboard therefore shows both live queue
state and locally combined SRM statistics from a single file.

Add `--no-srm-stats` if reading the combined matrices every cycle becomes expensive; task counts and
event totals are still reported.

Remote data is **not** deleted by default. Once you trust the pipeline, add `--purge-after-pull` to
delete verified task directories from Lustre and keep the 2 TB allocation as a rolling buffer.

Install both timers from `submit_slurm/systemd/` (edit the paths first). They run at different
cadences on purpose: progress every minute over the mount, Globus pulls every 30 minutes.

```bash
mkdir -p ~/.config/systemd/user
cp submit_slurm/systemd/qmirt-*.service submit_slurm/systemd/qmirt-*.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now qmirt-progress-refresh.timer qmirt-globus-pull.timer
loginctl enable-linger "$USER"     # keep user units running after logout
systemctl --user list-timers 'qmirt-*'
journalctl --user -u qmirt-globus-pull.service -f
```

Both services put the `opengate` micromamba environment on `PATH`, since `globus` and `numpy` both
live there. With the timers installed, steps 4 and 5 below happen on their own — run them by hand
only for a one-off or when debugging.

If the sshfs mount goes stale the run aborts with a clear error rather than silently reporting an
empty campaign; remount and it recovers on the next tick.

### 4. Combine the campaign locally

With the timer from step 3 running, this happens automatically after every pull that brings in new
tasks. To do it manually — for a one-off pull or while debugging — run the combine wrapper directly.
It falls back to plain `python3` when no Apptainer image is present, so it needs no container:

```bash
CAMPAIGN=/data/fanghan/opengate_sim/data/brain_spect/<batch_id>
bash submit_slurm/wrapper_campaign_combine.sh \
  --campaign-dir "$CAMPAIGN" \
  --input-stage tasks \
  --expected-tasks 100
```

This writes `final_srm_{1mm,1p5mm,2mm}.npz` and `combined_srm_metadata.json` at the campaign root.
Re-running it after more tasks arrive simply folds in the larger input set, so you can combine
incrementally while the campaign is still running. Add `--require-complete` for the final pass.

### 5. Serve the dashboard from your local server

Because step 3 regenerates `progress.json` on local disk after every pull, the monitor just watches a
local file — no SSH, no remote command per poll. The file already carries both the locally combined
SRM statistics and the cluster's queue state, so the dashboard stays current on its own:

```bash
screen -S srm-monitor
python3 monitor/serve_monitor.py \
  --host 127.0.0.1 --port 8765 \
  --local-json "$CAMPAIGN/progress.json" \
  --interval-s 30
```

Detach with `Ctrl-A` then `D` (the process keeps running). To check on it or stop it later:

```bash
screen -ls                 # list sessions, confirm "srm-monitor" is there
screen -r srm-monitor       # reattach
# inside the session: Ctrl-C to stop the server, then `exit` to close the session
```

If the server reboots, the `screen` session won't survive — you'll need to start it again the same way; there's no auto-restart unless you add one (e.g. a `@reboot` crontab entry running the same `screen -dmS srm-monitor python3 monitor/serve_monitor.py ...` command).

Binding to `127.0.0.1` keeps the raw dev server off the network; only your reverse proxy (next step) should be reachable externally.

```bash
curl -s http://127.0.0.1:8765/api/progress | python3 -m json.tool | head -20
```

### 6. Put nginx in front with HTTPS

Add an nginx site that redirects `80 → 443` and reverse-proxies to the local `127.0.0.1:8765` monitor:

```bash
sudo mkdir -p /etc/nginx/ssl
# copy a cert/key pair to /etc/nginx/ssl (self-signed for internal testing,
# or one issued by your institution's internal CA for a warning-free experience)
sudo cp /etc/nginx/sites-available/srm-monitor /etc/nginx/sites-available/srm-monitor  # edit server_name for your host
sudo ln -s /etc/nginx/sites-available/srm-monitor /etc/nginx/sites-enabled/srm-monitor
sudo nginx -t && sudo systemctl reload nginx
```

See [monitor/README.md](monitor/README.md) for the exact `server { ... }` blocks (HTTP→HTTPS redirect plus the `proxy_pass http://127.0.0.1:8765;` block) and the detector pixel-mapping conventions the dashboard uses.

### 7. Open the firewall

```bash
sudo ufw allow 443/tcp
sudo ufw allow 80/tcp   # needed for the redirect to 443 to be reachable at all
sudo ufw status verbose
```

Confirm `8765/tcp` is **not** in the allow list — it should only be reachable via nginx on `127.0.0.1`, not directly from the network.

### 8. View it

Browse to `https://<your-local-server-hostname>` from a machine on the same internal network/VPN. The dashboard polls `/api/progress` every 15 seconds and reflects whatever the last Globus pull placed in the local `progress.json`.

## 40 trillion-event simulation plan

The long-run goal is to reach a total of $4 \times 10^{13}$ simulated events. This is not a single-job target; it must be treated as a staged campaign built from many independent array chunks.

### ERIS partition policy

The exact scheduler output from ERIS is:

| Partition | Max job duration    | Total CPUs | Total nodes | Default | Notes                                          |
| --------- | ------------------- | ---------: | ----------: | ------- | ---------------------------------------------- |
| `normal`  | 1-00:00:00 (1 day)  |        768 |           8 | Yes     | Default partition for most jobs                |
| `long`    | 7-00:00:00 (7 days) |       1360 |          17 | No      | Recommended for compute jobs longer than 1 day |
| `bigmem`  | 2-00:00:00 (2 days) |       2184 |          23 | No      | Reserved for large-memory jobs                 |
| `debug`   | 00:30:00 (30 min)   |         80 |           1 | No      | Intended for short tests and debugging         |

The `interactive` partition was not included in the exact scheduler output you shared, so the scheduler-confirmed values above are the ones to rely on for job planning. The key point is that the scheduler reports the real total CPU and node capacity directly, not an approximate table.

This policy affects the plan in two ways:

- do not ask for a job time that exceeds the partition limit
- choose the partition before sizing the run, because the total node and CPU capacity constrain how much work can be launched in one job

### Critical constraint: GATE event counter overflow

GATE uses a 32-bit integer for event counts in some internal structures. That means a single long run can overflow if the number of events in one simulation exceeds the safe range for that counter. In practice, long simulations should not try to accumulate all events in a single run.

This is why the required operational rule is:

- do not run a single job beyond a chunk size that is safely below the 32-bit event limit
- use many short chunks and accumulate totals across many independent tasks
- validate each chunk before moving to the next stage

For planning purposes, the chunk size should be chosen conservatively so that it is well below the overflow threshold even under the largest realistic event count for a task.

### Recommended strategy

1. Benchmark first with a small pilot run.
   - Measure throughput in events/sec.
   - Measure wall time, output size, memory use, and failures.
   - Run on a small subset of tasks before scaling.

2. Use chunked array jobs rather than one massive job.
   - Each array task should produce a moderate number of events, for example $10^8$ to $10^9$ events per task when the runtime is stable.
   - The full campaign then becomes:

   $$
   \text{required tasks} = \frac{4 \times 10^{13}}{\text{events per task}}
   $$

3. Keep each chunk comfortably below the GATE overflow limit.
   - Use many short chunks instead of a single long run.
   - If the simulation produces a large number of events per second, reduce the chunk size so that a task never approaches the 32-bit event counter boundary.
   - The overflow risk is the main reason to prefer numerous short batch segments.

4. Keep task durations short enough to recover from failures and to remain within partition policy.
   - Avoid very long single-task jobs unless throughput has already been benchmarked and the chunk size has been explicitly checked against the event counter bound.
   - Prefer many moderate chunks over a single fragile run.
   - Use `normal` for routine work, `long` only for jobs that genuinely require >1 day, and reserve `bigmem` for jobs that require the large-memory node class.

5. Use ERIS defaults consistent with the available partitions.
   - Default partition: `normal`
   - Other valid partitions: `long`, `bigmem`, `interactive`, `debug`
   - Start with `normal` for production jobs; use `long` only when your measured runtime exceeds 1 day and the requested memory fits the node profile.

6. Use `/scratch/f/fh890` as the batch staging area.
   - Output layout should follow:

   ```bash
   /scratch/f/fh890/<sim_name>/<batch_id>
   ```

   - Example: `brain_spect_sim` or `cardiac_spect_sim`.

7. Validate each batch before expanding.
   - Check event totals, output file counts, root-file integrity, timing metadata, and task completion.
   - Re-run only failed chunks instead of restarting the entire campaign.

### Suggested execution flow

1. Run a dry-run first to inspect the generated batch script:

```bash
./submit_slurm/run_spect_sim_slurm.sh brain \
  --job-count 5 \
  --cpus-per-task 1 \
  --time-limit 01:00:00 \
  --mem-gb 4 \
  --partition normal \
  --source-activity-bq 3.7e5 \
  --chunk-duration-s 1.0 \
  --num-chunks 10 \
  --dry-run
```

2. Run a short pilot batch without dry-run:

```bash
./submit_slurm/run_spect_sim_slurm.sh brain \
  --job-count 20 \
  --cpus-per-task 1 \
  --time-limit 04:00:00 \
  --mem-gb 4 \
  --partition normal \
  --source-activity-bq 3.7e5 \
  --chunk-duration-s 1.0 \
  --num-chunks 10
```

3. Read the per-task sim stats and confirm the event count per chunk stays safely below the overflow threshold.
4. Scale up the number of tasks while reducing each chunk size if needed.
5. Repeat until the total campaign reaches the $4 \times 10^{13}$ event goal.

### How to decide how many CPUs you can request

The cluster policy is node-based, not just CPU-based. The safe rule is:

- check the partition limits first
- then request no more CPUs than the partition allows per node
- and do not assume you can request more than the node can provide in one job

For this cluster, the exact scheduler values are:

- `normal`: 768 CPUs total across 8 nodes, MaxTime 1 day
- `long`: 1360 CPUs total across 17 nodes, MaxTime 7 days
- `bigmem`: 2184 CPUs total across 23 nodes, MaxTime 2 days
- `debug`: 80 CPUs total on 1 node, MaxTime 30 minutes

So a practical rule is:

$$
\text{total requested CPUs} \le \text{partition total CPUs}
$$

and, for a job that requests multiple nodes, you should also keep the request within the scheduler’s node and user limits. The scheduler output is the authoritative source; in this case, `scontrol show partition ...` is the correct way to determine the exact capacity.

In this project, we generally default to `cpus-per-task=1` because the workflow is dominated by many independent array tasks rather than a single large multithreaded job. That is the simplest way to keep throughput and chunking under control while staying comfortably inside the cluster policy.

### Operational guidance

- Use `--dry-run` before every new campaign configuration.
- Prefer `cpus-per-task=1` unless a benchmark shows a clear benefit to using more.
- Keep a log of batch IDs, task counts, event totals per chunk, runtime, output directories, and seed state to support restart and debugging.
- Keep the full campaign in independent batches so each stage can be validated and archived separately.
- Treat chunking as a safety requirement, not just a convenience.
- Check the partition policy before choosing the job time, memory, and CPU request.

### Summary

The 40-trillion-event target should be approached as a large, validated campaign of many short, independent, overflow-safe chunks. The main design constraint is the GATE 32-bit event counter limit, and the operational constraints are the ERIS partition limits on runtime, memory, and CPUs. The correct plan is: choose the partition, keep each chunk safely below the overflow threshold, and scale the number of array tasks without exceeding the node-level CPU policy.

### Suggested execution flow

1. Run a dry-run first to inspect the generated batch script:

```bash
./submit_slurm/run_spect_sim_slurm.sh brain \
  --job-count 5 \
  --cpus-per-task 1 \
  --time-limit 01:00:00 \
  --mem-gb 4 \
  --partition normal \
  --source-activity-bq 3.7e5 \
  --chunk-duration-s 1.0 \
  --num-chunks 10 \
  --dry-run
```

2. Run a short pilot batch without dry-run:

```bash
./submit_slurm/run_spect_sim_slurm.sh brain \
  --job-count 20 \
  --cpus-per-task 1 \
  --time-limit 04:00:00 \
  --mem-gb 4 \
  --partition normal \
  --source-activity-bq 3.7e5 \
  --chunk-duration-s 1.0 \
  --num-chunks 10
```

3. Measure throughput and adjust the per-task target.
4. Submit a larger batch with the tuned values.
5. Repeat until the total campaign reaches the $4 \times 10^{13}$ event goal.

### Operational guidance

- Use `--dry-run` before every new campaign configuration.
- Prefer `cpus-per-task=1` unless a benchmark shows a clear benefit to using more.
- Keep a log of batch IDs, task counts, runtime, output directories, and seed state to support restart and debugging.
- Keep the full campaign in independent batches so each stage can be validated and archived separately.

### Summary

The safest path is to treat 40 trillion events as a validated campaign of many independent simulation chunks, not one run. Benchmark the per-task throughput, choose a stable event budget, and scale the number of array tasks until the cumulative total reaches the target.

### Runtime analysis plots from collected task logs

The following plots were generated from:

- `results/brain_spect/brain_spect_runtime_summary.csv`

Dataset summary used for captions:

- Tasks: 30
- Total simulated events: 25,905,360
- Events per task range: 369,016 to 1,482,372
- Mean pure simulation duration: 20.74 s (median 19.62 s)
- Mean wall time: 56.8 s (median 52.0 s)
- Mean overhead (wall - pure simulation): 36.06 s (median 32.85 s)
- Mean event rate: 35,825 events/s (median 37,370 events/s)
- 2 outlier tasks show very short pure simulation durations (<5 s)

#### Task wall time vs number of simulated events

![Task wall time vs number of simulated events](submit_slurm/plots/task_time_vs_events.png)

Caption: Across 30 tasks, wall time clusters around 48 to 54 s for most points and extends to 76 to 86 s in slower cases, while event counts span 369k to 1.48M. The mean wall time is 56.8 s (median 52.0 s), indicating a substantial near-fixed overhead component compared with pure simulation time.

#### Pure simulation time vs number of simulated events

![Pure simulation time vs number of simulated events](submit_slurm/plots/pure_simulation_time_vs_events.png)

Caption: Pure simulation duration scales approximately linearly with events in three visible clusters (~370k, ~740k, ~1.48M events), with mean 20.74 s and median 19.62 s. Two short-duration outliers (~1.05 s and ~1.22 s at ~1.48M events) are present and should be treated as anomalous when fitting production throughput.

```bash
./submit_slurm/run_spect_sim_slurm.sh brain \
  --job-count 400 \
  --cpus-per-task 1 \
  --time-limit 1-00:00:00 \
  --mem-gb 4 \
  --partition long \
  --source-activity-bq 2e7 \
  --chunk-duration-s 1.0 \
  --num-chunks 100
```

Event Rate: ~36,000 events/s per job (single-threaded, 1 CPU per job), 400 jobs in parallel, total throughput $\sim 14.4 \times 10^6$ events/s across the cluster.

ETA: 55555.6 seconds (approximately 15.4 hours) for 400 tasks, 16 to 20 hours.
Event per batch: 400 tasks $\times$ 100 chunks $\times 2\times 10^7$ events = $$8 \times 10^{11}$$ events
Total number of batches to reach $4 \times 10^{13}$ events: $50$ batches

### Running on the PSC Bridges2 cluster

#### Estimated throughput and time for a 40 trillion-event campaign

Each node has 128 CPUs, and in this set up we should be able to simulated $\sim 20$ billion events per node per job, each job should be able to finish in $\sim 3$ hours. Each batch can run 32 jobs in parallel, we will need to run about 63 batchs. That is a total of $\sim 189$ hours, or $\sim 8$ days to finish the whole campaign.

#### Disk space considerations

Each job will produce a single output root file, and the size of each output file is $\sim 2.5$ GB. Each batch will produce 32 output files, that is $\sim 80$ GB per batch. The total output size for the whole campaign will be $\sim 5$ TB. Make sure you have enough disk space to store the output files.

#### Split 2000 jobs into 4 campaigns

Do the following for 4 times.

```bash
./run_spect_sim_slurm.sh brain --cluster bridges2 --job-count 500 --cpus-per-task 128 --concurrent-limit 32 --source-activity-bq 6.25e6 --chunk-duration-s 1.0 --num-chunks 25
```

## Globus transfer

### Local workstaion

- Globus Connect Personal must be installed and running.
- Endpoint ID for `rpil64c`:

  ```plain
  c88668d2-ab9f-11f1-b472-0afff7074b21
  ```

### SDSC Expanse

| ID                                   | Display Name              |
| :----------------------------------- | :------------------------ |
| 8735b734-00dc-4659-be0d-ff96beaff17b | SDSC HPC - Expanse Lustre |
| ba05bd44-4422-48ab-a110-842e0edd107c | SDSC HPC Data Movers      |

### Path conventions

The Expanse Lustre collection is rooted at `/expanse/lustre`, so Globus paths are
collection-relative and differ from the POSIX paths Slurm and the wrappers use:

| Purpose | Globus path | POSIX path on Expanse |
| :--- | :--- | :--- |
| Projects allocation (2 TB, campaign buffer) | `/projects/<group>/$USER/` | `/expanse/lustre/projects/<group>/$USER/` |
| Lustre scratch (free, ~90-day purge) | `/scratch/$USER/temp_project/` | `/expanse/lustre/scratch/$USER/temp_project/` |

`<group>` is your SDSC unix group (`id -Gn` on Expanse), which is **not** the ACCESS allocation ID
used for `--account`. Node-local NVMe on compute nodes is `/scratch/$USER/job_$SLURM_JOB_ID` and is
not reachable over Globus.

Verify access before running a campaign:

```bash
globus ls 8735b734-00dc-4659-be0d-ff96beaff17b:/projects/<group>/$USER/
globus ls c88668d2-ab9f-11f1-b472-0afff7074b21:/data/fanghan/opengate_sim/data/brain_spect/
```

Both commands must succeed; see step 3 of the Expanse campaign walkthrough for the consent and
Globus Connect Personal path setup they depend on.
