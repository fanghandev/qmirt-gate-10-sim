# SRM production runbook

How to run brain-SPECT SRM production on SDSC Expanse and cardiac-SPECT SRM
production on OSPool, and what was verified before the first production campaign.

Validated 2026-09-09 against commit `703e3c2`.

| | Brain SPECT | Cardiac SPECT |
| :--- | :--- | :--- |
| Compute | SDSC Expanse (Slurm) | OSPool (HTCondor) |
| Detector heads | 73 | 80 |
| Pixels per head | 625 (25 × 25) | 625 (25 × 25) |
| Job model | self-contained task: loops of sim → SRM → per-task combine | one partial SRM per job |
| Aggregation | workstation, Globus pull every 30 min | workstation, hourly over the sshfs mount |

Both scanners share the same source, energy and grid settings:

| Setting | Value |
| :--- | :--- |
| Source | uniform sphere, **210 mm diameter** (105 mm radius) |
| Reconstruction grid | ±105 mm cube |
| Resolutions | 1, 1.5, 2 mm → 210³, 140³, 105³ voxels |
| Energy window | 20% at 140 keV (126–154 keV) |
| Energy blurring | 10% FWHM Gaussian at 140 keV |
| Output | `final_srm_<label>_head_NN.npz`, CSR `(625, grid³)`, int64 |

---

## Prerequisites

All three hosts must be on the same commit:

```bash
cd ~/Work/RPIL/QMIRT/qmirt-gate-10-sim && git pull
ssh ospool  'cd ~/qmirt-gate-10-sim && git pull'
ssh expanse 'cd ~/qmirt-gate-10-sim && git pull'
```

Workstation services (installed, enabled, `Linger=yes`):

| Unit | Role |
| :--- | :--- |
| `qmirt-globus-connect.service` | Globus Connect Personal endpoint |
| `qmirt-monitor.service` | dashboard backend on `127.0.0.1:8765` |
| `qmirt-progress-refresh.timer` | 1 min — dashboard refresh over the Expanse mount |
| `qmirt-globus-pull.timer` | 30 min — pull + combine + report for brain |
| `qmirt-cardiac-combine.timer` | 1 h — extract + combine + report for cardiac |

nginx publishes the dashboard at <http://rpil64c.partners.org> (redirects to HTTPS).
The backend binds to loopback only.

---

## Cardiac production on OSPool

### 1. Time one job first

This is the step that cannot be skipped. **Measured wall time for identical work
was 144 s on one node and 1565 s on another — an 11× spread.** Size for the slow
end, because a job preempted past the eviction window loses its entire simulation.

```bash
ssh ospool
cd ~/qmirt-gate-10-sim
bash submit_htcondor/run_cardiac_sparse_srm_batch.sh \
    --job-count 4 --num-loops 1 --num-chunks 20 --source-activity-bq 5e6
condor_history <cluster> -af ProcId ExitCode RemoteWallClockTime
```

Chunks scale linearly. Target 2–4 h against the **slowest** observed job:

```
--num-chunks ≈ 20 × (3 h / slowest_measured_hours)
```

### 2. Submit

```bash
bash submit_htcondor/run_cardiac_sparse_srm_batch.sh \
    --job-count 10000 \
    --num-loops 1 \
    --num-chunks <tuned> \
    --chunk-duration-s 1.0 \
    --source-activity-bq 5e6
```

Add `--dry-run` to inspect the submit file first. Record the batch id.

The payload is published to OSDF automatically and pulled through the site cache,
so the access point does not re-send it per job. The URL and all resolved
parameters are written to `campaign_manifest.json` in the batch directory.

Useful options:

- `--no-osdf` — stage from the access point instead (2.3 MB/job), if OSDF misbehaves
- `--payload-url osdf:///...` — pin an exact payload for reproducibility
- `--fov-size-mm`, `--resolutions-mm` — must divide evenly; checked at submit time

### 3. Watch

```bash
condor_q <cluster>
condor_q -better-analyze <cluster>                       # if jobs stay idle
condor_q -held -af ProcId HoldReason | sort | uniq -c    # hold reasons
```

### 4. Aggregation is automatic

`qmirt-cardiac-combine.timer` runs hourly: extracts new job tarballs incrementally,
rebuilds the 80 per-head SRMs, and refreshes `progress.json`. It skips itself when
no new partials arrived, and exits cleanly when the campaign has returned nothing yet.

```bash
journalctl --user -u qmirt-cardiac-combine -f
```

Manual run, e.g. to force a rebuild:

```bash
./submit_htcondor/run_cardiac_campaign_combine.sh --latest --force
```

> **At scale, switch to sharded combining.** The default is a single pass over every
> partial, which will not fit in memory at 1 mm with thousands of partials. Once past
> a few hundred, add `--shard-count 16` to
> `~/.config/systemd/user/qmirt-cardiac-combine.service` and `systemctl --user daemon-reload`.
> This path is correct on small inputs but **has not been measured at 10k scale** —
> trial it at a few hundred partials first.

---

## Brain production on Expanse

### 1. Submit

```bash
ssh expanse
cd ~/qmirt-gate-10-sim
./submit_slurm/run_spect_sim_slurm.sh brain --account mgh102 \
    --job-count 100 --cpus-per-task 128 --time-limit 12:00:00 \
    --num-loops 50 --num-chunks 10 --chunk-duration-s 1.0 \
    --auto-report --report-interval-s 60
```

Add `--dry-run` first and confirm `Cluster mode: expanse`, the partition, the account
and `LOCAL_SCRATCH_ROOT`. Record the batch id.

`--num-loops` is independent Gate invocations per task; `--num-chunks` is timing
intervals inside one invocation, which bounds Geant4 event-number growth.

### 2. Point the puller at the new campaign

Easy to forget, and the pull timer will keep polling the previous campaign without it:

```bash
# on the workstation
./submit_slurm/select_campaign.sh --latest      # or --batch batch_YYYYMMDD_HHMMSS
```

Confirm `Expected tasks` and `Tasks ready`.

### 3. The rest is automatic

- `qmirt-progress-refresh` (1 min) updates the dashboard over the sshfs mount; it
  creates no Globus task.
- `qmirt-globus-pull` (30 min) transfers, verifies SHA-256, combines and reports.
  Only tasks with a valid `TASK_COMPLETE.json` are pulled, so a task that failed
  after simulating is never transferred.

Once the cycle is trusted, add `--purge-after-pull` to `qmirt-globus-pull.service`
so the Lustre allocation behaves as a rolling buffer.

---

## Reading the output

```python
import json
from scipy.sparse import load_npz

A = load_npz("final_srm_1mm_head_07.npz").astype("float32")   # (625, 9_261_000)

meta = json.load(open("combined_srm_metadata.json"))["resolutions"]["1mm"]
sensitivity = meta["accumulated_counts"] / meta["simulated_primaries"]
```

Voxel column index is `x·g² + y·g + z`. `save_npz` cannot carry extra arrays, so
grid size, voxel size, extent, energy window, per-head totals and
`simulated_primaries` live in `combined_srm_metadata.json` beside the NPZs.

**Stragglers are expected and harmless.** A campaign is combined while jobs are
still finishing, so counts are only meaningful against `simulated_primaries`, which
is summed from exactly the job stats that were combined and advances in step with
the counts. Check `input_count` against `expected_input_count` before treating an
SRM as final. A job that dies before returning its tarball contributes neither
counts nor primaries, so the ratio stays unbiased.

The metadata file is also a consistency guard: the reader verifies each head's total
against it and refuses a set that a concurrent combine is midway through replacing.

---

## Verification record

Everything below was executed on the real systems, not simulated locally.

### Environment

Container `qmirt-gate-10-sim.sif` on OSPool: opengate 10.1.0, numpy 2.5.1,
scipy 1.18.0, uproot 5.7.5, `DigitizerBlurringActor` present.

### Physics changes

| Check | Result |
| :--- | :--- |
| Energy blurring FWHM | 14.39 keV measured = 10.3% of 140 keV, as configured |
| Photopeak centroid | 139.77 keV |
| Energy window acceptance | 8.2% (old 1% window) → 87.7% (20% window) |
| Source sphere radius | 105.00 mm, 100% of emissions inside the ±105 mm grid |

The FOV fix was significant: the source sphere had been built at radius 210 mm
against a ±105 mm grid, so **only 0.3% of in-window events landed in the
reconstruction volume**. After the fix, acceptance is 84–100%.

### End-to-end runs

| Campaign | Result |
| :--- | :--- |
| Cardiac, OSPool cluster 14996548 | 2 jobs, `ExitCode 0`, OSDF payload unpacked on a UConn node |
| Cardiac, OSPool cluster 14996552 | 2 jobs, `ExitCode 0`, wall 144 s and 1565 s |
| Brain, Expanse array 54238113 | task_1 completed; task_0 **failed** — see below |
| Brain, Expanse array 54238298 | 2/2 `COMPLETED`, pulled over Globus, `0 failed verification` |

### Aggregation

| Check | Result |
| :--- | :--- |
| Per-head split lossless | identical `(head, pixel, voxel) → count` vs single-file combine |
| Cardiac output | 240 files = 80 heads × 3 resolutions |
| Brain output | 219 files = 73 heads × 3 resolutions |
| Matrix shape | `(625, 9261000)` at 1 mm, both scanners |
| Tree reduction | 2 shards → final merge, identical totals to a direct combine |
| Incremental extraction | re-run reports `Extracted 0, already present N` |
| Empty partial | combines cleanly with a real one |
| Torn read | rejected with a clear error, not silently blended |

### Measured sensitivity

| Campaign | Primaries | Counts | Counts/primary |
| :--- | ---: | ---: | ---: |
| Cardiac (2 jobs) | 3,000,283 | 1,672 | 5.573e-4 |
| Brain (test-mode, 2 tasks) | 15,980 | 11 | 6.884e-4 |

### Dashboard

Both campaigns served on port 80 with a selector. Per-head sums and per-pixel SRM
maps are byte-identical whether read from per-head files or a single combined file
(only `path` and `generated_at` differ). `progress.json` is a lossy summary —
projections capped at 20k points, top 15 elements — not a second copy of the SRM.

### Payload transfer

| Approach | Per job from the AP | 10,000 jobs |
| :--- | ---: | ---: |
| whole repo | 15.8 MB | 154 GB |
| trimmed staging (`--no-osdf`) | 2.3 MB | 22 GB |
| OSDF (default) | ~0 | ~0, cached per site |

61% of the naive payload was brain-SPECT STL and a stale 3.7 MB CSV that cardiac
never opens.

---

## Bugs found and fixed during validation

**A zero-count run killed the whole task.** Gate writes no ROOT file when a run
detects nothing, and the SRM generator raised `FileNotFoundError`, failing the task
(Expanse `54238113_0`, `FAILED 1:0`). At campaign scale this turns ordinary
low-statistics luck into lost work. Fixed with `--allow-empty`; without the flag the
generator still fails loudly, so a genuinely broken run is not masked. — `b008590`

**The OSDF payload hash was not reproducible across machines.** Identical sources
published under different names from the workstation and the access point, because
the hash covered the compressed archive (tar 1.34 vs 1.35 differ) and the staged
tree picked up local `qmirt.egg-info`. Now hashed over file contents with build
artifacts excluded. — `b008590`

**The dashboard service restart-looped when no campaign existed.** `latest_progress()`
returned non-zero when it found nothing, and `set -e` killed the launcher, so systemd
retried every 10 s — the normal state after a cleanup or on a fresh machine. — `f1e2d73`

**The Globus pull had no working dependency on the endpoint.**
`qmirt-globus-pull.service` ordered itself after `globus-connect-personal.service`,
which does not exist; systemd silently ignores unknown units. Globus Connect Personal
was only ever started by hand, so the timer would hang on every fire after a reboot.
Added `qmirt-globus-connect.service` and corrected the dependency. — `813a58e`

**Simulated primaries were not recorded next to the SRM.** The normalization factor
existed only in `progress.json` and the raw per-job stats, neither of which travels
with the matrix. — `703e3c2`

**Per-head splitting introduced a torn-read window.** The combine wrote 219 files in
place while the 1-minute dashboard refresh read them, and a mixed set loads without
error — silently wrong totals. Now staged and renamed, with the reader verifying each
head against the metadata.

---

## Known gaps

- **Sharded combining is unmeasured at 10k partials.** Correct on small inputs; memory
  behaviour at 1 mm with a realistic partial count is unknown. Trial before relying on it.
- **Job sizing is untuned.** Every test so far was deliberately tiny. The 11× node
  spread on OSPool means `--num-chunks` must be set against the slowest node.
- **`payload/python/combine_brain_sparse_srm.py` is a dead duplicate** of
  `combine_spect_sparse_srm.py`, byte-identical and referenced by nothing. It will
  drift and mislead; remove with `git rm` when convenient.
- **Cardiac has no per-job completion marker.** The brain path gates transfers on
  `TASK_COMPLETE.json`; cardiac relies on the tarball being present plus
  `exit_code.txt` inside it. A partially written tarball would need manual screening.
- **`select_campaign.sh` must be run for each new brain campaign**, or the pull timer
  keeps polling the previous one.
