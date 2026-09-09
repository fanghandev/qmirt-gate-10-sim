# Cardiac-SPECT sparse SRM production on OSPool

This mirrors the brain-SPECT workflow that runs on SDSC Expanse
([submit_slurm/README.md](../submit_slurm/README.md)), adapted to HTCondor and the
OSPool execution model.

The key property is that **ROOT files never leave the execute node**. Each job
simulates, converts its own output to a sparse SRM, deletes the ROOT files, and
returns a small tarball. A campaign of 10,000 jobs therefore moves megabytes rather
than terabytes.

| Stage | Runs on | Output |
| :--- | :--- | :--- |
| simulation + sparse SRM | OSPool execute node | `srm_c_<cluster>_p_<proc>.tar.gz` |
| campaign combine + 80-head split | workstation | `final_srm_<label>_head_01..80.npz` |

Cardiac geometry is 80 heads × 625 pixels (25 × 25). The sparse coordinates inside a
partial SRM are `(CrystalID, PixelID, x_bin, y_bin, z_bin)`.

## Submit a campaign

```bash
./submit_htcondor/run_cardiac_sparse_srm_batch.sh \
    --job-count 10000 \
    --num-loops 1 \
    --num-chunks 100 \
    --chunk-duration-s 1.0 \
    --source-activity-bq 5e6
```

Inspect the generated submit file without queueing anything:

```bash
./submit_htcondor/run_cardiac_sparse_srm_batch.sh --job-count 2 --dry-run
```

Each job runs `--num-loops` independent simulations, so the campaign produces
`job_count × num_loops` partial SRMs. Keep a single job comfortably inside the OSPool
eviction window; loops exist to amortize the container and geometry setup cost, not to
extend runtime.

### How the payload reaches the execute nodes

`transfer_input_files` is re-sent by the access point for **every job**, with no reuse
between jobs even on the same node. Shipping the repo directly would mean 15.8 MB per
job — 154 GB across a 10,000-job campaign, most of it brain-SPECT STL and a stale
3.7 MB CSV that cardiac never opens.

So the launcher publishes the payload to OSDF and jobs pull it through the site cache,
exactly like the `.sif`: the first job at a site fetches it, the rest read it locally.

```bash
./submit_htcondor/publish_payload_osdf.sh
# osdf:///ospool/ap40/data/fang.han/payload/qmirt-cardiac-payload-<hash>.tar.gz
```

The launcher calls this automatically, so normally you never run it by hand. The
archive holds `payload/python`, `persistent_data/{cardiac_spect,GateMaterials.db}` and
`qmirt/src` — 993 KB compressed — and the wrapper unpacks it in the sandbox.

| Approach | Per job from the AP | 10,000 jobs |
| :--- | ---: | ---: |
| whole repo | 15.8 MB | 154 GB |
| trimmed staging (`--no-osdf`) | 2.3 MB | 22 GB |
| OSDF (default) | ~0 | ~0, cached per site |

The filename embeds a **content hash**, which matters: OSDF caches key on path, so
republishing under a fixed name can leave execute nodes reading a stale payload for
hours. Changed sources produce a new hash and therefore a new path, which can never be
stale. Identical sources reproduce the same hash — the archive is built with fixed
ownership and mtime and `gzip -n` — so re-publishing is a no-op.

Options:

- `--no-osdf` stages the trimmed tree from the access point instead. Useful when OSDF
  is misbehaving, or for a two-job pilot where cache warm-up is pointless.
- `--payload-url osdf:///...` pins an exact payload, so a campaign can be reproduced
  later against the code it actually ran with.

The resolved input is recorded as `payload_input` in `campaign_manifest.json`.

Defaults:

| Setting | Value |
| :--- | :--- |
| FOV | 150 mm sphere **diameter** (grid is the ±75 mm cube) |
| Resolutions | 1, 1.5, 2 mm → 150³, 100³, 75³ voxels |
| Energy window | 20% at 140 keV (126–154 keV) |
| Energy blurring | 10% FWHM Gaussian at 140 keV |
| Activity | 5e6 Bq × 100 chunks × 1 s |

The launcher rejects a `--fov-size-mm` that any requested resolution does not divide
evenly, so that failure surfaces at submit time rather than after the simulation.

Outputs land in `/ospool/ap40/data/fang.han/cardiac_spect_srm/<batch_id>/`, together
with a `campaign_manifest.json` recording the resolved parameters and the git commit.

## Combine on the workstation

The OSPool data area is mounted over sshfs at `~/ospool`. The combine script reads the
tarballs from that mount but does all work on local disk, and refuses to write results
back onto the mount.

```bash
ma opengate
./submit_htcondor/run_cardiac_campaign_combine.sh --latest
```

Select a specific campaign instead:

```bash
./submit_htcondor/run_cardiac_campaign_combine.sh --batch batch_20260909_004518
```

This extracts to `<local-dir>/partials/` and writes 80 per-head CSR matrices per
resolution. Extraction is incremental — each archive gets a marker file — so the
script can be re-run while a campaign is still draining and it will only unpack jobs
that finished since the last run.

Paths are configurable by environment variable:

| Variable | Default |
| :--- | :--- |
| `OSPOOL_MOUNT` | `~/ospool` |
| `CAMPAIGN_SUBDIR` | `cardiac_spect_srm` |
| `LOCAL_ROOT` | `/data/fanghan/opengate_sim/data/cardiac_spect` |

### Large campaigns

A single combine pass folds every partial into one accumulator. At 1 mm with tens of
thousands of partials that will not fit in memory, so use a tree reduction:

```bash
./submit_htcondor/run_cardiac_campaign_combine.sh --latest --shard-count 16
```

That merges the partials in 16 independent groups, then merges the 16 group results and
splits the outcome per head. The result is bit-identical to the single-pass combine.

## Output format

Each per-head file is a `scipy.sparse` CSR matrix of shape
`(625 pixels, grid³ voxels)` with `int64` counts, where the voxel column index is
`x·g² + y·g + z`:

```python
from scipy.sparse import load_npz
A = load_npz("final_srm_1mm_head_07.npz").astype("float32")   # (625, 3_375_000)
```

`save_npz` cannot store extra arrays, so grid size, voxel size, extent, energy window
and per-head totals live in `combined_srm_metadata.json` next to the NPZs. That file is
also what makes the set self-checking: the reader verifies each head's total against it
and refuses a set that a concurrent combine is midway through replacing.

## Event accounting

`raw_events` in the per-loop `srm_metadata_*.json` is the number of singles read from
the ROOT trees; `accepted_events` is how many passed the energy window and landed
inside the FOV grid. Neither is the number of simulated primaries — that comes from the
GATE stats file, retained per loop under `stats/` in each job tarball.

## Relationship to the other scripts here

`run_cardiac_spect_sim_batch.sh` and `wrapper_cardiac_spect_sim_batch.sh` still run the
older flow that returns full ROOT archives. They are kept for phantom studies and any
analysis that needs the raw singles; they are not part of SRM production.
