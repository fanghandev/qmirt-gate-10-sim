# Sparse SRM generation

This workflow processes the GATE result archives produced by the cardiac
simulation jobs and creates sparse 5D SRM chunks. The sparse coordinates are:

```text
(CrystalID, PixelID, x_bin, y_bin, z_bin)
```

The worker runs inside the configured Singularity image on OSPool execute
nodes. The login node only discovers archives, creates Condor manifests, and
submits jobs.

## Generate SRMs

Run from any directory. The script resolves its own location before reading
the submit files.

For the existing cardiac simulation batch:

```bash
./submit_htcondor/run_sparse_batch.sh \
	--input-source /ospool/ap40/data/fang.han/batch_20260813_213906
```

`--input-source` may be either:

- a directory containing `*.tar.gz` result archives; only direct children are
  discovered
- a text file containing one archive path per line

If `--input-source` is omitted, the script uses
`submit_htcondor/filenames.txt`.

The default submission generates all three requested resolutions over a
150 mm centered cubic FOV:

| Voxel size | Bins per axis |
| ---------- | ------------- |
| 1 mm       | 150           |
| 1.5 mm     | 100           |
| 2 mm       | 75            |

Submit only selected resolutions with positional voxel-size arguments:

```bash
./submit_htcondor/run_sparse_batch.sh \
	--input-source /ospool/ap40/data/fang.han/batch_20260813_213906 \
	1.5
```

The FOV can also be changed explicitly:

```bash
./submit_htcondor/run_sparse_batch.sh \
	--input-source /path/to/archives \
	--fov-size 150 \
	1 1.5 2
```

Resource profiling is enabled by default for sparse workers. Disable it or
change the sampling interval from the launcher:

```bash
./submit_htcondor/run_sparse_batch.sh \
	--input-source /path/to/archives \
	--no-profile-resources

./submit_htcondor/run_sparse_batch.sh \
	--input-source /path/to/archives \
	--profile-interval-s 10
```

Use `--help` to display the available launcher options.

## Job chunking

Archives are sorted and grouped into Condor manifest entries with 100 archives
per worker. For `N` discovered archives:

$$
\mathrm{workers} = \left\lceil \frac{N}{100} \right\rceil
$$

Each worker reads its archive slice once and accumulates all requested voxel
sizes during that pass. Therefore, the default three-resolution run submits
one Condor cluster containing the calculated number of worker processes.

For example, 1,001 archives produce 11 workers total, with each worker writing
all requested resolutions.

Each worker transfers one bundle containing the requested resolution files to
the batch output directory:

```text
/ospool/ap40/data/fang.han/<sparse_batch_id>/
	job_<cluster>_<proc>_sparse_5d_srm_outputs.tar.gz
	job_<cluster>_<proc>_resource_profile.tsv
	job_<cluster>_<proc>_resource_profile_summary.txt
```

The profiling files report time-series samples and per-job aggregates (average
CPU usage, CPU usage relative to the requested cores, peak RSS memory, and
final read/write volume).

## Event accounting

The worker reads the GATE stats member matching `*_sim_stats.txt` in every
archive. The simulated-event total comes from the stats JSON field:

```json
"events": {"value": 499974131}
```

The worker requires exactly one stats member per archive and stores the sum in
both `simulated_events` and the backward-compatible `total_events` NPZ fields.
The combiner sums these metadata values across workers.

`accumulated_counts` is a separate value. It is the number of detector hits
that passed the energy and FOV filters and were inserted into sparse bins; it
is not the number of simulated events.

## Combine one resolution

Combine each resolution separately. Do not combine different voxel sizes into
one SRM because their coordinate grids are incompatible. The third argument
selects the requested grid from each worker bundle.

```bash
./submit_htcondor/run_sparse_combine.sh \
	/ospool/ap40/data/fang.han/<sparse_batch_id> \
	/ospool/ap40/data/fang.han/<sparse_batch_id>/combined_1mm \
	1
```

Repeat for the other grids:

```bash
./submit_htcondor/run_sparse_combine.sh \
	/ospool/ap40/data/fang.han/<sparse_batch_id> \
	/ospool/ap40/data/fang.han/<sparse_batch_id>/combined_1.5mm \
	1.5

./submit_htcondor/run_sparse_combine.sh \
	/ospool/ap40/data/fang.han/<sparse_batch_id> \
	/ospool/ap40/data/fang.han/<sparse_batch_id>/combined_2mm \
	2
```

The destination argument is optional. If omitted, a new
`sparse_combine_<timestamp>` directory is created under the OSPool data path.
The final transferred file is `combined_sparse_5d_srm.npz`.

The combiner validates that all chunks have matching histogram, FOV, voxel,
and energy metadata. It also accepts the legacy worker output naming pattern
from earlier batches.
