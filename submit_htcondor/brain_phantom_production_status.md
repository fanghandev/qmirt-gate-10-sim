# Brain SPECT phantom acquisitions: production status

Launched 2026-10-09, about 22:55 UTC. The code is commit `d806a78` (branch `brain-288mm-csg-production`), in
the `~/qmirt-brain-phantom` worktree on both clusters. The payload is
`qmirt-brain-phantom-payload-ccd31d70e0f2.tar.gz` (on OSDF, and in `~/brain_phantom/payloads/` on ERIS).

## Settings (all campaigns)

- Brain SPECT, 288 mm FOV geometry, CSG shield, merged digitizer, `G4EmStandardPhysics_option4`.
- Tc-99m gamma lines (ICRP-107, 0.891 photons per decay) with the 6.0067 h half-life, a 15 min
  acquisition (0–900 s), and time slices per job.
- Singles are stored unblurred (true deposited energy) with phantom-scatter truth and the decay
  position. Each job reduces its singles to `listmode.npz`; the ROOT files are deleted.
- Merge (`payload/python/merge_phantom_listmode.py`):
  - energy blur, FWHM 10 % at 140.5 keV, ∝ √E;
  - 1 keV projections over 10–200 keV, and the main and TEW windows;
  - split by decay inside or outside the SRM FOV (r < 144 mm), and by scattered or not in the phantom.
- Phantoms (specs in the payload's `payload/phantom_specs/`):
  - `small_jaszczak`: flipped, z −25 mm, rotated −150°, activity in the water;
  - `mesh50_head`: Auer's mesh50_XCAT head, brain perfusion, brain centre at the FOV centre.
- The two clusters are independent backups of the same target; seeds differ per job.

## Campaigns

| Campaign | Activity | Photons | OSPool | ERIS |
| --- | --- | --- | --- | --- |
| small Jaszczak | 15 mCi (5.55e8 Bq) in the water | 4.39e11 | cluster 15868854: 1800 × 0.5 s | array 5735283: 1200 × 0.75 s (25 tasks × 48) |
| mesh50 clinical | 27.75 MBq in the brain (5 % of 15 mCi); 44.4 MBq in total | 3.51e10 | cluster 15868855: 1800 × 0.5 s | array 5735285: 720 × 1.25 s (15 × 48) |
| mesh50 10× | 277.5 MBq in the brain | 3.51e11 | cluster 15868856: 9000 × 0.1 s | array 5735286: 7200 × 0.125 s (150 × 48) |

- OSPool outputs go to `/ospool/ap40/data/fang.han/brain_phantom/full_*_<UTC>/phantom_c_<cluster>_p_<proc>.tar.gz`.
  Logs are in `~/qmirt-brain-phantom/submit_htcondor/logs/brain_phantom/`.
- ERIS outputs go to `/scratch/f/fh890/brain_phantom/full_*_<UTC>/slices/slice_<s>/`; each slice holds
  `exit_code.txt` and `job_info.json`. Logs are in `logs/`. Copy them home when finished (scratch).

## Test runs (2026-10-09)

All jobs exited 0, primaries matched the expected counts, and the time windows were contiguous.

| Cost per primary | Small Jaszczak | mesh50 head |
| --- | --- | --- |
| OSPool (EPYC / Xeon) | about 23 µs | about 230 µs (161–314) |
| ERIS (Xeon Gold 5318Y) | 49 µs | 324 µs |

- Memory per process: 0.45 GB (Jaszczak), 1.3 GB (head).
- Estimated cost: OSPool about 27k core-hours, ERIS about 41k core-hours.
- The test outputs are in `test_*` next to the full campaigns.

## Notes

- The cluster container has opengate 10.1.0. The sim falls back to its actor-based phantom-scatter
  counter, which gives the same counts as 10.1.1 but refuses multi-threading. ERIS therefore runs
  single-threaded processes side by side.
- mesh50: 42 % of the main-window counts come from decays outside the FOV (neck and shoulders), mostly
  on the bottom-ring heads 0, 9 and 1. The merge stores them separately.
