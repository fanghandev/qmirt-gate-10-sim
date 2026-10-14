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

## Watcher (since 2026-10-09 23:10 UTC)

- `qmirt-brain-phantom-harvest.timer` (systemd user, at :15 and :45) runs
  `campaign_tools/watch_brain_phantom_campaigns.py`. It pulls new outputs to
  `/data/fanghan/opengate_sim/data/brain_phantom/{ospool,eris}/<campaign>/`, checks every slice,
  and writes `progress_phantom.json`, plus a line per campaign in `watch.log` there. It gives
  held OSPool jobs a PeriodicRelease, and merges a campaign once it is complete on a cluster
  (`merged/projections.npz`).
- `qmirt-ssh-keepalive.timer` (every 4 min) keeps the OSPool and ERIS SSH masters open
  (`ControlPersist 600`). After a reboot or a lost connection the user must reconnect
  (`ssh ospool` with MFA, `ssh eris`).
- Unit templates are in `submit_slurm/systemd/`; the installed copies are in `~/.config/systemd/user/`.
- Resubmit failed slices on OSPool (on the access point, `~/qmirt-brain-phantom`):
  `bash submit_htcondor/run_brain_phantom_batch.sh --into <campaign dir> --slices 3,17,...`;
  the failed slices are listed in `progress_phantom.json`.

## Incident 2026-10-09: SIGILL on old OSPool nodes

- 74 Jaszczak jobs exited 132 (illegal instruction) about 60 s in, on x86_64-v2 nodes
  (palmetto.clemson.edu, aglt2, uconn). The container's polars needs v3.
- Fix: the queued clusters were edited to require `Microarch` x86_64-v3/v4 (93 % of slots) and to
  return failed jobs to the queue (up to 5 starts); the launcher now does both.
- The 74 slices were resubmitted as cluster 15868860.
- The cardiac OSPool workflow only requires Singularity and could hit the same problem if its jobs
  import polars.

## Status 2026-10-13 (13:40 EDT)

| Campaign | OSPool | ERIS |
| --- | --- | --- |
| small Jaszczak | 1797/1800; the last resubmitted jobs (cluster 16040967) are finishing | **complete, merged** |
| mesh50 clinical | **complete, merged** | **abandoned**: 288/720 done (see below) |
| mesh50 10× | 8994/9000; the last runaway resubmissions (cluster 16040968) are finishing | **abandoned**: 0/7200 done (see below) |

## Incident: ERIS scratch purge deleted the payload (2026-10-12)

- The payload is unpacked with its archive's fixed 2020 mtime (kept for a reproducible hash), so
  the scratch purge saw it as old and deleted it while the jobs ran. The simulations finished
  (about 7 h each), but `reduce_phantom_singles.py` was gone, so the slices exited 2 with no
  list-mode. Their ROOT files were kept, because cleanup only runs after a successful reduction.
- Fix: the launcher unpacks with `--touch`, and every slice refreshes the payload mtimes.
- Damage (slice scan 2026-10-13):
  - clinical: 288 done; 384 finished the simulation and kept their ROOT files (reduction only
    missing); 48 (slices 240–287, array task 5) were cancelled at the Slurm time limit
    (2026-10-10 09:36) with partial ROOT files.
  - 10×: slices 0–383 finished the simulation and kept their ROOT files; from slice 384 on the
    purge had already happened, so the slices exited 2 within seconds
    (`can't open file .../gate_sim_brain_spect_boolean.py`) and have no data.

## Decision 2026-10-13: ERIS mesh50 route abandoned

- The ERIS mesh50 clinical and 10× campaigns are not recovered or re-simulated. ERIS was the
  backup for the same targets, and OSPool has the clinical campaign merged and the 10× campaign
  at 8994/9000. Recovering the 10× campaign would mean simulating about 6800 slices again
  (about 30k ERIS core-hours) for a duplicate.
- The ERIS small Jaszczak campaign is complete and merged, and stays usable.
- Nothing is queued on ERIS. The failed slices are left on scratch for the purge; the harvested
  copies stay in `/data/fanghan/opengate_sim/data/brain_phantom/eris/`. The watcher still
  lists these campaigns as incomplete.
- Also seen 2026-10-13: Slurm commands failed on `erishpc-login-002` (DNS SRV lookup of the
  controller), so the watcher could not read the ERIS queue.

## Incident: radioactive decay not applied across slices (found 2026-10-13)

- opengate (10.1.0 and 10.1.1) starts the decay clock at each simulation's first time window,
  not at t = 0. Every time-sliced job therefore restarted the decay: primaries are flat over the
  900 s instead of 2.8 % lower at the end.
- Finished data: the merge keeps each single of a slice starting at t0 with probability
  exp(−λ t0) (Poisson thinning). The result is exactly a source decaying from t = 0: decay-weighted
  primaries 4.3858e11 (Jaszczak) and 3.5082e10 (head), as expected. The merges without this
  correction are kept as `merged/projections_nodecay.npz`.
- Sim fix (`df3f39b`): a job starts with the activity decayed to its slice start and records
  `phantom_activity_decay_origin = acquisition`; the merge does not thin such slices.
- The running and resubmitted jobs use the old payload; the merge corrects them.

## Note: stuck-track warnings in the mesh head

- About 1 GeomNav1002 warning per 10⁷ photons, where the brain or air cavity meets the skeleton
  or body (the 20 µm cut gaps). Geant4 pushes the track 1e-7 mm and continues; the physical effect
  is negligible.
- 9 of the 9000 OSPool 10× jobs ran past 20 h, presumably tracks that never got unstuck; their
  slices are resubmitted with new seeds.

## Bug found 2026-10-13: mesh50 activity placed unrotated (all mesh50 campaigns)

- opengate's `VoxelSource` ignores `position.rotation`: only `position.translation` places the
  image, in the frame of the `attached_to` volume (`sources/voxelsources.py`, 10.1.0 to master;
  the docs only describe the translation). `phantom_gate.add_activity_source` set the rotation
  on a source attached to the world for the mesh geometry, so the activity image was placed
  unrotated: face towards −y, brain at about (−12, +18) mm, while the head meshes (the
  attenuating anatomy) were placed correctly (face towards +y, the face window).
- Evidence:
  - the decay positions of the clinical OSPool singles match the unrotated image (smoothed
    correlation 0.95) and not the intended pose (0.61);
  - a minimal GATE test (hot voxel, 180° about z): `position.rotation` changes nothing;
    attaching the source to a rotated volume rotates it;
  - `dev/python/check_gate_phantom.py`, now also run on the mesh geometry: the old code has
    17.9 % of the emissions outside the activity (FAIL), the fix 0 % (PASS), also for a
    flipped, rotated and shifted pose.
- Affected: mesh50 clinical and 10× on OSPool and ERIS. Not affected: the small Jaszczak
  (analytic `GenericSource`) and the voxel geometry (the source follows the ImageVolume).
- Fix: the source is attached to the body mesh (the image frame) and translated to the image
  centre. Payload `qmirt-brain-phantom-payload-477dd53bd115` (mesh50_head only) has the fix and
  the decay origin fix (`df3f39b`).
- The old mesh50 data stay usable as a mismatched case only: the true activity is the
  unrotated image (`dev/python/brain_phantom_recon/activity_as_simulated_2mm.npy`).

## Expanse campaign 2026-10-13 (fixed mesh50 clinical)

- Launcher: `submit_slurm/run_brain_phantom_eris.sh --cluster expanse` (Singularity, account
  mgh102, whole 128-core `compute` nodes, `--slice-timeout` per slice). Run from
  `~/brain_phantom_launcher` on Expanse with `SIF=~/qmirt-gate-10-sim/submit_slurm/qmirt-gate-10-sim-sif_v1.0.0.sif`
  (the same image as ERIS and OSPool, sha256 7c1b929e…).
- Pilot (debug, job 54899130, 128 × 0.01 s): all exit 0; 282 µs per primary with 128
  single-threaded processes per node (EPYC 7742); about 30 s start-up per slice. Decays match
  the intended pose (correlation 0.92 vs 0.60 for the old placement).
- Full: array 54899180, `/expanse/lustre/projects/mgh102/fhan1/brain_phantom/full_mesh50_clinical_fixed_20261013T183201Z`,
  1280 slices × 0.703125 s (10 nodes × 128), 5 h limit, 4 h slice timeout. Estimate about
  2.2 h per slice, 3,000–3,800 SU. Not watched by the harvest timer; copy the slices home
  when done.
