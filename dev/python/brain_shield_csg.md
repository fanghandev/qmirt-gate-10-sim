# Brain SPECT lead shield: CSG model vs STL

Status as of 2026-10-02. This note describes the constructive solid geometry (CSG)
model of the brain SPECT lead shield, how its parameters were measured from the
STL, how the two were compared, and a Gate 10 benchmark of CSG vs STL shielding.
Read it before changing the shield geometry in the brain simulation.

## Summary

- The shield STL (`BrainFrame.008.Lead_Shield.STL`, 186,772 facets) is described
  almost exactly by a spherical shell with a cone cut at the bottom, a two-step face
  window and 73 identical tapered square apertures. Every parameter is a round
  number or fits the STL to within its own facet accuracy.
- The STL has rounded edges (4.73 / 6.0 / 14.0 mm). An optional patch adds them to
  the manifold3d model; they cannot yet be built in Gate 10 (see Limitations).
- In Gate 10, the CSG built as 73 tiles costs 9 µs per primary per thread in a
  production-like scene (shield + 73 collimators + 73 crystals, 64 threads), vs 30 µs
  with the simplified STL used in production today and 403 µs with the full STL.
  The shield alone costs ~2 µs as CSG tiles vs 24 µs (simplified STL) and 329 µs
  (full STL).
- Physics: in a bare shell detector the sharp CSG passes 2.0% more unscattered
  photons through the apertures than the STL (the STL's holes are slightly
  narrower at the lip and have small corner fillets) and 0.14% fewer at the window
  and bottom edges (missing rounds). With the collimators in place (production-like
  scene) the counts entering the crystals agree with the full STL to +0.08% in total
  (paired per-head scatter 0.8%, max 3.3%): the collimators mask the apertures.
- Recommendation: switch production to the tiled CSG after one A/B benchmark on
  Expanse. The remaining geometric differences (rounds, aperture fillets) change the
  crystal counts by < 0.1%; the speed-up is 3.3x over the simplified STL and ~45x
  over the full STL in the production-like scene.

## Files

| File | Role |
| --- | --- |
| `payload/python/brain_spect_shield_csg.py` | **Source of the CSG parameters** (`SHIELD_CSG`) and the Gate 10 builder `add_csg_shield_to_gate_sim(sim, pl_df, layout="tiles" \| "single")` |
| `dev/python/brain_shield_csg.py` | manifold3d model of the same CSG (`build_csg(segments, rounded=False)`), STL loader, exact signed distance, radial lead thickness by ray casting |
| `dev/python/plot_brain_shield_csg_vs_stl.py` | 2D/3D comparison plots; `--rounded` for the rounded model |
| `dev/python/brain_shield_csg_comparison[_rounded]/` | outputs: `summary.json`, thickness maps, sections, `apertures.pdf` (one page per aperture), `whole_3d.html`, `regions_3d.html` |
| `dev/python/benchmark_brain_shield_models.py` | Gate 10 benchmark (`run`, `analyse`) |
| `dev/python/split_brain_shield_stl.py` | the 0.05 mm simplified STL used in production today |

## Frames and conventions

- World frame of the Gate simulation; the STL is placed with
  `get_shielding_rotation_matrix()` (Rx(-90) Rz(180)) and no translation.
- Azimuth is measured from +x toward +y (90 deg = +y, the face-window side);
  elevation from the xy plane. G4Sphere uses theta = 90 deg - elevation.
- Module (aperture) frame: origin at the pinhole, +z from the pinhole toward the FOV
  centre, rotation `get_head_rotation_matrix()`. Depth z > 0 is inside the pinhole
  plane (toward the centre).
- Aperture N = module N = `Collimator_N` / `DetectorCrystal_N` = SRM head N. Order:
  elevation rings bottom to top (slow index), azimuth counter-clockwise starting just
  past 90 deg (fast index). Quirk: the module exactly at 90 deg comes last in its ring
  (apertures 41, 56, 67, 72) because the azimuth is rounded to 6 decimals before
  subtracting pi/2. Do not renumber: existing SRMs use this order.

## The CSG model

| Part | Definition |
| --- | --- |
| Shell | r = 145.0 to 155.0 mm, elevation >= -47.0 deg (G4Sphere theta 0 to 137 deg) |
| Face window, upper | azimuth 56.25 to 123.75 deg (90 +/- 33.75), elevation -26.5 to -5.75 deg |
| Face window, lower | azimuth 50.524 to 129.476 deg (90 +/- 39.476), elevation -47.0 to -26.5 deg |
| Aperture (x73) | module frame: square, width 27.528 - 1.5861 z mm for z <= 3.8 mm (G4Trd), then a 21.65 mm square lip (G4Box) through the inner surface. Identical for all 72 ring modules; the top module's aperture is rotated 2.5 deg about its axis |
| Rounds (optional, manifold3d only) | 4.73 mm on every inner edge of the window and bottom cut (where a cut meets r = 145); 6.0 mm at the 6 window outline corners; 14.0 mm at the 2 window-bottom junctions |

Notes:

- Ring elevations are -37, -16, 4.5, 25, 45.5, 67 and 90 deg; the window edges
  (-26.5, -5.75) are exactly the midpoints between the first three rings.
- The aperture taper opens faster than the collimator's outer wall: at the pinhole
  plane the hole is 27.5 mm wide and the collimator nozzle 18.3 mm, a 4.6 mm open gap
  per side. The unused `BrainSPECT_Module.008` STL may be the hardware that fills it;
  worth checking against the real design.

## How the parameters were measured

All measurements use the STL in the world frame (Gate rotation applied).

- **Shell radii.** Vertex radii span 145.0000 to 155.0000 mm.
- **Bottom cut and window angles.** Facets whose plane contains the centre (normal
  perpendicular to the radial direction) are cuts that G4Sphere can express. Their
  vertices give the exact angles: azimuth 50.5240 / 56.2500 / 123.7500 / 129.4760
  deg (zero spread) and elevation -46.9964 +/- 0.0091 (bottom), -26.5000, -5.7500 deg.
- **Apertures.** For each module, a 70 x 70 mm patch of the STL was voxelized at
  0.05-0.1 mm in the module frame (exact parity fill of the triangles) and the open
  square around the axis measured at depths -3.4 to +4.6 mm. All 72 ring apertures
  give the same widths: 27.528 - 1.5861 z (residual <= 0.05 mm, the voxel size),
  constant 21.65 mm from z = 3.8 to 4.3, then a small flare at the inner surface.
  The top aperture fits the same law after a 2.5 deg in-plane rotation.
- **Rounds.** Circle fits to STL sections perpendicular to each edge: 4.73 mm on
  every inner edge (centre at r = 149.73 = 145 + 4.73 and 4.73 mm from the cut, i.e.
  tangent to both); 6.0 mm at all window corners (fixed-radius test: mean deviation
  0.01-0.05 mm at 6.0 vs >= 0.11 mm at 5.5 or 6.5); 14.0 mm at the window-bottom
  junctions. Corner radii are the same at r = 152 and 154 mm, so these are straight
  cylinders along the corner's radial direction, not cones.

## How volumes and differences are computed

- **Solids.** The STL is a closed manifold; it is loaded into manifold3d in double
  precision. The CSG is built in manifold3d from the same parameters (spheres and
  cones with 720 segments, hulls for the aperture trd/box). Volumes are exact mesh
  volumes of these solids.
- **Differences.** `STL - CSG` and `CSG - STL` are exact boolean differences,
  decomposed into connected pieces. The largest piece of each is a thin film over a
  sphere surface: the STL's flat facets sit up to 0.22 mm inside the true spheres
  (inner surface bulges inward, outer surface falls short), so these films measure
  the STL's faceting, not a CSG error. The remaining pieces are assigned to the
  nearest aperture (module-frame box +/- 32 mm, |z| < 12 mm).
- **Signed surface distance.** Every triangle is sampled at <= 0.5 mm; all triangles
  owning a sample within (nearest-sample distance + 0.5 mm) are tested exactly, so the
  closest point is exact (checked against brute force). An earlier version picked
  candidates by nearest centroid and overstated distances on the long aperture-wall
  triangles; plots and numbers in this note use the exact version.
- **Lead thickness maps.** Radial rays from the centre (r = 140-160 mm) on a 0.25 deg
  azimuth x elevation grid, ray-cast against both solids; the lead path length is the
  sum of entry-exit segments.

## Findings (manifold3d comparison)

| | STL | CSG sharp | CSG rounded |
| --- | --- | --- | --- |
| Volume (cm³) | 1721.62 | 1722.73 (+1.11) | 1718.26 (-3.37) |
| STL - CSG, film / other (cm³) | | 20.75 / 1.47 | 21.19 / 1.14 |
| CSG - STL, film / other (cm³) | | 22.87 / 0.46 | 18.50 / 0.46 |
| Radial thickness differs by > 0.5 mm | | 1.17% of directions | 0.30% of directions |
| Largest non-film piece | | 74 mm³ (window corner) | 20 mm³ (aperture) |
| STL vertex to CSG surface, max (99.9%) | | 3.36 mm (1.63 mm) | 1.05 mm (0.55 mm) |

- The rounds are the largest *shape* difference (about 4.4 cm³ of lead at the window
  and bottom edges), but the net volume is a poor measure: the STL's faceting adds
  about 2.3 cm³ of lead (inner film 20.7 cm³ vs outer film 18.5 cm³) and the aperture
  details (hole-corner fillets, lip chamfer) about 1.1 cm³ (~15 mm³ per aperture).
  Without rounds these effects happen to cancel to +1.1 cm³; with rounds the CSG is
  3.4 cm³ lighter, because the STL carries the extra facet lead.
- Every aperture has the same small residual (~15 mm³ STL-only, ~6 mm³ CSG-only).
- See `brain_shield_csg_comparison/apertures.pdf`: per aperture, a horizontal cut at
  the pinhole height (fixed scanner x-y), a vertical cut at its azimuth, zooms, and
  12 slices perpendicular to its axis with its own collimator and bore.

## Gate 10 implementation

`add_csg_shield_to_gate_sim(sim, pl_df, layout)` in
`payload/python/brain_spect_shield_csg.py`:

- `tiles` (recommended): for each ring, a G4Sphere segment per aperture between the
  mid-azimuths to its neighbours and between the mid-elevations to the next rings,
  minus that aperture (G4Trd united with G4Box). The window is where no tile is
  placed (its edges are band and sector edges). 73 world volumes, each one small
  boolean with a small bounding box.
- `single`: one G4Sphere minus the two window segments minus 73 apertures.
- opengate passes a boolean's rotation to Geant4 as a frame (passive) rotation, so an
  object placed with rotation R (local to world) is given `rotation=R.T`. This was
  checked by sampling surface points of a test boolean.

Validation:

- Geant4 surface points (`GetPointOnSurface`) of both layouts lie on the manifold3d
  CSG surface to within 0.004 mm (all tile points off the surface lie on internal tile
  boundaries, which are inside the lead). Geant4's volume estimate: 1723.9 cm³
  (Monte Carlo, ~0.1% noise) vs 1722.7 cm³.
- Geant4 `CheckOverlaps`: no overlaps between the tiles, or between the tiles and the
  73 collimators. No G4Exceptions in any run.

## Benchmark (Gate 10)

`dev/python/benchmark_brain_shield_models.py`. Workstation: AMD Threadripper PRO
3995WX (Zen 2, the same architecture as Expanse's EPYC 7742), 64 threads pinned to
the 64 physical cores, one simulation at a time. Cost per primary is the slope
between two run sizes (setup cancels).

### Shield-only scene

World air; the shield only; a CsI shell detector r = 160-170 mm, elevation >= -50 deg
(a dome rather than a hemisphere, so it also sees the window and bottom edges, where
STL and CSG differ most). 140 keV photons. Timing with a uniform 210 mm sphere
source; physics with a point source at the centre (rays are radial, so differences
appear exactly where the shield differs). 3.2e7 primaries per model, same seed.

| Shield | µs per primary per thread (64 threads) |
| --- | --- |
| STL, 186,772 facets | 329 |
| STL simplified 0.05 mm (production today) | 23.8 |
| CSG, single boolean chain | 96 |
| CSG, 73 tiles | ~2 (at the noise floor of these run sizes; see Pending) |
| no shield | ~0 (below the run-to-run noise of ~0.5 s) |

| Unscattered photons entering the detector, per primary | STL | CSG tiles | CSG vs STL |
| --- | --- | --- | --- |
| through apertures (within 9 deg of a pinhole axis) | 0.12485 | 0.12734 | +2.0% |
| elsewhere (window, bottom opening) | 0.08091 | 0.08080 | -0.14% |

- The simplified STL matches the STL (-0.16% total, no structure).
- The CSG maps differ from the STL only at the aperture rims (thin outlines, more
  counts) and at the window corners (missing rounds). Scattered-photon maps agree
  within noise. Single and tiled CSG give identical results.
- The aperture excess comes from the STL's slightly narrower lip opening: the
  projected half-width at the pinhole plane is ~0.08 mm larger in the CSG at the 99th
  percentile (hole-corner fillets, lip chamfer, and possibly a lip a few hundredths
  of a mm narrower than the 0.05 mm-resolution fit).

### Production-like scene (collimators and crystals)

Shield + the 73 collimators + the 73 crystal boxes (no pixel grid, no digitizer),
uniform 210 mm sphere source, photons entering each crystal recorded.

| Shield | µs per primary per thread (64 threads) |
| --- | --- |
| STL, 186,772 facets | 403 |
| STL simplified 0.05 mm (production today) | 29.8 |
| CSG, 73 tiles | 9.0 |

The collimators (booleans) now dominate the CSG run; the shield model still sets
the difference between the rows.

Photons entering the 73 crystals, 1.28e8 primaries per model, same seed (paired),
relative to the full STL:

| Shield | unscattered (> 139.9 keV), per primary | photopeak 126-154 keV, per primary | total vs STL (photopeak) | per-head difference (std / max) |
| --- | --- | --- | --- | --- |
| STL | 5.6616e-4 | 5.6898e-4 | | |
| STL simplified 0.05 mm | 5.6614e-4 | 5.6896e-4 | -0.003% | 0.08% / 0.33% |
| CSG tiles | 5.6663e-4 | 5.6943e-4 | +0.08% | 0.77% / 3.3% |

About 990 photons per head per run; the runs share a seed, so the per-head
differences are far below independent Poisson noise (~4.5%) and reflect real but
small geometric differences.

## Implications for the Expanse SRM production

Cost per primary per thread on the workstation (64 threads):

| Configuration | µs | Source |
| --- | --- | --- |
| per-head actors + full STL (the 210 mm campaigns) | 518 | full sim, 2026-10-01 |
| merged actors + simplified STL (production scripts now) | 24-33 | full sim, 2026-10-01 |
| merged actors + CSG tiles | ~10 (estimate) | production-like scene 9.0 vs 29.8 for the simplified STL |

- The 210 mm campaigns cost 196 µs per primary per core on Expanse (vs 518 here), so
  the workstation probably overstates the old configuration's cost; expect the
  Expanse gain to be smaller than the workstation's ~50x. If it is 10-50x, the full
  288 mm target (1.04e14 primaries, ~5.4M SU at the old rate) needs roughly
  0.1-0.55M SU.
- Loops get short: a 7.94e9-primary loop took 3.4 h; at 10-50x it takes 4-20 min, so
  per-loop overheads (Gate start-up and geometry build, the volume overlap check,
  ROOT to SRM conversion) start to matter.

Implemented 2026-10-02 (not yet run on Expanse):

| Change | Where |
| --- | --- |
| `--shield-model {stl,pieces,csg}` (default `stl`; `--shield-pieces-dir` alone still means pieces) | `gate_sim_brain_spect_boolean.py` (`resolve_shield_model`) |
| `--check-overlaps`, off by default (Geant4's check used to rerun every loop) | `gate_sim_brain_spect_boolean.py` |
| `SHIELD_MODEL`, `CHECK_OVERLAPS` passed to Gate and the provenance writer, recorded in `campaign_manifest.json` | `wrapper_brain_spect_sim_slurm.sh`, `run_spect_sim_slurm.sh` |
| `shielding.model` (+ the CSG parameters, or the pieces manifest hash) in `geometry_provenance.json`; the STL stays as `shielding.source` | `write_brain_spect_geometry_provenance.py` |
| Production defaults: merged actors, `SHIELD_MODEL=csg`, `NUM_CHUNKS=40` (3.175e10 primaries per loop, 4x before; same 7.9e8 events per chunk) | `run_brain_expanse_127_production.sh` |
| Campaign plan: 10 parts x 64 tasks x 5 loops = 1.016e14 primaries, group `brain_288mm_csg_102t_<stamp>`; `NUM_CHUNKS`, `SHIELD_MODEL`, `ACTOR_LAYOUT` passed through; provisional `SU_PER_LOOP=175` | `launch_brain_expanse_127_production_remote.sh` |
| Configurable benchmark (activity, chunks, shield, actors) | `run_brain_expanse_127_benchmark.sh` |
| A/B benchmark: legacy / pieces / CSG at 2 and 8 chunks (~600 SU) | `launch_brain_expanse_shield_ab_benchmark.sh` |
| Cost per primary, fixed per-loop cost, recommended `NUM_CHUNKS`, `NUM_LOOPS`, `SU_PER_LOOP` | `analyze_brain_shield_ab_benchmark.py` |

Checks done: the production geometry (288 mm FOV, 73 collimators, pixelated
crystals, CSG tiles) has no Geant4 overlaps and no G4Exceptions, both locally
(opengate 10.1.1) and inside the production container `qmirt-gate-10-sim-sif_v1.0.0.sif`
(opengate 10.1.0, same boolean rotation convention); the SRM converter reads its
output; launcher dry runs carry the settings into the sbatch file and manifest.

Launch procedure:

1. Commit and pull the repo on Expanse (the CSG builder is
   `payload/python/brain_spect_shield_csg.py`; no new Python packages).
2. On an Expanse login node: `bash submit_slurm/launch_brain_expanse_shield_ab_benchmark.sh`
   (note the printed stamp).
3. When the six jobs finish:
   `python3 submit_slurm/analyze_brain_shield_ab_benchmark.py <PROJECT_DIR>/brain_spect_sim/brain_ab_*_<stamp>`
   (`--loop-minutes`, `--time-limit-hours` to taste).
4. From the workstation, with the recommended values, e.g.
   `NUM_CHUNKS=<n> NUM_LOOPS=<n> SU_PER_LOOP=<n> DRY_RUN=1 BATCH_COUNT=<parts> bash submit_slurm/launch_brain_expanse_127_production_remote.sh`,
   check the SU estimate, then run it without `DRY_RUN`. If `NUM_CHUNKS` changes, set
   `CAMPAIGN_PART_COUNT` so parts x 64 x loops x chunks x 7.9375e8 is about 1.0e14.
5. Keep the CSG campaign in its own group; do not merge it with STL-shield SRMs
   without the SRM-level check (pending item 3).

## Expanse results and the NUMA split (2026-10-02 to 10-05)

A/B benchmark on Expanse (`launch_brain_expanse_shield_ab_benchmark.sh`, stamp
20261002T175708Z; 127 threads on the shared partition, 288 mm FOV, production
activity; slope between 2- and 8-chunk loops):

| Configuration | µs per primary per thread | Node throughput vs legacy |
| --- | --- | --- |
| legacy: per-head actors + full STL | 217 | 1x |
| merged actors + 0.05 mm simplified STL | 55.9 | 3.9x |
| merged actors + CSG, one Gate process | 50.6 | 4.3x |
| merged actors + CSG, 8 processes x 16 threads, `numactl`-pinned (test job 54591702) | ~15 | ~13x |

- The workstation gave 10 µs for merged + CSG (12 µs inside the production container),
  so the container is not the cause.
- **Why one process is slow on Expanse:** a node has 8 NUMA domains of 16 cores (2 x
  EPYC 7742, NPS4; `ThreadsPerCore=1`). Geant4's master thread allocates the shared
  geometry and physics tables, so they land in its domain (Linux first-touch); about
  111 of 127 workers then read them across the fabric on every step. Photon transport
  is latency-bound (navigation and cross-section lookups), so this dominates once the
  CSG makes the geometry cheap. The workstation is one NUMA domain. One process per
  domain, bound to its cores and memory, keeps every read local; the cost is 8 copies
  of the tables (a few GB each) and 8 start-ups. Not profiled with hardware counters;
  `numastat`/`perf` on one job would confirm it directly.
- The legacy setup was bandwidth-bound on the workstation (8 memory channels for 64
  cores) and ran faster on Expanse (217 vs 518 µs), which hid this effect before.
- Stuck tracks: the CSG run had 3 `GeomNav1002` warnings in 6.35e9 primaries, all on a
  tile's inner surface (r = 145.000 mm) next to an aperture lip; Geant4 pushes the
  photon on. About 5e-10 per primary, negligible for the SRM.

Projection for the full 288 mm target (1.016e14 primaries, 64 nodes, no queue time):
legacy ~5.6M SU / ~31 days; merged + CSG one process ~1.45M SU / ~7.3 days; merged +
CSG NUMA split ~0.47M SU / ~2.4 days.

### The NUMA-split workflow (implemented 2026-10-05)

| Piece | Role |
| --- | --- |
| `submit_slurm/numa_layout.py` | Groups the job's CPUs (`sched_getaffinity`) by NUMA node (`/sys/devices/system/node`); groups under 4 CPUs are merged into a neighbour. `NUMA_SPLIT=auto` (default), `off`, or `N` equal groups (testing). Runs inside the image. |
| `submit_slurm/run_parallel_commands.sh` | Runs one command per line in parallel, fails if any fails, forwards TERM/INT to every process tree. |
| `wrapper_brain_spect_sim_slurm.sh` | Per loop: one `numactl --physcpubind=<cpus> --membind=<node>` Gate process per group, each with its own thread count, task id (`<task>_loop_<n>_n<g>`, so its own seed), output folder `loop_<n>/numa<g>/` and log. Statistics and run manifests are copied from the subfolders; the SRM converter already searches them. Geant4 warnings go to `srm_chunks/geant4_warnings_loop_<n>.txt`; after a failure the log tails go to `srm_chunks/failed_loop_<n>/`. `PYTHONUNBUFFERED=1` keeps Gate's messages in the logs. |
| `report_campaign_progress.py`, `analyze_brain_shield_ab_benchmark.py` | Group statistics files by loop: a loop counts once, its primaries are summed and its time is its slowest process. |
| `NUMA_SPLIT` | Default `auto` in the production and benchmark scripts; recorded in `campaign_manifest.json`. A shared-partition job (127 CPUs) runs 7 x 16 + 1 x 15 threads. |

Local tests (workstation, production container, 64 CPUs, `NUMA_SPLIT=4`):

- layout on real and fake (8-domain) topologies; identical inside the image;
- launcher: success, one failure (others finish, non-zero exit), TERM kills every
  process tree;
- live pinning: each Gate process confined to its 16 CPUs and memory node;
- 2 loops x 4 processes: distinct seeds, all ROOT files converted, primaries in the
  combined SRM equal the sum of the statistics files, task marked complete;
- detection rate split vs one process: 358.5 +/- 2.2 vs 363.0 +/- 2.2 counts per 1e6
  primaries (-1.5 sigma);
- injected failure in one process of loop 1: loop 1 dropped, its log tails kept,
  partial SRM from loop 0, no completion marker;
- reporter and analyzer: 2 loops (not 8), simulation time = slowest process per loop.

Validation on Expanse (next step): `AB_CONFIGS=csg_numa bash submit_slurm/launch_brain_expanse_shield_ab_benchmark.sh`
(two jobs, ~30 SU), then the analyzer on the `brain_ab_csg_numa_*` folders; it
recommends `NUM_CHUNKS`, `NUM_LOOPS` and `SU_PER_LOOP` for the production launcher
(provisional `SU_PER_LOOP=135` per 40-chunk loop).

## Pending (for later pickup)

The results above are sufficient to adopt the tiled CSG. More rigorous checks that
were started or planned:

1. **Full-STL production-like count run.** Done 2026-10-02 (table above). To repeat:
   `python dev/python/benchmark_brain_shield_models.py run --model {stl,stl_simpl,csg_tiles} --source sphere --with-modules --threads 64 --per-thread 2e6 --out <dir>`
   then `... analyse --out <dir>` (the `modules` section of `benchmark_summary.json`).
2. **CSG-tiles timing at larger sizes.** The shield-only slope (~2 µs) is at the
   noise floor of 2e5/6e5 primaries per thread; repeat with 2e6 and 6e6 per thread.
3. **SRM-level check.** Run one short production loop per shield model (merged
   actors, same seed) and compare the sparse SRMs per head and per voxel
   neighbourhood, not just crystal counts.
4. **Expanse A/B benchmark** at 127 threads: scripts ready
   (`launch_brain_expanse_shield_ab_benchmark.sh`, `analyze_brain_shield_ab_benchmark.py`),
   not yet run.
5. **Regenerate the comparison folders.** Done 2026-10-02 with the exact signed
   distance (delete `cache.npz` and rerun the plotting script to repeat).

## Limitations and next steps

- **Rounds in Gate.** The rounds need G4Torus (edge rounds), G4GenericPolycone or
  G4ExtrudedSolid (corner slivers), which opengate_core does not expose; G4MultiUnion
  is exposed without `AddNode`. Adding these bindings would allow the rounded model.
  A polyhedral approximation (G4Polyhedra) is possible but not tried.
- **Aperture details.** The hole-corner fillets and the lip chamfer are not modelled
  (~15 mm³ per aperture). They are the source of the +2% bare-aperture flux.
- **Production defaults** assume the CSG speed-up holds on Expanse; `SU_PER_LOOP=175`
  and the 24 h limit are provisional until the A/B benchmark (launch procedure above).
- **Confirm on Expanse.** The speed-ups were measured on the workstation; confirm with
  one Expanse benchmark before resizing campaigns.
