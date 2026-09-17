#set page(margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(size: 11pt)
#set heading(numbering: "1.")
#set par(justify: true)
#show link: underline

#align(center)[
  #text(size: 16pt, weight: "bold")[Progress Report:High-Throughput Monte Carlo Simulations and System Response
  Matrix Modeling for a Novel Brain SPECT Architecture]
]

#v(0.6em)

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: (x: 0pt, y: 3pt),
  [*ACCESS allocation:*], [MDE260019 -- ACCESS Explore, 400,000 SU award (Slurm account `mgh102`, SDSC Expanse)],
  [*PI:*], [Fang Han],
  [*Reporting period:*], [2026-08-14 (allocation start) -- 2026-09-15],
  [*Report date:*], [2026-09-15],
)

#v(0.6em)
#line(length: 100%, stroke: 0.6pt)
#v(0.6em)

= Scientific objective

This allocation supports Monte Carlo simulation of a novel, fixed-gantry
brain-dedicated SPECT scanner (DB-SPECT) originally proposed by our group
[1] as an alternative to conventional dual-head gamma cameras and adapted
multi-pinhole systems for epilepsy, dementia, Alzheimer's, Parkinson's, and
tumor imaging. The initial design concept @feng_2022_ieee_brain used 118 large-area detector
modules on a spherical gantry viewing a 210 mm-diameter field of view
through one-collimator-per-detector optics, and demonstrated (via
analytical modeling and GATE Monte Carlo) up to 18.3x the sensitivity of a
dual-head gamma camera at matched resolution, and 4.05 mm FWHM resolution
at matched sensitivity. The current campaign simulates a refined 73-head
iteration of this architecture (625 pixels/head) in GATE 10 (OpenGATE) to
produce high-statistics system response matrices (SRMs) -- the
sensitivity/detection-probability mapping between image-space voxels and
detector pixels -- at three spatial resolutions (1, 1.5, 2 mm). These SRMs
are the forward-model operator required for quantitative iterative image
reconstruction (e.g. OSEM) and for evaluating the updated design's
reconstructed-image sensitivity and resolution ahead of hardware
fabrication. Producing them to the statistical precision needed for a
$210 "mm"$-diameter uniform source over a $plus.minus 105 "mm"$
reconstruction volume requires simulating on the order of $4 times 10^13$
primary photon histories, which is only tractable with a large-scale
allocation such as this one.

= Computational approach

Each SRM voxel-pixel entry is accumulated from single-photon Monte Carlo
histories tracked through the scanner geometry (collimators, crystal stack,
shielding) with GATE 10's Geant4 backend. The workload is embarrassingly
parallel across independent primaries, and is decomposed as follows:

- The full campaign requires $4.0005 times 10^13$ simulated primaries, split
  into *8 sequential parts* of $5.000625 times 10^12$ primaries each.
- Each part is a 126-way SLURM job array on Expanse's `shared` partition,
  127 CPUs per task (whole-node), 20-hour wall-clock limit per task.
- Each array task simulates 5 independent loops of
  $7.9375 times 10^9$ primaries, converts each loop's raw single events into
  a sparse (coordinate, count) representation immediately (sparse SRM
  accumulation), then combines its own 5 loops into a per-task partial SRM.
- Per-task partial SRMs are pulled to the home workstation over Globus and
  incrementally combined into per-part, then campaign-level, sparse SRMs,
  split into one compressed CSR matrix per detector head for efficient
  downstream reconstruction I/O.

= Progress to date

As of this report, the 8-part brain-SPECT campaign
(group `brain_40t_20260910T020837Z`, started 2026-09-09) stands at:

#table(
  columns: (auto, auto),
  align: (left, right),
  stroke: 0.5pt,
  [*Metric*], [*Value*],
  [Target primaries (full campaign)], [$4.0005 times 10^13$],
  [Committed primaries], [$1.2422 times 10^13$ (31.05 %)],
  [Committed + in-flight primaries], [$1.2573 times 10^13$ (31.43 %)],
  [Array tasks complete], [313 / 1,008 (31.05 %)],
  [Simulation loops complete], [1,565 / 5,040 (31.05 %)],
)

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (center, center, right, right, left),
  stroke: 0.5pt,
  [*Part*], [*Tasks complete*], [*Committed primaries*], [*In-flight primaries*], [*Status*],
  [1], [123 / 126], [4,881,560,589,347], [31,750,070,087], [Nearly done; 3 straggler tasks],
  [2], [126 / 126], [5,000,625,642,007], [0], [Complete],
  [3], [64 / 126], [2,539,997,583,860], [119,062,730,888], [In progress],
  [4--8], [0 / 126 each], [0], [0], [Not yet submitted],
)

Part 2 is fully complete and part 1 is 98% complete, validating the
end-to-end pipeline (simulation, sparse SRM extraction, per-task combine,
Globus transfer, workstation aggregation) at production scale across
multiple independent job arrays. Part 3 is roughly half complete and
actively running. Parts 4--8 (60% of the total campaign) remain to be
submitted.

= Resource utilization

Based on the 313 completed array tasks reported above (each requesting 127
CPUs for up to 20 hours):

#table(
  columns: (auto, auto),
  align: (left, right),
  stroke: 0.5pt,
  [*Metric*], [*Value*],
  [Core-hours consumed to date (estimated)], [$approx$ 682,400 core hours],
  [Mean core-hours per completed task], [$approx$ 2,180 core hours],
  [Projected core-hours, full 40,005-primary campaign], [$approx$ 2.20 M core hours],
  [Projected core-hours remaining (695 tasks)], [$approx$ 1.52 M core hours],
  [Current ACCESS Explore award], [400,000 core hours],
)

The consumed and projected figures above are our own extrapolation from
measured per-task wall-clock time on completed tasks (127 requested CPUs
$times$ task wall-clock seconds), *not* a pull from the ACCESS/XSEDE
usage ledger; the PI should confirm actual usage-to-date against the ACCESS
allocations portal before this report is finalized. Even accounting for
estimation error, the projected total need ($approx$ 2.2 M core hours) is roughly
5.5x the 400,000-SU Explore award, so this campaign cannot be completed on
an Explore-tier allocation and requires a resource increase. These
projections will be refined as parts 4--8 proceed. Storage on Expanse is
used only as a rolling buffer (per-task outputs are under 1 GB and are
pulled to the PI's workstation and purged within days of completion), so no
persistent-storage allocation growth is anticipated.

= Technical issues identified and resolved

Two defects were found during progress review this period and corrected in
the pipeline:

+ *Lost statistics from straggler tasks.* Array tasks that reach a SLURM
  time limit mid-campaign (3 of 126 in part 1; more expected in later parts)
  previously contributed zero data to the SRM, even when several of their 5
  loops had fully completed and been written to disk -- their task-level
  combine step never ran. A repair utility now recovers these completed
  loops into a valid partial per-task SRM instead of discarding them.
+ *Missing primaries denominator in SRM metadata.* The per-task combine step
  was not recording how many primaries were behind each SRM
  (`simulated_primaries` was stuck at 0), which silently breaks the
  sensitivity calculation
  ($"sensitivity" = "accumulated_counts" / "simulated_primaries"$). The
  simulation wrapper now computes this value from each task's own
  simulation-statistics files and records it correctly; a repair script
  patches already-produced metadata without altering the underlying SRM
  data.
+ *Dashboard progress percentage.* The live monitoring dashboard's
  multi-part aggregation only counted parts that had already produced a
  progress report, so it displayed 82.8% complete instead of the correct
  31.05%. This has been corrected to count all 8 manifested parts.

None of these affected already-published or already-transferred SRM data
beyond the metadata field noted above; all fixes are additive/corrective and
have been validated against the current campaign's mounted output.

= Remaining work and timeline

- Finalize the 3 straggler tasks in part 1 and continue monitoring part 3 to
  completion.
- Submit parts 4--8 (695 remaining array tasks, $approx$ 1.52 M SU estimated),
  sequenced to stay within per-part Lustre buffer limits.
- Combine the completed parts into the campaign-level, per-head sparse SRMs
  and validate reconstructed-image sensitivity against the expected
  84--100% acceptance range established during pipeline validation.
- Evaluate GPU-accelerated combine tooling on the workstation to shorten the
  aggregation step once multiple parts are combinable simultaneously.

= Resource request

The current award (ACCESS Explore, 400,000 core hours) is insufficient to complete
this campaign: our estimate of $approx$ 682,400 core hours already consumed for 31%
of the campaign already exceeds the Explore cap, and the full
$4.0005 times 10^13$-primary brain-SPECT SRM is projected to require
$approx$ 2.2 million core hours in total ($approx$ 1.52 million core hours to complete the
remaining 69%). We therefore request an increase beyond the Explore tier
(e.g. an ACCESS Discover- or Maximize-tier award, or a supplement covering
the shortfall) sized to the $approx$ 1.52 million SU remaining need above,
subject to confirming exact consumption-to-date against the ACCESS portal.
No change to storage or other resource types is requested; compute is the
only resource type consumed by this workload.

= Conclusion

The brain-SPECT SRM production pipeline is validated at scale: two of eight
campaign parts are complete or nearly complete, the transfer/aggregation
pipeline has processed over 300 array tasks without data loss once the
straggler-recovery fix is applied, and the remaining compute need is
well-characterized from measured per-task cost. We request continued access
to the awarded allocation to complete parts 4--8 and deliver the final
campaign-level system response matrices.

#bibliography("references.bib")
