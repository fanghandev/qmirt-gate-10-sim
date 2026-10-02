"""Gate 10 benchmark: brain SPECT lead shield as STL vs CSG.

Scene: world (air), the shield only (no collimators, no module crystals), and a CsI
spherical shell detector (r = 160-170 mm, elevation >= -50 deg) outside it. Photons
of 140 keV start either at the centre ("point") or uniformly in the 210 mm FOV
sphere ("sphere"). A phase-space actor records every photon entering the detector
(position, energy). Shield models:

  stl         BrainFrame.008.Lead_Shield.STL (186,772 facets), as in the brain sim
  stl_simpl   the 0.05 mm simplified STL (43,258 facets) used by the production run
  csg_tiles   analytic CSG, 73 G4Sphere-segment tiles each minus one aperture
  csg_single  analytic CSG, one G4Sphere minus window and 73 apertures (boolean chain)
  none        no shield (lower bound for the cost)

Usage (one configuration per call; then analyse):
  ma opengate && python dev/python/benchmark_brain_shield_models.py run \\
      --model csg_tiles --source point --threads 64 --per-thread 5e5 --out <dir>
  python dev/python/benchmark_brain_shield_models.py analyse --out <dir>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "payload" / "python"))

MODELS = ("stl", "stl_simpl", "csg_tiles", "csg_single", "none")
SIMPLIFIED_DIR = REPO / "persistent_data/brain_spect/stl/BrainFrame.008.Lead_Shield.simplified_0.05mm"
DETECTOR = {"r_inner_mm": 160.0, "r_outer_mm": 170.0, "min_elevation_deg": -50.0}


def run(args):
    import opengate as gate
    import qmirt

    import brain_spect_shield_csg as csg
    import gate_sim_brain_spect_boolean as brain

    mm, deg, keV, Bq, sec = (gate.g4_units.mm, gate.g4_units.deg, gate.g4_units.keV,
                             gate.g4_units.Bq, gate.g4_units.s)
    tag = (f"{args.model}_{args.source}{'_modules' if args.with_modules else ''}"
           f"_t{args.threads}_n{args.per_thread:g}_s{args.seed}")
    out = Path(args.out).resolve() / tag
    out.mkdir(parents=True, exist_ok=True)
    sim = gate.Simulation(output_dir=out)
    sim.random_seed = args.seed
    sim.number_of_threads = args.threads
    sim.check_volumes_overlap = args.check_overlaps
    sim.volume_manager.add_material_database(
        qmirt.utils.filesystem.search_dir_up("persistent_data", brain.__file__) / "GateMaterials.db")
    sim.world.size = [500 * mm] * 3
    sim.world.material = "Air"

    df = brain.get_geometry_definitions()
    if args.model == "stl":
        brain.add_shielding_to_gate_sim(sim, brain.get_geometry_base_definition(0))
    elif args.model == "stl_simpl":
        brain.add_shield_pieces_to_gate_sim(sim, SIMPLIFIED_DIR)
    elif args.model in ("csg_tiles", "csg_single"):
        csg.add_csg_shield_to_gate_sim(sim, df, layout=args.model.split("_")[1])

    if args.with_modules:
        # production-like: the 73 collimators and crystal boxes (no pixel grid); score
        # photons entering the crystals
        n = df.shape[0]
        crystals = []
        for k in range(n):
            cfg = brain.get_geometry_base_definition(brain.map_crystal_id(k, n, "sequential"))
            brain.add_collimator_to_gate_sim(sim, cfg, df, k)
            crystal = brain.add_crystal_box(sim, f"DetectorCrystal_{k + 1}")
            crystal.size = cfg["crystal definition"]["size_mm"]
            crystal.translation = [df.item(k, f"Crystal_{a}") for a in "xyz"]
            crystal.rotation = brain.get_head_rotation_matrix(df, k)
            crystal.material = "CsI"
            crystals.append(crystal.name)
        det_names = crystals
    else:
        det = sim.add_volume("Sphere", name="Detector")
        det.rmin, det.rmax = DETECTOR["r_inner_mm"] * mm, DETECTOR["r_outer_mm"] * mm
        det.stheta, det.dtheta = 0, (90 - DETECTOR["min_elevation_deg"]) * deg
        det.material = "CsI"
        det_names = det.name

    src = sim.add_source("GenericSource", "gamma")
    src.particle = "gamma"
    src.energy.type = "mono"
    src.energy.mono = 140 * keV
    src.direction.type = "iso"
    if args.source == "sphere":
        src.position.type = "sphere"
        src.position.radius = 105 * mm
    else:
        src.position.type = "point"
    src.activity = args.per_thread * Bq  # per thread, over a 1 s run
    sim.run_timing_intervals = [[0, 1 * sec]]

    if not args.no_phsp:
        phsp = sim.add_actor("PhaseSpaceActor", "Entering")
        phsp.attached_to = det_names
        phsp.steps_to_store = "entering"
        phsp.attributes = ["PrePosition", "KineticEnergy", "PreStepUniqueVolumeID"]
        phsp.output_filename = out / "entering.root"
    stats = sim.add_actor("SimulationStatisticsActor", "Stats")
    stats.output_filename = out / "stats.txt"

    t0 = time.time()
    sim.run(start_new_process=False)
    wall = time.time() - t0
    s = sim.get_actor("Stats")
    result = {"model": args.model, "source": args.source, "with_modules": args.with_modules,
              "threads": args.threads,
              "per_thread": args.per_thread, "seed": args.seed, "events": int(s.counts.events),
              "steps": int(s.counts.steps), "run_s": float(s.counts.duration) / sec, "wall_s": wall}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result))


def _load_entering(path):
    import uproot

    with uproot.open(path) as f:
        tree = f[[k for k in f.keys() if k.split(";")[0] == "Entering"][0]]
        d = tree.arrays(["PrePosition_X", "PrePosition_Y", "PrePosition_Z", "KineticEnergy"],
                        library="np")
    p = np.c_[d["PrePosition_X"], d["PrePosition_Y"], d["PrePosition_Z"]]
    r = np.linalg.norm(p, axis=1)
    return (np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360, np.degrees(np.arcsin(p[:, 2] / r)),
            d["KineticEnergy"] * 1e3)  # keV


def analyse(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    results = [json.loads(p.read_text()) for p in sorted(out.glob("*/result.json"))]
    summary = {"timing": [], "physics": {}, "modules": analyse_modules(out, results, args.map_source)}
    results_all, results = results, [r for r in results if not r.get("with_modules")]

    # timing: slope between two sizes of the same model/source/threads
    groups = {}
    for r in results:
        groups.setdefault((r["model"], r["source"], r["threads"]), []).append(r)
    for (model, source, threads), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["events"])
        if len(rs) >= 2 and rs[-1]["events"] > rs[0]["events"]:
            slope = (rs[-1]["run_s"] - rs[0]["run_s"]) / (rs[-1]["events"] - rs[0]["events"])
            summary["timing"].append({"model": model, "source": source, "threads": threads,
                                      "us_per_primary_per_thread": slope * threads * 1e6,
                                      "steps_per_primary": rs[-1]["steps"] / rs[-1]["events"]})

    # physics: detector entry maps for the largest point-source run of each model
    az_bins, el_bins = np.arange(0, 360.5, 0.5), np.arange(DETECTOR["min_elevation_deg"], 90.5, 0.5)
    maps = {}
    for model in MODELS:
        rs = [r for r in results if r["model"] == model and r["source"] == args.map_source]
        rs = [r for r in rs if (out / _tag(r) / "entering.root").exists()]
        if not rs:
            continue
        r = max(rs, key=lambda r: r["events"])
        az, el, e = _load_entering(out / _tag(r) / "entering.root")
        prim = e > 139.9
        maps[model] = {
            "events": r["events"],
            "primary": np.histogram2d(az[prim], el[prim], [az_bins, el_bins])[0],
            "scattered": np.histogram2d(az[~prim], el[~prim], [az_bins, el_bins])[0],
            "energy": np.histogram(e, np.arange(0, 145, 1.0))[0],
        }
        summary["physics"][model] = {
            "events": r["events"],
            "entering_per_primary": float(len(e) / r["events"]),
            "unscattered_per_primary": float(prim.sum() / r["events"]),
            "scattered_per_primary": float((~prim).sum() / r["events"]),
        }
    ref = "stl"
    if ref in maps:
        for model, m in maps.items():
            if model == ref:
                continue
            for kind in ("primary", "scattered"):
                a = maps[ref][kind] / maps[ref]["events"]
                b = m[kind] / m["events"]
                var = maps[ref][kind] / maps[ref]["events"] ** 2 + m[kind] / m["events"] ** 2
                ok = var > 0
                z = (b - a)[ok] / np.sqrt(var[ok])
                summary["physics"][model][f"{kind}_map_vs_stl"] = {
                    "bins_compared": int(ok.sum()),
                    "chi2_per_bin": float(np.mean(z ** 2)),
                    "bins_beyond_5_sigma": int(np.sum(np.abs(z) > 5)),
                    "relative_total_difference": float(b.sum() / a.sum() - 1),
                }
        _plot_maps(maps, az_bins, el_bins, out, plt)
    (out / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def _tag(r):
    return (f"{r['model']}_{r['source']}{'_modules' if r.get('with_modules') else ''}"
            f"_t{r['threads']}_n{r['per_thread']:g}_s{r['seed']}")


def _per_crystal(path, n_heads=73):
    import re

    import uproot

    with uproot.open(path) as f:
        tree = f[[k for k in f.keys() if k.split(";")[0] == "Entering"][0]]
        d = tree.arrays(["KineticEnergy", "PreStepUniqueVolumeID"], library="np")
    vid = d["PreStepUniqueVolumeID"].astype(str)
    head = np.array([int(re.match(r"DetectorCrystal_(\d+)", v).group(1)) for v in vid]) - 1
    e = d["KineticEnergy"] * 1e3
    window = (e >= 126) & (e <= 154)
    return (np.bincount(head[e > 139.9], minlength=n_heads),
            np.bincount(head[window], minlength=n_heads), np.bincount(head, minlength=n_heads))


def analyse_modules(out, results, source):
    rows = {r["model"]: r for r in results if r.get("with_modules") and r["source"] == source}
    if "stl" not in rows:
        return {}
    ref = _per_crystal(out / _tag(rows["stl"]) / "entering.root")
    n_ref = rows["stl"]["events"]
    summary = {}
    for model, r in rows.items():
        cur = _per_crystal(out / _tag(r) / "entering.root")
        entry = {"events": r["events"]}
        for name, a, b in zip(("unscattered", "photopeak_window_126_154keV", "all"), ref, cur):
            ra, rb = a / n_ref, b / r["events"]
            rel = np.where(a > 0, rb / np.where(ra > 0, ra, 1) - 1, 0)
            sig = np.where(a + b > 0, np.abs(b - a * r["events"] / n_ref) / np.sqrt(np.maximum(a + b, 1)), 0)
            entry[name] = {"per_primary": float(b.sum() / r["events"]),
                           "total_vs_stl": float(rb.sum() / ra.sum() - 1),
                           "per_head_max_abs_relative": float(np.abs(rel).max()),
                           "heads_beyond_3_sigma_if_independent": int(np.sum(sig > 3))}
        summary[model] = entry
    return summary


def _plot_maps(maps, az_bins, el_bins, out, plt):
    ext = [az_bins[0], az_bins[-1], el_bins[0], el_bins[-1]]
    others = [m for m in maps if m != "stl"]
    fig, axes = plt.subplots(1 + len(others), 2, figsize=(17, 4.2 * (1 + len(others))),
                             constrained_layout=True, squeeze=False)
    ref = maps["stl"]
    for col, kind in enumerate(("primary", "scattered")):
        im = axes[0, col].imshow((ref[kind] / ref["events"]).T, origin="lower", extent=ext,
                                 aspect="auto", cmap="Greys")
        axes[0, col].set_title(f"STL: {'unscattered' if kind == 'primary' else 'scattered'} "
                               "photons entering the detector, per primary")
        fig.colorbar(im, ax=axes[0, col])
        for row, model in enumerate(others, start=1):
            m = maps[model]
            a, b = ref[kind] / ref["events"], m[kind] / m["events"]
            var = ref[kind] / ref["events"] ** 2 + m[kind] / m["events"] ** 2
            z = np.where(var > 0, (b - a) / np.sqrt(np.where(var > 0, var, 1)), 0.0)
            im = axes[row, col].imshow(z.T, origin="lower", extent=ext, aspect="auto",
                                       cmap="RdBu_r", vmin=-5, vmax=5)
            axes[row, col].set_title(f"{model} − STL, in standard deviations (red: more counts)")
            fig.colorbar(im, ax=axes[row, col])
    for ax in axes.ravel():
        ax.set_xlabel("azimuth (deg)")
        ax.set_ylabel("elevation (deg)")
    fig.savefig(out / "benchmark_maps.png", dpi=100)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", choices=MODELS, required=True)
    r.add_argument("--source", choices=("point", "sphere"), default="point")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--per-thread", type=float, default=1e5, help="primaries per thread")
    r.add_argument("--seed", type=int, default=12345)
    r.add_argument("--no-phsp", action="store_true", help="timing only, no phase space")
    r.add_argument("--check-overlaps", action="store_true")
    r.add_argument("--with-modules", action="store_true",
                   help="add the 73 collimators and crystal boxes and score entering photons "
                        "per crystal instead of the outer shell detector")
    r.add_argument("--out", required=True)
    a = sub.add_parser("analyse")
    a.add_argument("--out", required=True)
    a.add_argument("--map-source", default="point")
    args = ap.parse_args(argv)
    run(args) if args.cmd == "run" else analyse(args)


if __name__ == "__main__":
    main()
