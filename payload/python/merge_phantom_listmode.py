"""Merge brain SPECT phantom list-mode files (reduce_phantom_singles.py) into one
acquisition: energy blur, energy-binned projections and window projections.

Energy resolution is applied here (the simulation stores the true deposited energy):
FWHM(E) = R * E_ref * sqrt(E / E_ref), Gaussian, R = --fwhm (default 0.10 at 140.5 keV).

Every single is sorted by where its decay was: inside the SRM FOV (a sphere of --fov-radius-mm
at the scanner centre; 144 mm = the 288 mm SRM) or outside it (e.g. neck and shoulders below
the helmet), and by whether its photon scattered in the phantom. The full acquisition is the
sum of the in-FOV and out-of-FOV sets.

Output (one .npz):
    projections_{in_fov,out_fov}_total    uint32 (heads, 625, bins)  all singles
    projections_{in_fov,out_fov}_primary  uint32 (heads, 625, bins)  no Compton/Rayleigh in the
                         phantom (phantom_scatter 0); collimator and detector scatter,
                         penetration and lead X-rays are part of the system response, as in the SRM
    energy_edges_kev     bin edges (--emin..--emax, --bin-kev)
    windows_{w}_{in_fov,out_fov}_{total,primary}  (heads, 625) counts per window w:
                         main 126.45-154.55 keV (20% at 140.5); TEW tew_lower 120.83-126.45,
                         tew_upper 154.55-160.17 (4% each)
    meta                 JSON: jobs, primaries, time windows (coverage checked), resolution, FOV
With --listmode the merged, blurred list-mode is written too (<out>_listmode.npz, with the
decay positions and an in_fov flag).

    python merge_phantom_listmode.py OUT.npz JOB_LISTMODE.npz... [--fwhm 0.10] [--seed 1]
"""

import argparse
import json
from pathlib import Path

import numpy as np

E_REF = 140.511
WINDOWS = {"main": (126.46, 154.56), "tew_lower": (120.84, 126.46), "tew_upper": (154.56, 160.18)}
HEADS, PIXELS = 73, 625


def blur(energy_kev: np.ndarray, fwhm: float, rng) -> np.ndarray:
    if fwhm <= 0:
        return energy_kev.astype(np.float32)
    sigma = fwhm * E_REF * np.sqrt(np.maximum(energy_kev, 0) / E_REF) / 2.3548
    return (energy_kev + rng.standard_normal(len(energy_kev)) * sigma).astype(np.float32)


REGIONS = ("in_fov", "out_fov")
CLASSES = ("total", "primary")


def merge(out: Path, files: list[Path], fwhm: float, seed: int, emin: float, emax: float, bin_kev: float,
          heads: int = HEADS, write_listmode: bool = False, fov_radius_mm: float = 144.0) -> dict:
    rng = np.random.default_rng(seed)
    edges = np.arange(emin, emax + bin_kev / 2, bin_kev)
    nb = len(edges) - 1
    proj = {f"projections_{r}_{c}": np.zeros((heads, PIXELS, nb), np.uint32) for r in REGIONS for c in CLASSES}
    win = {f"windows_{w}_{r}_{c}": np.zeros((heads, PIXELS), np.uint32) for w in WINDOWS for r in REGIONS for c in CLASSES}
    metas, kept = [], {k: [] for k in ("head", "pixel", "energy_kev", "time_s", "phantom_compton", "phantom_rayleigh",
                                         "phantom_scatter", "event_mm", "in_fov")}
    for f in files:
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        metas.append({k: meta[k] for k in ("job_dir", "singles", "primaries", "time_window_s", "random_seed")})
        e = blur(d["energy_kev"], fwhm, rng)
        unscattered = d["phantom_scatter"] == 0
        in_fov = np.linalg.norm(d["event_mm"], axis=1) < fov_radius_mm
        flat = d["head"].astype(np.int64) * PIXELS + d["pixel"]
        b = np.floor((e - emin) / bin_kev).astype(np.int64)
        ok = (b >= 0) & (b < nb)
        for r, rm in zip(REGIONS, (in_fov, ~in_fov)):
            for c, cm in zip(CLASSES, (np.ones_like(rm), unscattered)):
                m = rm & cm
                np.add.at(proj[f"projections_{r}_{c}"].reshape(-1), flat[m & ok] * nb + b[m & ok], 1)
                for w, (lo, hi) in WINDOWS.items():
                    inw = m & (e >= lo) & (e < hi)
                    np.add.at(win[f"windows_{w}_{r}_{c}"].reshape(-1), flat[inw], 1)
        if write_listmode:
            for k in kept:
                kept[k].append(e if k == "energy_kev" else in_fov if k == "in_fov" else d[k])
    spans = sorted(tuple(m["time_window_s"]) for m in metas)
    gaps = [(a[1], b[0]) for a, b in zip(spans, spans[1:]) if b[0] - a[1] > 1e-9]
    overlaps = [(a, b) for a, b in zip(spans, spans[1:]) if b[0] < a[1] - 1e-9]
    meta = {"schema": "qmirt.phantom_projections/1", "jobs": len(files), "primaries": sum(m["primaries"] for m in metas),
            "singles": sum(m["singles"] for m in metas), "time_covered_s": [spans[0][0], spans[-1][1]] if spans else None,
            "time_gaps_s": gaps, "time_overlaps": len(overlaps),
            "energy_resolution": {"fwhm_at_ref": fwhm, "ref_kev": E_REF, "model": "FWHM proportional to sqrt(E)", "seed": seed},
            "energy_bins_kev": [emin, emax, bin_kev], "windows_kev": WINDOWS,
            "fov": {"radius_mm": fov_radius_mm, "centre": [0.0, 0.0, 0.0], "split_by": "decay position (event_mm)"},
            "window_counts": {k: int(v.sum()) for k, v in win.items() if k.startswith("windows_main")},
            "job_list": metas}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, energy_edges_kev=edges, meta=np.array(json.dumps(meta, default=str)), **proj, **win)
    if write_listmode:
        np.savez_compressed(out.with_name(out.stem + "_listmode.npz"),
                            **{k: np.concatenate(v) for k, v in kept.items()})
    return meta


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--fwhm", type=float, default=0.10, help="FWHM / E at 140.5 keV (0: no blur)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--emin", type=float, default=10.0)
    ap.add_argument("--emax", type=float, default=200.0)
    ap.add_argument("--bin-kev", type=float, default=1.0)
    ap.add_argument("--listmode", action="store_true")
    ap.add_argument("--fov-radius-mm", type=float, default=144.0, help="SRM FOV radius (288 mm SRM: 144)")
    a = ap.parse_args()
    m = merge(a.out, a.files, a.fwhm, a.seed, a.emin, a.emax, a.bin_kev, write_listmode=a.listmode,
              fov_radius_mm=a.fov_radius_mm)
    print(f"{m['jobs']} jobs, {m['primaries']:,} primaries, {m['singles']:,} singles, time {m['time_covered_s']} s, "
          f"gaps {m['time_gaps_s']}, overlaps {m['time_overlaps']}")
    print("main window:", m["window_counts"])


if __name__ == "__main__":
    main()
