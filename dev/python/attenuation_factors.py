"""Attenuation factors for the brain SPECT SRM: A[m, j] = exp(-integral of mu) along the straight
line from reconstruction voxel j's centre to module m's pinhole centre, through the phantom.

Every photon a module detects passes its pinhole, so the path through the phantom from a voxel
is (to within the aperture's angular size) the voxel-to-pinhole segment; one factor per module
and voxel then multiplies all 625 rows of that module's SRM block (attenuated SRM = H * A).

Two engines compute the line integrals:
- grid (default, opengate env): (phantom_models.material_volume, nearest-voxel
  labels), linear attenuation at --energy-kev from the bundle's attenuation.json (total:
  photoelectric + Compton + Rayleigh, i.e. the survival of an unscattered photon).
  Line integral: midpoint rule with --samples points per ray, trilinear sampling of the mu
  grid on the GPU (torch.grid_sample). The 1 mm staircase misplaces surfaces that rays graze.
- gammart (raytracing env): OptiX traces each ray through the phantom's own surfaces
  (phantom_models.gammart_scene; solids meshed at gammart's tolerance, image phantoms as their
  label grid) and returns per-material path lengths; mu from gammart's tables (xraylib, the
  same source as attenuation.json). Rays run from the pinhole to the voxel (origins outside
  the phantom, as in check_gammart_phantom.py).
- Check: --check rays against the exact analytic path lengths (phantom_models.path_lengths).

    python attenuation_factors.py --spec SPEC.json --attenuation-json BUNDLE/attenuation.json \
        --out factors.npy [--grid 144 --voxel-mm 2] [--mu-grid 288 --mu-voxel-mm 1]
    micromamba run -n raytracing python attenuation_factors.py --engine gammart --spec SPEC.json \
        --attenuation-json BUNDLE/attenuation.json --out factors.npy
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "payload" / "python"))
import phantom_models as pm  # noqa: E402

PROVENANCE = Path("/data/fanghan/opengate_sim/data/brain_spect/brain_288mm_csg_102t_20261005T211647Z_01/geometry_provenance.json")


def mu_per_mm(table: dict, energy_kev: float) -> dict:
    E = np.log(np.asarray(table["energies_kev"], float))
    return {name: float(np.interp(np.log(energy_kev), E, np.asarray(m["mu_total_per_mm"], float)))
            for name, m in table["materials"].items()}


def material_key(name: str, mu: dict) -> str:
    """phantom_models material names (G4 names or short role names) to attenuation.json keys."""
    if name in mu:
        return name
    g4 = pm.MATERIALS.get(name)
    if g4 in mu:
        return g4
    raise KeyError(f"no attenuation for material {name!r} (have {sorted(mu)})")


def mu_grid(spec: pm.Spec, n: int, voxel_mm: float, mu: dict) -> np.ndarray:
    labels, names = pm.material_volume(spec, n, voxel_mm)
    lut = np.array([mu[material_key(nm, mu)] for nm in names], np.float32)
    return lut[labels]


def line_integrals(mu_xyz: np.ndarray, mu_voxel_mm: float, starts: torch.Tensor, end: torch.Tensor,
                   samples: int, batch: int = 200_000) -> torch.Tensor:
    """Integral of mu (1/mm) from each start (N, 3) to end (3,), mm, world = grid frame."""
    dev = starts.device
    # grid_sample wants (1, C, D, H, W) with D = z, H = y, W = x and coordinates (x, y, z) in [-1, 1]
    vol = torch.from_numpy(np.ascontiguousarray(mu_xyz.transpose(2, 1, 0)))[None, None].to(dev)
    half = mu_xyz.shape[0] * mu_voxel_mm / 2
    t = (torch.arange(samples, device=dev, dtype=torch.float32) + 0.5) / samples
    out = torch.empty(len(starts), device=dev)
    for a in range(0, len(starts), batch):
        s = starts[a:a + batch]
        d = end - s
        pts = s[:, None, :] + t[None, :, None] * d[:, None, :]  # (B, S, 3)
        g = (pts / half).reshape(1, -1, 1, 1, 3)  # align_corners=False: voxel centres at -1 + (2i+1)/n
        vals = F.grid_sample(vol, g, mode="bilinear", padding_mode="zeros", align_corners=False)
        out[a:a + batch] = vals.reshape(len(s), samples).mean(1) * d.norm(dim=1)
    return out


def gammart_tracer(spec: pm.Spec, tmp: Path):
    from gammart.ops import GammaTracer
    from gammart.scene.spec import SceneSpec

    names = list(pm.describe(spec)["materials"])
    db = tmp / "materials.db"
    db.write_text(pm.materials_db_text(names))
    cells = None
    if pm.is_image(spec.phantom):  # the grid's material labels, [x, y, z]
        cells = tmp / "cells.npy"
        np.save(cells, pm.image_data(spec)["labels"])
    return GammaTracer(SceneSpec.from_dict(pm.gammart_scene(spec, str(db), cell_material_file=str(cells) if cells else None)))


def gammart_integrals(tracer, mu_vec: torch.Tensor, ends: np.ndarray, pinhole: np.ndarray,
                      batch: int = 4_000_000) -> np.ndarray:
    """Integral of mu from the pinhole to each end point (N, 3)."""
    out = np.empty(len(ends))
    for a in range(0, len(ends), batch):
        e = ends[a:a + batch]
        d = e - pinhole
        L = np.linalg.norm(d, axis=1)
        o = np.broadcast_to(pinhole, e.shape)
        lengths = tracer.path_lengths(np.ascontiguousarray(o), d / L[:, None], tmax=L)  # (N, materials), mm
        out[a:a + batch] = (lengths.to(mu_vec) @ mu_vec).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--attenuation-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--energy-kev", type=float, default=140.511)
    ap.add_argument("--grid", type=int, default=144)
    ap.add_argument("--voxel-mm", type=float, default=2.0)
    ap.add_argument("--mu-grid", type=int, default=288)
    ap.add_argument("--mu-voxel-mm", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=600, help="points per ray (0.5 mm at the 300 mm maximum)")
    ap.add_argument("--support", type=Path, default=None,
                    help="optional sensitivity .npy; voxels at or below --min-sensitivity get factor 1")
    ap.add_argument("--min-sensitivity", type=float, default=0.0)
    ap.add_argument("--provenance", type=Path, default=PROVENANCE)
    ap.add_argument("--check", type=int, default=2000, help="rays checked against analytic path lengths")
    ap.add_argument("--engine", choices=["grid", "gammart"], default="grid")
    a = ap.parse_args()

    spec_dict = json.loads(a.spec.read_text())
    spec = pm.Spec.from_dict(spec_dict)
    mu = mu_per_mm(json.loads(a.attenuation_json.read_text()), a.energy_kev)
    print("mu (1/cm):", {k: round(10 * v, 4) for k, v in mu.items()})
    t0 = time.perf_counter()
    if a.engine == "grid":
        mu_xyz = mu_grid(spec, a.mu_grid, a.mu_voxel_mm, mu)
        print(f"mu grid {mu_xyz.shape} at {a.mu_voxel_mm} mm in {time.perf_counter() - t0:.0f} s")
    else:
        import tempfile
        tmpdir = tempfile.TemporaryDirectory()
        tracer = gammart_tracer(spec, Path(tmpdir.name))
        mats = tracer.scene.materials.names
        mu_vec = torch.as_tensor(np.asarray(tracer.scene.materials.mu_total(a.energy_kev), float), device="cuda")
        print("gammart mu (1/cm):", {m: round(10 * float(v), 4) for m, v in zip(mats, mu_vec.cpu().numpy())},
              f"scene compiled in {time.perf_counter() - t0:.0f} s")

    pinholes = np.asarray(json.loads(a.provenance.read_text())["analytic_geometry"]["modules"]["pinhole_mm"], float)
    c = pm.grid_centers(a.grid, a.voxel_mm)
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    centres = np.stack([X, Y, Z], -1).reshape(-1, 3)
    support = np.ones(len(centres), bool)
    if a.support is not None:
        support = np.load(a.support).reshape(-1) > a.min_sensitivity
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    starts = torch.from_numpy(centres[support].astype(np.float32)).to(dev)

    factors = np.ones((len(pinholes), len(centres)), np.float32)
    t0 = time.perf_counter()
    ends = centres[support]
    for m, p in enumerate(pinholes):
        if a.engine == "grid":
            li = line_integrals(mu_xyz, a.mu_voxel_mm, starts, torch.tensor(p, dtype=torch.float32, device=dev), a.samples)
            factors[m, support] = torch.exp(-li).cpu().numpy()
        else:
            factors[m, support] = np.exp(-gammart_integrals(tracer, mu_vec, ends, p))
    print(f"factors for {len(pinholes)} modules x {support.sum():,} voxels in {time.perf_counter() - t0:.0f} s")

    # exact check: analytic per-material path lengths along random (voxel, module) rays
    # (all attenuated voxels, and voxels with activity: only those matter for the reconstruction)
    rng = np.random.default_rng(1)
    active = pm.activity_volume(spec, a.grid, a.voxel_mm).reshape(-1) > 0
    check = {}
    for label, pool in (("attenuated_voxels", support & (factors.min(0) < 0.999)), ("activity_voxels", support & active)):
        pool = np.flatnonzero(pool)
        ks = rng.choice(pool, size=min(a.check, len(pool)), replace=False)
        ms = rng.integers(0, len(pinholes), size=len(ks))
        o = centres[ks]
        d = pinholes[ms] - o
        L = np.linalg.norm(d, axis=1)
        lengths = pm.path_lengths(spec, o, d / L[:, None], tmax=L)
        exact = np.exp(-sum(mu[material_key(nm, mu)] * v for nm, v in lengths.items()))
        err = factors[ms, ks] - exact
        check[label] = {"rays": int(len(ks)), "mean_abs_diff": float(np.abs(err).mean()),
                        "p99_abs_diff": float(np.percentile(np.abs(err), 99)), "max_abs_diff": float(np.abs(err).max()),
                        "mean_rel_diff": float(np.mean(err / exact)), "exact_range": [float(exact.min()), float(exact.max())]}
        print(f"check vs analytic, {label}:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in check[label].items()})

    np.save(a.out, factors)
    a.out.with_suffix(".json").write_text(json.dumps({
        "spec": spec_dict, "energy_kev": a.energy_kev, "mu_per_mm": mu, "grid": [a.grid, a.voxel_mm],
        "engine": a.engine, "mu_grid": [a.mu_grid, a.mu_voxel_mm] if a.engine == "grid" else None,
        "samples": a.samples if a.engine == "grid" else None, "provenance": str(a.provenance),
        "ray": "voxel centre to module pinhole centre", "check": check}, indent=1) + "\n")
    print(f"Wrote {a.out}: {factors.shape}, min {factors.min():.3f}")


if __name__ == "__main__":
    main()
