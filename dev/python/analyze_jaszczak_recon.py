"""Cold-insert contrast for brain SPECT Jaszczak reconstructions (recon_brain_phantom_projections.py).

Activity is in the water only, so every rod and sphere is cold (true contrast 1).
Per data set and saved iteration:
    sphere_<d>mm   1 - mean(sphere voxels) / mean(water near the sphere plane)
    rods_<d>mm     1 - mean(rod voxels of that sector) / mean(water between the rods)
Masks come from phantom_models.voxelize on the reconstruction grid; a voxel is assigned to
the nearest part centre of its role. The background is local, so a non-uniform image (e.g.
without attenuation correction) does not bias it: for a sphere, the water in a shell 4-12 mm
outside its surface, > 4 mm from every other insert; for a rod sector, the water in the rod
slab, inside the sector's footprint (within 1.5 rod pitches of a rod centre), > 2 mm from a rod.

    python analyze_jaszczak_recon.py <recon output dir> --spec SPEC.json [--plot-iteration 50]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import distance_transform_edt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "payload" / "python"))
import phantom_models as pm  # noqa: E402

ROLE = {name: k for k, (name, _) in pm.LABELS.items()}


def masks(spec: pm.Spec, n: int, voxel_mm: float) -> tuple[dict, dict]:
    parts, _ = pm.build(spec.phantom, spec.options)
    roles = pm.voxelize(spec, n, voxel_mm)
    c = pm.grid_centers(n, voxel_mm)
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    water = roles == ROLE["water"]
    insert = np.isin(roles, [ROLE["rod"], ROLE["sphere"], ROLE["stem"]])
    dist_insert = distance_transform_edt(~insert) * voxel_mm
    out, planes = {}, {}

    spheres = [p for p in parts if p.role == "sphere"]
    sphere_mask = roles == ROLE["sphere"]
    centres = np.array([spec.pose.to_world(np.asarray(p.center, float)) for p in spheres])
    z_s = float(centres[:, 2].mean())
    planes["spheres"] = z_s
    near = np.argmin([(X[sphere_mask] - x) ** 2 + (Y[sphere_mask] - y) ** 2 + (Z[sphere_mask] - z) ** 2
                      for x, y, z in centres], axis=0)
    idx = np.argwhere(sphere_mask)
    for k, p in sorted(enumerate(spheres), key=lambda kp: kp[1].radius):
        m = np.zeros_like(sphere_mask)
        sel = idx[near == k]
        m[tuple(sel.T)] = True
        x, y, z = centres[k]
        r = np.sqrt((X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2)
        others = insert & ~m & (r > p.radius + 1)
        bg = water & (r >= p.radius + 4) & (r <= p.radius + 12) & (distance_transform_edt(~others) * voxel_mm > 4)
        out[f"sphere_{2 * p.radius:.1f}mm"] = (m, bg)

    rods = [p for p in parts if p.role == "rod"]
    rod_mask = roles == ROLE["rod"]
    rod_centres = np.array([spec.pose.to_world(np.asarray(p.center, float)) for p in rods])
    z_r = float(rod_centres[:, 2].mean())
    planes["rods"] = z_r
    half = rods[0].length / 2
    slab = np.abs(Z - z_r) <= half - 4  # clear of the plates
    dist_rod = distance_transform_edt(~rod_mask) * voxel_mm
    for radius in sorted({round(p.radius, 3) for p in rods}):
        cs = rod_centres[[abs(p.radius - radius) < 1e-6 for p in rods]]
        d2 = np.min([(X - x) ** 2 + (Y - y) ** 2 for x, y, _ in cs], axis=0)
        pitch = 4 * radius  # rods are spaced by twice their diameter
        footprint = d2 <= (1.5 * pitch) ** 2
        rm = rod_mask & slab & (d2 <= radius**2 + 1e-9)
        bg_r = water & slab & footprint & (dist_rod > 2)
        out[f"rods_{2 * radius:.1f}mm"] = (rm, bg_r)
    return out, planes


def contrast(volume: np.ndarray, region: dict) -> dict:
    return {name: float(1 - volume[m].mean() / volume[b].mean()) for name, (m, b) in region.items()}


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("recon_dir", type=Path)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--plot-iteration", type=int, default=50)
    a = ap.parse_args()

    spec = pm.Spec.from_dict(json.loads(a.spec.read_text()))
    recons = {p.name.removesuffix("_reconstruction.npz"): p for p in sorted(a.recon_dir.glob("*_reconstruction.npz"))}
    with np.load(next(iter(recons.values()))) as d:
        n = int(d["volume_shape"][0])
    voxel_mm = 288.0 / n
    region, planes = masks(spec, n, voxel_mm)
    reference = pm.activity_volume(spec, n, voxel_mm)

    table, shown = {}, {"reference": reference}
    for name, path in recons.items():
        with np.load(path) as d:
            its = d["iterations"].tolist()
            est = d["estimates"]
            table[name] = {str(i): contrast(est[k].reshape(n, n, n), region) for k, i in enumerate(its)}
            shown[name] = est[its.index(a.plot_iteration)].reshape(n, n, n)
    (a.recon_dir / "jaszczak_contrast.json").write_text(json.dumps({"true": 1.0, "planes_mm": planes, "contrast": table}, indent=1) + "\n")

    # slices: rod plane, sphere plane, coronal through the phantom axis
    c = pm.grid_centers(n, voxel_mm)
    kr, ks = int(np.argmin(np.abs(c - planes["rods"]))), int(np.argmin(np.abs(c - planes["spheres"])))
    fig = Figure(figsize=(10, 3.2 * len(shown)), layout="constrained")
    axes = fig.subplots(len(shown), 3, squeeze=False)
    lim = 80
    for row, (name, v) in enumerate(shown.items()):
        sel = np.abs(c) <= lim
        panels = ((f"rods, z = {c[kr]:.0f} mm", v[:, :, kr - 1:kr + 2].mean(2)),
                  (f"spheres, z = {c[ks]:.0f} mm", v[:, :, ks - 1:ks + 2].mean(2)),
                  ("coronal, y = 0", v[:, n // 2 - 1:n // 2 + 1, :].mean(1)))
        water_level = float(np.percentile(v[region[next(iter(region))][1]], 50))
        for ax, (title, img) in zip(axes[row], panels):
            if title.startswith("coronal"):
                ax.imshow(img[sel][:, (c >= -110) & (c <= 60)].T, origin="lower", cmap="gray",
                          extent=(-lim, lim, -110, 60), vmin=0, vmax=1.3 * water_level)
                ax.set_xlabel("x (mm)", fontsize=8); ax.set_ylabel("z (mm)", fontsize=8)
            else:
                ax.imshow(img[np.ix_(sel, sel)].T, origin="lower", cmap="gray", extent=(-lim, lim, -lim, lim),
                          vmin=0, vmax=1.3 * water_level)
                ax.set_xlabel("x (mm)", fontsize=8); ax.set_ylabel("y (mm)", fontsize=8)
            ax.set_title(f"{name}: {title}", fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"{a.recon_dir.name}: {a.plot_iteration} iterations (3-slice average; grey scale 0 to 1.3 x water)", fontsize=10)
    fig.savefig(a.recon_dir / f"jaszczak_slices_iteration_{a.plot_iteration:03d}.png", dpi=140)

    keys = list(region)
    fig = Figure(figsize=(12, 3.4), layout="constrained")
    axes = fig.subplots(1, len(table))
    for ax, (name, rows) in zip(np.atleast_1d(axes), table.items()):
        its = sorted(int(i) for i in rows)
        for key in keys:
            ax.plot(its, [rows[str(i)][key] for i in its], marker="o", ms=3, label=key,
                    ls="-" if key.startswith("sphere") else "--")
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xscale("log"); ax.set_ylim(-0.1, 1.05)
        ax.set_title(name, fontsize=9); ax.set_xlabel("iterations", fontsize=8); ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel("cold contrast (true 1)", fontsize=8)
    np.atleast_1d(axes)[-1].legend(fontsize=6, ncol=2)
    fig.savefig(a.recon_dir / "jaszczak_contrast.png", dpi=140)

    it = str(a.plot_iteration)
    print(f"{'':16s}" + "".join(f"{k:>16s}" for k in keys))
    for name, rows in table.items():
        print(f"{name + ' @' + it:16s}" + "".join(f"{rows[it][k]:16.2f}" for k in keys))


if __name__ == "__main__":
    main()
