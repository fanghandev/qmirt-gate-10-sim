"""Compare Jaszczak reconstructions at several SRM resolutions (2, 1.5, 1 mm voxels).

For each reconstruction directory (analyze_jaszczak_recon.py must have written
jaszczak_contrast.json there) and data set: per saved iteration, the cold contrast of every
insert and the noise in the uniform section (z -80 to -60 mm, r < 50 mm) as the coefficient of
variation, per voxel and after averaging into 6 mm blocks (a common multiple of the voxel sizes,
so the noise is compared at the same scale). Writes a contrast-vs-noise figure, the rod and
sphere planes side by side at --plot-iteration, and a JSON summary.

    python compare_jaszczak_resolutions.py DIR_2MM DIR_1P5MM DIR_1MM --spec SPEC.json --out OUT_DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

KEYS = ["sphere_9.5mm", "sphere_12.7mm", "sphere_25.4mm", "rods_6.4mm", "rods_7.9mm", "rods_9.5mm"]
SLAB = (-80.0, -60.0, 50.0)  # z range and radius (mm) of the uniform section
BLOCK_MM = 6.0


def noise(volume: np.ndarray, voxel_mm: float) -> tuple[float, float]:
    n = volume.shape[0]
    c = (np.arange(n) - n / 2 + 0.5) * voxel_mm
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    sel = (Z > SLAB[0]) & (Z < SLAB[1]) & (np.hypot(X, Y) < SLAB[2])
    v = volume[sel]
    k = round(BLOCK_MM / voxel_mm)
    m = n // k
    blocks = volume[:m * k, :m * k, :m * k].reshape(m, k, m, k, m, k).mean(axis=(1, 3, 5))
    cb = (np.arange(m) - m / 2 + 0.5) * BLOCK_MM
    Xb, Yb, Zb = np.meshgrid(cb, cb, cb, indexing="ij")
    selb = (Zb - BLOCK_MM / 2 > SLAB[0]) & (Zb + BLOCK_MM / 2 < SLAB[1]) & (np.hypot(Xb, Yb) + BLOCK_MM < SLAB[2])
    b = blocks[selb]
    return float(v.std() / v.mean()), float(b.std() / b.mean())


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["in_fov_primary", "full_tew"])
    ap.add_argument("--plot-iteration", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    summary, planes, shown = {}, None, {}
    for d in a.dirs:
        info = json.loads((d / "jaszczak_contrast.json").read_text())
        planes = info["planes_mm"]
        for ds in a.datasets:
            with np.load(d / f"{ds}_reconstruction.npz") as z:
                n = int(z["volume_shape"][0])
                voxel_mm = 288.0 / n
                label = f"{voxel_mm:g} mm"
                rows = {}
                for k, it in enumerate(z["iterations"].tolist()):
                    vol = z["estimates"][k].reshape(n, n, n)
                    cov, cov6 = noise(vol, voxel_mm)
                    rows[it] = {"cov_voxel": cov, f"cov_{BLOCK_MM:g}mm": cov6,
                                **{key: info["contrast"][ds][str(it)][key] for key in KEYS}}
                    if it == a.plot_iteration and ds == a.datasets[0]:
                        shown[label] = (vol, voxel_mm)
                summary.setdefault(ds, {})[label] = rows
    (a.out / "resolution_comparison.json").write_text(json.dumps(summary, indent=1) + "\n")

    # contrast vs 6 mm-block noise, one panel per insert, curves over iterations per resolution
    for ds, by_res in summary.items():
        fig = Figure(figsize=(13, 6.5), layout="constrained")
        axes = fig.subplots(2, 3).ravel()
        for ax, key in zip(axes, KEYS):
            for label, rows in by_res.items():
                its = sorted(rows)
                ax.plot([rows[i][f"cov_{BLOCK_MM:g}mm"] for i in its], [rows[i][key] for i in its], marker="o", ms=3, label=label)
                ax.annotate(str(its[-1]), (rows[its[-1]][f"cov_{BLOCK_MM:g}mm"], rows[its[-1]][key]), fontsize=6)
            ax.set_title(key, fontsize=9)
            ax.set_xlabel(f"noise: CoV of {BLOCK_MM:g} mm blocks", fontsize=8)
            ax.set_ylabel("cold contrast (true 1)", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Small Jaszczak, {ds}, attenuation corrected: contrast vs noise over iterations (last iteration labelled)", fontsize=10)
        fig.savefig(a.out / f"contrast_vs_noise_{ds}.png", dpi=140)

    # rod and sphere planes at --plot-iteration, one row per resolution (3-slice / 6 mm slab average)
    fig = Figure(figsize=(7, 3.4 * len(shown)), layout="constrained")
    axes = fig.subplots(len(shown), 2, squeeze=False)
    lim = 75
    for row, (label, (vol, voxel_mm)) in enumerate(shown.items()):
        n = vol.shape[0]
        c = (np.arange(n) - n / 2 + 0.5) * voxel_mm
        sel = np.abs(c) <= lim
        water = None
        for col, (name, z0) in enumerate((("rods", planes["rods"]), ("spheres", planes["spheres"]))):
            k = np.abs(c - z0) <= 3.0
            img = vol[:, :, k].mean(2)[np.ix_(sel, sel)]
            if water is None:
                water = float(np.median(img[img > 0.2 * img.max()]))
            axes[row, col].imshow(img.T, origin="lower", cmap="gray", extent=(-lim, lim, -lim, lim), vmin=0, vmax=1.3 * water)
            axes[row, col].set_title(f"{label} voxels: {name}, z = {z0:.0f} mm", fontsize=9)
            axes[row, col].tick_params(labelsize=7)
    fig.suptitle(f"{a.datasets[0]}, {a.plot_iteration} iterations, attenuation corrected (6 mm slab average)", fontsize=10)
    fig.savefig(a.out / f"planes_iteration_{a.plot_iteration:03d}.png", dpi=140)

    for ds, by_res in summary.items():
        print(ds)
        print(f"  {'res':7s} {'it':>4s} {'CoV vox':>8s} {'CoV 6mm':>8s} " + " ".join(f"{k.replace('mm', ''):>12s}" for k in KEYS))
        for label, rows in by_res.items():
            for it in sorted(rows):
                r = rows[it]
                print(f"  {label:7s} {it:4d} {r['cov_voxel']:8.3f} {r[f'cov_{BLOCK_MM:g}mm']:8.3f} " + " ".join(f"{r[k]:12.2f}" for k in KEYS))


if __name__ == "__main__":
    main()
