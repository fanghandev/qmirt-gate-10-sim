"""Region metrics for the brain phantom reconstructions (recon_brain_phantom_projections.py).

Regions come from the reference activity on the reconstruction grid (mesh50 XCAT head,
ratio x 100): grey matter 920, white matter 230, striatum 1012, soft-tissue background
100 (exact levels only, so partial-volume voxels are left out). Per data set and saved
iteration:
    gm_wm, str_wm, str_gm   ratios of region means (true 4.00, 4.40, 1.10)
    wm_cov                  standard deviation / mean in white matter (noise)
    nrmse_head              RMS error over the head (reference > 0) after the least-squares
                            scale of the estimate to the reference, divided by the RMS reference
    gm_inner_outer          mean grey matter within r < 40 mm of the brain centre over grey
                            matter beyond 60 mm (true 1; attenuation lowers it)

    python analyze_brain_phantom_recon.py <recon output dir> [--plot-iteration N] [--reference X.npy]

--reference replaces reference_activity.nrrd with another [x, y, z] volume on the same grid
(e.g. the activity as actually simulated); outputs then get a _<reference stem> suffix.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
import SimpleITK as sitk
from recon_brain_phantom_projections import plot_slices

LEVELS = {"gm": 920.0, "wm": 230.0, "str": 1012.0, "bg": 100.0}
TRUE = {"gm_wm": 4.0, "str_wm": 1012 / 230, "str_gm": 1012 / 920, "gm_inner_outer": 1.0}


def read_reference(path: Path) -> np.ndarray:
    """reference_activity.nrrd back to [x, y, z]."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).transpose(2, 1, 0)


def region_masks(reference: np.ndarray, voxel_mm: float) -> dict[str, np.ndarray]:
    masks = {name: np.isclose(reference, level) for name, level in LEVELS.items()}
    brain = masks["gm"] | masks["wm"] | masks["str"]
    n = reference.shape[0]
    c = (np.arange(n) - n / 2 + 0.5) * voxel_mm
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    centre = [float((axis * brain).sum() / brain.sum()) for axis in (x, y, z)]
    r = np.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2 + (z - centre[2]) ** 2)
    masks["gm_inner"] = masks["gm"] & (r < 40)
    masks["gm_outer"] = masks["gm"] & (r > 60)
    masks["head"] = reference > 0
    return masks


def metrics(estimate: np.ndarray, reference: np.ndarray, masks: dict) -> dict:
    mean = {k: float(estimate[m].mean()) for k, m in masks.items() if m.any()}
    wm = estimate[masks["wm"]]
    head = masks["head"]
    ref, est = reference[head], estimate[head]
    scale = float(est @ ref / (est @ est)) if np.any(est) else 0.0
    return {
        "gm_wm": mean["gm"] / mean["wm"],
        "str_wm": mean["str"] / mean["wm"],
        "str_gm": mean["str"] / mean["gm"],
        "wm_cov": float(wm.std() / wm.mean()),
        "nrmse_head": float(np.sqrt(np.mean((scale * est - ref) ** 2)) / np.sqrt(np.mean(ref**2))),
        "gm_inner_outer": mean["gm_inner"] / mean["gm_outer"],
    }


def plot_metrics(table: dict, title: str) -> Figure:
    keys = ["gm_wm", "str_wm", "gm_inner_outer", "wm_cov", "nrmse_head"]
    fig = Figure(figsize=(12, 3.2), layout="constrained")
    axes = fig.subplots(1, len(keys))
    for ax, key in zip(axes, keys):
        for name, rows in table.items():
            its = sorted(int(i) for i in rows)
            ax.plot(its, [rows[str(i)][key] for i in its], marker="o", ms=3, label=name)
        if key in TRUE:
            ax.axhline(TRUE[key], color="k", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("iterations", fontsize=8)
        ax.set_title(key, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle(title + " (dashed: truth)", fontsize=10)
    return fig


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("recon_dir", type=Path)
    ap.add_argument("--plot-iteration", type=int, default=None)
    ap.add_argument("--reference", type=Path, default=None)
    a = ap.parse_args()

    if a.reference is None:
        reference, suffix = read_reference(a.recon_dir / "reference_activity.nrrd"), ""
    else:
        reference, suffix = np.load(a.reference).astype(np.float32), f"_{a.reference.stem}"
    voxel_mm = 288.0 / reference.shape[0]
    masks = region_masks(reference, voxel_mm)
    table, shown = {}, {"reference": reference}
    for path in sorted(a.recon_dir.glob("*_reconstruction.npz")):
        name = path.name.removesuffix("_reconstruction.npz")
        with np.load(path) as data:
            iterations = data["iterations"].tolist()
            shape = tuple(data["volume_shape"])
            estimates = data["estimates"]
            table[name] = {str(i): metrics(estimates[k].reshape(shape), reference, masks)
                           for k, i in enumerate(iterations)}
            it = a.plot_iteration or max(iterations)
            shown[name] = estimates[iterations.index(it)].reshape(shape)

    (a.recon_dir / f"metrics{suffix}.json").write_text(json.dumps({"true": TRUE, "metrics": table}, indent=1) + "\n")
    plot_metrics(table, a.recon_dir.name).savefig(a.recon_dir / f"metrics{suffix}.png", dpi=150)
    plot_slices(shown, voxel_mm, f"{a.recon_dir.name}: {it} iterations").savefig(
        a.recon_dir / f"slices_iteration_{it:03d}{suffix}.png", dpi=150)
    for name, rows in table.items():
        best = min(rows, key=lambda i: rows[i]["nrmse_head"])
        last = rows[str(max(int(i) for i in rows))]
        print(f"{name:15s} it {max(int(i) for i in rows):3d}: " + ", ".join(f"{k} {v:.3f}" for k, v in last.items())
              + f" | lowest nrmse at it {best}")


if __name__ == "__main__":
    main()
