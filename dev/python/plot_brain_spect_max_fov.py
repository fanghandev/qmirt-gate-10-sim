"""Plot the outputs of brain_spect_max_fov.py.

Usage:  ma opengate && python dev/python/plot_brain_spect_max_fov.py [--input DIR]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

CURRENT_D = 210.0
SOLID_GRAY = "#8a8a8a"


def count_cmap():
    base = plt.get_cmap("Blues")
    colors = base(np.linspace(0.15, 1.0, 256))
    cmap = ListedColormap(colors)
    cmap.set_under("white")
    return cmap


def slice_panel(ax, img, solid, extent, title, rmax, cmap, norm, circles=True):
    ax.imshow(img.T, origin="lower", extent=extent, cmap=cmap, norm=norm,
              interpolation="nearest")
    ax.imshow(np.where(solid.T, 1.0, np.nan), origin="lower", extent=extent,
              cmap=ListedColormap([SOLID_GRAY]), interpolation="nearest")
    x = np.linspace(extent[0], extent[1], img.shape[0])
    y = np.linspace(extent[2], extent[3], img.shape[1])
    ax.contour(x, y, img.T, levels=[72.5], colors="#d1495b", linewidths=1.2)
    if circles:
        ax.add_patch(Circle((0, 0), CURRENT_D / 2, fill=False, ls="--", lw=1.2,
                            ec="black"))
        ax.add_patch(Circle((0, 0), rmax, fill=False, lw=1.2, ec="#e08f00"))
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path(__file__).resolve().parent / "brain_fov_analysis")
    args = ap.parse_args(argv)
    d = np.load(args.input / "brain_fov_maps.npz")
    summary = json.loads((args.input / "brain_fov_summary.json").read_text())
    x, y, z = d["x"], d["y"], d["z"]
    count, solid, sens = d["count"], d["solid"], d["sens"]
    rmax = summary["nozzle_tip_min_radius_mm"]
    cmap = count_cmap()
    norm = BoundaryNorm(np.arange(0.5, 74.5, 1), cmap.N)

    ix0, iy0 = np.argmin(abs(x)), np.argmin(abs(y))
    iz = {zz: np.argmin(abs(z - zz)) for zz in (0, -60, -130, -200)}
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), constrained_layout=True)
    ext_xz = [x[0], x[-1], z[0], z[-1]]
    ext_yz = [y[0], y[-1], z[0], z[-1]]
    ext_xy = [x[0], x[-1], y[0], y[-1]]
    slice_panel(axes[0, 0], count[:, iy0, :], solid[:, iy0, :], ext_xz,
                "x–z plane (y = 0)", rmax, cmap, norm)
    axes[0, 0].set(xlabel="x (mm)", ylabel="z (mm)")
    slice_panel(axes[0, 1], count[ix0, :, :], solid[ix0, :, :], ext_yz,
                "y–z plane (x = 0)  · face window is +y", rmax, cmap, norm)
    axes[0, 1].set(xlabel="y (mm)", ylabel="z (mm)")
    slice_panel(axes[0, 2], count[:, :, iz[0]], solid[:, :, iz[0]], ext_xy,
                "x–y plane (z = 0)", rmax, cmap, norm)
    for ax, zz in zip(axes[1], (-60, -130, -200)):
        k = iz[zz]
        slice_panel(ax, count[:, :, k], solid[:, :, k], ext_xy,
                    f"x–y plane (z = {z[k]:.0f} mm)", rmax, cmap, norm,
                    circles=(zz == -60))
        ax.set(xlabel="x (mm)", ylabel="y (mm)")
    axes[0, 2].set(xlabel="x (mm)", ylabel="y (mm)")
    for ax in axes[0, :2]:
        ax.axhline(-113.4, color=SOLID_GRAY, lw=0.8, ls=":")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, ticks=[1, 12, 24, 36, 48, 60, 73])
    cb.set_label("number of modules that see the point (white = none)")
    handles = [
        plt.Line2D([], [], color="black", ls="--", label=f"current FOV sphere D = {CURRENT_D:.0f} mm"),
        plt.Line2D([], [], color="#e08f00", label=f"max sphere without overlap D = {2 * rmax:.1f} mm"),
        plt.Line2D([], [], color="#d1495b", label="seen by all 73 modules"),
        plt.Rectangle((0, 0), 1, 1, color=SOLID_GRAY, label="lead shield / tungsten"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.45, -0.035))
    fig.suptitle("Brain SPECT detectable region: module coverage count", fontsize=13)
    fig.savefig(args.input / "brain_fov_coverage_slices.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # radial profiles inside the cavity: two panels, no shared y-scale
    prof = summary["cavity_radial_profile"]
    r = np.array([p["r_mm"] for p in prof])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    a1.plot(r, [p["mean_count"] for p in prof], color="#2c6fbb", lw=2, label="mean")
    a1.plot(r, [p["min_count"] for p in prof], color="#2c6fbb", lw=2, ls="--", label="minimum")
    a1.set(xlabel="radius from system centre (mm)", ylabel="modules seeing the point",
           ylim=(0, 76), title="Coverage vs radius (spherical shells, 5 mm)")
    s0 = prof[0]["mean_sens"]
    a2.plot(r, np.array([p["mean_sens"] for p in prof]) / s0, color="#2c6fbb", lw=2)
    a2.set(xlabel="radius from system centre (mm)", ylabel="relative to centre",
           ylim=(0, None), title="Mean geometric sensitivity vs radius")
    for ax in (a1, a2):
        ax.axvline(CURRENT_D / 2, color="black", ls="--", lw=1)
        ax.axvline(rmax, color="#e08f00", lw=1)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    a1.legend(frameon=False)
    a1.text(CURRENT_D / 2 + 2, 4, "D = 210", fontsize=8)
    a1.text(rmax - 2, 4, f"D = {2 * rmax:.0f}", fontsize=8, ha="right")
    fig.savefig(args.input / "brain_fov_radial_profile.png", dpi=130)
    plt.close(fig)
    print("wrote figures to", args.input)


if __name__ == "__main__":
    main()
