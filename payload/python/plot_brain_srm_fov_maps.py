#!/usr/bin/env python3
"""Plot per-crystal and per-pixel SPECT SRM maps (brain or cardiac) in the FOV.

Consumes the per-head CSR matrices (``final_srm_{resolution}_head_{NN}.npz``)
and ``combined_srm_metadata.json`` written by ``hybrid_combine_partial_srm.py``
(both the ospool/cardiac and slurm/brain paths use the same per-head layout).
Each head's matrix has shape (pixels_per_head, grid_size**3); voxel columns are
linearized as ``x * grid_size**2 + y * grid_size + z``. Per-crystal plots sum
every pixel row; per-pixel plots use one row. Both can be projected onto any of
the FOV's XY/XZ/YZ planes (summing over the remaining axis), migrated from the
cardiac ``plot_sparse_srm.ipynb`` notebook's 3-panel projection + WRL crystal-
outline overlay, generalized to also support the brain scanner's pinhole-CSV
overlay.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import trimesh
from matplotlib.axes import Axes
from qmirt.plot.wrl import parse_vrml_indexed_face_sets
from scipy.sparse import load_npz
from scipy.spatial.transform import Rotation

# (row axis, column axis, xlabel, ylabel) for each projection, matching the
# cardiac notebook's plot_sparse_3d_projections (axis=2-i for i in 0,1,2).
PROJECTION_AXES = {
    "xy": (0, 1, "X (mm)", "Y (mm)"),
    "xz": (0, 2, "X (mm)", "Z (mm)"),
    "yz": (1, 2, "Y (mm)", "Z (mm)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--srm-dir",
        type=Path,
        required=True,
        help="Directory containing final_srm_{resolution}_head_{NN}.npz and "
        "combined_srm_metadata.json (hybrid_combine_partial_srm.py output).",
    )
    parser.add_argument(
        "--resolution",
        default="1mm",
        help="Resolution label to plot, e.g. 1mm, 1p5mm, 2mm (default: 1mm).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figs/srm_fov_maps"),
        help="Directory for generated PNG files.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=None,
        help="1-indexed head IDs to plot per-crystal maps for (default: all).",
    )
    parser.add_argument(
        "--skip-crystal-plots",
        action="store_true",
        help="Skip the per-crystal (summed) SRM maps.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="1-indexed head ID to plot per-pixel maps for. Omit to skip.",
    )
    parser.add_argument(
        "--pixels",
        type=int,
        nargs="+",
        default=None,
        help="1-indexed pixel IDs within --head to plot (default: all "
        "nonzero pixels in that head).",
    )
    parser.add_argument(
        "--skip-pixel-plots",
        action="store_true",
        help="Skip the per-pixel SRM maps, even if --head is set.",
    )
    parser.add_argument(
        "--projections",
        choices=["xy", "xz", "yz", "all"],
        default="xy",
        help="FOV projection(s) to plot (default: xy). 'all' renders a 3-panel "
        "XY/XZ/YZ figure like the cardiac SRM notebook.",
    )
    parser.add_argument(
        "--fov-size-mm",
        type=float,
        nargs="+",
        default=None,
        help="Override the FOV extent per axis (1 or 3 values, mm). Defaults "
        "to grid_size * voxel_size_mm from the SRM metadata.",
    )
    parser.add_argument(
        "--plot-range-mm",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Fix the displayed axis range (both x and y) on every panel, e.g. "
        "-160 160, so plots are directly comparable across heads/pixels "
        "regardless of the underlying FOV extent (default: no clipping).",
    )
    parser.add_argument(
        "--wrl-path",
        type=Path,
        default=None,
        help="Optional VRML geometry file to overlay crystal outlines "
        "(DetectorCrystal_{id}:0 solids), as in the cardiac SRM notebook.",
    )
    parser.add_argument(
        "--crystals-per-head",
        type=int,
        default=1,
        help="Physical WRL crystals grouped per head/module for the overlay "
        "(default: 1; cardiac SPECT with 20 crystals/head module uses 20).",
    )
    parser.add_argument(
        "--geometry-csv",
        type=Path,
        default=None,
        help="Optional BrainSPECT point-cloud CSV (Pinhole_x/y/z, Crystal_x/y/z "
        "columns). When set, per-pixel plots draw the principal axis from the "
        "pixel's own center through the pinhole, extended to the FOV border, "
        "instead of the default crystal-centroid-to-origin line.",
    )
    parser.add_argument(
        "--pixel-count-xy",
        type=int,
        default=25,
        help="Pixels per side of the (square) crystal pixel grid (default: 25).",
    )
    parser.add_argument(
        "--crystal-size-mm",
        type=float,
        default=50.0,
        help="Physical crystal side length in mm, for locating pixel centers "
        "(default: 50.0).",
    )
    return parser.parse_args()


def load_srm_metadata(srm_dir: Path, resolution: str) -> dict:
    with open(srm_dir / "combined_srm_metadata.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    entry = metadata["resolutions"].get(resolution)
    if entry is None:
        raise KeyError(
            f"Resolution {resolution!r} not found in combined_srm_metadata.json "
            f"(available: {sorted(metadata['resolutions'])})"
        )
    return entry


def load_head_matrix(srm_dir: Path, resolution: str, head_id: int):
    """Load the (pixels_per_head, grid_size**3) CSR matrix for one head (1-indexed)."""
    path = srm_dir / f"final_srm_{resolution}_head_{head_id:02d}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-head SRM file: {path}")
    return load_npz(path)


def reshape_cube(voxel_counts_flat: np.ndarray, grid_size: int) -> np.ndarray:
    """Reshape a flat (grid_size**3,) voxel vector to (x, y, z)."""
    return np.asarray(voxel_counts_flat).reshape(grid_size, grid_size, grid_size)


def project_axes(cube: np.ndarray) -> dict[str, np.ndarray]:
    """Sum an (x, y, z) voxel cube onto each pair of axes (drop the third)."""
    return {"xy": cube.sum(axis=2), "xz": cube.sum(axis=1), "yz": cube.sum(axis=0)}


def _triangulate_polygons(polygons: list[list[int]]) -> np.ndarray:
    """Fan-triangulate WRL polygons into an (m, 3) face array for trimesh."""
    triangles = []
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        for i in range(1, len(polygon) - 1):
            triangles.append([polygon[0], polygon[i], polygon[i + 1]])
    return np.array(triangles)


# Brain and cardiac WRL geometry both name their per-head solids with these
# prefixes (brain also has "pixel_{n}_param" solids, which are not overlaid).
CRYSTAL_SOLID_PREFIXES = ("DetectorCrystal", "Collimator")


def _crystal_meshes(meshes: list[dict], crystal_id: int) -> list[trimesh.Trimesh]:
    """Return every solid (crystal and/or collimator) for one 1-indexed crystal ID."""
    found = []
    for prefix in CRYSTAL_SOLID_PREFIXES:
        mesh_data = next(
            (m for m in meshes if m["name"] == f"{prefix}_{crystal_id}:0"), None
        )
        if mesh_data is None:
            continue
        faces = _triangulate_polygons(mesh_data["polygons"])
        if faces.size == 0:
            continue
        found.append(
            trimesh.Trimesh(vertices=mesh_data["vertices"], faces=faces, process=True)
        )
    return found


def load_brain_geometry(geometry_csv: Path) -> pl.DataFrame:
    """Load per-head Pinhole/Crystal positions ordered to match head IDs 1..N
    (same elevation/azimuth sort convention as the BrainSPECT analysis notebooks).
    """
    geometry_df = pl.read_csv(geometry_csv)
    geometry_df = geometry_df.with_columns(
        Pinhole_y=pl.col("Pinhole_z"),
        Pinhole_z=pl.col("Pinhole_y"),
        Crystal_y=pl.col("Crystal_z"),
        Crystal_z=pl.col("Crystal_y"),
    )
    elevation = np.arctan2(
        geometry_df["Pinhole_z"],
        np.sqrt(geometry_df["Pinhole_x"] ** 2 + geometry_df["Pinhole_y"] ** 2),
    )
    azimuth = (
        np.arctan2(geometry_df["Pinhole_y"], geometry_df["Pinhole_x"]) + 2 * np.pi
    ) % (2 * np.pi)
    geometry_df = geometry_df.with_columns(
        pl.Series("elevation", elevation).round(6),
        pl.Series("azimuth", azimuth).round(6),
    ).sort(["elevation", "azimuth"])
    return geometry_df


def _head_rotation_matrix(azimuth: float, elevation: float) -> np.ndarray:
    r_base_x = Rotation.from_euler("x", -90, degrees=True)
    r_base_z = Rotation.from_euler("z", 90, degrees=True)
    r_dyn_z = Rotation.from_euler("z", azimuth, degrees=False)
    r_dyn_x = Rotation.from_euler("x", -elevation, degrees=False)
    return (r_dyn_z * r_base_z * r_dyn_x * r_base_x).as_matrix()


def pixel_and_pinhole_positions(
    geometry_df: pl.DataFrame,
    head_id: int,
    pixel_id: int,
    *,
    pixel_count_xy: int = 25,
    crystal_size_mm: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pixel_center_xyz, pinhole_xyz) in world coordinates for one pixel."""
    row = head_id - 1
    crystal_center = np.array(
        [
            geometry_df.item(row, "Crystal_x"),
            geometry_df.item(row, "Crystal_y"),
            geometry_df.item(row, "Crystal_z"),
        ],
        dtype=float,
    )
    pinhole_position = np.array(
        [
            geometry_df.item(row, "Pinhole_x"),
            geometry_df.item(row, "Pinhole_y"),
            geometry_df.item(row, "Pinhole_z"),
        ],
        dtype=float,
    )
    rotation = _head_rotation_matrix(
        geometry_df.item(row, "azimuth"), geometry_df.item(row, "elevation")
    )

    pixel_index = pixel_id - 1
    pixel_col = pixel_index // pixel_count_xy
    pixel_row = pixel_index % pixel_count_xy
    pitch = crystal_size_mm / pixel_count_xy
    local_pixel_center = np.array(
        [
            0.5 * crystal_size_mm - (pixel_col + 0.5) * pitch,
            -0.5 * crystal_size_mm + (pixel_row + 0.5) * pitch,
            0.0,
        ]
    )
    pixel_center = crystal_center + rotation @ local_pixel_center
    return pixel_center, pinhole_position


def _ray_box_segment_2d(
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
    bounds_xy: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip a ray (origin + t*direction, t>=0) to an axis-aligned box.

    Returns None if the ray never enters the box.
    """
    x_min, x_max, y_min, y_max = bounds_xy
    lowers, uppers = (x_min, y_min), (x_max, y_max)
    epsilon = 1e-12
    t_enter, t_exit = -np.inf, np.inf
    for axis in range(2):
        origin_value = float(origin_xy[axis])
        direction_value = float(direction_xy[axis])
        if abs(direction_value) < epsilon:
            if origin_value < lowers[axis] or origin_value > uppers[axis]:
                return None
            continue
        t0 = (lowers[axis] - origin_value) / direction_value
        t1 = (uppers[axis] - origin_value) / direction_value
        t_enter = max(t_enter, min(t0, t1))
        t_exit = min(t_exit, max(t0, t1))
        if t_enter > t_exit:
            return None
    if t_exit < 0:
        return None
    t_start = max(t_enter, 0.0)
    return origin_xy + t_start * direction_xy, origin_xy + t_exit * direction_xy


def overlay_crystal_wireframes(
    axs: dict[str, Axes],
    meshes: list[dict] | None,
    crystal_ids: list[int] | None,
    focus_crystal_id: int | None,
    *,
    principal_axis_3d: tuple[np.ndarray, np.ndarray] | None = None,
    plot_bounds: dict[str, tuple[float, float, float, float]] | None = None,
) -> None:
    """Draw sharp-edge wireframes + centroid labels for each crystal.

    Also draws a principal-axis line: by default from the focus crystal's
    centroid to the origin, or, when ``principal_axis_3d`` is given (start,
    end), a ray from ``start`` through ``end`` extended to the FOV border
    (e.g. a pixel center through the pinhole) instead.
    """
    focus_centroid = None
    for crystal_id in crystal_ids or []:
        assert meshes is not None
        crystal_meshes = _crystal_meshes(meshes, crystal_id)
        if not crystal_meshes:
            print(
                f"Warning: no WRL mesh for crystal {crystal_id} "
                f"(tried {', '.join(CRYSTAL_SOLID_PREFIXES)})"
            )
            continue
        for mesh in crystal_meshes:
            is_sharp = mesh.face_adjacency_angles > 0.1
            sharp_edges = mesh.face_adjacency_edges[is_sharp]
            vertices = mesh.vertices
            for view, (axis_i, axis_j, *_labels) in PROJECTION_AXES.items():
                ax = axs.get(view)
                if ax is None:
                    continue
                coords_2d = vertices[:, [axis_i, axis_j]]
                for edge in sharp_edges:
                    p1, p2 = coords_2d[edge[0]], coords_2d[edge[1]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="red", linewidth=1.0)
        # Label once per crystal, at the first available solid's centroid.
        centroid = crystal_meshes[0].centroid
        for view, (axis_i, axis_j, *_labels) in PROJECTION_AXES.items():
            ax = axs.get(view)
            if ax is None:
                continue
            ax.text(
                centroid[axis_i],
                centroid[axis_j],
                str(crystal_id),
                color="blue",
                fontsize=8,
                ha="center",
                va="center",
            )
        if crystal_id == focus_crystal_id:
            focus_centroid = centroid

    if principal_axis_3d is not None:
        start, end = principal_axis_3d
        direction = end - start
        for view, (axis_i, axis_j, *_labels) in PROJECTION_AXES.items():
            ax = axs.get(view)
            if ax is None:
                continue
            bounds = (
                plot_bounds[view] if plot_bounds else (*ax.get_xlim(), *ax.get_ylim())
            )
            segment = _ray_box_segment_2d(
                np.array([start[axis_i], start[axis_j]]),
                np.array([direction[axis_i], direction[axis_j]]),
                bounds,
            )
            if segment is None:
                continue
            p1, p2 = segment
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color="green",
                linewidth=1.5,
                linestyle="--",
            )
    elif focus_centroid is not None:
        for view, (axis_i, axis_j, *_labels) in PROJECTION_AXES.items():
            ax = axs.get(view)
            if ax is None:
                continue
            ax.plot(
                [focus_centroid[axis_i], 0.0],
                [focus_centroid[axis_j], 0.0],
                color="green",
                linewidth=1.5,
                linestyle="--",
            )


def plot_projection_maps(
    cube: np.ndarray,
    fov_size_mm: Sequence[float],
    projections: list[str],
    title: str,
    output_path: Path,
    *,
    meshes: list[dict] | None = None,
    crystal_ids: list[int] | None = None,
    focus_crystal_id: int | None = None,
    plot_range_mm: tuple[float, float] | None = None,
    principal_axis_3d: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Plot one or more FOV projections (XY/XZ/YZ) of a voxel cube."""
    maps = project_axes(cube)
    fig, ax_row = plt.subplots(
        1, len(projections), figsize=(7 * len(projections), 6), squeeze=False
    )
    axs: dict[str, Axes] = {}
    plot_bounds: dict[str, tuple[float, float, float, float]] = {}
    for view, ax in zip(projections, ax_row[0]):
        axis_i, axis_j, xlabel, ylabel = PROJECTION_AXES[view]
        extent = (
            -fov_size_mm[axis_i] / 2,
            fov_size_mm[axis_i] / 2,
            -fov_size_mm[axis_j] / 2,
            fov_size_mm[axis_j] / 2,
        )
        ax.imshow(
            maps[view].T,
            cmap="viridis",
            interpolation="none",
            extent=extent,
            origin="lower",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{view.upper()} projection")
        ax.set_aspect("equal", adjustable="box")
        if plot_range_mm is not None:
            ax.set_xlim(*plot_range_mm)
            ax.set_ylim(*plot_range_mm)
            extent = (*plot_range_mm, *plot_range_mm)
        axs[view] = ax
        plot_bounds[view] = extent

    fig.colorbar(
        ax_row[0][0].images[0],
        ax=list(ax_row[0]),
        orientation="vertical",
        fraction=0.02,
        pad=0.04,
    )
    fig.suptitle(title)

    if (
        meshes is not None and crystal_ids is not None
    ) or principal_axis_3d is not None:
        overlay_crystal_wireframes(
            axs,
            meshes,
            crystal_ids,
            focus_crystal_id,
            principal_axis_3d=principal_axis_3d,
            plot_bounds=plot_bounds,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    entry = load_srm_metadata(args.srm_dir, args.resolution)
    grid_size = int(entry["grid_size"])
    voxel_size_mm = float(entry["voxel_size_mm"])
    num_heads = int(entry["num_heads"])
    pixels_per_head = int(entry["pixels_per_head"])

    fov_size_mm = args.fov_size_mm or [grid_size * voxel_size_mm] * 3
    if len(fov_size_mm) == 1:
        fov_size_mm = fov_size_mm * 3
    if len(fov_size_mm) != 3:
        raise ValueError("--fov-size-mm needs 1 or 3 values")

    projections = (
        ["xy", "xz", "yz"] if args.projections == "all" else [args.projections]
    )

    meshes = parse_vrml_indexed_face_sets(args.wrl_path) if args.wrl_path else None
    geometry_df = load_brain_geometry(args.geometry_csv) if args.geometry_csv else None

    def crystal_ids_for_head(head_id: int) -> list[int]:
        row = (head_id - 1) // args.crystals_per_head
        return list(
            range(
                row * args.crystals_per_head + 1, (row + 1) * args.crystals_per_head + 1
            )
        )

    if not args.skip_crystal_plots:
        crystal_dir = args.output_dir / "per_crystal"
        heads = args.heads or list(range(1, num_heads + 1))
        for head_id in heads:
            matrix = load_head_matrix(args.srm_dir, args.resolution, head_id)
            total_counts = np.asarray(matrix.sum(axis=0)).ravel()
            cube = reshape_cube(total_counts, grid_size)
            plot_projection_maps(
                cube,
                fov_size_mm,
                projections,
                f"Head {head_id}: crystal-summed SRM sensitivity",
                crystal_dir / f"srm_head_{head_id:02d}.png",
                meshes=meshes,
                crystal_ids=crystal_ids_for_head(head_id) if meshes else None,
                focus_crystal_id=head_id if meshes else None,
                plot_range_mm=args.plot_range_mm,
            )
        print(f"Saved {len(heads)} per-crystal SRM maps to {crystal_dir}")

    if args.head is not None and not args.skip_pixel_plots:
        pixel_dir = args.output_dir / "per_pixel" / f"head_{args.head:02d}"
        matrix = load_head_matrix(args.srm_dir, args.resolution, args.head).tocsr()
        pixel_ids = args.pixels or [
            pixel_id + 1
            for pixel_id in range(pixels_per_head)
            if matrix.getrow(pixel_id).nnz > 0
        ]
        head_crystal_ids = crystal_ids_for_head(args.head) if meshes else None
        saved = 0
        for pixel_id in pixel_ids:
            row = matrix.getrow(pixel_id - 1)
            if row.nnz == 0:
                continue
            cube = reshape_cube(np.asarray(row.todense()).ravel(), grid_size)
            principal_axis_3d = (
                pixel_and_pinhole_positions(
                    geometry_df,
                    args.head,
                    pixel_id,
                    pixel_count_xy=args.pixel_count_xy,
                    crystal_size_mm=args.crystal_size_mm,
                )
                if geometry_df is not None
                else None
            )
            plot_projection_maps(
                cube,
                fov_size_mm,
                projections,
                f"Head {args.head}, Pixel {pixel_id}: SRM backprojection",
                pixel_dir / f"srm_pixel_{pixel_id:03d}.png",
                meshes=meshes,
                crystal_ids=head_crystal_ids,
                focus_crystal_id=args.head if meshes else None,
                plot_range_mm=args.plot_range_mm,
                principal_axis_3d=principal_axis_3d,
            )
            saved += 1
        print(f"Saved {saved} per-pixel SRM maps to {pixel_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
