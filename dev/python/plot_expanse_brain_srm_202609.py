import argparse
import hashlib
import json
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, Iterator, Literal

import numpy as np
import trimesh
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from scipy import sparse
from tqdm.auto import tqdm

# (horizontal, vertical) volume axes for the XY, XZ and YZ projection panels
PROJECTION_AXES = ((0, 1), (0, 2), (1, 2))
AXIS_NAMES = "XYZ"


class SRMLoader:
    def __init__(
        self,
        subdir: str = "batch_20260909_012610",
        modality: Literal["cardiac_spect", "brain_spect"] = "cardiac_spect",
        resolution: Literal["1mm", "1p5mm", "2mm"] = "1mm",
    ):
        self.subdir = subdir
        self.modality = modality
        basedir = Path("/data/fanghan/opengate_sim/data")
        self.datadir = basedir / self.modality / self.subdir
        self.resolution = resolution

    def get_module_srm_filenames(self) -> dict[int, str]:
        pattern = re.compile(
            rf"final_srm_{re.escape(self.resolution)}_head_(\d+)\.npz$"
        )

        srm_dict = {}
        for entry in os.scandir(self.datadir):
            match = pattern.search(entry.name)
            if match:
                head_id = int(match.group(1)) - 1
                srm_dict[head_id] = entry.name

        # Sort the dictionary by key (head_id) before returning
        return dict(sorted(srm_dict.items()))

    @cached_property
    def srm_filenames(self) -> dict[int, str]:
        """Directory scan done once; get_module_srm_filenames() always rescans."""
        return self.get_module_srm_filenames()

    def load_single_srm(self, module_index: int) -> sparse.csr_matrix:
        fnames = self.srm_filenames
        if module_index not in fnames:
            raise IndexError(f"Module index out of range: {module_index}")
        return sparse.load_npz(self.datadir / fnames[module_index])

    def iter_srms_parallel(
        self,
        module_indices: Iterable[int] | None = None,
        max_workers: int | None = 16,
    ) -> Iterator[tuple[int, sparse.csr_matrix]]:
        """Load SRMs in a thread pool and yield (module_index, srm) as each
        finishes (completion order, not index order). Failed loads are warned
        about and skipped. No progress bar here; wrap the caller's loop in tqdm."""

        if module_indices is None:
            module_indices = self.srm_filenames.keys()

        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = {
                executor.submit(self.load_single_srm, mod_idx): mod_idx
                for mod_idx in module_indices
            }

            for future in as_completed(futures):
                mod_idx = futures[future]

                try:
                    srm = future.result()
                except Exception as e:
                    print(f"\n[Warning] Failed to load SRM for module {mod_idx}: {e}")
                    continue
                yield mod_idx, srm
        finally:
            # Runs on normal exhaustion and when the caller breaks out early;
            # drop queued loads instead of waiting for them.
            executor.shutdown(wait=True, cancel_futures=True)

    def load_srms_parallel(
        self,
        module_indices: Iterable[int] | None = None,
        max_workers: int | None = 16,
    ) -> dict[int, sparse.csr_matrix]:

        if module_indices is None:
            module_indices = self.srm_filenames.keys()

        module_indices = list(module_indices)
        srm_iter = tqdm(
            self.iter_srms_parallel(module_indices, max_workers=max_workers),
            total=len(module_indices),
            desc="Loading SRMs",
        )
        # Results arrive in finish-order; sort by key so the dict is sequential.
        return dict(sorted(srm_iter))


# The 12 edges of a box/trd, as index pairs into primitive_corners' 8 corners
# (0-3: the -z face, 4-7: the +z face, each in ring order).
PRIMITIVE_EDGES = np.array(
    [[i, (i + 1) % 4] for i in range(4)]
    + [[4 + i, 4 + (i + 1) % 4] for i in range(4)]
    + [[i, 4 + i] for i in range(4)]
)


def primitive_corners(primitive: dict) -> np.ndarray:
    """(8, 3) local-frame corners of a Geant4 box or trd primitive."""
    if primitive["type"] == "box":
        half_x, half_y, half_z = np.asarray(primitive["size_mm"], dtype=float) * 0.5
        bottom = top = (half_x, half_y)
    elif primitive["type"] == "trd":
        half_z = primitive["dz"]
        bottom = (primitive["dx1"], primitive["dy1"])
        top = (primitive["dx2"], primitive["dy2"])
    else:
        raise ValueError(f"Unsupported primitive type: {primitive['type']!r}")
    ring = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    corners = np.vstack(
        [
            np.column_stack([ring * bottom, np.full(4, -half_z)]),
            np.column_stack([ring * top, np.full(4, half_z)]),
        ]
    )
    return corners + np.asarray(primitive["offset_mm"], dtype=float)


def placed_part_segments(
    primitives: list[dict], center: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    """(N, 2, 3) world-frame outline segments of one placed part."""
    return np.concatenate(
        [
            (center + primitive_corners(primitive) @ rotation.T)[PRIMITIVE_EDGES]
            for primitive in primitives
        ]
    )


def projected_silhouettes(
    points: np.ndarray, pixel_mm: float = 1.0, closing_iterations: int = 3
) -> list[list[np.ndarray]]:
    """Outer outline of a surface's shadow on the XY/XZ/YZ planes, from dense
    surface samples: rasterize, close sampling gaps, fill holes, then trace the
    boundary. Returns per panel a list of (N, 2) polylines."""
    from contourpy import contour_generator
    from scipy import ndimage

    extent = np.abs(points).max() + 2 * closing_iterations * pixel_mm
    edges = np.arange(-extent, extent + pixel_mm, pixel_mm)
    pixel_centers = 0.5 * (edges[:-1] + edges[1:])
    silhouettes = []
    for h, v in PROJECTION_AXES:
        occupied, _, _ = np.histogram2d(points[:, h], points[:, v], bins=(edges, edges))
        closed = ndimage.binary_closing(occupied > 0, iterations=closing_iterations)
        mask = np.asarray(ndimage.binary_fill_holes(closed), dtype=float)
        # contourpy takes z indexed [y, x], i.e. [v, h]
        lines = contour_generator(pixel_centers, pixel_centers, mask.T).lines(0.5)
        silhouettes.append([np.asarray(line) for line in lines])
    return silhouettes


def add_outlines(axs, edges: np.ndarray, **line_kwargs):
    """Draw (N, 2, 3) 3D segments on the XY/XZ/YZ panels, one collection per panel."""
    for ax, (h, v) in zip(axs, PROJECTION_AXES):
        ax.add_collection(LineCollection(list(edges[:, :, [h, v]]), **line_kwargs))


def nice_axis_limit(
    max_abs_mm: float, margin: float = 0.05, step_mm: float = 10.0
) -> float:
    """Symmetric axis limit covering max_abs_mm plus a margin, rounded up to step_mm."""
    return float(np.ceil(max_abs_mm * (1 + margin) / step_mm) * step_mm)


def compute_projections(srm, grid_size: int) -> list[np.ndarray]:
    """Sum the SRM over detector pixels into a (x, y, z) volume, then project
    it along z, y and x to get the XY, XZ and YZ images (indexed [h, v])."""
    volume = np.asarray(srm.sum(axis=0)).reshape((grid_size,) * 3)
    return [volume.sum(axis=2), volume.sum(axis=1), volume.sum(axis=0)]


def plot_sparse_3d_projections(
    srm_crystal_sparse,
    grid_size,
    label_id=1,
    *,
    prefix: str = "Crystal",
    fov_size_mm=(100, 100, 100),
    axis_limit_mm: float | None = None,
):
    """axis_limit_mm: symmetric +/- limit for every panel; None fits the FOV.
    Each panel gets its own color scale and color bar: the projections sum along
    different axes, so a shared scale would only wash out the lower-peaked ones."""
    projections = compute_projections(srm_crystal_sparse, grid_size)
    if axis_limit_mm is None:
        axis_limit_mm = nice_axis_limit(0.5 * max(fov_size_mm))

    # Use the OO Figure API (not pyplot) so this is safe in worker threads/processes
    # constrained layout keeps each panel's color bar clear of its neighbour's labels
    fig = Figure(figsize=(18, 5), layout="constrained")
    axs = fig.subplots(1, 3)

    for ax, projection, (h, v) in zip(axs, projections, PROJECTION_AXES):
        half_h, half_v = fov_size_mm[h] * 0.5, fov_size_mm[v] * 0.5
        image = ax.imshow(
            projection.T,
            cmap="hot",
            interpolation="none",
            extent=(-half_h, half_h, -half_v, half_v),
            origin="lower",
            vmin=0,
            vmax=max(1, projection.max()),
        )
        fig.colorbar(image, ax=ax, orientation="vertical", shrink=1)
        name_h, name_v = AXIS_NAMES[h], AXIS_NAMES[v]
        ax.set_title(f"{name_h}{name_v} Projection ({prefix} {label_id})")
        ax.set_xlabel(f"{name_h}-axis")
        ax.set_ylabel(f"{name_v}-axis")
        ax.set_aspect("equal")
        ax.set_xlim(-axis_limit_mm, axis_limit_mm)
        ax.set_ylim(-axis_limit_mm, axis_limit_mm)

    return fig, axs


@dataclass
class PlotContext:
    """Everything per-module plotting needs besides the SRM itself. Built once
    and pickled to each worker, so it holds plain arrays rather than meshes."""

    grid_size: int
    num_heads: int
    pixels_per_head: int
    fov_size_mm: tuple[float, float, float]
    ring_index: np.ndarray  # (num_heads,) ring id per module, grouped by elevation
    crystal_centers: np.ndarray  # (num_heads, 3)
    crystal_rotations: np.ndarray  # (num_heads, 3, 3) crystal local -> world
    # (pixels_per_head, 3) pixel centres in the crystal's local frame, indexed
    # by SRM row; None if the provenance's pixel grid doesn't match the SRM
    pixel_offsets_local: np.ndarray | None
    pixel_size_mm: np.ndarray  # (3,) pixel box full lengths (x, y, crystal depth)
    # (num_heads,) square pinhole aperture side per module; None if unknown
    pinhole_widths_mm: np.ndarray | None
    pinholes: np.ndarray  # (num_heads, 3)
    # part name -> per-module (N, 2, 3) outline segments
    part_segments: dict[str, list[np.ndarray]]
    # per panel, (N, 2) polylines of the shielding's projected outline; None if off
    shielding_silhouettes: list[list[np.ndarray]] | None
    axis_limit_mm: float  # shared +/- limit for every panel of every head
    output_dir: Path


def load_geometry_provenance(loader: SRMLoader) -> dict:
    path = loader.datadir / "geometry_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if "analytic_geometry" not in provenance:
        raise ValueError(
            f"{path} is schema {provenance.get('schema_version')} without "
            "analytic_geometry; upgrade it with payload/python/"
            "write_brain_spect_geometry_provenance.py --output <file> --from-existing"
        )
    return provenance


def load_shielding_silhouettes(
    provenance: dict, modality: str
) -> list[list[np.ndarray]] | None:
    """Per-panel silhouettes of the shielding STL recorded in the provenance,
    placed in the world frame, after checking the file on disk is the one the
    simulation used. None when the run had no shielding."""
    import qmirt

    shielding = provenance["shielding"]
    if not shielding.get("enabled"):
        return None
    source = shielding["source"]
    stl_path = (
        qmirt.utils.filesystem.search_dir_up("persistent_data", __file__)
        / modality
        / source["path"]
    )
    digest = hashlib.sha256(stl_path.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise ValueError(f"{stl_path} does not match the recorded shielding sha256")

    mesh = trimesh.load_mesh(stl_path)
    placement = shielding["placement"]
    rotation = np.asarray(placement["rotation"], dtype=float).reshape(3, 3)
    # Fixed seed so every run (and every worker) draws the same outline
    points, _ = trimesh.sample.sample_surface_even(mesh, 400_000, seed=0)
    points = points @ rotation.T + np.asarray(placement["translation_mm"])
    return projected_silhouettes(points)


def crystal_pixel_offsets(crystal: dict, pixels_per_head: int) -> np.ndarray | None:
    """(pixels_per_head, 3) pixel centres in the crystal's local frame, indexed
    by SRM row, or None if the recorded grid doesn't match the SRM.

    Row r is pixel (r // n_y, r % n_y): local y varies fastest, no axis flips.
    Verified 2026-09-30 on brain_merged_123 heads 1, 22 and 73 by tracing
    near-centre pixels' SRM beams back through the pinhole onto the crystal."""
    n_x, n_y, n_z = crystal.get("n_pixels", (0, 0, 0))
    if n_z != 1 or n_x * n_y != pixels_per_head or len(crystal["variants"]) != 1:
        print("Warning: crystal pixel grid doesn't match the SRM; rays use crystal centres")
        return None
    (box,) = crystal["variants"][0]
    pitch = np.asarray(box["size_mm"], dtype=float)[:2] / (n_x, n_y)
    row = np.arange(pixels_per_head)
    ix, iy = row // n_y, row % n_y
    return np.column_stack(
        [
            (ix - (n_x - 1) / 2) * pitch[0],
            (iy - (n_y - 1) / 2) * pitch[1],
            np.zeros(pixels_per_head),
        ]
    ) + np.asarray(box["offset_mm"], dtype=float)


def module_pinhole_widths(provenance: dict) -> np.ndarray | None:
    """(num_heads,) pinhole aperture side of each module's collimator variant,
    or None when the run didn't record per-module variants."""
    variant_index = provenance["analytic_geometry"]["parts"]["collimator"][
        "variant_index"
    ]
    variants = provenance.get("pinhole_pattern", {}).get("collimator_variants")
    if variant_index is None or not variants:
        return None
    widths = np.array([variant["w_pinhole"] for variant in variants], dtype=float)
    return widths[np.asarray(variant_index)]


def crystal_pixel_size(crystal: dict) -> np.ndarray:
    """(3,) full lengths of one pixel box: pitch in x/y, the crystal depth in z."""
    box = crystal["variants"][0][0]
    n_x, n_y, _ = crystal.get("n_pixels", (1, 1, 1))
    return np.asarray(box["size_mm"], dtype=float) / (n_x, n_y, 1)


def build_plot_context(
    loader: SRMLoader,
    output_dir: Path = Path("Brain_SRM_12t"),
    with_shielding: bool = True,
    with_collimators: bool = True,
    axis_limit_mm: float | None = None,
) -> PlotContext:
    """with_shielding / with_collimators toggle those overlays independently;
    a disabled overlay is not computed at all. Crystals are always drawn.

    axis_limit_mm: None fits everything that will be drawn (FOV, enabled outlines
    of all modules, pinholes, shielding), shared by all heads so plots compare
    at the same scale."""
    with (loader.datadir / "combined_srm_metadata.json").open(
        encoding="utf-8"
    ) as handle:
        metadata = json.load(handle)["resolutions"][loader.resolution]
    grid_size = int(metadata["grid_size"])
    num_heads = int(metadata["num_heads"])
    pixels_per_head = int(metadata["pixels_per_head"])
    voxel_size_mm = {"1mm": 1.0, "1p5mm": 1.5, "2mm": 2.0}[loader.resolution]
    fov_mm = grid_size * voxel_size_mm

    provenance = load_geometry_provenance(loader)
    geometry = provenance["analytic_geometry"]
    modules = geometry["modules"]
    if modules["count"] != num_heads:
        raise ValueError(
            f"Geometry has {modules['count']} modules but the SRM has {num_heads} heads"
        )
    rotations = np.asarray(modules["rotation"], dtype=float).reshape(-1, 3, 3)

    part_segments = {}
    for name, part in geometry["parts"].items():
        if name == "collimator" and not with_collimators:
            continue
        if part["variant_index"] is None:
            print(f"Warning: no per-module placement recorded for {name}; skipped")
            continue
        centers = np.asarray(part["center_mm"], dtype=float)
        part_segments[name] = [
            part_segments_for(part, index, centers[index], rotations[index])
            for index in range(num_heads)
        ]

    shielding_silhouettes = None
    if with_shielding:
        shielding_silhouettes = load_shielding_silhouettes(provenance, loader.modality)

    pinholes = np.asarray(modules["pinhole_mm"], dtype=float)
    if axis_limit_mm is None:
        drawn = [np.array([0.5 * fov_mm]), pinholes]
        drawn += [np.concatenate(segments) for segments in part_segments.values()]
        if shielding_silhouettes is not None:
            drawn += [line for lines in shielding_silhouettes for line in lines]
        axis_limit_mm = nice_axis_limit(max(np.abs(a).max() for a in drawn))

    output_dir.mkdir(parents=True, exist_ok=True)
    return PlotContext(
        grid_size=grid_size,
        num_heads=num_heads,
        pixels_per_head=pixels_per_head,
        fov_size_mm=(fov_mm, fov_mm, fov_mm),
        ring_index=np.asarray(modules["ring_index"]),
        crystal_centers=np.asarray(geometry["parts"]["crystal"]["center_mm"]),
        crystal_rotations=rotations,
        pixel_offsets_local=crystal_pixel_offsets(
            geometry["parts"]["crystal"], pixels_per_head
        ),
        pixel_size_mm=crystal_pixel_size(geometry["parts"]["crystal"]),
        pinhole_widths_mm=module_pinhole_widths(provenance),
        pinholes=pinholes,
        part_segments=part_segments,
        shielding_silhouettes=shielding_silhouettes,
        axis_limit_mm=axis_limit_mm,
        output_dir=output_dir,
    )


def part_segments_for(
    part: dict, index: int, center: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    primitives = part["variants"][part["variant_index"][index]]
    return placed_part_segments(primitives, center, rotation)


# Outline style per part; parts not listed fall back to grey
PART_STYLES = {
    "crystal": {"colors": "red", "linewidths": 1.5},
    "collimator": {"colors": "tab:blue", "linewidths": 0.8},
}


def fov_exit_point(
    start: np.ndarray, through: np.ndarray, fov_size_mm: tuple[float, float, float]
) -> np.ndarray:
    """Where the ray from start through `through` leaves the origin-centred FOV
    box; `through` itself if the ray misses the box."""
    direction = through - start
    half = 0.5 * np.asarray(fov_size_mm)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_low = (-half - start) / direction
        t_high = (half - start) / direction
    t_near = np.nanmax(np.minimum(t_low, t_high))
    t_far = np.nanmin(np.maximum(t_low, t_high))
    if t_far < max(t_near, 1.0):
        return through
    return start + t_far * direction


def pixel_center(
    ctx: PlotContext,
    head_id: int,
    pixel_index: int,
    depth_mm: float | None = None,
) -> np.ndarray:
    """World position of an SRM row's pixel centre; the crystal centre if the
    pixel grid is unknown. depth_mm: depth below the crystal's front (pinhole
    side) face; None for mid-depth."""
    center = ctx.crystal_centers[head_id]
    if ctx.pixel_offsets_local is None:
        return center
    local = ctx.pixel_offsets_local[pixel_index].copy()
    if depth_mm is not None:
        # Local +z points toward the FOV, so the front face is at +depth/2
        local[2] += 0.5 * ctx.pixel_size_mm[2] - depth_mm
    return center + ctx.crystal_rotations[head_id] @ local


def aperture_corners(
    ctx: PlotContext, head_id: int, margin_mm: float = 0.0
) -> np.ndarray | None:
    """(4, 3) world corners, in ring order, of the head's square pinhole aperture
    (side w_pinhole + margin_mm, in the module's local xy plane at the pinhole),
    or None if the aperture size is unknown."""
    if ctx.pinhole_widths_mm is None:
        return None
    half_width = 0.5 * (ctx.pinhole_widths_mm[head_id] + margin_mm)
    square = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    rotation = ctx.crystal_rotations[head_id]
    return ctx.pinholes[head_id] + (half_width * square) @ rotation.T


def pixel_beam_outlines(
    ctx: PlotContext,
    head_id: int,
    pixel_index: int,
    aperture_margin_mm: float = 0.0,
) -> list[np.ndarray] | None:
    """Per panel, the (M, 2) closed outline of the pixel's geometric beam: every
    line from a point in the pixel box through the square pinhole aperture, up
    to where it leaves the FOV. None if the pixel grid or aperture is unknown.

    Drawn as the convex hull of the 8 x 4 corner-to-corner rays, which is a
    slight over-estimate near the edges. Checked on brain_merged_123 heads 1 and
    22: 93-99% of a pixel's SRM counts fall inside with the nominal aperture,
    98-99% with aperture_margin_mm=1 (knife-edge penetration)."""
    from scipy.spatial import ConvexHull

    aperture = aperture_corners(ctx, head_id, aperture_margin_mm)
    if ctx.pixel_offsets_local is None or aperture is None:
        return None
    rotation = ctx.crystal_rotations[head_id]
    corner_signs = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    pixel_corners = ctx.crystal_centers[head_id] + (
        ctx.pixel_offsets_local[pixel_index] + 0.5 * corner_signs * ctx.pixel_size_mm
    ) @ rotation.T
    far_points = [
        fov_exit_point(start, through, ctx.fov_size_mm)
        for start in pixel_corners
        for through in aperture
    ]
    points = np.vstack([aperture, far_points])

    outlines = []
    for h, v in PROJECTION_AXES:
        projected = points[:, [h, v]]
        hull = ConvexHull(projected)
        outlines.append(projected[np.append(hull.vertices, hull.vertices[0])])
    return outlines


def add_module_overlays(
    axs,
    head_id: int,
    outlined_modules: np.ndarray,
    ctx: PlotContext,
    ray_start: np.ndarray | None = None,
    beam_outlines: list[np.ndarray] | None = None,
):
    """Draw the shielding silhouette, the part outlines of outlined_modules
    (labelled on the XY panel), and head_id's pinhole with a ray from ray_start
    (default: the crystal centre) through the pinhole to where it leaves the FOV.
    beam_outlines: optional per-panel closed outlines (see pixel_beam_outlines).
    With collimators on, each collimator also gets its pinhole aperture square."""
    if ctx.shielding_silhouettes is not None:
        for ax, lines in zip(axs, ctx.shielding_silhouettes):
            ax.add_collection(
                LineCollection(lines, colors="0.5", linewidths=0.8, linestyles="--")
            )

    for name, segments in ctx.part_segments.items():
        style = PART_STYLES.get(name, {"colors": "0.3", "linewidths": 0.8})
        add_outlines(
            axs,
            np.concatenate([segments[index] for index in outlined_modules]),
            capstyle="projecting",
            **style,
        )
    if "collimator" in ctx.part_segments:
        apertures = [aperture_corners(ctx, index) for index in outlined_modules]
        # (4, 2, 3) closed-square edges per module, drawn like the collimator
        edges = [
            corners[[0, 1, 1, 2, 2, 3, 3, 0]].reshape(4, 2, 3)
            for corners in apertures
            if corners is not None
        ]
        if edges:
            add_outlines(axs, np.concatenate(edges), **PART_STYLES["collimator"])
    for index in outlined_modules:
        center = ctx.crystal_centers[index]
        axs[0].text(
            center[0],
            center[1],
            f"{index + 1}",
            color="blue",
            fontsize=8,
            ha="center",
            va="center",
        )

    pinhole = ctx.pinholes[head_id]
    start: np.ndarray = (
        ctx.crystal_centers[head_id] if ray_start is None else ray_start
    )
    ray_end = fov_exit_point(start, pinhole, ctx.fov_size_mm)
    ray = np.array([start, ray_end])
    for panel, (ax, (h, v)) in enumerate(zip(axs, PROJECTION_AXES)):
        ax.plot(ray[:, h], ray[:, v], color="green", linewidth=1.5, linestyle="--")
        ax.plot(pinhole[h], pinhole[v], "o", color="green", markersize=3)
        if beam_outlines is not None:
            ax.plot(*beam_outlines[panel].T, color="cyan", linewidth=0.8)


def plot_single_module_srm(
    head_id: int, head_srm: sparse.csr_matrix, ctx: PlotContext
) -> Path:
    """Plot one module's SRM projections and save the PNG under
    <output_dir>/modules/. Pure plotting (no loading), safe to run in a thread
    or process pool."""
    if not 0 <= head_id < ctx.num_heads:
        raise IndexError(f"Module index out of range: {head_id}")
    expected_shape = (ctx.pixels_per_head, ctx.grid_size**3)
    if head_srm.shape != expected_shape:
        raise ValueError(
            f"Head {head_id + 1}: expected SRM shape {expected_shape}, "
            f"got {head_srm.shape}"
        )

    fig, axs = plot_sparse_3d_projections(
        head_srm,
        ctx.grid_size,
        label_id=head_id + 1,
        prefix="Head",
        fov_size_mm=ctx.fov_size_mm,
        axis_limit_mm=ctx.axis_limit_mm,
    )

    # Outline every module in this head's ring and label it on the XY panel
    ring_modules = np.flatnonzero(ctx.ring_index == ctx.ring_index[head_id])
    add_module_overlays(axs, head_id, ring_modules, ctx)

    n_ring = len(ring_modules)
    fig.suptitle(
        f"SRM Projections for Head ID {(head_id + 1):02} "
        f"(ring {ctx.ring_index[head_id]}, {n_ring} module{'s' * (n_ring != 1)})"
    )
    output_dir = ctx.output_dir / "modules"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"SRM_Head_{(head_id + 1):02}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return output_path


def load_and_plot_module_srm(loader: SRMLoader, head_id: int, ctx: PlotContext) -> Path:
    """Process-pool task: load inside the worker so the SRM (~180 MB pickled)
    never has to be shipped from the parent process."""
    return plot_single_module_srm(head_id, loader.load_single_srm(head_id), ctx)


def plot_per_module_srms(
    loader: SRMLoader,
    *,
    output_dir: Path = Path("Brain_SRM_12t"),
    module_indices: Iterable[int] | None = None,
    max_workers: int | None = 32,
    with_shielding: bool = True,
    with_collimators: bool = True,
    axis_limit_mm: float | None = None,
) -> list[int]:
    """Load and plot each module in its own worker process. Returns the head
    ids (0-based) that failed. axis_limit_mm=None fits the drawn geometry."""
    ctx = build_plot_context(
        loader,
        output_dir,
        with_shielding=with_shielding,
        with_collimators=with_collimators,
        axis_limit_mm=axis_limit_mm,
    )

    if module_indices is None:
        module_indices = range(ctx.num_heads)
    module_indices = list(module_indices)
    for head_id in module_indices:
        if not 0 <= head_id < ctx.num_heads:
            raise IndexError(f"Module index out of range: {head_id}")

    # Plotting is CPU-bound (GIL-held), so use processes to parallelize over
    # modules. forkserver avoids fork() from a process that already runs
    # threads (e.g. tqdm's monitor), which can deadlock.
    failed = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("forkserver"),
    ) as executor:
        futures = {
            executor.submit(load_and_plot_module_srm, loader, head_id, ctx): head_id
            for head_id in module_indices
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Plotting SRMs"
        ):
            head_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"\n[Warning] Failed to plot SRM for head {head_id + 1}: {e}")
                failed.append(head_id)

    if failed:
        print(
            f"[Warning] {len(failed)}/{len(module_indices)} heads not plotted: "
            f"{sorted(head_id + 1 for head_id in failed)}"
        )
    return sorted(failed)


def plot_single_pixel_srm(
    head_id: int,
    pixel_index: int,
    pixel_srm: sparse.csr_matrix,
    ctx: PlotContext,
    *,
    beam_outline: bool = True,
    aperture_margin_mm: float = 0.0,
    ray_depth_mm: float | None = None,
) -> Path:
    """Plot one detector pixel's SRM row (its sensitivity map over the FOV) as
    XY/XZ/YZ projections, with this head's outlines and central ray, and save
    the PNG under <output_dir>/pixels/Head_<hh>/.

    pixel_index is the SRM row within the head; the green ray runs from that
    pixel's centre (see crystal_pixel_offsets) through the pinhole, starting
    ray_depth_mm below the crystal's front face (None: mid-depth). Mean
    interaction depth puts the observed beam axis about 1.8 mm below the front
    face, so ray_depth_mm=1.8 follows the beam more closely than mid-depth.
    beam_outline draws the predicted geometric beam (see pixel_beam_outlines).
    pixel_srm: that single row, shape (1, grid_size**3). Pure plotting, safe to
    run in a thread or process pool."""
    if not 0 <= head_id < ctx.num_heads:
        raise IndexError(f"Module index out of range: {head_id}")
    if not 0 <= pixel_index < ctx.pixels_per_head:
        raise IndexError(f"Pixel index out of range: {pixel_index}")
    expected_shape = (1, ctx.grid_size**3)
    if pixel_srm.shape != expected_shape:
        raise ValueError(
            f"Head {head_id + 1} pixel {pixel_index}: expected SRM row shape "
            f"{expected_shape}, got {pixel_srm.shape}"
        )

    fig, axs = plot_sparse_3d_projections(
        pixel_srm,
        ctx.grid_size,
        label_id=pixel_index,
        prefix=f"Head {head_id + 1} Pixel",
        fov_size_mm=ctx.fov_size_mm,
        axis_limit_mm=ctx.axis_limit_mm,
    )
    pixel = pixel_center(ctx, head_id, pixel_index, depth_mm=ray_depth_mm)
    outlines = (
        pixel_beam_outlines(ctx, head_id, pixel_index, aperture_margin_mm)
        if beam_outline
        else None
    )
    add_module_overlays(
        axs,
        head_id,
        np.array([head_id]),
        ctx,
        ray_start=pixel,
        beam_outlines=outlines,
    )
    for ax, (h, v) in zip(axs, PROJECTION_AXES):
        ax.plot(pixel[h], pixel[v], "s", color="green", markersize=3)

    fig.suptitle(
        f"SRM Projections for Head ID {(head_id + 1):02}, Pixel {pixel_index} "
        f"(nnz {pixel_srm.nnz}, total {pixel_srm.sum():.4g})"
    )
    output_dir = ctx.output_dir / "pixels" / f"Head_{(head_id + 1):02}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"SRM_Head_{(head_id + 1):02}_Pixel_{pixel_index:03}.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return output_path


def plot_per_pixel_srms(
    loader: SRMLoader,
    head_id: int,
    *,
    pixel_indices: Iterable[int] | None = None,
    output_dir: Path = Path("Brain_SRM_12t"),
    max_workers: int | None = 32,
    with_shielding: bool = True,
    with_collimators: bool = True,
    axis_limit_mm: float | None = None,
    beam_outline: bool = True,
    aperture_margin_mm: float = 0.0,
    ray_depth_mm: float | None = None,
) -> list[int]:
    """Plot the SRM row of each pixel in pixel_indices (default: every pixel)
    of one head, one worker process per plot. Returns the pixel indices that
    failed. The head's SRM is loaded once here and only the single rows (a
    few hundred kB each, not the ~180 MB head) are shipped to the workers."""
    ctx = build_plot_context(
        loader,
        output_dir,
        with_shielding=with_shielding,
        with_collimators=with_collimators,
        axis_limit_mm=axis_limit_mm,
    )
    if not 0 <= head_id < ctx.num_heads:
        raise IndexError(f"Module index out of range: {head_id}")
    if pixel_indices is None:
        pixel_indices = range(ctx.pixels_per_head)
    pixel_indices = list(pixel_indices)
    for pixel_index in pixel_indices:
        if not 0 <= pixel_index < ctx.pixels_per_head:
            raise IndexError(f"Pixel index out of range: {pixel_index}")

    head_srm = loader.load_single_srm(head_id).tocsr()

    # Same pool setup as plot_per_module_srms (see the comment there)
    failed = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("forkserver"),
    ) as executor:
        futures = {
            executor.submit(
                plot_single_pixel_srm,
                head_id,
                pixel_index,
                head_srm[pixel_index : pixel_index + 1],
                ctx,
                beam_outline=beam_outline,
                aperture_margin_mm=aperture_margin_mm,
                ray_depth_mm=ray_depth_mm,
            ): pixel_index
            for pixel_index in pixel_indices
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Head {head_id + 1:02}: plotting pixels",
        ):
            pixel_index = futures[future]
            try:
                future.result()
            except Exception as e:
                print(
                    f"\n[Warning] Failed to plot head {head_id + 1} "
                    f"pixel {pixel_index}: {e}"
                )
                failed.append(pixel_index)

    if failed:
        print(
            f"[Warning] {len(failed)}/{len(pixel_indices)} pixels of head "
            f"{head_id + 1} not plotted: {sorted(failed)}"
        )
    return sorted(failed)


def parse_index_list(text: str, *, one_based: bool) -> list[int] | None:
    """Parse "all" or a comma list of ints and inclusive ranges ("1-10,22,70")
    into sorted unique 0-based indices; None means all. one_based=True
    subtracts 1, so users can type head numbers as they appear in filenames."""
    if text.strip().lower() == "all":
        return None
    offset = 1 if one_based else 0
    indices = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", token)
        if match is None:
            raise argparse.ArgumentTypeError(
                f"bad index {token!r}; use e.g. 'all', '5', '1-10,22'"
            )
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if last < first:
            raise argparse.ArgumentTypeError(f"empty range {token!r}")
        if first < offset:
            raise argparse.ArgumentTypeError(
                f"{token!r}: indices start at {offset} here"
            )
        indices.update(range(first - offset, last - offset + 1))
    if not indices:
        raise argparse.ArgumentTypeError("no indices given")
    return sorted(indices)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize and plot per-module / per-pixel SRM projections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Options shared by every command, so they can go after the command name
    common = argparse.ArgumentParser(add_help=False)
    data = common.add_argument_group("data")
    data.add_argument(
        "--modality",
        choices=["brain_spect", "cardiac_spect"],
        default="brain_spect",
        help="data folder under /data/fanghan/opengate_sim/data",
    )
    data.add_argument(
        "--subdir",
        default="brain_merged_123",
        help="campaign folder under the modality folder",
    )
    data.add_argument(
        "--resolution",
        choices=["1mm", "1p5mm", "2mm"],
        default="1mm",
        help="voxel size of the SRMs to load",
    )
    data.add_argument(
        "--heads",
        type=lambda text: parse_index_list(text, one_based=True),
        default="all",
        metavar="LIST",
        help="1-based head numbers, e.g. 'all', '1', '1-10,22,70'",
    )
    data.add_argument(
        "--workers", type=int, default=32, help="worker processes for plotting"
    )

    plotting = argparse.ArgumentParser(add_help=False)
    plot = plotting.add_argument_group("plotting")
    plot.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Brain_SRM_12t"),
        help="relative to the current directory; plots go in <dir>/modules and "
        "<dir>/pixels/Head_<hh>",
    )
    plot.add_argument(
        "--shielding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="draw the shielding silhouette (adds ~3 s per context build)",
    )
    plot.add_argument(
        "--collimators",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw collimator outlines",
    )
    plot.add_argument(
        "--axis-limit-mm",
        type=float,
        default=None,
        help="symmetric +/- panel limit; default fits the drawn geometry",
    )

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "summary",
        parents=[common],
        help="print nnz and total counts of the selected heads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    commands.add_parser(
        "modules",
        parents=[common, plotting],
        help="plot each selected head's summed SRM projections",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pixels = commands.add_parser(
        "pixels",
        parents=[common, plotting],
        help="plot single detector pixels (SRM rows) of the selected heads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pixels.add_argument(
        "--pixels",
        type=lambda text: parse_index_list(text, one_based=False),
        default="all",
        metavar="LIST",
        help="0-based SRM rows within a head, e.g. 'all', '312', '0-24,312'",
    )
    pixels.add_argument(
        "--beam-outline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="outline the predicted geometric beam (pixel box through the aperture)",
    )
    pixels.add_argument(
        "--aperture-margin-mm",
        type=float,
        default=0.0,
        help="widen the pinhole aperture of the beam outline, e.g. 1 to cover "
        "knife-edge penetration",
    )
    pixels.add_argument(
        "--ray-depth-mm",
        type=float,
        default=None,
        help="start the central ray this far below the crystal front face; "
        "default: mid-depth. ~1.8 follows the mean interaction depth",
    )
    return parser


def print_srm_summary(loader: SRMLoader, heads: list[int] | None) -> None:
    # Keyed by head id (not positional), so gaps in the file numbering are fine
    srms = loader.load_srms_parallel(heads)
    for head_id, srm in srms.items():
        print(f"Head {head_id + 1:02}: nnz {srm.nnz}, counts {srm.data.sum():.6g}")
    print(f"Total number of non-zero elements: {sum(s.nnz for s in srms.values())}")
    print(f"Total counts: {sum(s.data.sum() for s in srms.values())}")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    loader = SRMLoader(
        subdir=args.subdir, modality=args.modality, resolution=args.resolution
    )

    if args.command == "summary":
        print_srm_summary(loader, args.heads)
        return 0

    plot_options = dict(
        output_dir=args.output_dir,
        max_workers=args.workers,
        with_shielding=args.shielding,
        with_collimators=args.collimators,
        axis_limit_mm=args.axis_limit_mm,
    )
    if args.command == "modules":
        failed = plot_per_module_srms(
            loader, module_indices=args.heads, **plot_options
        )
        return 1 if failed else 0

    # pixels: one context build and one head load per selected head
    heads = args.heads if args.heads is not None else list(loader.srm_filenames)
    any_failed = False
    for head_id in heads:
        failed = plot_per_pixel_srms(
            loader,
            head_id,
            pixel_indices=args.pixels,
            beam_outline=args.beam_outline,
            aperture_margin_mm=args.aperture_margin_mm,
            ray_depth_mm=args.ray_depth_mm,
            **plot_options,
        )
        any_failed |= bool(failed)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
