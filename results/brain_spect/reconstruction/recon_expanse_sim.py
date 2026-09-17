#!/usr/bin/env python3
"""Quick sparse MLEM test reconstruction on pseudo data.

Run with the project env activated:
    ma opengate && python results/brain_spect/reconstruction/recon_expanse_sim.py \
      --campaign-dir /path/to/campaign \
      --resolution 2mm \
      --head-slice 0:1

This script loads the per-head CSR matrices produced by merge_brain_campaigns.py,
vertically stacks the selected heads, builds a synthetic truth image, forward
projects it via A @ x to generate pseudo measured data, and runs a small MLEM
reconstruction using mlem_sparse.reconstruct.SparseMLEMReconstructor.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mlem_sparse.reconstruct import SparseMLEMReconstructor
from scipy.sparse import load_npz, vstack


def parse_head_slice(raw: str | None, n_heads: int) -> slice:
    if raw is None:
        return slice(0, min(1, n_heads))
    if raw == ":":
        return slice(0, n_heads)
    if ":" not in raw:
        start = int(raw)
        return slice(start, min(start + 1, n_heads))
    start_s, stop_s = raw.split(":", 1)
    start = int(start_s) if start_s else 0
    stop = int(stop_s) if stop_s else n_heads
    start = max(0, min(start, n_heads))
    stop = max(start, min(stop, n_heads))
    return slice(start, stop)


def make_center_sphere_phantom(
    grid_size: int, *, radius: int | None = None, intensity: float = 1.0
) -> np.ndarray:
    """Create a single sphere centered in the volume."""
    grid = int(grid_size)
    if radius is None:
        radius = max(6, grid // 10)
    image = np.zeros((grid, grid, grid), dtype=np.float32)
    x = np.arange(grid)
    y = np.arange(grid)
    z = np.arange(grid)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    cx = cy = cz = grid // 2
    image[(xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius**2] = intensity
    return image.reshape(-1, order="C").astype(np.float32)


def make_three_rods_phantom(
    grid_size: int,
    voxel_size_mm: float,
    *,
    intensity: float = 1.0,
) -> tuple[np.ndarray, list[str]]:
    """Create three central-slice rods with diameters 4, 8, and 12 mm."""
    grid = int(grid_size)
    if voxel_size_mm <= 0:
        raise ValueError("voxel_size_mm must be positive")

    volume = np.zeros((grid, grid, grid), dtype=np.float32)
    center = grid // 2
    mid = center
    x_coords = (np.arange(grid) - center) * voxel_size_mm
    y_coords = (np.arange(grid) - center) * voxel_size_mm
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")

    rod_diameters_mm = (4.0, 8.0, 12.0)
    rod_radius_from_z_mm = 18.0
    rod_angles = (np.pi / 2.0, 7.0 * np.pi / 6.0, 11.0 * np.pi / 6.0)
    rod_centers_xy_mm = tuple(
        (
            rod_radius_from_z_mm * np.cos(angle),
            rod_radius_from_z_mm * np.sin(angle),
        )
        for angle in rod_angles
    )
    half_fov_mm = 0.5 * grid * voxel_size_mm
    if (
        max(
            np.hypot(x, y) + diameter / 2.0
            for (x, y), diameter in zip(rod_centers_xy_mm, rod_diameters_mm)
        )
        > half_fov_mm
    ):
        raise ValueError(
            "Three-rod layout exceeds the FOV for the selected resolution."
        )

    annotation_texts = [
        f"rod {idx}: diameter {diameter_mm:.0f} mm"
        for idx, diameter_mm in enumerate(rod_diameters_mm, start=1)
    ]
    for diameter_mm, (x0, y0) in zip(rod_diameters_mm, rod_centers_xy_mm):
        radius_vox = max(1, int(round((diameter_mm / 2.0) / voxel_size_mm)))
        circle = (xx - x0) ** 2 + (yy - y0) ** 2 <= radius_vox**2
        volume[:, :, mid] = np.maximum(
            volume[:, :, mid], circle.astype(np.float32) * intensity
        )

    return volume.reshape(-1, order="C").astype(np.float32), annotation_texts


def build_iteration_schedule(max_iter: int, save_every: int = 10) -> list[int]:
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if save_every <= 0:
        raise ValueError("save_every must be positive")
    schedule = list(range(save_every, max_iter + 1, save_every))
    if schedule[-1] != max_iter:
        schedule.append(max_iter)
    return schedule


def plot_iteration(
    true_image: np.ndarray,
    estimate: np.ndarray,
    iteration: int,
    grid_size: int,
    output_dir: Path,
    resolution: str,
    head_slice: slice,
    phantom_name: str,
    phantom_annotations: list[str] | None = None,
):
    true_3d = true_image.reshape(grid_size, grid_size, grid_size)
    mid = grid_size // 2
    true_slice = true_3d[:, :, mid]

    estimate_3d = estimate.reshape(grid_size, grid_size, grid_size)
    recon_slice = estimate_3d[:, :, mid]
    vmin = min(float(true_slice.min()), float(recon_slice.min()))
    vmax = max(float(true_slice.max()), float(recon_slice.max()))
    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(
        f"{resolution} {phantom_name}, iteration {iteration}, heads {head_slice}"
    )

    if phantom_annotations:
        annotation_text = "\n".join(phantom_annotations)
    else:
        annotation_text = "sphere radius"

    axes[0].imshow(true_slice, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"phantom, z={mid}")
    axes[0].axis("off")
    axes[0].text(
        0.02,
        0.98,
        annotation_text,
        transform=axes[0].transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "0.7"},
    )

    axes[1].imshow(recon_slice, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"recon iter {iteration}")
    axes[1].axis("off")
    if phantom_annotations:
        axes[1].text(
            0.02,
            0.98,
            annotation_text,
            transform=axes[1].transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "0.7"},
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"iteration_{iteration:03d}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def replot_saved_reconstruction(npz_path: Path, output_dir: Path | None = None):
    data = np.load(npz_path)
    true_image = np.asarray(data["true_image"])
    estimate_history = np.asarray(data["estimate_history"])
    saved_iters = [int(value) for value in data["iterations"]]
    grid_size = int(np.asarray(data["grid_size"]).reshape(-1)[0])
    resolution = str(np.asarray(data["resolution"]).reshape(-1)[0])
    name = npz_path.parent.name
    phantom_name = "rods" if "rod" in name else "sphere"
    output_dir = output_dir or npz_path.parent / "per_iteration"
    annotations = (
        ["rod 1: diameter 4 mm", "rod 2: diameter 8 mm", "rod 3: diameter 12 mm"]
        if phantom_name == "rods"
        else [f"sphere, center voxel = {grid_size // 2}"]
    )
    head_slice = slice(0, int(npz_path.stem.rsplit("_", 2)[-2]))
    if npz_path.stem.endswith("_0_73"):
        head_slice = slice(0, 73)
    elif npz_path.stem.endswith("_0_1"):
        head_slice = slice(0, 1)
    for iteration, estimate in zip(saved_iters, estimate_history):
        plot_iteration(
            true_image,
            estimate,
            iteration,
            grid_size,
            output_dir,
            resolution,
            head_slice,
            phantom_name,
            annotations,
        )
    return output_dir, saved_iters


def load_campaign_metadata(campaign_dir: Path, resolution: str) -> dict:
    metadata_path = campaign_dir / "combined_srm_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing campaign metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if "resolutions" not in metadata:
        raise ValueError(f"{metadata_path} does not contain a 'resolutions' block")
    if resolution not in metadata["resolutions"]:
        raise KeyError(
            f"Resolution {resolution!r} not found in {metadata_path}; "
            f"available: {sorted(metadata['resolutions'])}"
        )
    return metadata["resolutions"][resolution]


def select_head_paths(campaign_dir: Path, resolution_meta: dict, head_slice: slice):
    output_names = resolution_meta.get("outputs") or resolution_meta.get("heads", [])
    if not output_names:
        pattern = f"final_srm_{resolution_meta.get('resolution_label', '*')}_head_*.npz"
        raise ValueError(
            f"No head outputs listed in metadata for this resolution. "
            f"Expected file pattern like {pattern!r}."
        )

    if isinstance(output_names[0], dict):
        names = [entry["output"] for entry in output_names]
    else:
        names = list(output_names)

    selected = names[head_slice]
    return [campaign_dir / name for name in selected]


def build_system_matrix(campaign_dir: Path, resolution: str, head_slice: slice):
    meta = load_campaign_metadata(campaign_dir, resolution)
    paths = select_head_paths(campaign_dir, meta, head_slice)
    if not paths:
        raise ValueError(
            f"No per-head SRMs matched the selected head slice {head_slice!r}"
        )

    matrices = [load_npz(path).astype(np.float32).tocsr() for path in paths]
    if len(matrices) == 1:
        return matrices[0], meta
    return vstack(matrices, format="csr").astype(np.float32), meta


def pseudo_forward_project(
    system_matrix: np.ndarray, true_image: np.ndarray
) -> np.ndarray:
    measured = system_matrix @ true_image
    measured = np.asarray(measured, dtype=np.float32).reshape(-1)
    measured = np.maximum(measured, 0.0)
    return measured


def main():
    parser = argparse.ArgumentParser(
        description="Test sparse MLEM reconstruction on pseudo data from per-head SRMs."
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        required=False,
        help="Directory containing final_srm_<res>_head_*.npz and combined_srm_metadata.json",
    )
    parser.add_argument(
        "--replot-npz",
        type=Path,
        default=None,
        help="Replot an existing pseudo_recon_*.npz as one compact PNG per iteration.",
    )
    parser.add_argument(
        "--resolution",
        choices=["1mm", "1p5mm", "2mm"],
        default="2mm",
        help="Resolution to load. Use 2mm for quickest testing.",
    )
    parser.add_argument(
        "--head-slice",
        default="0:1",
        help="Slice of head outputs to use, e.g. '0:1' for the first head or '0:5' for five heads.",
    )
    parser.add_argument(
        "--iterations",
        nargs="+",
        type=int,
        default=None,
        help="Explicit MLEM iteration numbers to save. Overrides --max-iters/--save-every.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=200,
        help="Maximum MLEM iteration count for a full test reconstruction.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save reconstruction estimates every N iterations during the MLEM run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for the pseudo-data and reconstructed output arrays.",
    )
    parser.add_argument(
        "--phantom-type",
        choices=["sphere", "rods"],
        default="sphere",
        help="Digital phantom type: a single centered sphere or three rods.",
    )
    parser.add_argument(
        "--phantom-intensity",
        type=float,
        default=1.0,
        help="Peak intensity of the synthetic phantom used to create pseudo data.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use for the reconstructor, e.g. 'cpu' or 'cuda'.",
    )
    args = parser.parse_args()

    if args.replot_npz is not None:
        output_dir, saved_iters = replot_saved_reconstruction(args.replot_npz.resolve())
        print(f"Saved {len(saved_iters)} per-iteration plots to {output_dir}")
        return

    if args.campaign_dir is None:
        parser.error("--campaign-dir is required unless --replot-npz is used")

    campaign_dir = args.campaign_dir.resolve()
    if not campaign_dir.exists():
        raise FileNotFoundError(f"Campaign directory does not exist: {campaign_dir}")

    metadata = load_campaign_metadata(campaign_dir, args.resolution)
    num_heads = int(metadata.get("num_heads", 0) or 0)
    pixels_per_head = int(metadata.get("pixels_per_head", 0) or 0)
    head_slice = parse_head_slice(args.head_slice, num_heads)

    if num_heads <= 0 or pixels_per_head <= 0:
        raise ValueError(
            f"Metadata for {args.resolution} is missing a valid num_heads or pixels_per_head: "
            f"num_heads={num_heads}, pixels_per_head={pixels_per_head}"
        )

    system_matrix, metadata = build_system_matrix(
        campaign_dir, args.resolution, head_slice
    )
    grid_size = int(metadata.get("grid_size", 0))
    if grid_size <= 0:
        raise ValueError(
            f"Metadata for {args.resolution} does not contain a valid grid_size"
        )

    expected_rows = num_heads * pixels_per_head
    if head_slice.start != 0 or head_slice.stop != num_heads:
        expected_rows = (head_slice.stop - head_slice.start) * pixels_per_head
    if system_matrix.shape[0] != expected_rows:
        raise ValueError(
            f"Selected head set gives {system_matrix.shape[0]} rows, but expected {expected_rows} "
            f"for {num_heads} heads and {pixels_per_head} pixels/head."
        )

    num_voxels = grid_size**3
    expected_cols = num_voxels
    if system_matrix.shape[1] != expected_cols:
        raise ValueError(
            f"System matrix column count is {system_matrix.shape[1]}, expected {expected_cols} "
            f"for grid_size={grid_size}."
        )

    voxel_size_mm = float(
        metadata.get(
            "voxel_size_mm",
            2.0
            if args.resolution == "2mm"
            else 1.5
            if args.resolution == "1p5mm"
            else 1.0,
        )
    )
    if args.phantom_type == "sphere":
        true_image = make_center_sphere_phantom(
            grid_size,
            radius=max(6, grid_size // 10),
            intensity=args.phantom_intensity,
        )
        phantom_annotations = [
            f"sphere radius = {max(6, grid_size // 10)} voxels ({max(6, grid_size // 10) * voxel_size_mm:.1f} mm)"
        ]
    else:
        true_image, phantom_annotations = make_three_rods_phantom(
            grid_size,
            voxel_size_mm,
            intensity=args.phantom_intensity,
        )
    measured = pseudo_forward_project(system_matrix, true_image)

    if measured.size != system_matrix.shape[0]:
        raise ValueError(
            f"Forward projection length mismatch: got {measured.size}, expected {system_matrix.shape[0]}"
        )

    print(
        f"Selected SRM shape: {system_matrix.shape} = ({system_matrix.shape[0]}, {system_matrix.shape[1]})"
    )
    print(
        f"Expected 2mm full campaign shape: ({num_heads * pixels_per_head}, {num_voxels})"
    )
    print(f"True phantom voxels: {true_image.size}")
    print(f"Pseudo-measured projection length: {measured.size}")
    print(f"Selected head slice: {head_slice}")

    if args.iterations is None:
        args.iterations = build_iteration_schedule(args.max_iters, args.save_every)
    args.iterations = sorted(set(int(v) for v in args.iterations))

    recon = SparseMLEMReconstructor(
        system_matrix,
        measured,
        device=args.device,
        use_streaming=True,
        chunk_size=5_000_000,
    )

    output_dir = (
        args.output_dir.resolve() if args.output_dir else campaign_dir / "recon_test"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    estimates = recon.reconstruct(
        args.iterations, initial_estimate=np.ones(num_voxels, dtype=np.float32)
    )
    recon.destroy()

    final_estimate = (
        estimates[-1] if estimates else np.ones(num_voxels, dtype=np.float32)
    )

    estimate_history = np.asarray(
        [e.detach().cpu().numpy().reshape(-1) for e in estimates],
        dtype=np.float32,
    )
    np.savez_compressed(
        output_dir
        / f"pseudo_recon_{args.resolution}_{head_slice.start}_{head_slice.stop}.npz",
        true_image=true_image,
        measured=measured,
        final_estimate=final_estimate.detach().cpu().numpy()
        if hasattr(final_estimate, "detach")
        else final_estimate,
        iterations=np.asarray(args.iterations, dtype=np.int32),
        estimate_history=estimate_history,
        grid_size=np.asarray([grid_size], dtype=np.int32),
        resolution=np.asarray(args.resolution, dtype=str),
    )

    plot_dir = output_dir / "per_iteration"
    for iteration, estimate in zip(args.iterations, estimate_history):
        plot_iteration(
            true_image,
            estimate,
            iteration,
            grid_size,
            plot_dir,
            args.resolution,
            head_slice,
            args.phantom_type,
            phantom_annotations,
        )

    print(f"Saved pseudo-data test recon outputs to {output_dir}")
    print(f"Saved per-iteration plots to {plot_dir}")
    print(
        f"Final estimate size: {final_estimate.numel() if hasattr(final_estimate, 'numel') else final_estimate.size}"
    )
    print(f"Iterations written: {args.iterations}")


if __name__ == "__main__":
    main()
