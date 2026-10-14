"""Forward project a sparse digital phantom through every brain SPECT module, then
reconstruct it with MLEM from those noiseless projections.

There is no brain SPECT measurement, so the simulated projections are the data.
The projections are plotted one physical detector ring per row.

Run from dev/python:
    python fp_jaszczak_and_recon.py --phantom nema_triple_line_phantom.npz
"""

import argparse
from math import isqrt
from pathlib import Path
from typing import cast

import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from compact_gpu_mlem import CompactGpuMLEM, compact_matrix_bytes
from matplotlib.figure import Figure
from mlem_sparse.reconstruct import SparseMLEMReconstructor
from plot_expanse_brain_srm_202609 import SRMLoader, load_geometry_provenance
from scipy.sparse import csr_matrix, vstack
from volume_export import export_reconstruction, write_volume

SCRIPT_DIR = Path(__file__).resolve().parent
VOXEL_SIZE_MM = {"1mm": 1.0, "1p5mm": 1.5, "2mm": 2.0}


def matrix_shape(matrix: csr_matrix) -> tuple[int, int]:
    """(rows, columns); SciPy types .shape as optional."""
    return cast(tuple[int, int], matrix.shape)


def load_sparse_phantom(
    filename: str | Path, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Load a sparse phantom saved as indices/values/shape in a NumPy archive."""
    data = np.load(filename)
    return torch.sparse_coo_tensor(
        indices=torch.from_numpy(data["indices"]),
        values=torch.from_numpy(data["values"]),
        size=tuple(data["shape"]),
        device=device,
    ).coalesce()


def build_system_matrix(
    loader: SRMLoader, num_heads: int, max_workers: int
) -> csr_matrix:
    """Stack every module's SRM in module order, one detector image per block."""
    srms = loader.load_srms_parallel(range(num_heads), max_workers=max_workers)
    missing = sorted(set(range(num_heads)) - set(srms))
    if missing:
        raise ValueError(f"SRMs failed to load for heads {[i + 1 for i in missing]}")
    system_matrix = cast(
        csr_matrix,
        vstack([srms[i].astype(np.float32) for i in range(num_heads)], format="csr"),
    )
    print(
        f"System matrix: shape={system_matrix.shape}, "
        f"nnz={system_matrix.nnz:,}, dtype={system_matrix.dtype}"
    )
    return system_matrix


def forward_project(system_matrix: csr_matrix, phantom: torch.Tensor) -> np.ndarray:
    """Noiseless projections of the phantom, flattened in system-matrix row order."""
    phantom = phantom.coalesce().cpu()
    num_voxels = int(np.prod(phantom.shape))
    _, columns = matrix_shape(system_matrix)
    if columns != num_voxels:
        raise ValueError(
            f"System matrix columns {columns} do not match phantom "
            f"voxels {num_voxels} (shape {tuple(phantom.shape)})"
        )
    volume = np.zeros(num_voxels, dtype=np.float32)
    volume[np.ravel_multi_index(phantom.indices().numpy(), phantom.shape)] = (
        phantom.values().numpy()
    )
    return np.asarray(system_matrix @ volume, dtype=np.float32)


# Module order runs by azimuth starting at 90 degrees, so the plot's azimuth axis
# spans [90, 450): 0-90 degrees are drawn after 360
AZIMUTH_START_DEG = 90.0


def plot_projections_by_ring(
    projections: np.ndarray,
    ring_index: np.ndarray,
    ring_elevation_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    title: str,
) -> Figure:
    """One row per physical ring, highest elevation on top, each detector image
    centred on its module's azimuth so modules at the same azimuth line up across
    rings. Images are as wide as 90% of the closest module spacing in any ring and
    share one colour scale. Each image is drawn with the first SRM pixel axis
    horizontal; the pixel frame is not yet verified against the world frame."""
    num_modules, pixels_per_head = projections.shape
    side = isqrt(pixels_per_head)
    images = projections.reshape(num_modules, side, side)
    x_deg = AZIMUTH_START_DEG + (azimuth_deg - AZIMUTH_START_DEG) % 360
    rings = np.unique(ring_index)[::-1]
    closest = min(
        float(np.diff(np.sort(x_deg[ring_index == ring])).min())
        for ring in rings
        if np.sum(ring_index == ring) > 1
    )
    width = 0.9 * closest
    vmax = max(float(images.max()), float(np.finfo(np.float32).tiny))

    fig = Figure(figsize=(20, 1.2 * len(rings) + 1.5), layout="constrained")
    axes = fig.subplots(len(rings), 1, sharex=True, squeeze=False)[:, 0]
    for ax, ring in zip(axes, rings):
        for module in np.flatnonzero(ring_index == ring):
            x = x_deg[module]
            ax.imshow(
                images[module].T,
                origin="lower",
                cmap="hot",
                interpolation="none",
                extent=(x - width / 2, x + width / 2, 0, width),
                vmin=0,
                vmax=vmax,
            )
            ax.text(x, -0.08 * width, str(module + 1), ha="center", va="top", fontsize=7)
        ax.set_ylim(-0.3 * width, width)
        ax.set_yticks([])
        ax.set_ylabel(
            f"Ring {ring}\n{ring_elevation_deg[ring]:g}°", rotation=0, ha="right"
        )
        for spine in ("left", "right", "top"):
            ax.spines[spine].set_visible(False)
    ticks = np.arange(AZIMUTH_START_DEG, AZIMUTH_START_DEG + 361, 30)
    axes[-1].set_xticks(ticks, [f"{tick % 360:g}" for tick in ticks])
    axes[-1].set_xlim(AZIMUTH_START_DEG - width / 2, AZIMUTH_START_DEG + 360 + width / 2)
    axes[-1].set_xlabel("Module azimuth (degrees)")
    fig.colorbar(
        ScalarMappable(Normalize(0, vmax), "hot"),
        ax=axes,
        shrink=0.8,
        label="Projected counts",
    )
    fig.suptitle(title)
    return fig


def load_projections(path: Path, num_heads: int) -> np.ndarray:
    """Saved forward projections, flattened in system-matrix row order."""
    with np.load(path) as data:
        module_indices = data["module_indices"]
        projections = np.asarray(data["projections"], dtype=np.float32)
    if not np.array_equal(module_indices, np.arange(num_heads)):
        raise ValueError(f"{path} does not hold modules 0..{num_heads - 1} in order")
    return projections.reshape(-1)


def pick_device(requested: str | None, system_matrix: csr_matrix) -> torch.device:
    """The requested device, else CUDA when the whole matrix fits in free GPU
    memory (the reconstructor keeps it resident there), else CPU."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        rows, columns = matrix_shape(system_matrix)
        # int32 column indices + float32 values, plus a few voxel-length buffers
        needed = 8 * system_matrix.nnz + 4 * rows + 4 * 8 * columns
        free, _ = torch.cuda.mem_get_info()
        if needed < free:
            return torch.device("cuda")
        print(
            f"System matrix needs ~{needed / 2**30:.1f} GiB, GPU has "
            f"{free / 2**30:.1f} GiB free; reconstructing on CPU"
        )
    return torch.device("cpu")


def compact_gpu_fits(system_matrix: csr_matrix) -> bool:
    """Whether CompactGpuMLEM fits in free GPU memory: the compact matrix, a few
    voxel-length buffers and ~1.5 GiB of per-chunk temporaries."""
    if not torch.cuda.is_available():
        return False
    rows, columns = matrix_shape(system_matrix)
    needed = compact_matrix_bytes(system_matrix.nnz, rows) + 4 * 8 * columns + 1.5 * 2**30
    free, _ = torch.cuda.mem_get_info()
    print(f"Compact GPU MLEM needs ~{needed / 2**30:.1f} GiB, {free / 2**30:.1f} GiB free")
    return needed < free


def make_reconstructor(args, system_matrix: csr_matrix, measured: np.ndarray):
    """compact-gpu keeps a 1-byte-value copy of the matrix on the GPU (exact for
    integer counts <= 255); mlem-sparse is SparseMLEMReconstructor on --device."""
    engine = args.engine
    if engine == "auto":
        engine = "compact-gpu" if compact_gpu_fits(system_matrix) else "mlem-sparse"
    print(f"Reconstruction engine: {engine}")
    if engine == "compact-gpu":
        return CompactGpuMLEM(
            system_matrix, measured, min_sensitivity=args.min_sensitivity
        )
    if args.min_sensitivity > 0:
        raise ValueError("--min-sensitivity needs --engine compact-gpu")
    return SparseMLEMReconstructor(
        system_matrix,
        measured,
        device=pick_device(args.device, system_matrix),
        use_streaming=True,
        chunk_size=5_000_000,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--phantom",
        type=Path,
        default=SCRIPT_DIR / "jaszczak_phantom_hot_volume.npz",
        help="Sparse phantom npz (indices/values/shape).",
    )
    parser.add_argument("--subdir", default="brain_merged_123")
    parser.add_argument("--resolution", choices=["1mm", "1p5mm", "2mm"], default="1mm")
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50, 100, 150, 200],
        help="MLEM iterations at which to save the estimate.",
    )
    parser.add_argument("--load-workers", type=int, default=16)
    parser.add_argument(
        "--no-recon",
        action="store_true",
        help="Stop after saving and plotting the forward projections.",
    )
    parser.add_argument(
        "--recon-only",
        action="store_true",
        help="Reconstruct from the forward_projections.npz already in the output "
        "directory instead of projecting again.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for --engine mlem-sparse. Default: CUDA if the system matrix "
        "fits, else CPU.",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "compact-gpu", "mlem-sparse"],
        default="auto",
        help="auto: compact-gpu when it fits in GPU memory, else mlem-sparse.",
    )
    parser.add_argument(
        "--min-sensitivity",
        type=float,
        default=0.0,
        help="compact-gpu only: freeze voxels whose sensitivity is at or below this "
        "(0 freezes only zero-sensitivity voxels, as mlem-sparse does).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <script dir>/<phantom name>_brain_spect",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    phantom_name = args.phantom.stem
    output_dir = args.output_dir or SCRIPT_DIR / f"{phantom_name}_brain_spect"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = SRMLoader(
        modality="brain_spect", subdir=args.subdir, resolution=args.resolution
    )
    modules = load_geometry_provenance(loader)["analytic_geometry"]["modules"]
    num_heads = int(modules["count"])
    ring_index = np.asarray(modules["ring_index"])
    ring_elevation_deg = np.round(np.degrees(modules["ring_elevation_rad"]), 2)

    phantom = load_sparse_phantom(args.phantom)
    print(f"Phantom {args.phantom.name}: shape={tuple(phantom.shape)}")
    system_matrix = build_system_matrix(loader, num_heads, args.load_workers)
    projections_path = output_dir / "forward_projections.npz"
    if args.recon_only:
        measured = load_projections(projections_path, num_heads)
        print(f"Loaded forward projections from {projections_path}")
    else:
        measured = forward_project(system_matrix, phantom)
        projections = measured.reshape(num_heads, -1)
        np.savez_compressed(
            projections_path,
            module_indices=np.arange(num_heads, dtype=np.int32),
            ring_index=ring_index,
            projections=projections,
        )
        fig = plot_projections_by_ring(
            projections,
            ring_index,
            ring_elevation_deg,
            np.degrees(modules["azimuth_rad"]),
            f"Brain SPECT forward projections of {phantom_name} "
            f"({num_heads} modules, {args.resolution})",
        )
        fig.savefig(output_dir / "forward_projections_by_ring.png", dpi=200)
        print(f"Saved forward projections to {output_dir}")
        if args.no_recon:
            return

    # Reconstruction from the noiseless simulated projections, all modules. Start
    # at zero where MLEM never updates (no sensitivity, or at most
    # --min-sensitivity): a uniform start would leave those voxels at 1 (about
    # half the 210^3 grid).
    support = (
        np.asarray(system_matrix.sum(axis=0)).reshape(-1) > args.min_sensitivity
    )
    print(f"Reconstruction support: {support.sum():,} of {support.size:,} voxels")
    reconstructor = make_reconstructor(args, system_matrix, measured)
    estimates = reconstructor.reconstruct(
        args.iterations, initial_estimate=support.astype(np.float32)
    )
    reconstructor.destroy()

    reconstruction_path = output_dir / f"{phantom_name}_reconstruction.npz"
    np.savez_compressed(
        reconstruction_path,
        iterations=np.asarray(args.iterations, dtype=np.int32),
        estimates=np.stack(
            [estimate.detach().cpu().numpy().reshape(-1) for estimate in estimates]
        ),
        measured=measured,
        module_indices=np.arange(num_heads, dtype=np.int32),
        volume_shape=np.asarray(phantom.shape, dtype=np.int32),
    )
    print(f"Saved reconstruction to {reconstruction_path}")

    # NRRD copies for 3D Slicer: every saved iteration, plus the phantom itself
    voxel_mm = VOXEL_SIZE_MM[args.resolution]
    nrrd_paths = export_reconstruction(reconstruction_path, voxel_mm)
    write_volume(
        nrrd_paths[0].parent / f"{phantom_name}_phantom.nrrd",
        phantom.to_dense().numpy(),
        voxel_mm,
    )


if __name__ == "__main__":
    main()
