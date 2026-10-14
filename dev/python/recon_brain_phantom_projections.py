"""MLEM reconstruction of a merged brain SPECT phantom acquisition
(merge_phantom_listmode.py, schema qmirt.phantom_projections/1) with the Gate SRM.

The SRM is loaded and put on the GPU once (CompactGpuMLEM); each data set below swaps
only the measured vector:
    in_fov_primary  main window, decays inside the SRM FOV, no scatter in the phantom
                    (what the SRM models, apart from attenuation)
    in_fov_total    main window, decays inside the FOV, phantom scatter included
    full_total      main window, the whole acquisition (in + out of the FOV)
    full_tew        full_total minus the triple-energy-window scatter estimate,
                    (L / wL + U / wU) * wM / 2, clipped at 0

The SRM was simulated in air. With --attenuation-factors (attenuation_factors.py: (heads,
voxels) transmissions from each voxel to each module's pinhole) every module's SRM block is
multiplied by its factors, which models the phantom's attenuation (attenuation correction).
The reference
volume is the phantom's relative activity on the reconstruction grid
(phantom_models.activity_volume).

--engine compact (default) builds the whole matrix and keeps it on the GPU (CompactGpuMLEM; the
2 mm SRM). --engine streaming (StreamingCompactMLEM) loads it module by module into pinned host
memory and streams what does not fit (the 1.5 mm and 1 mm SRMs); the air sensitivity for the
support is cached as brain_phantom_recon/sensitivity_<subdir>_<resolution>.npy.

Run from dev/python:
    python recon_brain_phantom_projections.py \
        --projections /data/fanghan/opengate_sim/data/brain_phantom/ospool/<campaign>/merged/projections.npz \
        --spec mesh50_head.json
"""

import argparse
import json
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from compact_gpu_mlem import CompactGpuMLEM, StreamingCompactMLEM
from fp_jaszczak_and_recon import VOXEL_SIZE_MM, build_system_matrix
from matplotlib.figure import Figure
from plot_expanse_brain_srm_202609 import SRMLoader
from volume_export import export_reconstruction, write_volume

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "payload" / "python"))
import phantom_models  # noqa: E402

NUM_HEADS = 73
DATASETS = ("in_fov_primary", "in_fov_total", "full_total", "full_tew")


def measured_sets(data, meta: dict) -> dict[str, np.ndarray]:
    """Every data set as a float32 vector in system-matrix row order (head-major)."""
    def window(name: str, fov: str, part: str) -> np.ndarray:
        return np.asarray(data[f"windows_{name}_{fov}_fov_{part}"], dtype=np.float64)

    def full(name: str) -> np.ndarray:
        return window(name, "in", "total") + window(name, "out", "total")

    widths = {k: hi - lo for k, (lo, hi) in meta["windows_kev"].items()}
    scatter = (full("tew_lower") / widths["tew_lower"]
               + full("tew_upper") / widths["tew_upper"]) * widths["main"] / 2
    sets = {
        "in_fov_primary": window("main", "in", "primary"),
        "in_fov_total": window("main", "in", "total"),
        "full_total": full("main"),
        "full_tew": np.clip(full("main") - scatter, 0, None),
    }
    for name, values in sets.items():
        if values.shape != (NUM_HEADS, 625):
            raise ValueError(f"{name}: shape {values.shape}, expected ({NUM_HEADS}, 625)")
    return {name: values.astype(np.float32).reshape(-1) for name, values in sets.items()}


def apply_attenuation(system_matrix, factors: np.ndarray) -> None:
    """In place: rows of head h (625 per head, head-major) times factors[h, column]."""
    rows_per_head = system_matrix.shape[0] // NUM_HEADS
    if factors.shape != (NUM_HEADS, system_matrix.shape[1]):
        raise ValueError(f"factors {factors.shape}, expected ({NUM_HEADS}, {system_matrix.shape[1]})")
    for h in range(NUM_HEADS):
        a, b = system_matrix.indptr[h * rows_per_head], system_matrix.indptr[(h + 1) * rows_per_head]
        system_matrix.data[a:b] *= np.asarray(factors[h])[system_matrix.indices[a:b]]


def srm_blocks(loader: SRMLoader, factors=None, workers: int = 6):
    """(row0, float32 CSR block) per module in order, at most `workers` modules loaded ahead;
    each block times its attenuation factors when given."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = deque(pool.submit(loader.load_single_srm, h) for h in range(min(workers, NUM_HEADS)))
        queued = len(futures)
        for k in range(NUM_HEADS):
            block = futures.popleft().result().astype(np.float32)
            if queued < NUM_HEADS:
                futures.append(pool.submit(loader.load_single_srm, queued))
                queued += 1
            if factors is not None:
                block.data *= np.asarray(factors[k])[block.indices]
            yield k * block.shape[0], block


def air_sensitivity(loader: SRMLoader, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    sens = None
    for _, block in srm_blocks(loader):
        s = np.bincount(block.indices, weights=block.data, minlength=block.shape[1])
        sens = s if sens is None else sens + s
    np.save(cache, sens.astype(np.float32))
    return sens.astype(np.float32)


def plot_slices(volumes: dict[str, np.ndarray], voxel_mm: float, title: str) -> Figure:
    """Central axial (z), coronal (y) and sagittal (x) slices, one row per volume,
    each row scaled to its own maximum."""
    names = list(volumes)
    fig = Figure(figsize=(9, 3 * len(names)), layout="constrained")
    axes = fig.subplots(len(names), 3, squeeze=False)
    n = next(iter(volumes.values())).shape[0]
    half = n * voxel_mm / 2
    extent = (-half, half, -half, half)
    for row, name in enumerate(names):
        volume = volumes[name]
        vmax = float(np.percentile(volume[volume > 0], 99.9)) if np.any(volume > 0) else 1.0
        panels = (
            ("axial z = 0", volume[:, :, n // 2].T, "x (mm)", "y (mm)"),
            ("coronal y = 0", volume[:, n // 2, :].T, "x (mm)", "z (mm)"),
            ("sagittal x = 0", volume[n // 2, :, :].T, "y (mm)", "z (mm)"),
        )
        for ax, (label, image, xlabel, ylabel) in zip(axes[row], panels):
            ax.imshow(image, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=vmax)
            ax.set_title(f"{name}: {label}", fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--projections", type=Path, required=True, help="merged projections.npz")
    parser.add_argument("--spec", type=Path, required=True, help="phantom spec .json of the campaign")
    parser.add_argument("--subdir", default="brain_288mm_csg_102t_20261005T211647Z_merged_p01-10")
    parser.add_argument("--resolution", choices=["1mm", "1p5mm", "2mm"], default="2mm")
    parser.add_argument("--iterations", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--min-sensitivity", type=float, default=0.0,
                        help="on the air SRM's sensitivity (the same support with or without attenuation)")
    parser.add_argument("--attenuation-factors", type=Path, default=None, help="(heads, voxels) .npy")
    parser.add_argument("--load-workers", type=int, default=16)
    parser.add_argument("--engine", choices=["compact", "streaming"], default="compact")
    parser.add_argument("--gpu-budget-gib", type=float, default=7.0, help="streaming: matrix bytes kept on the GPU")
    parser.add_argument("--plot-iteration", type=int, default=None, help="Default: the last saved")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: <script dir>/brain_phantom_recon/<campaign>_<resolution>")
    return parser.parse_args()


def main():
    args = parse_args()
    campaign = args.projections.parent.parent.name
    output_dir = args.output_dir or SCRIPT_DIR / "brain_phantom_recon" / f"{campaign}_{args.resolution}"
    output_dir.mkdir(parents=True, exist_ok=True)
    voxel_mm = VOXEL_SIZE_MM[args.resolution]
    plot_iteration = args.plot_iteration or max(args.iterations)

    with np.load(args.projections) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("schema") != "qmirt.phantom_projections/1":
            raise ValueError(f"{args.projections}: schema {meta.get('schema')}")
        measured = measured_sets(data, meta)
    for name in args.datasets:
        print(f"{name}: {measured[name].sum():,.0f} counts")

    loader = SRMLoader(modality="brain_spect", subdir=args.subdir, resolution=args.resolution)
    if args.engine == "compact":
        system_matrix = build_system_matrix(loader, NUM_HEADS, args.load_workers)
        columns = system_matrix.shape[1]
    else:
        sens_air = air_sensitivity(loader, SCRIPT_DIR / "brain_phantom_recon" / f"sensitivity_{args.subdir}_{args.resolution}.npy")
        columns = sens_air.size
    n = round(columns ** (1 / 3))
    if n**3 != columns:
        raise ValueError(f"SRM has {columns} columns, not a cube")

    spec_dict = json.loads(args.spec.read_text())
    spec = phantom_models.Spec.from_dict(spec_dict)
    reference = phantom_models.activity_volume(spec, n, voxel_mm)
    write_volume(output_dir / "reference_activity.nrrd", reference, voxel_mm)

    factors = np.load(args.attenuation_factors, mmap_mode="r") if args.attenuation_factors else None
    if args.engine == "compact":
        support = np.asarray(system_matrix.sum(axis=0)).reshape(-1) > args.min_sensitivity
        print(f"Reconstruction support: {support.sum():,} of {support.size:,} voxels")
        if factors is not None:
            apply_attenuation(system_matrix, factors)
            print(f"Attenuation factors applied from {args.attenuation_factors}")
        # freeze by the support (air SRM), not by the attenuated sensitivity
        reconstructor = CompactGpuMLEM(system_matrix, measured[args.datasets[0]], min_sensitivity=0.0)
        reconstructor.update_mask &= reconstructor.update_mask.new_tensor(support)
        del system_matrix
    else:
        support = sens_air > args.min_sensitivity
        print(f"Reconstruction support: {support.sum():,} of {support.size:,} voxels")
        reconstructor = StreamingCompactMLEM(srm_blocks(loader, factors), measured[args.datasets[0]], support=support,
                                             gpu_budget_bytes=args.gpu_budget_gib * 2**30)

    shown = {"reference": reference}
    for name in args.datasets:
        reconstructor.measured.copy_(
            reconstructor.measured.new_tensor(measured[name])
        )
        estimates = reconstructor.reconstruct(args.iterations, initial_estimate=support.astype(np.float32))
        path = output_dir / f"{name}_reconstruction.npz"
        np.savez_compressed(
            path,
            iterations=np.asarray(args.iterations, dtype=np.int32),
            estimates=np.stack([e.numpy().reshape(-1) for e in estimates]),
            measured=measured[name],
            module_indices=np.arange(NUM_HEADS, dtype=np.int32),
            volume_shape=np.asarray((n, n, n), dtype=np.int32),
            projections=str(args.projections),
            srm=str(loader.datadir),
            attenuation_factors=str(args.attenuation_factors or ""),
            spec=json.dumps(spec_dict),
        )
        export_reconstruction(path, voxel_mm)
        shown[name] = estimates[args.iterations.index(plot_iteration)].numpy().reshape(n, n, n)
        print(f"Saved {path}")
    reconstructor.destroy()

    fig = plot_slices(shown, voxel_mm,
                      f"{campaign}: MLEM {plot_iteration} iterations, {args.resolution} SRM, "
                      + ("attenuation corrected" if args.attenuation_factors else "no attenuation correction"))
    fig.savefig(output_dir / f"slices_iteration_{plot_iteration:03d}.png", dpi=150)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
