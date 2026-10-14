"""Write reconstruction volumes as NRRD for 3D Slicer.

Volumes in this project are indexed [x, y, z] on a grid centred on the origin. The
files are written so that the RAS coordinates Slicer shows are the simulation's
(x, y, z) in mm: ITK stores physical points in LPS, so the x and y directions are
flipped (direction diag(-1, -1, 1)) to cancel Slicer's LPS -> RAS conversion.
Whether simulation +z is the patient's superior direction is not established; the
axes are the simulation axes.

Convert an existing reconstruction (all saved iterations, or some):
    python volume_export.py -d <name>_reconstruction.npz -v 1.0 [-i 50 100]
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Direction cosines (row-major 3x3) mapping index axes to LPS space
SIM_TO_LPS_DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)


def write_volume(path: Path, volume_xyz: np.ndarray, voxel_mm: float) -> Path:
    """Write a [x, y, z] float volume centred on the origin as a gzip NRRD."""
    volume_xyz = np.asarray(volume_xyz, dtype=np.float32)
    if volume_xyz.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {volume_xyz.shape}")
    # SimpleITK reads arrays as [z, y, x]
    image = sitk.GetImageFromArray(np.ascontiguousarray(volume_xyz.transpose(2, 1, 0)))
    image.SetSpacing((voxel_mm,) * 3)
    image.SetDirection(SIM_TO_LPS_DIRECTION)
    # Centre of voxel [0, 0, 0] in simulation mm, expressed in LPS
    first = [(-n / 2 + 0.5) * voxel_mm for n in volume_xyz.shape]
    image.SetOrigin((-first[0], -first[1], first[2]))
    path = Path(path)
    sitk.WriteImage(image, str(path), useCompression=True)
    return path


def export_reconstruction(
    npz_path: Path,
    voxel_mm: float,
    output_dir: Path | None = None,
    iterations: list[int] | None = None,
) -> list[Path]:
    """Write <stem>_iteration_<N>.nrrd for each requested saved iteration."""
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        saved = data["iterations"].tolist()
        shape = tuple(int(n) for n in data["volume_shape"])
        estimates = data["estimates"]
    wanted = saved if iterations is None else iterations
    missing = sorted(set(wanted) - set(saved))
    if missing:
        raise ValueError(f"Iterations {missing} not saved; available: {saved}")
    output_dir = Path(output_dir or npz_path.parent / f"{npz_path.stem}_nrrd")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for iteration in wanted:
        volume = estimates[saved.index(iteration)].reshape(shape)
        paths.append(
            write_volume(
                output_dir / f"{npz_path.stem}_iteration_{iteration:03d}.nrrd",
                volume,
                voxel_mm,
            )
        )
    print(f"Wrote {len(paths)} NRRD volumes to {output_dir}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("-d", "--data", type=Path, required=True, help="Reconstruction npz.")
    parser.add_argument(
        "-v",
        "--voxel_size",
        type=float,
        default=1.0,
        help="Voxel size in mm (1.0, 1.5 or 2.0 to match the SRM resolution).",
    )
    parser.add_argument("-i", "--iterations", type=int, nargs="+", help="Default: all saved.")
    parser.add_argument("-o", "--output-dir", type=Path, help="Default: <npz stem>_nrrd/")
    args = parser.parse_args()
    export_reconstruction(args.data, args.voxel_size, args.output_dir, args.iterations)


if __name__ == "__main__":
    main()
