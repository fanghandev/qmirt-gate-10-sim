#!/usr/bin/env python3
"""Combine sparse SRM chunk outputs into one global 5D sparse file."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine sparse SRM chunks into one 5D sparse output."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing sparse NPZ chunk files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the combined sparse output is written.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Only combine chunks for this voxel size in mm.",
    )
    return parser.parse_args()


def read_scalar(data: np.lib.npyio.NpzFile, key: str, default: float | int) -> float | int:
    value = data.get(key, np.array([default]))
    return np.asarray(value).reshape(-1)[0].item()


def combine_crystal_npz_files(
    input_dir: Path,
    voxel_size: float | None = None,
) -> tuple[
    dict[tuple[int, int, int, int, int], int],
    dict[str, object],
    list[str],
    int,
    float,
    int,
    int,
]:
    combined: dict[tuple[int, int, int, int, int], int] = defaultdict(int)
    source_files: list[str] = []
    reference_metadata: dict[str, object] | None = None
    total_events = 0
    total_sim_time = 0.0
    total_files = 0
    total_root_members = 0

    npz_paths = sorted(
        path
        for path in input_dir.rglob("*.npz")
        if path.name.endswith(("_sparse_5d_srm.npz", "_sparse_5d_histograms.npz"))
        or path.name.startswith("sparse_5d_srm_")
    )
    if not npz_paths:
        raise FileNotFoundError(f"No sparse NPZ files found in {input_dir}")

    for npz_path in npz_paths:
        with np.load(npz_path, allow_pickle=False) as data:
            file_voxel_size = float(read_scalar(data, "voxel_size", 0.0))
            if voxel_size is not None and not np.isclose(
                file_voxel_size, voxel_size
            ):
                continue

            coords = np.asarray(data["coords"], dtype=np.int32)
            counts = np.asarray(data["counts"], dtype=np.int64)

            if coords.ndim != 2 or coords.shape[1] != 5:
                raise ValueError(f"Expected coords to have shape (N, 5), got {coords.shape!r}")
            if coords.shape[0] != counts.shape[0]:
                raise ValueError("coords and counts must have the same number of rows")

            for coord, count in zip(coords, counts):
                key = (
                    int(coord[0]),
                    int(coord[1]),
                    int(coord[2]),
                    int(coord[3]),
                    int(coord[4]),
                )
                combined[key] += int(count)

            file_metadata = {
                "hist_bins": int(read_scalar(data, "hist_bins", 80)),
                "hist_range": np.asarray(data.get("hist_range", np.array([-80.0, 80.0]))).reshape(-1).tolist(),
                "voxel_size": file_voxel_size,
                "fov_size": float(read_scalar(data, "fov_size", 0.0)),
                "energy_min": float(read_scalar(data, "energy_min", 0.0)),
                "energy_max": float(read_scalar(data, "energy_max", 0.0)),
            }
            if reference_metadata is None:
                reference_metadata = file_metadata
            elif file_metadata != reference_metadata:
                raise ValueError(
                    f"Incompatible SRM grid or energy metadata in {npz_path.name}: "
                    f"{file_metadata} != {reference_metadata}"
                )

            total_events += int(
                read_scalar(data, "simulated_events", read_scalar(data, "total_events", 0))
            )
            total_sim_time += float(read_scalar(data, "total_sim_time", 0.0))
            total_files += int(read_scalar(data, "file_count", 1))
            total_root_members += int(read_scalar(data, "processed_root_members", 0))

            source_files.append(str(npz_path))

    if reference_metadata is None:
        requested = (
            f" for voxel size {voxel_size:g} mm" if voxel_size is not None else ""
        )
        raise FileNotFoundError(f"No compatible sparse NPZ files found{requested}")
    return (
        combined,
        reference_metadata,
        source_files,
        total_events,
        total_sim_time,
        total_files,
        total_root_members,
    )

def save_combined_sparse_file(
    output_dir: Path,
    coord_map: dict[tuple[int, int, int, int, int], int],
    metadata: dict[str, object],
    total_events: int,
    total_sim_time: float,
    file_count: int,
    processed_root_members: int,
    source_files: list[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    coords = np.array(list(coord_map.keys()), dtype=np.int32)
    counts = np.array(list(coord_map.values()), dtype=np.int64)
    output_path = output_dir / "sparse_5d_histograms.npz"
    np.savez_compressed(
        output_path,
        coords=coords,
        counts=counts,
        simulated_events=np.array([total_events], dtype=np.int64),
        hist_bins=np.array([metadata.get("hist_bins", 80)], dtype=np.int32),
        hist_range=np.array(metadata.get("hist_range", [-80.0, 80.0]), dtype=np.float32),
        voxel_size=np.array([metadata.get("voxel_size", 0.0)], dtype=np.float32),
        fov_size=np.array([metadata.get("fov_size", 0.0)], dtype=np.float32),
        energy_min=np.array([metadata.get("energy_min", 0.0)], dtype=np.float32),
        energy_max=np.array([metadata.get("energy_max", 0.0)], dtype=np.float32),
        file_count=np.array([file_count], dtype=np.int64),
        total_events=np.array([total_events], dtype=np.int64),
        total_sim_time=np.array([total_sim_time], dtype=np.float64),
        processed_root_members=np.array([processed_root_members], dtype=np.int64),
        crystal_count=np.array([len(np.unique(coords[:, 0]))], dtype=np.int64),
        accumulated_counts=np.array([int(counts.sum())], dtype=np.int64),
        source_files=np.array(source_files, dtype=str),
    )
    return output_path


def main() -> int:
    args = parse_args()
    (
        combined,
        metadata,
        source_files,
        total_events,
        total_sim_time,
        total_files,
        total_root_members,
    ) = combine_crystal_npz_files(args.input_dir, args.voxel_size)

    output_path = save_combined_sparse_file(
        args.output_dir,
        combined,
        metadata,
        total_events,
        total_sim_time,
        total_files,
        total_root_members,
        source_files,
    )

    summary_path = args.output_dir / "combined_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "chunk_count": len(source_files),
                "file_count": total_files,
                "root_member_count": total_root_members,
                "total_events": total_events,
                "total_sim_time": total_sim_time,
                "crystal_count": len({coord[0] for coord in combined}),
                "outputs": [str(output_path)],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        f"Combined {len(source_files)} sparse chunks into {output_path}"
    )
    print(f"Total events: {total_events:,d}")
    print(f"Total simulation time: {total_sim_time:.2f} seconds")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
