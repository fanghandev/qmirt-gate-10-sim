#!/usr/bin/env python3
"""Process a slice of OSPool files into one sparse SRM NPZ.

This worker reads a shared file list, processes a contiguous slice of entries,
and writes one sparse NPZ containing 5D sparse coordinates as
``(CrystalID, PixelID, x_bin, y_bin, z_bin)`` plus the associated counts and
metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import uproot

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a file-list slice into sparse per-crystal SRM chunks."
    )
    parser.add_argument(
        "--input-list",
        type=Path,
        required=True,
        help="Text file containing one OSPool tar archive path per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the sparse NPZ output will be written.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start line index in the file list, inclusive.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End line index in the file list, exclusive.",
    )
    parser.add_argument(
        "--job-tag",
        type=str,
        default=None,
        help="Optional job tag used to group chunk outputs.",
    )
    parser.add_argument(
        "--energy-min",
        type=float,
        default=0.14 * 0.995,
        help="Lower energy threshold in MeV.",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=0.14 * 1.005,
        help="Upper energy threshold in MeV.",
    )
    parser.add_argument(
        "--fov-size",
        type=float,
        default=150.0,
        help="Cubic FOV size in mm when --voxel-size is used.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Single voxel edge length in mm. Derives a centered grid from --fov-size.",
    )
    parser.add_argument(
        "--voxel-sizes",
        type=float,
        nargs="+",
        default=None,
        help="Multiple voxel edge lengths in mm to accumulate in one pass.",
    )
    parser.add_argument(
        "--hist-min",
        type=float,
        default=-105.0,
        help="Lower bound for the histogram window in mm.",
    )
    parser.add_argument(
        "--hist-max",
        type=float,
        default=105.0,
        help="Upper bound for the histogram window in mm.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=140,
        help="Number of bins per axis.",
    )
    return parser.parse_args()


def load_file_paths(
    input_list: Path, start_index: int, end_index: int | None
) -> list[Path]:
    file_paths = [
        Path(line.strip())
        for line in input_list.read_text().splitlines()
        if line.strip()
    ]
    return file_paths[start_index:end_index]


def resolve_histogram_grid(
    args: argparse.Namespace, voxel_size: float | None = None
) -> tuple[float, float, int, float]:
    selected_voxel_size = args.voxel_size if voxel_size is None else voxel_size
    if selected_voxel_size is None:
        hist_min = args.hist_min
        hist_max = args.hist_max
        hist_bins = args.hist_bins
    else:
        if args.fov_size <= 0 or selected_voxel_size <= 0:
            raise ValueError("--fov-size and --voxel-size must be positive")
        bins = args.fov_size / selected_voxel_size
        hist_bins = round(bins)
        if not np.isclose(bins, hist_bins):
            raise ValueError("--fov-size must be divisible by --voxel-size")
        hist_min = -args.fov_size / 2.0
        hist_max = args.fov_size / 2.0

    if hist_max <= hist_min or hist_bins <= 0:
        raise ValueError("Histogram range and bin count must be valid")
    return hist_min, hist_max, hist_bins, (hist_max - hist_min) / hist_bins


def parse_stats_member(member_file) -> tuple[int, float]:
    text_content = member_file.read().decode("utf-8")
    stats_dict = json.loads(text_content)
    raw_events = float(stats_dict["events"]["value"])
    if not math.isfinite(raw_events) or raw_events < 0 or not raw_events.is_integer():
        raise ValueError(f"Invalid simulated event count in stats member: {raw_events!r}")
    total_events = int(raw_events)
    total_sim_time = float(stats_dict["sim_stop_time"]["value"]) - float(
        stats_dict["sim_start_time"]["value"]
    )
    return total_events, total_sim_time


def extract_crystal_and_pixel_ids(
    unique_volume_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    crystal_ids = []
    pixel_ids = []
    for value in unique_volume_ids:
        if isinstance(value, bytes):
            text = value.decode("utf-8")
        else:
            text = str(value)

        parts = text.split("_")
        if len(parts) < 2:
            raise ValueError(f"Could not parse crystal/pixel IDs from {text!r}")

        crystal_ids.append(int(parts[1])-1)  # Convert to 0-based index
        pixel_ids.append(int(parts[-1]))

    return (
        np.asarray(crystal_ids, dtype=np.int32),
        np.asarray(pixel_ids, dtype=np.int32),
    )


def accumulate_sparse(
    sparse_store: defaultdict[tuple[int, int, int, int, int], int],
    crystal_ids: np.ndarray,
    pixel_ids: np.ndarray,
    x_bin: np.ndarray,
    y_bin: np.ndarray,
    z_bin: np.ndarray,
) -> None:
    coords = np.column_stack((crystal_ids, pixel_ids, x_bin, y_bin, z_bin))
    unique_coords, counts = np.unique(coords, axis=0, return_counts=True)
    for coord, count in zip(unique_coords, counts):
        coord_key = (
            int(coord[0]),
            int(coord[1]),
            int(coord[2]),
            int(coord[3]),
            int(coord[4]),
        )
        sparse_store[coord_key] += int(count)


def main() -> int:
    args = parse_args()
    if args.voxel_size is not None and args.voxel_sizes is not None:
        raise ValueError("Use either --voxel-size or --voxel-sizes, not both")

    requested_voxel_sizes = args.voxel_sizes
    if requested_voxel_sizes is None:
        requested_voxel_sizes = [args.voxel_size]

    file_paths = load_file_paths(args.input_list, args.start_index, args.end_index)
    if not file_paths:
        raise FileNotFoundError("The selected file-list slice is empty.")

    grid_specs = []
    for voxel_size in requested_voxel_sizes:
        hist_min, hist_max, hist_bins, hist_width = resolve_histogram_grid(
            args, voxel_size
        )
        label = (
            f"{voxel_size:g}mm"
            if voxel_size is not None
            else ""
        )
        grid_specs.append(
            {
                "label": label,
                "hist_min": hist_min,
                "hist_max": hist_max,
                "hist_bins": hist_bins,
                "hist_width": hist_width,
                "sparse_store": defaultdict(int),
            }
        )

    total_events = 0
    total_sim_time = 0.0
    processed_files = 0
    processed_members = 0
    stats_member_count = 0

    job_tag = (
        args.job_tag
        or f"chunk_{args.start_index:05d}_{args.end_index if args.end_index is not None else 'end'}"
    )
    branches = [
        "TotalEnergyDeposit",
        "EventPosition_X",
        "EventPosition_Y",
        "EventPosition_Z",
        "PreStepUniqueVolumeID",
    ]

    for archive_path in file_paths:
        if not archive_path.exists():
            print(f"Warning: missing archive, skipping -> {archive_path}")
            continue

        processed_files += 1
        archive_stats_count = 0
        with (
            open(archive_path, "rb") as file_obj,
            tarfile.open(fileobj=file_obj, mode="r:gz") as tar,
        ):
            for member in tar:
                if not member.isfile():
                    continue

                member_name = member.name
                if Path(member_name).name.endswith("_sim_stats.txt"):
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        archive_stats_count += 1
                        if archive_stats_count > 1:
                            raise ValueError(
                                f"Multiple GATE stats members found in {archive_path}"
                            )
                        events, sim_time = parse_stats_member(extracted)
                        total_events += events
                        total_sim_time += sim_time
                        stats_member_count += 1
                    continue

                if not member_name.endswith(".root"):
                    continue

                extracted_root_file = tar.extractfile(member)
                if extracted_root_file is None:
                    continue

                processed_members += 1
                with uproot.open(extracted_root_file) as root_file:
                    tree_names = [
                        name
                        for name, class_name in root_file.classnames(
                            cycle=False
                        ).items()
                        if class_name == "TTree"
                    ]

                    for tree_name in tree_names:
                        tree = root_file[tree_name]
                        if tree.num_entries == 0:
                            continue

                        data = tree.arrays(branches, library="np")
                        energy = np.asarray(data["TotalEnergyDeposit"])
                        energy_mask = (energy >= args.energy_min) & (
                            energy <= args.energy_max
                        )
                        if not np.any(energy_mask):
                            continue

                        event_x = np.asarray(data["EventPosition_X"])[energy_mask]
                        event_y = np.asarray(data["EventPosition_Y"])[energy_mask]
                        event_z = np.asarray(data["EventPosition_Z"])[energy_mask]
                        crystal_ids, pixel_ids = extract_crystal_and_pixel_ids(
                            np.asarray(data["PreStepUniqueVolumeID"])
                        )
                        crystal_ids = crystal_ids[energy_mask]
                        pixel_ids = pixel_ids[energy_mask]

                        valid_mask = (
                            (event_x >= hist_min)
                            & (event_x < hist_max)
                            & (event_y >= hist_min)
                            & (event_y < hist_max)
                            & (event_z >= hist_min)
                            & (event_z < hist_max)
                        )
                        if not np.any(valid_mask):
                            continue

                        event_x = event_x[valid_mask]
                        event_y = event_y[valid_mask]
                        event_z = event_z[valid_mask]
                        crystal_ids = crystal_ids[valid_mask]
                        pixel_ids = pixel_ids[valid_mask]

                        for grid in grid_specs:
                            grid_hist_min = grid["hist_min"]
                            grid_hist_width = grid["hist_width"]
                            x_bin = np.floor(
                                (event_x - grid_hist_min) / grid_hist_width
                            ).astype(np.int32)
                            y_bin = np.floor(
                                (event_y - grid_hist_min) / grid_hist_width
                            ).astype(np.int32)
                            z_bin = np.floor(
                                (event_z - grid_hist_min) / grid_hist_width
                            ).astype(np.int32)

                            accumulate_sparse(
                                grid["sparse_store"],
                                crystal_ids,
                                pixel_ids,
                                x_bin,
                                y_bin,
                                z_bin,
                            )

        if archive_stats_count != 1:
            raise ValueError(
                f"Expected exactly one *_sim_stats.txt member in {archive_path}, "
                f"found {archive_stats_count}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for grid in grid_specs:
        coords = np.array(list(grid["sparse_store"].keys()), dtype=np.int32)
        counts = np.array(list(grid["sparse_store"].values()), dtype=np.int64)
        label_suffix = f"_{grid['label']}" if grid["label"] else ""
        output_path = args.output_dir / (
            f"{job_tag}{label_suffix}_sparse_5d_srm.npz"
        )
        np.savez_compressed(
            output_path,
            coords=coords,
            counts=counts,
            simulated_events=np.array([total_events], dtype=np.int64),
            hist_bins=np.array([grid["hist_bins"]], dtype=np.int32),
            hist_range=np.array(
                [grid["hist_min"], grid["hist_max"]], dtype=np.float32
            ),
            voxel_size=np.array([grid["hist_width"]], dtype=np.float32),
            fov_size=np.array(
                [grid["hist_max"] - grid["hist_min"]], dtype=np.float32
            ),
            energy_min=np.array([args.energy_min], dtype=np.float32),
            energy_max=np.array([args.energy_max], dtype=np.float32),
            file_count=np.array([processed_files], dtype=np.int64),
            total_events=np.array([total_events], dtype=np.int64),
            total_sim_time=np.array([total_sim_time], dtype=np.float64),
            processed_root_members=np.array([processed_members], dtype=np.int64),
            stats_member_count=np.array([stats_member_count], dtype=np.int64),
            job_tag=np.array([job_tag]),
            accumulated_counts=np.array([int(counts.sum())], dtype=np.int64),
        )
        output_paths.append(output_path)

    print(
        f"Processed {processed_files} archives, {processed_members} ROOT members, "
        f"and wrote {len(output_paths)} SRM grids: "
        + ", ".join(str(path) for path in output_paths)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
