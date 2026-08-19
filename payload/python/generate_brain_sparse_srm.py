#!/usr/bin/env python3
"""Convert one Gate ROOT chunk into sparse brain-SPECT SRM files."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import uproot

Coordinate = tuple[int, int, int, int, int]


def parse_resolutions(value: str) -> list[float]:
    resolutions = [float(item) for item in value.split(",") if item.strip()]
    if not resolutions or any(item <= 0 for item in resolutions):
        raise argparse.ArgumentTypeError("resolutions must be positive numbers")
    return resolutions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Gate brain-SPECT ROOT files into sparse SRMs."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resolutions-mm",
        type=parse_resolutions,
        default=[1.0, 1.5, 2.0],
        help="Comma-separated voxel sizes, for example 1,1.5,2.",
    )
    parser.add_argument(
        "--fov-size-mm",
        type=float,
        default=210.0,
        help="Cartesian reconstruction extent in mm; default is -105 to +105.",
    )
    parser.add_argument("--energy-min-kev", type=float, default=139.3)
    parser.add_argument("--energy-max-kev", type=float, default=140.7)
    parser.add_argument("--step-size", default="50 MB")
    parser.add_argument("--job-id", default="local")
    parser.add_argument("--task-id", default="0")
    parser.add_argument("--loop-id", default="0")
    return parser.parse_args()


def resolution_label(resolution_mm: float) -> str:
    text = f"{resolution_mm:g}".replace(".", "p")
    return f"{text}mm"


def detector_ids(tree_name: str, volume_ids: np.ndarray) -> tuple[int, np.ndarray]:
    match = re.search(r"Pixel_(\d+)_Singles", tree_name)
    if match is None:
        raise ValueError(f"Cannot determine crystal ID from tree name {tree_name!r}")
    crystal_id = int(match.group(1)) - 1

    pixel_ids = []
    for value in volume_ids:
        text = value.decode() if isinstance(value, bytes) else str(value)
        try:
            pixel_ids.append(int(text.rsplit("_", 1)[-1]))
        except ValueError as error:
            raise ValueError(f"Cannot parse pixel ID from {text!r}") from error
    return crystal_id, np.asarray(pixel_ids, dtype=np.int32)


def validate_grid(fov_size_mm: float, resolutions_mm: list[float]) -> dict[float, int]:
    if fov_size_mm <= 0:
        raise ValueError("fov_size_mm must be positive")
    grid_sizes = {}
    for resolution_mm in resolutions_mm:
        grid_size = fov_size_mm / resolution_mm
        if not np.isclose(grid_size, round(grid_size), rtol=0, atol=1e-6):
            raise ValueError(
                f"fov_size_mm={fov_size_mm} is not evenly divisible by "
                f"resolution={resolution_mm}"
            )
        grid_sizes[resolution_mm] = int(round(grid_size))
    return grid_sizes


def accumulate_batch(
    accumulators: dict[float, defaultdict[Coordinate, int]],
    resolutions_mm: list[float],
    grid_sizes: dict[float, int],
    crystal_id: int,
    pixel_ids: np.ndarray,
    event_x: np.ndarray,
    event_y: np.ndarray,
    event_z: np.ndarray,
    fov_size_mm: float,
) -> int:
    valid_events = 0
    half_size = fov_size_mm * 0.5
    for resolution_mm in resolutions_mm:
        grid_size = grid_sizes[resolution_mm]
        index_x = np.floor((event_x + half_size) / resolution_mm).astype(np.int32)
        index_y = np.floor((event_y + half_size) / resolution_mm).astype(np.int32)
        index_z = np.floor((event_z + half_size) / resolution_mm).astype(np.int32)
        valid_mask = (
            (index_x >= 0)
            & (index_x < grid_size)
            & (index_y >= 0)
            & (index_y < grid_size)
            & (index_z >= 0)
            & (index_z < grid_size)
            & (pixel_ids >= 0)
            & (pixel_ids < 625)
        )
        valid_events = max(valid_events, int(np.count_nonzero(valid_mask)))
        for pixel_id, x_bin, y_bin, z_bin in zip(
            pixel_ids[valid_mask],
            index_x[valid_mask],
            index_y[valid_mask],
            index_z[valid_mask],
        ):
            accumulators[resolution_mm][
                (crystal_id, int(pixel_id), int(x_bin), int(y_bin), int(z_bin))
            ] += 1
    return valid_events


def save_outputs(
    output_dir: Path,
    accumulators: dict[float, defaultdict[Coordinate, int]],
    resolutions_mm: list[float],
    grid_sizes: dict[float, int],
    args: argparse.Namespace,
    root_files: list[Path],
    raw_events: int,
    accepted_events: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = []
    for resolution_mm in resolutions_mm:
        entries = accumulators[resolution_mm]
        coords = np.asarray(list(entries), dtype=np.int32).reshape(-1, 5)
        counts = np.asarray(list(entries.values()), dtype=np.int64)
        output_path = output_dir / f"srm_{resolution_label(resolution_mm)}.npz"
        np.savez_compressed(
            output_path,
            coords=coords,
            counts=counts,
            voxel_size_mm=np.asarray([resolution_mm], dtype=np.float32),
            grid_size=np.asarray([grid_sizes[resolution_mm]], dtype=np.int32),
            hist_range=np.asarray(
                [-args.fov_size_mm / 2, args.fov_size_mm / 2], dtype=np.float32
            ),
            energy_min_kev=np.asarray([args.energy_min_kev], dtype=np.float32),
            energy_max_kev=np.asarray([args.energy_max_kev], dtype=np.float32),
            raw_events=np.asarray([raw_events], dtype=np.int64),
            accepted_events=np.asarray([accepted_events], dtype=np.int64),
        )
        output_names.append(output_path.name)

    metadata = {
        "job_id": str(args.job_id),
        "task_id": str(args.task_id),
        "loop_id": str(args.loop_id),
        "resolutions_mm": resolutions_mm,
        "fov_size_mm": args.fov_size_mm,
        "energy_min_kev": args.energy_min_kev,
        "energy_max_kev": args.energy_max_kev,
        "raw_events": raw_events,
        "accepted_events": accepted_events,
        "root_files": [str(path) for path in root_files],
        "outputs": output_names,
    }
    (output_dir / "srm_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    resolutions_mm = list(dict.fromkeys(args.resolutions_mm))
    grid_sizes = validate_grid(args.fov_size_mm, resolutions_mm)
    root_files = sorted(args.input_dir.rglob("pixel_singles_*.root"))
    if not root_files:
        raise FileNotFoundError(
            f"No pixel_singles_*.root files found in {args.input_dir}"
        )

    accumulators = {resolution_mm: defaultdict(int) for resolution_mm in resolutions_mm}
    branches = [
        "TotalEnergyDeposit",
        "EventPosition_X",
        "EventPosition_Y",
        "EventPosition_Z",
        "PreStepUniqueVolumeID",
    ]
    raw_events = 0
    accepted_events = 0
    energy_min_mev = args.energy_min_kev / 1000.0
    energy_max_mev = args.energy_max_kev / 1000.0

    for root_path in root_files:
        with uproot.open(root_path) as root_file:
            tree_names = [
                name
                for name, class_name in root_file.classnames(cycle=False).items()
                if class_name == "TTree" and "Pixel_" in name and "_Singles" in name
            ]
            for tree_name in tree_names:
                tree = root_file[tree_name]
                for data in tree.iterate(
                    branches, library="np", step_size=args.step_size
                ):
                    raw_events += len(data["TotalEnergyDeposit"])
                    energy = np.asarray(data["TotalEnergyDeposit"])
                    energy_mask = (energy >= energy_min_mev) & (
                        energy <= energy_max_mev
                    )
                    if not np.any(energy_mask):
                        continue
                    crystal_id, pixel_ids = detector_ids(
                        tree_name,
                        np.asarray(data["PreStepUniqueVolumeID"])[energy_mask],
                    )
                    event_x = np.asarray(data["EventPosition_X"])[energy_mask]
                    event_y = np.asarray(data["EventPosition_Y"])[energy_mask]
                    event_z = np.asarray(data["EventPosition_Z"])[energy_mask]
                    accepted_events += accumulate_batch(
                        accumulators,
                        resolutions_mm,
                        grid_sizes,
                        crystal_id,
                        pixel_ids,
                        event_x,
                        event_y,
                        event_z,
                        args.fov_size_mm,
                    )

    save_outputs(
        args.output_dir,
        accumulators,
        resolutions_mm,
        grid_sizes,
        args,
        root_files,
        raw_events,
        accepted_events,
    )
    print(
        f"Processed {len(root_files)} ROOT files, {raw_events} raw events, "
        f"and wrote {len(resolutions_mm)} sparse SRMs to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
