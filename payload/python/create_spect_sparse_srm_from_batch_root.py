#!/usr/bin/env python3
"""Convert a batch of Gate SPECT ROOT files into sparse SRM coordinate files."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import polars as pl
import uproot

COORDINATE_COLUMNS = ["crystal", "pixel", "x", "y", "z"]
TREE_PATTERN = re.compile(r"^Pixel_(\d+)_Singles$")
BRANCHES = [
    "PreStepUniqueVolumeID",
    "TotalEnergyDeposit",
    "EventPosition_X",
    "EventPosition_Y",
    "EventPosition_Z",
]


def parse_resolutions(value: str) -> list[float]:
    try:
        resolutions = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "resolutions must be comma-separated numbers"
        ) from error
    if not resolutions or any(
        not np.isfinite(item) or item <= 0 for item in resolutions
    ):
        raise argparse.ArgumentTypeError("resolutions must be positive finite numbers")
    return resolutions


def get_parsed_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Gate SPECT ROOT files into sparse SRMs."
    )
    parser.add_argument("-i", "--input-dir", type=Path, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
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
    parser.add_argument(
        "--pixels-per-head",
        type=int,
        default=625,
        help="Detector pixels per head; singles outside this range are dropped.",
    )
    parser.add_argument(
        "--output-stem",
        default="srm",
        help="Prefix for output files, e.g. 'srm' creates srm_1mm.npz.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write empty SRMs instead of failing when no ROOT files are found.",
    )
    parser.add_argument("--photopeak-kev", type=float, default=140.0)
    parser.add_argument(
        "--energy-window-percent",
        type=float,
        default=20.0,
        help="Total window width as a percentage of --photopeak-kev.",
    )
    parser.add_argument("--energy-min-kev", type=float, default=None)
    parser.add_argument("--energy-max-kev", type=float, default=None)
    parser.add_argument(
        "--step-size",
        default="50 MB",
        help="Maximum uncompressed data read from one ROOT tree at a time.",
    )
    parser.add_argument("--job-id", default="local")
    parser.add_argument("--task-id", default="0")
    parser.add_argument("--loop-id", default="0")
    parser.add_argument("--print-trees-only", action="store_true")
    parser.add_argument("--print-branch-names", action="store_true")
    args = parser.parse_args()

    if not np.isfinite(args.photopeak_kev) or args.photopeak_kev <= 0:
        parser.error("--photopeak-kev must be positive and finite")
    if not np.isfinite(args.energy_window_percent) or args.energy_window_percent <= 0:
        parser.error("--energy-window-percent must be positive and finite")
    half_width_kev = args.photopeak_kev * args.energy_window_percent / 200.0
    if args.energy_min_kev is None:
        args.energy_min_kev = args.photopeak_kev - half_width_kev
    if args.energy_max_kev is None:
        args.energy_max_kev = args.photopeak_kev + half_width_kev
    if not np.isfinite(args.energy_min_kev) or not np.isfinite(args.energy_max_kev):
        parser.error("energy bounds must be finite")
    if args.energy_min_kev < 0:
        parser.error("--energy-min-kev must be non-negative")
    if args.energy_min_kev >= args.energy_max_kev:
        parser.error("--energy-min-kev must be below --energy-max-kev")
    if args.pixels_per_head <= 0:
        parser.error("--pixels-per-head must be positive")
    return args


def iter_root_files(directory: Path) -> Iterator[Path]:
    """Yield matching ROOT files recursively in deterministic order."""
    yield from sorted(directory.rglob("pixel_singles_*.root"))


def print_root_file_contents(root_file: Any, print_branch_names: bool = False) -> None:
    for key, obj_type in root_file.classnames(cycle=False).items():
        obj = root_file[key]
        branch_names = list(obj.keys()) if hasattr(obj, "keys") else []
        print(f"Found object: {key} (Type: {obj_type}), {len(branch_names)} branches")
        if print_branch_names:
            print(textwrap.fill(f"[{', '.join(branch_names)}]", width=160))


def resolution_label(resolution_mm: float) -> str:
    return f"{resolution_mm:g}".replace(".", "p") + "mm"


def validate_grid(fov_size_mm: float, resolutions_mm: list[float]) -> dict[float, int]:
    if not np.isfinite(fov_size_mm) or fov_size_mm <= 0:
        raise ValueError("fov_size_mm must be positive and finite")
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


def get_tree_names(root_file: Any) -> list[tuple[str, int]]:
    trees = []
    for name, class_name in root_file.classnames(cycle=False).items():
        match = TREE_PATTERN.fullmatch(name)
        if class_name == "TTree" and match is not None:
            trees.append((name, int(match.group(1)) - 1))
    return sorted(trees, key=lambda item: item[1])


def histogram_chunk(
    data: dict[str, np.ndarray],
    crystal_id: int,
    resolutions_mm: list[float],
    grid_sizes: dict[float, int],
    fov_size_mm: float,
    pixels_per_head: int,
    energy_min_mev: float,
    energy_max_mev: float,
) -> tuple[dict[float, pl.DataFrame], int]:
    """Filter and histogram one Uproot chunk without returning event positions."""
    volume_ids = np.asarray(data["PreStepUniqueVolumeID"])
    if volume_ids.dtype.kind == "S":
        volume_ids = np.char.decode(volume_ids, "utf-8")
    filtered = (
        pl.DataFrame(
            {
                "energy_mev": np.asarray(data["TotalEnergyDeposit"]),
                "event_x": np.asarray(data["EventPosition_X"]),
                "event_y": np.asarray(data["EventPosition_Y"]),
                "event_z": np.asarray(data["EventPosition_Z"]),
                "volume_id": volume_ids,
            }
        )
        .with_columns(
            pl.lit(crystal_id, dtype=pl.Int32).alias("crystal"),
            pl.col("volume_id")
            .cast(pl.String)
            .str.extract(r"_(\d+)$", 1)
            .cast(pl.Int32, strict=False)
            .alias("pixel"),
        )
        .filter(
            pl.col("energy_mev").is_between(
                energy_min_mev, energy_max_mev, closed="both"
            )
            & pl.col("pixel").is_not_null()
            & pl.col("pixel").is_between(0, pixels_per_head - 1, closed="both")
        )
    )
    if filtered.is_empty():
        return {}, 0

    half_size = fov_size_mm * 0.5
    histograms = {}
    accepted_events = 0
    for resolution_mm in resolutions_mm:
        grid_size = grid_sizes[resolution_mm]
        indexed = (
            filtered.with_columns(
                ((pl.col("event_x") + half_size) / resolution_mm)
                .floor()
                .cast(pl.Int32)
                .alias("x"),
                ((pl.col("event_y") + half_size) / resolution_mm)
                .floor()
                .cast(pl.Int32)
                .alias("y"),
                ((pl.col("event_z") + half_size) / resolution_mm)
                .floor()
                .cast(pl.Int32)
                .alias("z"),
            )
            .filter(
                pl.col("x").is_between(0, grid_size - 1, closed="both")
                & pl.col("y").is_between(0, grid_size - 1, closed="both")
                & pl.col("z").is_between(0, grid_size - 1, closed="both")
            )
            .group_by(COORDINATE_COLUMNS)
            .agg(pl.len().cast(pl.Int64).alias("counts"))
        )
        if not indexed.is_empty():
            accepted_events = max(
                accepted_events, int(indexed.get_column("counts").sum())
            )
            histograms[resolution_mm] = indexed
    return histograms, accepted_events


def save_outputs(
    output_dir: Path,
    accumulators: dict[float, list[pl.DataFrame]],
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
        batches = accumulators[resolution_mm]
        if batches:
            entries = (
                pl.concat(batches, rechunk=False)
                .group_by(COORDINATE_COLUMNS)
                .agg(pl.col("counts").sum())
                .sort(COORDINATE_COLUMNS)
            )
            coords = (
                entries.select(COORDINATE_COLUMNS)
                .to_numpy()
                .astype(np.int32, copy=False)
            )
            counts = (
                entries.get_column("counts").to_numpy().astype(np.int64, copy=False)
            )
        else:
            coords = np.empty((0, 5), dtype=np.int32)
            counts = np.empty((0,), dtype=np.int64)

        output_path = (
            output_dir / f"{args.output_stem}_{resolution_label(resolution_mm)}.npz"
        )
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
        "photopeak_kev": args.photopeak_kev,
        "energy_window_percent": args.energy_window_percent,
        "energy_min_kev": args.energy_min_kev,
        "energy_max_kev": args.energy_max_kev,
        "raw_events": raw_events,
        "accepted_events": accepted_events,
        "root_files": [str(path) for path in root_files],
        "outputs": output_names,
    }
    metadata_path = output_dir / "srm_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    args = get_parsed_args()
    resolutions_mm = list(dict.fromkeys(args.resolutions_mm))
    grid_sizes = validate_grid(args.fov_size_mm, resolutions_mm)
    root_files = list(iter_root_files(args.input_dir))

    if args.print_trees_only:
        for root_path in root_files:
            print(f"\n{root_path}")
            root_file: Any = uproot.open(root_path)
            try:
                print_root_file_contents(root_file, args.print_branch_names)
            finally:
                root_file.close()
        return 0

    if not root_files and not args.allow_empty:
        raise FileNotFoundError(
            f"No pixel_singles_*.root files found in {args.input_dir}"
        )
    if not root_files:
        print(f"Warning: no ROOT files in {args.input_dir}; writing empty SRMs")

    accumulators: dict[float, list[pl.DataFrame]] = {
        resolution_mm: [] for resolution_mm in resolutions_mm
    }
    raw_events = 0
    accepted_events = 0
    energy_min_mev = args.energy_min_kev / 1000.0
    energy_max_mev = args.energy_max_kev / 1000.0

    for root_path in root_files:
        root_file: Any = uproot.open(root_path)
        try:
            for tree_name, crystal_id in get_tree_names(root_file):
                tree: Any = root_file[tree_name]
                for data in tree.iterate(
                    BRANCHES, library="np", step_size=args.step_size
                ):
                    raw_events += len(data["TotalEnergyDeposit"])
                    histograms, chunk_accepted = histogram_chunk(
                        data,
                        crystal_id,
                        resolutions_mm,
                        grid_sizes,
                        args.fov_size_mm,
                        args.pixels_per_head,
                        energy_min_mev,
                        energy_max_mev,
                    )
                    accepted_events += chunk_accepted
                    for resolution_mm, histogram in histograms.items():
                        accumulators[resolution_mm].append(histogram)
        finally:
            root_file.close()

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
        f"{accepted_events} accepted events, and wrote {len(resolutions_mm)} "
        f"sparse SRMs to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
