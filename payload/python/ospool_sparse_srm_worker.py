#!/usr/bin/env python3
"""Process a slice of OSPool files into per-crystal sparse SRM chunks.

This worker reads a shared file list, processes a contiguous slice of entries,
and writes one sparse NPZ per crystal plus a chunk summary JSON. The per-crystal
NPZ stores sparse coordinates as ``(PixelID, x_bin, y_bin, z_bin)`` so the
crystal identity can live in the filename / directory layout instead of in the
coordinate payload.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import uproot


@dataclass(frozen=True)
class ChunkSummary:
    file_count: int
    total_events: int
    total_sim_time: float
    crystal_count: int


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
        help="Directory where per-crystal chunk outputs will be written.",
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
        default=0.14 * 0.999,
        help="Lower energy threshold in MeV.",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=0.14 * 1.001,
        help="Upper energy threshold in MeV.",
    )
    parser.add_argument(
        "--hist-min",
        type=float,
        default=-80.0,
        help="Lower bound for the histogram window in mm.",
    )
    parser.add_argument(
        "--hist-max",
        type=float,
        default=80.0,
        help="Upper bound for the histogram window in mm.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=80,
        help="Number of bins per axis.",
    )
    parser.add_argument(
        "--crystal-id-regex",
        type=str,
        default=r"_a_(\d+)_j_",
        help=(
            "Regular expression with one capture group that extracts the crystal ID "
            "from a ROOT member name or archive stem."
        ),
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


def extract_crystal_id(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    fallback = re.search(r"(?:Crystal|crystal|head|Head|a_)(\d+)", text)
    if fallback:
        return int(fallback.group(1))
    return None


def parse_stats_member(member_file) -> tuple[int, float]:
    text_content = member_file.read().decode("utf-8")
    stats_dict = json.loads(text_content)
    total_events = int(float(stats_dict["events"]["value"]))
    total_sim_time = float(stats_dict["sim_stop_time"]["value"]) - float(
        stats_dict["sim_start_time"]["value"]
    )
    return total_events, total_sim_time


def accumulate_sparse(
    sparse_store: dict[int, defaultdict[tuple[int, int, int, int], int]],
    crystal_id: int,
    pixel_ids: np.ndarray,
    x_bin: np.ndarray,
    y_bin: np.ndarray,
    z_bin: np.ndarray,
) -> None:
    coords = np.column_stack((pixel_ids, x_bin, y_bin, z_bin))
    unique_coords, counts = np.unique(coords, axis=0, return_counts=True)
    crystal_store = sparse_store[crystal_id]
    for coord, count in zip(unique_coords, counts):
        crystal_store[tuple(int(value) for value in coord)] += int(count)


def save_crystal_chunk(
    output_dir: Path,
    job_tag: str,
    crystal_id: int,
    coord_map: defaultdict[tuple[int, int, int, int], int],
    hist_bins: int,
    hist_min: float,
    hist_max: float,
    energy_min: float,
    energy_max: float,
) -> Path:
    crystal_dir = output_dir / job_tag
    crystal_dir.mkdir(parents=True, exist_ok=True)

    coords = np.array(list(coord_map.keys()), dtype=np.int32)
    counts = np.array(list(coord_map.values()), dtype=np.int64)
    output_path = crystal_dir / f"crystal_{crystal_id:03d}.npz"
    np.savez_compressed(
        output_path,
        crystal_id=np.array([crystal_id], dtype=np.int32),
        coords=coords,
        counts=counts,
        hist_bins=np.array([hist_bins], dtype=np.int32),
        hist_range=np.array([hist_min, hist_max], dtype=np.float32),
        energy_min=np.array([energy_min], dtype=np.float32),
        energy_max=np.array([energy_max], dtype=np.float32),
        accumulated_counts=np.array([int(counts.sum())], dtype=np.int64),
    )
    return output_path


def main() -> int:
    args = parse_args()
    file_paths = load_file_paths(args.input_list, args.start_index, args.end_index)
    if not file_paths:
        raise FileNotFoundError("The selected file-list slice is empty.")

    hist_width = (args.hist_max - args.hist_min) / args.hist_bins
    sparse_store: dict[int, defaultdict[tuple[int, int, int, int], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    total_events = 0
    total_sim_time = 0.0
    processed_files = 0
    processed_members = 0

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
        with (
            open(archive_path, "rb") as file_obj,
            tarfile.open(fileobj=file_obj, mode="r:gz") as tar,
        ):
            archive_crystal_id = extract_crystal_id(
                archive_path.stem, args.crystal_id_regex
            )

            for member in tar:
                if not member.isfile():
                    continue

                member_name = member.name
                if member_name.endswith(".txt"):
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        events, sim_time = parse_stats_member(extracted)
                        total_events += events
                        total_sim_time += sim_time
                    continue

                if not member_name.endswith(".root"):
                    continue

                extracted_root_file = tar.extractfile(member)
                if extracted_root_file is None:
                    continue

                crystal_id = extract_crystal_id(member_name, args.crystal_id_regex)
                if crystal_id is None:
                    crystal_id = archive_crystal_id
                if crystal_id is None:
                    raise ValueError(
                        f"Could not infer a crystal ID from {archive_path.name} / {member_name}."
                    )

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
                        df = pl.DataFrame(data).with_columns(
                            pl.col("PreStepUniqueVolumeID")
                            .str.split("_")
                            .list.get(-1)
                            .cast(pl.Int32)
                            .alias("PixelID"),
                        )

                        df = df.filter(
                            (pl.col("TotalEnergyDeposit") >= args.energy_min)
                            & (pl.col("TotalEnergyDeposit") <= args.energy_max)
                        )
                        if df.is_empty():
                            continue

                        event_x = df["EventPosition_X"].to_numpy()
                        event_y = df["EventPosition_Y"].to_numpy()
                        event_z = df["EventPosition_Z"].to_numpy()
                        pixel_ids = df["PixelID"].to_numpy().astype(np.int32)

                        valid_mask = (
                            (event_x >= args.hist_min)
                            & (event_x < args.hist_max)
                            & (event_y >= args.hist_min)
                            & (event_y < args.hist_max)
                            & (event_z >= args.hist_min)
                            & (event_z < args.hist_max)
                        )
                        if not np.any(valid_mask):
                            continue

                        event_x = event_x[valid_mask]
                        event_y = event_y[valid_mask]
                        event_z = event_z[valid_mask]
                        pixel_ids = pixel_ids[valid_mask]

                        x_bin = np.floor((event_x - args.hist_min) / hist_width).astype(
                            np.int32
                        )
                        y_bin = np.floor((event_y - args.hist_min) / hist_width).astype(
                            np.int32
                        )
                        z_bin = np.floor((event_z - args.hist_min) / hist_width).astype(
                            np.int32
                        )

                        accumulate_sparse(
                            sparse_store,
                            crystal_id,
                            pixel_ids,
                            x_bin,
                            y_bin,
                            z_bin,
                        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    crystal_outputs: list[str] = []
    for crystal_id in sorted(sparse_store):
        output_path = save_crystal_chunk(
            args.output_dir,
            job_tag,
            crystal_id,
            sparse_store[crystal_id],
            args.hist_bins,
            args.hist_min,
            args.hist_max,
            args.energy_min,
            args.energy_max,
        )
        crystal_outputs.append(str(output_path))

    summary = ChunkSummary(
        file_count=processed_files,
        total_events=total_events,
        total_sim_time=total_sim_time,
        crystal_count=len(crystal_outputs),
    )
    summary_path = args.output_dir / job_tag / "chunk_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "job_tag": job_tag,
                "file_count": summary.file_count,
                "total_events": summary.total_events,
                "total_sim_time": summary.total_sim_time,
                "crystal_count": summary.crystal_count,
                "processed_root_members": processed_members,
                "crystal_outputs": crystal_outputs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        f"Processed {processed_files} archives, {processed_members} ROOT members, "
        f"and {len(crystal_outputs)} crystals into {summary_path.parent}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
