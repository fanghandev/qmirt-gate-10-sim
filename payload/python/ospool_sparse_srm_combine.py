#!/usr/bin/env python3
"""Combine per-crystal sparse SRM chunk outputs into final per-crystal files."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine sparse per-crystal SRM chunks into final outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing per-job subdirectories with crystal NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where final per-crystal sparse outputs are written.",
    )
    return parser.parse_args()


def extract_crystal_id(path: Path, data: dict[str, np.ndarray]) -> int:
    if "crystal_id" in data:
        return int(np.asarray(data["crystal_id"]).reshape(-1)[0])
    match = re.search(r"crystal_(\d+)", path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not infer crystal ID from {path}")


def load_chunk_summaries(input_dir: Path) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for summary_path in sorted(input_dir.rglob("chunk_summary.json")):
        summaries.append(json.loads(summary_path.read_text()))
    return summaries


def combine_crystal_npz_files(
    input_dir: Path,
) -> dict[int, dict[tuple[int, int, int, int], int]]:
    combined: dict[int, dict[tuple[int, int, int, int], int]] = defaultdict(dict)
    for npz_path in sorted(input_dir.rglob("crystal_*.npz")):
        data = np.load(npz_path)
        crystal_id = extract_crystal_id(npz_path, data)
        coords = np.asarray(data["coords"], dtype=np.int32)
        counts = np.asarray(data["counts"], dtype=np.int64)

        crystal_store = combined[crystal_id]
        for coord, count in zip(coords, counts):
            key = tuple(int(value) for value in coord)
            crystal_store[key] = crystal_store.get(key, 0) + int(count)

    return combined


def save_final_crystal_file(
    output_dir: Path,
    crystal_id: int,
    coord_map: dict[tuple[int, int, int, int], int],
    summary: dict[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    coords = np.array(list(coord_map.keys()), dtype=np.int32)
    counts = np.array(list(coord_map.values()), dtype=np.int64)
    output_path = output_dir / f"crystal_{crystal_id:03d}_srm.npz"
    np.savez_compressed(
        output_path,
        crystal_id=np.array([crystal_id], dtype=np.int32),
        coords=coords,
        counts=counts,
        hist_bins=np.array([summary.get("hist_bins", 80)], dtype=np.int32),
        hist_range=np.array(summary.get("hist_range", [-80.0, 80.0]), dtype=np.float32),
        energy_min=np.array([summary.get("energy_min", 0.0)], dtype=np.float32),
        energy_max=np.array([summary.get("energy_max", 0.0)], dtype=np.float32),
        total_events=np.array([summary["total_events"]], dtype=np.int64),
        total_sim_time=np.array([summary["total_sim_time"]], dtype=np.float64),
        accumulated_counts=np.array([int(counts.sum())], dtype=np.int64),
    )
    return output_path


def main() -> int:
    args = parse_args()
    chunk_summaries = load_chunk_summaries(args.input_dir)
    combined = combine_crystal_npz_files(args.input_dir)

    total_events = int(sum(int(summary["total_events"]) for summary in chunk_summaries))
    total_sim_time = float(
        sum(float(summary["total_sim_time"]) for summary in chunk_summaries)
    )
    total_files = int(sum(int(summary["file_count"]) for summary in chunk_summaries))
    total_root_members = int(
        sum(
            int(summary.get("processed_root_members", 0)) for summary in chunk_summaries
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_outputs: list[str] = []
    for crystal_id in sorted(combined):
        reference_summary = {
            "total_events": total_events,
            "total_sim_time": total_sim_time,
            "hist_bins": 80,
            "hist_range": [-80.0, 80.0],
            "energy_min": 0.14 * 0.999,
            "energy_max": 0.14 * 1.001,
        }
        output_path = save_final_crystal_file(
            args.output_dir,
            crystal_id,
            combined[crystal_id],
            reference_summary,
        )
        final_outputs.append(str(output_path))

    summary_path = args.output_dir / "combined_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "chunk_count": len(chunk_summaries),
                "file_count": total_files,
                "root_member_count": total_root_members,
                "total_events": total_events,
                "total_sim_time": total_sim_time,
                "crystal_count": len(final_outputs),
                "outputs": final_outputs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        f"Combined {len(chunk_summaries)} chunks into {len(final_outputs)} crystal outputs"
    )
    print(f"Total events: {total_events:,d}")
    print(f"Total simulation time: {total_sim_time:.2f} seconds")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
