#!/usr/bin/env python3
"""Prepare a lightweight dashboard fixture from OSPool sparse SRM chunks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def write_fixture(source: Path, output_dir: Path, task_count: int) -> None:
    with np.load(source, allow_pickle=False) as data:
        coords = np.asarray(data["coords"], dtype=np.int32)
        counts = np.asarray(data["counts"], dtype=np.int64)
        hist_range = np.asarray(data["hist_range"], dtype=np.float32).reshape(-1)
        hist_bins = int(np.asarray(data["hist_bins"]).reshape(-1)[0])
        energy_min = float(np.asarray(data["energy_min"]).reshape(-1)[0])
        energy_max = float(np.asarray(data["energy_max"]).reshape(-1)[0])
        total_events = int(np.asarray(data["total_events"]).reshape(-1)[0])
        total_sim_time = float(np.asarray(data["total_sim_time"]).reshape(-1)[0])

    voxel_size = float((hist_range[1] - hist_range[0]) / hist_bins)
    srm_dir = output_dir / "srm"
    campaign_dir = output_dir / "campaign"
    srm_dir.mkdir(parents=True, exist_ok=True)
    campaign_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        srm_dir / "final_srm_2mm.npz",
        coords=coords,
        counts=counts,
        grid_size=np.array([hist_bins], dtype=np.int32),
        voxel_size_mm=np.array([voxel_size], dtype=np.float32),
        hist_range=hist_range,
        energy_min_kev=np.array([energy_min * 1000.0], dtype=np.float32),
        energy_max_kev=np.array([energy_max * 1000.0], dtype=np.float32),
        chunk_count=np.array([1], dtype=np.int64),
        input_count=np.array([1], dtype=np.int64),
        total_events=np.array([total_events], dtype=np.int64),
        total_sim_time=np.array([total_sim_time], dtype=np.float64),
    )

    now = time.time()
    for task_id in range(task_count):
        task_dir = campaign_dir / f"task_{task_id}"
        chunk_dir = task_dir / "srm_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        events = max(1, total_events // task_count)
        stats = {
            "events": {"value": events},
            "tracks": {"value": events * 4},
            "steps": {"value": events * 12},
            "duration": {"value": 90 + task_id % 20, "unit": "s"},
            "init": {"value": 3, "unit": "s"},
        }
        (chunk_dir / f"task_sim_stats_loop_{task_id:05d}.txt").write_text(
            json.dumps(stats) + "\n"
        )
        (chunk_dir / f"srm_metadata_loop_{task_id:05d}.json").write_text(
            json.dumps({"raw_events": events // 2, "accepted_events": events // 3}) + "\n"
        )
        (chunk_dir / f"resource_profile_summary_loop_{task_id:05d}.txt").write_text(
            f"peak_rss_mb: {1200 + task_id}\n"
            f"average_cpu_pct: {350 + task_id % 30}\n"
            "average_cpu_pct_of_allocation: 87\n"
            "requested_cpus: 4\n"
        )
        start = now - (task_count - task_id) * 100
        wall = "\n".join(
            [
                f"start_epoch_s: {start}",
                f"end_epoch_s: {start + 90 + task_id % 20}",
                f"wall_time_seconds: {90 + task_id % 20}",
            ]
        )
        (task_dir / f"task_{task_id}_wall_time.txt").write_text(wall + "\n")

    print(f"Converted representative source chunk: {source}")
    print(f"Wrote SRM: {srm_dir / 'final_srm_2mm.npz'}")
    print(f"Wrote dummy task campaign: {campaign_dir} ({task_count} tasks)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=100)
    args = parser.parse_args()
    sources = sorted(args.source_dir.glob("*_sparse_5d_srm.npz"))
    if not sources:
        raise SystemExit(f"No sparse SRM chunks found in {args.source_dir}")
    write_fixture(sources[0], args.output_dir, args.task_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
