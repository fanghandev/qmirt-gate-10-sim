#!/usr/bin/env python3
"""Additive SRM summaries ("snapshots") per campaign part, for the dashboard.

For every pulled part (<root>/<prefix>*_NN) and resolution this keeps
<part>/srm_snapshot_<label>.npz with the sums the dashboard plots:

  xy, yz, zx       orthogonal projections (counts summed over the third axis and
                   over all detector pixels), as in report_campaign_progress.py
  detector_sums    counts per detector pixel, shape (heads, pixels)
  voxel_sums       counts per voxel, flattened x*g*g + y*g + z
  total_counts, tasks (the task ids included), grid_size, voxel_size_mm, hist_range

These are sums, so a part's snapshot is the sum over its complete tasks, and a
campaign's is the sum over its parts: no SRM merge is needed. A snapshot is
updated incrementally: only complete tasks not yet included are read.

Usage: srm_projection_snapshot.py [--root DIR] [--prefix brain_288mm_csg_102t_]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("/data/fanghan/opengate_sim/data/brain_spect")
LABELS = ("1mm", "1p5mm", "2mm")
N_HEADS, N_PIXELS = 73, 625


def complete_tasks(part_dir: Path, label: str) -> list[int]:
    ids = []
    for task_dir in part_dir.glob("task_*"):
        suffix = task_dir.name.removeprefix("task_")
        if (suffix.isdigit() and (task_dir / "TASK_COMPLETE.json").is_file()
                and (task_dir / f"final_srm_{label}.npz").is_file()):
            ids.append(int(suffix))
    return sorted(ids)


def task_sums(path: Path) -> dict:
    """Additive sums of one task's sparse SRM (coords: head, pixel, x, y, z)."""
    with np.load(path) as data:
        coords = data["coords"]
        counts = data["counts"].astype(np.int64, copy=False)
        g = int(data["grid_size"][0])
        meta = {
            "grid_size": g,
            "voxel_size_mm": float(data["voxel_size_mm"][0]) if "voxel_size_mm" in data else None,
            "hist_range": data["hist_range"].astype(float).tolist() if "hist_range" in data else None,
            "energy_window_kev": [float(data["energy_min_kev"][0]), float(data["energy_max_kev"][0])]
            if "energy_min_kev" in data and "energy_max_kev" in data else None,
            "chunk_count": int(data["chunk_count"][0]) if "chunk_count" in data else 0,
        }
    x = coords[:, 2].astype(np.int64)
    y = coords[:, 3].astype(np.int64)
    z = coords[:, 4].astype(np.int64)
    detector = coords[:, 0].astype(np.int64) * N_PIXELS + coords[:, 1].astype(np.int64)
    out = {
        "xy": np.bincount(x * g + y, weights=counts, minlength=g * g).astype(np.int64),
        "yz": np.bincount(y * g + z, weights=counts, minlength=g * g).astype(np.int64),
        "zx": np.bincount(x * g + z, weights=counts, minlength=g * g).astype(np.int64),
        "detector_sums": np.bincount(detector, weights=counts,
                                     minlength=N_HEADS * N_PIXELS).astype(np.int64),
        "voxel_sums": np.bincount((x * g + y) * g + z, weights=counts,
                                  minlength=g ** 3).astype(np.int64),
        "total_counts": int(counts.sum()),
        **meta,
    }
    return out


def load_snapshot(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            snap = {key: data[key] for key in data.files}
        snap["meta"] = json.loads(str(snap["meta"]))
        return snap
    except (OSError, ValueError, KeyError):
        return None


def save_snapshot(path: Path, arrays: dict, meta: dict) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temp = handle.name
    np.savez_compressed(temp, meta=json.dumps(meta), **arrays)
    os.chmod(temp, 0o644)
    os.replace(temp, path)


def update_part(part_dir: Path, label: str, workers: int) -> tuple[int, int]:
    """(tasks added, tasks in snapshot)."""
    path = part_dir / f"srm_snapshot_{label}.npz"
    tasks = complete_tasks(part_dir, label)
    snap = load_snapshot(path)
    included = set(snap["tasks"].tolist()) if snap is not None else set()
    if not included <= set(tasks) or (snap is not None and "chunk_count" not in snap["meta"]):
        # a task vanished locally, or the snapshot predates a field: rebuild
        snap, included = None, set()
    new = [t for t in tasks if t not in included]
    if not new:
        return 0, len(included)
    keys = ("xy", "yz", "zx", "detector_sums", "voxel_sums")
    sums = {k: snap[k].astype(np.int64) for k in keys} if snap is not None else None
    meta = dict(snap["meta"]) if snap is not None else {}
    total = int(meta.get("total_counts", 0))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(
            lambda t: task_sums(part_dir / f"task_{t}" / f"final_srm_{label}.npz"), new
        ):
            if sums is None:
                sums = {k: result[k] for k in keys}
                meta = {k: result[k] for k in ("grid_size", "voxel_size_mm", "hist_range",
                                               "energy_window_kev")}
            else:
                if result["grid_size"] != meta["grid_size"]:
                    raise ValueError(f"{part_dir.name}: grid size differs between tasks")
                for k in keys:
                    sums[k] += result[k]
            total += result["total_counts"]
            meta["chunk_count"] = int(meta.get("chunk_count", 0)) + result["chunk_count"]
    meta["total_counts"] = total
    meta["label"] = label
    all_tasks = sorted(included | set(new))
    save_snapshot(path, {**sums, "tasks": np.asarray(all_tasks, dtype=np.int32)}, meta)
    return len(new), len(all_tasks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--prefix", default="brain_288mm_csg_102t_")
    ap.add_argument("--labels", default=",".join(LABELS))
    ap.add_argument("--workers", type=int, default=4, help="tasks read in parallel (~4 GB each at 1 mm)")
    args = ap.parse_args()
    labels = [label for label in args.labels.split(",") if label]
    for part_dir in sorted(args.root.glob(f"{args.prefix}*_[0-9][0-9]")):
        if not part_dir.is_dir() or "_merged_" in part_dir.name:
            continue
        for label in labels:
            added, total = update_part(part_dir, label, args.workers)
            if added:
                print(f"{part_dir.name} {label}: added {added} tasks ({total} in snapshot)",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
