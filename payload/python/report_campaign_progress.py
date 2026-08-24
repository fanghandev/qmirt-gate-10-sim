#!/usr/bin/env python3
"""Summarize sparse-SRM campaign progress into a small JSON status document."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

DURATION_UNITS_S = {
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "sec": 1.0,
    "min": 60.0,
    "h": 3600.0,
    "hour": 3600.0,
    "d": 86400.0,
}

FAILED_STATES = (
    "FAILED",
    "TIMEOUT",
    "CANCELLED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
)

PIXELS_PER_CRYSTAL = 625


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report sparse-SRM campaign progress as JSON."
    )
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON here (atomically). Defaults to stdout.",
    )
    parser.add_argument(
        "--expected-tasks",
        type=int,
        default=0,
        help="Array size, used for percentage complete.",
    )
    parser.add_argument(
        "--job-id",
        default="",
        help="Slurm array job id; enables sacct wait/wall-clock timing.",
    )
    parser.add_argument(
        "--srm-labels",
        default="1mm,1p5mm,2mm",
        help="Comma-separated resolution labels to summarize.",
    )
    parser.add_argument(
        "--srm-dir",
        type=Path,
        help="Directory holding final_srm_*.npz. Defaults to the campaign dir.",
    )
    parser.add_argument(
        "--no-srm-stats",
        action="store_true",
        help="Skip reading combined SRMs (cheaper, for frequent polling).",
    )
    parser.add_argument(
        "--max-projection-points",
        type=int,
        default=20000,
        help="Cap on non-zero points sent per 2D projection.",
    )
    parser.add_argument(
        "--top-elements",
        type=int,
        default=15,
        help="How many hottest matrix elements to include.",
    )
    parser.add_argument(
        "--pixel-query",
        action="store_true",
        help="Emit the SRM row for one detector pixel instead of a campaign report.",
    )
    parser.add_argument(
        "--crystal",
        type=int,
        default=0,
        help="Crystal (detector head) id for --pixel-query.",
    )
    parser.add_argument(
        "--pixel",
        type=int,
        default=-1,
        help="Pixel id for --pixel-query; -1 sums every pixel of the crystal.",
    )
    parser.add_argument(
        "--geometry-wrl",
        type=Path,
        help="Gate VRML export; adds detector outlines and pixel geometry to the report.",
    )
    parser.add_argument(
        "--geometry-crystal-offset",
        type=int,
        default=1,
        help="WRL solid number for SRM crystal id 0 (Gate names are 1-based).",
    )
    args = parser.parse_args()
    if not args.pixel_query and args.campaign_dir is None:
        parser.error("--campaign-dir is required unless --pixel-query is given")
    if args.pixel_query and args.campaign_dir is None and args.srm_dir is None:
        parser.error("--pixel-query needs --srm-dir (or --campaign-dir)")
    return args


def duration_to_seconds(entry: dict) -> float:
    value = float(entry.get("value", 0.0))
    unit = (entry.get("unit") or "s").strip().lower()
    return value * DURATION_UNITS_S.get(unit, 1.0)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def parse_wall_time(path: Path) -> dict:
    fields: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                fields[key.strip()] = value.strip()
    except OSError:
        return {}
    parsed: dict[str, float | str] = {}
    for key in ("start_epoch_s", "end_epoch_s", "wall_time_seconds"):
        if key in fields:
            try:
                parsed[key] = float(fields[key])
            except ValueError:
                pass
    return parsed


def parse_key_values(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    try:
        for line in path.read_text().splitlines():
            key, _, raw = line.partition(":")
            try:
                values[key.strip()] = float(raw.strip())
            except ValueError:
                continue
    except OSError:
        return {}
    return values


def scan_task(task_dir: Path, srm_label: str) -> dict:
    chunk_dir = task_dir / "srm_chunks"
    final_srm = task_dir / f"final_srm_{srm_label}.npz"

    primaries = 0
    tracks = 0
    steps = 0
    sim_seconds = 0.0
    init_seconds = 0.0
    loops_done = 0
    for stats_path in sorted(chunk_dir.glob("*_sim_stats_loop_*.txt")):
        stats = read_json(stats_path)
        if not stats:
            continue
        loops_done += 1
        primaries += int(stats.get("events", {}).get("value", 0))
        tracks += int(stats.get("tracks", {}).get("value", 0))
        steps += int(stats.get("steps", {}).get("value", 0))
        sim_seconds += duration_to_seconds(stats.get("duration", {}))
        init_seconds += duration_to_seconds(stats.get("init", {}))

    raw_singles = 0
    accepted_singles = 0
    for meta_path in sorted(chunk_dir.glob("srm_metadata_loop_*.json")):
        meta = read_json(meta_path)
        if not meta:
            continue
        raw_singles += int(meta.get("raw_events", 0))
        accepted_singles += int(meta.get("accepted_events", 0))

    peak_rss_mb = 0.0
    cpu_pct_values = []
    cpu_alloc_values = []
    requested_cpus = None
    for profile_path in sorted(chunk_dir.glob("resource_profile_summary_loop_*.txt")):
        profile = parse_key_values(profile_path)
        if not profile:
            continue
        peak_rss_mb = max(peak_rss_mb, profile.get("peak_rss_mb", 0.0))
        if "average_cpu_pct" in profile:
            cpu_pct_values.append(profile["average_cpu_pct"])
        if "average_cpu_pct_of_allocation" in profile:
            cpu_alloc_values.append(profile["average_cpu_pct_of_allocation"])
        if "requested_cpus" in profile:
            requested_cpus = int(profile["requested_cpus"])

    task_id = task_dir.name.removeprefix("task_")
    wall = parse_wall_time(task_dir / f"task_{task_id}_wall_time.txt")

    return {
        "task_id": task_id,
        "complete": final_srm.is_file(),
        "loops_done": loops_done,
        "primaries": primaries,
        "tracks": tracks,
        "steps": steps,
        "simulation_seconds": sim_seconds,
        "init_seconds": init_seconds,
        "raw_singles": raw_singles,
        "accepted_singles": accepted_singles,
        "peak_rss_mb": peak_rss_mb,
        "mean_cpu_pct": (
            sum(cpu_pct_values) / len(cpu_pct_values) if cpu_pct_values else None
        ),
        "mean_cpu_pct_of_allocation": (
            sum(cpu_alloc_values) / len(cpu_alloc_values) if cpu_alloc_values else None
        ),
        "requested_cpus": requested_cpus,
        "start_epoch_s": wall.get("start_epoch_s"),
        "end_epoch_s": wall.get("end_epoch_s"),
        "wall_time_seconds": wall.get("wall_time_seconds"),
    }


def union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Wall-clock duration covered by intervals, counting overlap only once."""
    spans = sorted((s, e) for s, e in intervals if e > s)
    if not spans:
        return 0.0
    total = 0.0
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + (current_end - current_start)


def parse_slurm_time(value: str) -> float | None:
    value = value.strip()
    if not value or value in {"Unknown", "None", "N/A"}:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def collect_sacct(job_id: str) -> dict:
    command = [
        "sacct",
        "-j",
        job_id,
        "-X",
        "-n",
        "-P",
        "--format=JobID,State,Submit,Start,End",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "reason": "sacct not available"}
    if completed.returncode != 0:
        return {"available": False, "reason": completed.stderr.strip()[:200]}

    elements = []
    for line in completed.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        raw_id, state, submit, start, end = parts[:5]
        elements.append(
            {
                "job_id": raw_id,
                "task_id": raw_id.split("_")[-1] if "_" in raw_id else None,
                "state": state.split()[0],
                "submit": parse_slurm_time(submit),
                "start": parse_slurm_time(start),
                "end": parse_slurm_time(end),
            }
        )
    if not elements:
        return {"available": False, "reason": "no sacct records"}

    now = time.time()
    wait_intervals = []
    run_intervals = []
    waits = []
    states: dict[str, int] = {}
    for element in elements:
        states[element["state"]] = states.get(element["state"], 0) + 1
        submit, start, end = element["submit"], element["start"], element["end"]
        if submit is not None:
            wait_end = start if start is not None else now
            wait_intervals.append((submit, wait_end))
            waits.append(max(0.0, wait_end - submit))
        if start is not None:
            run_intervals.append((start, end if end is not None else now))

    submits = [e["submit"] for e in elements if e["submit"] is not None]
    ends = [e["end"] for e in elements if e["end"] is not None]
    starts = [e["start"] for e in elements if e["start"] is not None]

    campaign_start = min(submits) if submits else None
    latest = max(ends) if ends else now

    return {
        "available": True,
        "element_count": len(elements),
        "states": states,
        "failed": sum(count for s, count in states.items() if s in FAILED_STATES),
        "running": states.get("RUNNING", 0),
        "pending": states.get("PENDING", 0),
        "completed": states.get("COMPLETED", 0),
        "wait_seconds_sum": float(sum(waits)),
        "wait_seconds_max": float(max(waits)) if waits else 0.0,
        "wait_seconds_mean": float(sum(waits) / len(waits)) if waits else 0.0,
        # Overlapping queue time is counted once, so this is real elapsed waiting.
        "wait_wallclock_seconds": union_seconds(wait_intervals),
        "run_wallclock_seconds": union_seconds(run_intervals),
        "campaign_span_seconds": (latest - campaign_start) if campaign_start else 0.0,
        "first_submit_epoch_s": campaign_start,
        "first_start_epoch_s": min(starts) if starts else None,
        "wait_seconds_per_task": {
            e["task_id"]: max(
                0.0,
                (e["start"] if e["start"] is not None else now) - e["submit"],
            )
            for e in elements
            if e["submit"] is not None and e["task_id"] is not None
        },
    }


def histogram_counts(counts: np.ndarray, bin_count: int = 24) -> dict:
    if counts.size == 0:
        return {"edges": [], "values": []}
    maximum = int(counts.max())
    if maximum <= 1:
        return {"edges": [1, 2], "values": [int(counts.size)]}
    edges = np.unique(
        np.round(np.logspace(0, np.log10(maximum + 1), bin_count + 1)).astype(np.int64)
    )
    values, _ = np.histogram(counts, bins=edges)
    return {"edges": edges.tolist(), "values": values.tolist()}


def project_plane(
    coords: np.ndarray,
    counts: np.ndarray,
    grid_size: int,
    first_axis: int,
    second_axis: int,
    max_points: int,
) -> dict:
    """Sum counts over the remaining spatial axis, returned sparse."""
    flat = coords[:, first_axis] * grid_size + coords[:, second_axis]
    sums = np.bincount(flat, weights=counts, minlength=grid_size * grid_size)
    nonzero = np.flatnonzero(sums)
    truncated = False
    if nonzero.size > max_points:
        strongest = np.argpartition(sums[nonzero], -max_points)[-max_points:]
        nonzero = nonzero[strongest]
        truncated = True
    values = sums[nonzero]
    return {
        "i": (nonzero // grid_size).astype(np.int32).tolist(),
        "j": (nonzero % grid_size).astype(np.int32).tolist(),
        "v": values.astype(np.int64).tolist(),
        "nonzero_bins": int(np.count_nonzero(sums)),
        "max_value": int(values.max()) if values.size else 0,
        "total": int(sums.sum()),
        "truncated": truncated,
    }


def load_srm(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        coords = np.asarray(data["coords"], dtype=np.int64)
        counts = np.asarray(data["counts"], dtype=np.int64)
        meta = {
            "grid_size": int(np.asarray(data["grid_size"]).reshape(-1)[0]),
            "voxel_size_mm": float(np.asarray(data["voxel_size_mm"]).reshape(-1)[0]),
            "extent_mm": np.asarray(data["hist_range"]).reshape(-1).tolist(),
            "energy_window_kev": [
                float(np.asarray(data["energy_min_kev"]).reshape(-1)[0]),
                float(np.asarray(data["energy_max_kev"]).reshape(-1)[0]),
            ],
            "chunk_count": (
                int(np.asarray(data["chunk_count"]).reshape(-1)[0])
                if "chunk_count" in data.files
                else None
            ),
            "input_count": (
                int(np.asarray(data["input_count"]).reshape(-1)[0])
                if "input_count" in data.files
                else None
            ),
        }
    return coords, counts, meta


def detector_inventory(coords: np.ndarray, counts: np.ndarray) -> dict:
    """Per-crystal pixel occupancy, used to populate the dashboard selectors."""
    inventory = {
        "pixels_per_crystal": PIXELS_PER_CRYSTAL,
        "pixel_grid": round(PIXELS_PER_CRYSTAL**0.5),
        "crystals": [],
    }
    if coords.size == 0:
        return inventory
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    crystal_sorted = coords[order, 0]
    pixel_sorted = coords[order, 1]
    counts_sorted = counts[order]
    boundaries = np.flatnonzero(np.diff(crystal_sorted)) + 1
    for start, stop in zip(
        np.concatenate(([0], boundaries)),
        np.concatenate((boundaries, [crystal_sorted.size])),
    ):
        pixels = pixel_sorted[start:stop]
        totals = np.bincount(pixels, weights=counts_sorted[start:stop]).astype(np.int64)
        hit = np.flatnonzero(totals)
        inventory["crystals"].append(
            {
                "id": int(crystal_sorted[start]),
                "total_counts": int(totals.sum()),
                "pixels_hit": int(hit.size),
                "pixel_ids": hit.astype(np.int32).tolist(),
                "pixel_counts": totals[hit].tolist(),
            }
        )
    return inventory


def summarize_pixel(
    path: Path, crystal: int, pixel: int, top_voxels: int, max_points: int
) -> dict:
    coords, counts, meta = load_srm(path)
    selection = coords[:, 0] == crystal
    if pixel >= 0:
        selection &= coords[:, 1] == pixel
    coords = coords[selection]
    counts = counts[selection]
    grid_size = meta["grid_size"]

    summary = {
        "available": True,
        "path": str(path),
        "crystal": crystal,
        "pixel": pixel,
        "nonzero_elements": int(counts.size),
        "total_counts": int(counts.sum()) if counts.size else 0,
        "distinct_pixels": int(np.unique(coords[:, 1]).size) if counts.size else 0,
        **{k: meta[k] for k in ("grid_size", "voxel_size_mm", "extent_mm")},
    }
    if counts.size == 0:
        summary["projections"] = {}
        summary["hottest_voxels"] = []
        return summary

    voxel_index = (
        coords[:, 2] * grid_size * grid_size + coords[:, 3] * grid_size + coords[:, 4]
    )
    voxel_ids, voxel_inverse = np.unique(voxel_index, return_inverse=True)
    voxel_sums = np.bincount(np.ravel(voxel_inverse), weights=counts).astype(np.int64)
    order = np.argsort(voxel_sums)[::-1][:top_voxels]
    summary.update(
        {
            "distinct_voxels": int(voxel_ids.size),
            "max_voxel_counts": int(voxel_sums.max()),
            "mean_voxel_counts": float(voxel_sums.mean()),
            "voxel_total_histogram": histogram_counts(voxel_sums),
            "hottest_voxels": [
                {
                    "voxel_index": int(voxel_ids[i]),
                    "voxel_xyz": [
                        int(voxel_ids[i] // (grid_size * grid_size)),
                        int((voxel_ids[i] // grid_size) % grid_size),
                        int(voxel_ids[i] % grid_size),
                    ],
                    "counts": int(voxel_sums[i]),
                }
                for i in order
            ],
            "projections": {
                "xy": project_plane(coords, counts, grid_size, 2, 3, max_points),
                "yz": project_plane(coords, counts, grid_size, 3, 4, max_points),
                "zx": project_plane(coords, counts, grid_size, 2, 4, max_points),
            },
        }
    )
    return summary


def summarize_srm(path: Path, top_elements: int, max_points: int) -> dict:
    coords, counts, meta = load_srm(path)
    grid_size = meta["grid_size"]

    summary = {
        "available": True,
        "path": str(path),
        "nonzero_elements": int(counts.size),
        "total_counts": int(counts.sum()) if counts.size else 0,
        **meta,
    }
    if coords.size == 0:
        return summary

    # coords columns: crystal_id, pixel_id, voxel_x, voxel_y, voxel_z
    detector_index = coords[:, 0] * PIXELS_PER_CRYSTAL + coords[:, 1]
    voxel_index = (
        coords[:, 2] * grid_size * grid_size + coords[:, 3] * grid_size + coords[:, 4]
    )

    detector_ids, detector_inverse = np.unique(detector_index, return_inverse=True)
    detector_sums = np.bincount(
        np.ravel(detector_inverse), weights=counts
    ).astype(np.int64)
    voxel_ids, voxel_inverse = np.unique(voxel_index, return_inverse=True)
    voxel_sums = np.bincount(np.ravel(voxel_inverse), weights=counts).astype(np.int64)

    order = np.argsort(counts)[::-1][:top_elements]
    summary.update(
        {
            "max_element_counts": int(counts.max()),
            "mean_element_counts": float(counts.mean()),
            "median_element_counts": float(np.median(counts)),
            "distinct_detector_pixels": int(detector_ids.size),
            "distinct_voxels": int(voxel_ids.size),
            "distinct_crystals": int(np.unique(coords[:, 0]).size),
            "counts_per_detector_mean": float(detector_sums.mean()),
            "counts_per_voxel_mean": float(voxel_sums.mean()),
            "element_count_histogram": histogram_counts(counts),
            "detector_total_histogram": histogram_counts(detector_sums),
            "voxel_total_histogram": histogram_counts(voxel_sums),
            "detector_map": detector_inventory(coords, counts),
            "hottest_elements": [
                {
                    "crystal": int(coords[i, 0]),
                    "pixel": int(coords[i, 1]),
                    "detector_index": int(detector_index[i]),
                    "voxel_xyz": [
                        int(coords[i, 2]),
                        int(coords[i, 3]),
                        int(coords[i, 4]),
                    ],
                    "voxel_index": int(voxel_index[i]),
                    "counts": int(counts[i]),
                }
                for i in order
            ],
            "projections": {
                "xy": project_plane(coords, counts, grid_size, 2, 3, max_points),
                "yz": project_plane(coords, counts, grid_size, 3, 4, max_points),
                "zx": project_plane(coords, counts, grid_size, 2, 4, max_points),
            },
        }
    )
    return summary


def parse_wrl_solids(path: Path) -> dict[str, tuple[np.ndarray, list[list[int]]]]:
    """Vertices and polygons of every named solid in a Gate VRML export."""
    solids: dict[str, tuple[np.ndarray, list[list[int]]]] = {}
    text = path.read_text()
    for block in re.split(r"#-+ SOLID:\s+", text)[1:]:
        name = block[: block.find("\n")].strip()
        points = re.search(r"point\s*\[(.*?)\]", block, re.DOTALL)
        if not points:
            continue
        vertices = [
            [float(v) for v in numbers]
            for numbers in (line.split() for line in points.group(1).split(","))
            if len(numbers) == 3
        ]
        if not vertices:
            continue
        polygons: list[list[int]] = []
        indices = re.search(r"coordIndex\s*\[(.*?)\]", block, re.DOTALL)
        if indices:
            current: list[int] = []
            for token in re.findall(r"-?\d+", indices.group(1)):
                value = int(token)
                if value == -1:
                    if current:
                        polygons.append(current)
                        current = []
                else:
                    current.append(value)
            if current:
                polygons.append(current)
        solids[name] = (np.asarray(vertices, dtype=float), polygons)
    return solids


def unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector / length if length else vector


def ring_order(
    points: np.ndarray, axis: np.ndarray, reference: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sort four coplanar points into a ring, optionally sharing another ring's phase."""
    centre = points.mean(axis=0)
    relative = points - centre
    first = relative[0]
    u = reference if reference is not None else unit(first - axis * float(first @ axis))
    v = np.cross(axis, u)
    angles = np.arctan2(relative @ v, relative @ u)
    return points[np.argsort(angles)], u


def unique_vertices(vertices: np.ndarray) -> np.ndarray:
    return np.unique(np.round(vertices, 2), axis=0)


def frustum_outline(vertices: np.ndarray) -> dict:
    """Near and far quads of a head-shaped solid, plus the edges joining them."""
    points = unique_vertices(vertices)
    order = np.argsort(np.linalg.norm(points, axis=1))
    axis = unit(points.mean(axis=0))
    near, phase = ring_order(points[order[:4]], axis)
    far, _ = ring_order(points[order[-4:]], axis, phase)
    loop = np.vstack((near, far))
    edges = [[i, (i + 1) % 4] for i in range(4)]
    edges += [[4 + i, 4 + (i + 1) % 4] for i in range(4)]
    edges += [[i, 4 + i] for i in range(4)]
    return {"vertices": np.round(loop, 2).tolist(), "edges": edges}


def box_outline(vertices: np.ndarray, polygons: list[list[int]]) -> dict:
    edges = sorted(
        {
            (min(a, b), max(a, b))
            for polygon in polygons
            for a, b in zip(polygon, polygon[1:] + polygon[:1])
        }
    )
    return {
        "vertices": np.round(vertices, 2).tolist(),
        "edges": [list(edge) for edge in edges],
    }


def crystal_pixel_frame(vertices: np.ndarray, pixel_grid: int) -> dict:
    """Corner and per-pixel step vectors of the crystal face turned toward the FOV."""
    points = unique_vertices(vertices)
    order = np.argsort(np.linalg.norm(points, axis=1))
    face, _ = ring_order(points[order[:4]], unit(points.mean(axis=0)))
    return {
        "pixel_origin": np.round(face[0], 3).tolist(),
        "pixel_u": np.round((face[1] - face[0]) / pixel_grid, 4).tolist(),
        "pixel_v": np.round((face[3] - face[0]) / pixel_grid, 4).tolist(),
        "face_centre": np.round(face.mean(axis=0), 3).tolist(),
    }


def collect_geometry(path: Path, crystal_offset: int, pixel_grid: int) -> dict:
    solids = parse_wrl_solids(path)
    heads = []
    for name, (vertices, polygons) in solids.items():
        match = re.fullmatch(r"DetectorCrystal_(\d+):0", name)
        if not match or vertices.shape[0] < 8:
            continue
        solid_id = int(match.group(1))
        head = {
            "crystal_id": solid_id - crystal_offset,
            "solid_id": solid_id,
            "crystal": box_outline(vertices, polygons),
        }
        head.update(crystal_pixel_frame(vertices, pixel_grid))
        collimator = solids.get(f"Collimator_{solid_id}:0")
        if collimator is not None and collimator[0].shape[0] >= 8:
            points = unique_vertices(collimator[0])
            radii = np.linalg.norm(points, axis=1)
            # The narrow end faces the FOV, so its vertices define the pinhole.
            inner = points[radii <= radii.min() + 8.0]
            head["collimator"] = frustum_outline(collimator[0])
            head["pinhole"] = np.round(inner.mean(axis=0), 3).tolist()
        heads.append(head)
    heads.sort(key=lambda h: h["crystal_id"])
    extent = max(
        (
            float(np.abs(np.asarray(h["crystal"]["vertices"])).max())
            for h in heads
        ),
        default=0.0,
    )
    return {
        "source": str(path),
        "pixel_grid": pixel_grid,
        "crystal_offset": crystal_offset,
        "extent_mm": extent,
        "heads": heads,
    }


def write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def task_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"\d+$", path.name)
    return (int(match.group()) if match else 0, path.name)


def main() -> int:
    args = parse_args()

    labels = [item.strip() for item in args.srm_labels.split(",") if item.strip()]
    if not labels:
        raise SystemExit("--srm-labels must list at least one label")

    if args.pixel_query:
        srm_dir = (args.srm_dir or args.campaign_dir).resolve()
        path = srm_dir / f"final_srm_{labels[0]}.npz"
        if not path.is_file():
            result = {"available": False, "path": str(path), "error": "SRM not found"}
        else:
            try:
                result = summarize_pixel(
                    path,
                    args.crystal,
                    args.pixel,
                    args.top_elements,
                    args.max_projection_points,
                )
            except (OSError, ValueError, KeyError) as exc:
                result = {
                    "available": False,
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        result["label"] = labels[0]
        result["generated_at"] = time.time()
        payload = json.dumps(result, default=str) + "\n"
        if args.output:
            write_atomic(args.output, payload)
        else:
            print(payload, end="")
        return 0

    campaign_dir = args.campaign_dir.resolve()
    if not campaign_dir.is_dir():
        raise SystemExit(f"campaign directory not found: {campaign_dir}")

    task_dirs = sorted(
        (path for path in campaign_dir.glob("task_*") if path.is_dir()),
        key=task_sort_key,
    )
    tasks = [scan_task(path, labels[0]) for path in task_dirs]

    complete = [t for t in tasks if t["complete"]]
    incomplete = [t for t in tasks if not t["complete"]]

    committed = {
        "primaries": sum(t["primaries"] for t in complete),
        "tracks": sum(t["tracks"] for t in complete),
        "steps": sum(t["steps"] for t in complete),
        "raw_singles": sum(t["raw_singles"] for t in complete),
        "accepted_singles": sum(t["accepted_singles"] for t in complete),
        "simulation_seconds": sum(t["simulation_seconds"] for t in complete),
        "init_seconds": sum(t["init_seconds"] for t in complete),
    }
    in_flight = {
        "primaries": sum(t["primaries"] for t in incomplete),
        "simulation_seconds": sum(t["simulation_seconds"] for t in incomplete),
        "loops_done": sum(t["loops_done"] for t in incomplete),
    }

    task_intervals = [
        (t["start_epoch_s"], t["end_epoch_s"])
        for t in tasks
        if t["start_epoch_s"] is not None and t["end_epoch_s"] is not None
    ]
    wall_sum = sum(t["wall_time_seconds"] or 0.0 for t in tasks)
    sim_seconds = committed["simulation_seconds"]

    def rate(numerator: float) -> float | None:
        return (numerator / sim_seconds) if sim_seconds > 0 else None

    cpu_values = [t["mean_cpu_pct"] for t in tasks if t["mean_cpu_pct"] is not None]
    alloc_values = [
        t["mean_cpu_pct_of_allocation"]
        for t in tasks
        if t["mean_cpu_pct_of_allocation"] is not None
    ]
    rss_values = [t["peak_rss_mb"] for t in tasks if t["peak_rss_mb"]]
    requested = [t["requested_cpus"] for t in tasks if t["requested_cpus"]]

    expected = args.expected_tasks or len(tasks)
    report = {
        "generated_at": time.time(),
        "campaign_dir": str(campaign_dir),
        "srm_labels": labels,
        "tasks": {
            "expected": expected,
            "observed": len(tasks),
            "complete": len(complete),
            "incomplete": len(incomplete),
            "percent_complete": (100.0 * len(complete) / expected) if expected else 0.0,
        },
        "events": {
            "committed_primaries": committed["primaries"],
            "in_flight_primaries": in_flight["primaries"],
            "committed_raw_singles": committed["raw_singles"],
            "committed_accepted_singles": committed["accepted_singles"],
            "committed_tracks": committed["tracks"],
            "committed_steps": committed["steps"],
            "loops_complete": sum(t["loops_done"] for t in complete),
            "loops_in_flight": in_flight["loops_done"],
        },
        "rates": {
            "primaries_per_second": rate(committed["primaries"]),
            "tracks_per_second": rate(committed["tracks"]),
            "steps_per_second": rate(committed["steps"]),
            "raw_singles_per_second": rate(committed["raw_singles"]),
            "accepted_singles_per_second": rate(committed["accepted_singles"]),
            "accepted_fraction_of_raw": (
                committed["accepted_singles"] / committed["raw_singles"]
                if committed["raw_singles"]
                else None
            ),
            "singles_per_primary": (
                committed["raw_singles"] / committed["primaries"]
                if committed["primaries"]
                else None
            ),
        },
        "time": {
            "committed_simulation_seconds": sim_seconds,
            "in_flight_simulation_seconds": in_flight["simulation_seconds"],
            "committed_init_seconds": committed["init_seconds"],
            "task_wall_seconds_sum": wall_sum,
            "task_wall_wallclock_seconds": union_seconds(task_intervals),
            "simulation_fraction_of_wall": (
                sim_seconds / wall_sum if wall_sum else None
            ),
        },
        "compute_profile": {
            "mean_cpu_pct": (sum(cpu_values) / len(cpu_values)) if cpu_values else None,
            "mean_cpu_pct_of_allocation": (
                sum(alloc_values) / len(alloc_values) if alloc_values else None
            ),
            "peak_rss_mb": max(rss_values) if rss_values else None,
            "mean_peak_rss_mb": (
                sum(rss_values) / len(rss_values) if rss_values else None
            ),
            "requested_cpus": max(requested) if requested else None,
            "tasks_with_profile": len(rss_values),
        },
        "per_task": tasks,
    }

    if args.job_id:
        report["slurm"] = collect_sacct(args.job_id)

    srm_dir = (args.srm_dir or campaign_dir).resolve()
    srm_reports = {}
    for label in labels:
        path = srm_dir / f"final_srm_{label}.npz"
        if args.no_srm_stats or not path.is_file():
            srm_reports[label] = {"available": False, "path": str(path)}
            continue
        try:
            srm_reports[label] = summarize_srm(
                path, args.top_elements, args.max_projection_points
            )
        except (OSError, ValueError, KeyError) as exc:
            srm_reports[label] = {
                "available": False,
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }
    report["srm"] = srm_reports

    if args.geometry_wrl:
        try:
            report["geometry"] = collect_geometry(
                args.geometry_wrl,
                args.geometry_crystal_offset,
                round(PIXELS_PER_CRYSTAL**0.5),
            )
        except (OSError, ValueError, np.linalg.LinAlgError) as exc:
            report["geometry"] = {
                "source": str(args.geometry_wrl),
                "error": f"{type(exc).__name__}: {exc}",
                "heads": [],
            }

    payload = json.dumps(report, default=str) + "\n"
    if args.output:
        write_atomic(args.output, payload)
        size_kb = len(payload) / 1024
        print(f"Wrote progress report to {args.output} ({size_kb:.1f} KB)")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
