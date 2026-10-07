#!/usr/bin/env python3
"""Serve a local dashboard for sparse-SRM campaign progress.

Fetches progress JSON produced by report_campaign_progress.py, either from a local
path or over SSH, caches it, and serves it to the browser. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import posixpath
import shlex
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

STATIC_DIR = Path(__file__).resolve().parent
FAILED_STATES = ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
                 "BOOT_FAIL", "DEADLINE")


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


SNAPSHOT_LABELS = ("1mm", "1p5mm", "2mm")
_SNAPSHOT_CACHE: dict = {}


def _histogram(values, bin_count: int = 24) -> dict:
    """Log-binned histogram, as report_campaign_progress.histogram_counts."""
    import numpy as np

    values = np.asarray(values)
    if values.size == 0:
        return {"edges": [], "values": []}
    maximum = int(values.max())
    if maximum <= 1:
        return {"edges": [1, 2], "values": [int(values.size)]}
    edges = np.unique(
        np.round(np.logspace(0, np.log10(maximum + 1), bin_count + 1)).astype(np.int64)
    )
    counts, _ = np.histogram(values, bins=edges)
    return {"edges": edges.tolist(), "values": counts.tolist()}


def _sparse_plane(full, grid_size: int, max_points: int) -> dict:
    import numpy as np

    nonzero = np.flatnonzero(full)
    truncated = False
    if nonzero.size > max_points:
        nonzero = nonzero[np.argpartition(full[nonzero], -max_points)[-max_points:]]
        truncated = True
    values = full[nonzero]
    return {
        "i": (nonzero // grid_size).astype(np.int32).tolist(),
        "j": (nonzero % grid_size).astype(np.int32).tolist(),
        "v": values.astype(np.int64).tolist(),
        "nonzero_bins": int(np.count_nonzero(full)),
        "max_value": int(values.max()) if values.size else 0,
        "total": int(full.sum()),
        "truncated": truncated,
    }


def snapshot_srm(root: Path, parts: list[str], max_points: int = 20000) -> dict:
    """SRM summaries for the dashboard from the parts' additive projection snapshots
    (payload/python/srm_projection_snapshot.py), summed over parts. Element-level
    statistics need the merged SRM and are left empty."""
    import numpy as np

    result = {}
    for label in SNAPSHOT_LABELS:
        paths = [root / part / f"srm_snapshot_{label}.npz" for part in parts]
        paths = [path for path in paths if path.is_file()]
        if not paths:
            result[label] = {"available": False, "reason": "no projection snapshots yet"}
            continue
        key = tuple((str(path), path.stat().st_mtime) for path in paths)
        cached = _SNAPSHOT_CACHE.get(label)
        if cached and cached[0] == key:
            result[label] = cached[1]
            continue
        sums: dict = {}
        tasks = 0
        meta: dict = {}
        for path in paths:
            with np.load(path) as data:
                for name in ("xy", "yz", "zx", "detector_sums", "voxel_sums"):
                    array = data[name].astype(np.int64)
                    sums[name] = array if name not in sums else sums[name] + array
                tasks += int(data["tasks"].size)
                meta = json.loads(str(data["meta"]))
        g = int(meta["grid_size"])
        detector = sums["detector_sums"].reshape(-1)
        voxels = sums["voxel_sums"]
        hit_detectors = detector[detector > 0]
        hit_voxels = voxels[voxels > 0]
        per_head = detector.reshape(-1, 625)
        crystals = []
        for head, totals in enumerate(per_head):
            hit = np.flatnonzero(totals)
            if hit.size:
                crystals.append({
                    "id": head,
                    "total_counts": int(totals.sum()),
                    "pixels_hit": int(hit.size),
                    "pixel_ids": hit.astype(np.int32).tolist(),
                    "pixel_counts": totals[hit].tolist(),
                })
        summary = {
            "available": True,
            "source": "projection snapshots",
            "path": None,  # no SRM file behind these sums: per-pixel queries need a merge
            "snapshot_parts": len(paths),
            "snapshot_tasks": tasks,
            "grid_size": g,
            "voxel_size_mm": meta.get("voxel_size_mm"),
            "hist_range": meta.get("hist_range"),
            "total_counts": int(detector.sum()),
            "nonzero_elements": None,
            "max_element_counts": None,
            "mean_element_counts": None,
            "median_element_counts": None,
            "distinct_detector_pixels": int(hit_detectors.size),
            "distinct_crystals": len(crystals),
            "distinct_voxels": int(hit_voxels.size),
            "counts_per_detector_mean": float(hit_detectors.mean()) if hit_detectors.size else None,
            "counts_per_voxel_mean": float(hit_voxels.mean()) if hit_voxels.size else None,
            "element_count_histogram": {"edges": [], "values": []},
            "detector_total_histogram": _histogram(hit_detectors),
            "voxel_total_histogram": _histogram(hit_voxels),
            "detector_map": {"pixels_per_crystal": 625, "pixel_grid": 25, "crystals": crystals},
            "hottest_elements": [],
            "projections": {
                name: _sparse_plane(sums[name], g, max_points) for name in ("xy", "yz", "zx")
            },
        }
        _SNAPSHOT_CACHE[label] = (key, summary)
        result[label] = summary
    return result


def database_summary(db_path: Path, campaigns: list[str]) -> dict:
    """Per-part task states, pull/merge status, counts and Slurm timing from the
    local job database (payload/python/update_local_slurm_campaign_database.py)."""
    if not db_path.is_file():
        return {"available": False, "reason": f"{db_path} not found"}
    summary: dict = {
        "available": True,
        "path": str(db_path),
        "updated_at": db_path.stat().st_mtime,
        "parts": [],
    }
    # read-only: the harvest timer may be writing
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5) as conn:
        for campaign in campaigns:
            rows = conn.execute(
                """SELECT array_task_id, state, exit_code, submit_epoch, start_epoch,
                          end_epoch, primaries, raw_singles, accepted_singles,
                          is_pulled, is_merged, eligible_epoch
                   FROM slurm_jobs WHERE campaign_name = ?""",
                (campaign,),
            ).fetchall()
            states: dict[str, int] = {}
            for row in rows:
                states[row[1] or "UNKNOWN"] = states.get(row[1] or "UNKNOWN", 0) + 1
            done = [r for r in rows if r[1] == "COMPLETED" and r[9]]
            # from eligibility (after the previous part), not submission; older rows
            # without it fall back to the submit time
            waits = [r[4] - (r[11] or r[3]) for r in rows if (r[11] or r[3]) and r[4]]
            runs = [r[5] - r[4] for r in rows if r[4] and r[5] and r[1] == "COMPLETED"]
            submits = [r[3] for r in rows if r[3]]
            ends = [r[5] for r in rows if r[5]]
            summary["parts"].append({
                "campaign": campaign,
                "tasks": len(rows),
                "states": states,
                "pulled": sum(1 for r in rows if r[9]),
                "merged": sum(1 for r in rows if r[10]),
                "failed": [
                    {"task": r[0], "state": r[1], "exit_code": r[2]}
                    for r in sorted(rows)
                    if r[1] in FAILED_STATES
                ],
                "completed_primaries": sum(r[6] or 0 for r in done),
                "completed_accepted_singles": sum(r[8] or 0 for r in done),
                "wait_seconds_mean": sum(waits) / len(waits) if waits else None,
                "wait_seconds_max": max(waits) if waits else None,
                "run_seconds_mean": sum(runs) / len(runs) if runs else None,
                "first_submit_epoch_s": min(submits) if submits else None,
                "last_end_epoch_s": max(ends) if ends else None,
            })
    parts = summary["parts"]
    submits = [p["first_submit_epoch_s"] for p in parts if p["first_submit_epoch_s"]]
    ends = [p["last_end_epoch_s"] for p in parts if p["last_end_epoch_s"]]
    summary["totals"] = {
        "tasks": sum(p["tasks"] for p in parts),
        "completed": sum(p["states"].get("COMPLETED", 0) for p in parts),
        "pulled": sum(p["pulled"] for p in parts),
        "merged": sum(p["merged"] for p in parts),
        "failed": sum(len(p["failed"]) for p in parts),
        "completed_primaries": sum(p["completed_primaries"] for p in parts),
        "completed_accepted_singles": sum(p["completed_accepted_singles"] for p in parts),
        "first_submit_epoch_s": min(submits) if submits else None,
        "last_end_epoch_s": max(ends) if ends else None,
    }
    return summary


def merge_state(db_path: Path, group_id: str | None) -> list:
    """Combined SRM sets written by campaign_tools/merge_brain_group.py --auto."""
    if not group_id:
        return []
    try:
        state = json.loads((db_path.parent / f"{group_id}_merge_state.json").read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return state.get("combined", []) if isinstance(state, dict) else []


def aggregate_slurm(reports: list[dict]) -> dict:
    """Combine the per-part sacct sections of a campaign group."""
    sections = [r.get("slurm") for r in reports if (r.get("slurm") or {}).get("available")]
    if not sections:
        return {"available": False, "reason": "no part report has sacct data"}
    states: dict[str, int] = {}
    for section in sections:
        for state, count in (section.get("states") or {}).items():
            states[state] = states.get(state, 0) + count
    starts = [s["first_submit_epoch_s"] for s in sections if s.get("first_submit_epoch_s")]
    ends = [
        s["first_submit_epoch_s"] + s.get("campaign_span_seconds", 0)
        for s in sections
        if s.get("first_submit_epoch_s")
    ]
    return {
        "available": True,
        "parts_with_sacct": len(sections),
        "states": states,
        "failed": sum(s.get("failed", 0) for s in sections),
        "running": sum(s.get("running", 0) for s in sections),
        "pending": sum(s.get("pending", 0) for s in sections),
        "completed": sum(s.get("completed", 0) for s in sections),
        "wait_seconds_sum": sum(s.get("wait_seconds_sum", 0) for s in sections),
        # parts run one after another, so their queue and run times add up
        "wait_wallclock_seconds": sum(s.get("wait_wallclock_seconds", 0) for s in sections),
        "run_wallclock_seconds": sum(s.get("run_wallclock_seconds", 0) for s in sections),
        "campaign_span_seconds": (max(ends) - min(starts)) if starts else 0.0,
        "first_submit_epoch_s": min(starts) if starts else None,
    }


class ProgressCache:
    def __init__(
        self,
        fetch_command: list[str] | None,
        local_path: Path | None,
        name: str = "default",
        label: str | None = None,
        ssh_host: str | None = None,
        remote_repo_root: str | None = None,
        root: Path | None = None,
    ):
        self.fetch_command = fetch_command
        self.local_path = local_path
        # When set, the newest batch under this root is re-resolved on every poll,
        # so a new campaign appears without restarting the service.
        self.root = root
        self.database: Path | None = None  # local job database (--database)
        self.name = name
        self.label = label or name
        self.ssh_host = ssh_host
        self.remote_repo_root = remote_repo_root
        self.group_id: str | None = None
        self.group_paths: list[Path] = []
        self.lock = threading.Lock()
        self.payload: dict = {"status": "starting"}
        self.fetched_at = 0.0
        self.error: str | None = None

    def resolve_path(self) -> Path | None:
        if self.root is None:
            return self.local_path
        candidates = sorted(self.root.glob("*/progress.json"), key=_mtime)
        return candidates[-1] if candidates else None

    def resolve_group(self) -> list[Path]:
        if self.root is None:
            return [self.local_path] if self.local_path is not None else []
        # newest = most recently written manifest (written at submission), not the
        # alphabetically last folder: names like brain_ab_<config>_* do not sort by time
        manifests = sorted(self.root.glob("*/campaign_manifest.json"), key=_mtime)
        # a campaign still waiting in the queue has no progress.json yet: show the
        # newest one that has started (or the newest overall if none has)
        started = [m for m in manifests if (m.parent / "progress.json").exists()]
        if started:
            manifests = manifests[: manifests.index(started[-1]) + 1]
        if not manifests:
            path = self.resolve_path()
            return [path] if path is not None else []
        latest_manifest = manifests[-1]
        try:
            latest = json.loads(latest_manifest.read_text())
        except (OSError, json.JSONDecodeError):
            return [latest_manifest.parent / "progress.json"]
        group_id = latest.get("campaign_group_id")
        if not group_id:
            return [latest_manifest.parent / "progress.json"]
        paths = []
        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("campaign_group_id") == group_id:
                paths.append(manifest_path.parent / "progress.json")
        self.group_id = group_id
        return sorted(paths)

    def _read_manifest(self, path: Path | None = None) -> dict:
        # campaign_manifest.json sits next to progress.json; num_loops isn't in
        # the report itself, so pull it here to compute a loop-level live
        # progress percentage that doesn't wait for whole tasks to finish.
        path = path or self.local_path
        if path is None:
            return {}
        manifest_path = path.parent / "campaign_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:  # noqa: BLE001 - manifest is optional context
            return {}
        result = {
            "num_loops": manifest.get("num_loops"),
            "job_count": manifest.get("job_count"),
            "expected_events_total": manifest.get("expected_events_total"),
            "campaign_part_index": manifest.get("campaign_part_index"),
            "campaign_part_count": manifest.get("campaign_part_count"),
        }
        provenance_name = manifest.get("geometry_provenance_file")
        if isinstance(provenance_name, str):
            provenance_path = manifest_path.parent / provenance_name
            try:
                result["geometry"] = json.loads(provenance_path.read_text())
            except Exception:  # noqa: BLE001 - provenance is optional context
                pass
        return result

    @staticmethod
    def _sum_fields(reports: list[dict], section: str, fields: list[str]) -> dict:
        return {
            field: sum((report.get(section, {}).get(field) or 0) for report in reports)
            for field in fields
        }

    def _aggregate_group(self, paths: list[Path]) -> dict:
        reports = []
        parts = []
        expected_tasks = 0
        expected_events = 0
        num_loops = None
        for path in paths:
            manifest = self._read_manifest(path)
            expected_tasks += manifest.get("job_count") or 0
            expected_events += manifest.get("expected_events_total") or 0
            num_loops = num_loops or manifest.get("num_loops")
            try:
                report = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                parts.append(
                    {
                        "campaign_dir": str(path.parent),
                        "manifest": manifest,
                        "progress": None,
                    }
                )
                continue
            report["manifest"] = manifest
            reports.append(report)
            parts.append(
                {
                    "campaign_dir": str(path.parent),
                    "manifest": manifest,
                    "progress": report,
                }
            )
        if not reports:
            raise FileNotFoundError("no readable campaign reports in group")

        task_fields = ["expected", "observed", "complete", "incomplete"]
        event_fields = [
            "committed_primaries",
            "in_flight_primaries",
            "committed_raw_singles",
            "committed_accepted_singles",
            "committed_tracks",
            "committed_steps",
            "loops_complete",
            "loops_in_flight",
        ]
        time_fields = [
            "committed_simulation_seconds",
            "in_flight_simulation_seconds",
            "committed_init_seconds",
            "task_wall_seconds_sum",
            "task_wall_wallclock_seconds",
        ]
        tasks = self._sum_fields(reports, "tasks", task_fields)
        events = self._sum_fields(reports, "events", event_fields)
        timing = self._sum_fields(reports, "time", time_fields)
        if expected_tasks:
            tasks["expected"] = expected_tasks
            tasks["incomplete"] = max(0, expected_tasks - tasks["complete"])
        expected = tasks["expected"]
        tasks["percent_complete"] = (
            100.0 * tasks["complete"] / expected if expected else 0.0
        )
        sim_seconds = timing["committed_simulation_seconds"]
        rates = {}
        for field, numerator in (
            ("primaries_per_second", events["committed_primaries"]),
            ("tracks_per_second", events["committed_tracks"]),
            ("steps_per_second", events["committed_steps"]),
            ("raw_singles_per_second", events["committed_raw_singles"]),
            ("accepted_singles_per_second", events["committed_accepted_singles"]),
        ):
            rates[field] = numerator / sim_seconds if sim_seconds else None
        rates["accepted_fraction_of_raw"] = (
            events["committed_accepted_singles"] / events["committed_raw_singles"]
            if events["committed_raw_singles"]
            else None
        )
        rates["singles_per_primary"] = (
            events["committed_raw_singles"] / events["committed_primaries"]
            if events["committed_primaries"]
            else None
        )
        latest = reports[-1]
        profile = {}
        for field in ("peak_rss_mb", "requested_cpus"):
            values = [r.get("compute_profile", {}).get(field) for r in reports]
            values = [value for value in values if value is not None]
            profile[field] = max(values) if values else None
        profile["tasks_with_profile"] = sum(
            r.get("compute_profile", {}).get("tasks_with_profile", 0) or 0
            for r in reports
        )
        profile["mean_cpu_pct"] = None
        profile["mean_cpu_pct_of_allocation"] = None
        profile["mean_peak_rss_mb"] = None
        return {
            "generated_at": time.time(),
            "campaign_dir": str(self.root),
            "campaign_group_id": self.group_id,
            "campaign_parts": len(paths),
            "campaign_parts_reported": len(reports),
            "parts": parts,
            "manifest": {
                "num_loops": num_loops,
                "job_count": expected_tasks,
                "expected_events_total": expected_events,
            },
            "srm_labels": latest.get("srm_labels", []),
            "tasks": tasks,
            "events": events,
            "rates": rates,
            "time": {
                **timing,
                "simulation_fraction_of_wall": (
                    sim_seconds / timing["task_wall_seconds_sum"]
                    if timing["task_wall_seconds_sum"]
                    else None
                ),
            },
            "compute_profile": profile,
            "slurm": aggregate_slurm(reports),
            "srm": latest.get("srm", {}),
            "group_srm_note": "SRM fields show the newest part until a grouped reduction is generated.",
        }

    def refresh(self) -> None:
        try:
            if self.fetch_command is None:
                paths = self.resolve_group()
                if not paths:
                    with self.lock:
                        self.error = f"no campaign report yet under {self.root}"
                    return
                if self.root is not None and len(paths) > 1:
                    self.group_paths = paths
                    data = self._aggregate_group(paths)
                    self._attach_database(data)
                    with self.lock:
                        self.payload = data
                        self.fetched_at = time.time()
                        self.error = None
                    return
                path = paths[0]
                self.group_paths = paths
                text = path.read_text()
                self.local_path = path
            else:
                completed = subprocess.run(
                    self.fetch_command,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=True,
                )
                text = completed.stdout
            data = json.loads(text)
            data["manifest"] = self._read_manifest()
            self._attach_database(data)
        except Exception as exc:  # noqa: BLE001 - surfaced to the dashboard
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
            return
        with self.lock:
            self.payload = data
            self.fetched_at = time.time()
            self.error = None

    def _attach_database(self, data: dict) -> None:
        if self.database is None:
            return
        campaigns = [path.parent.name for path in self.group_paths] or (
            [self.local_path.parent.name] if self.local_path else []
        )
        try:
            # projection snapshots live next to the database, in the pulled parts
            snapshots = snapshot_srm(self.database.parent, campaigns)
            if any(entry.get("available") for entry in snapshots.values()):
                data["srm"] = snapshots
                data["srm_labels"] = list(SNAPSHOT_LABELS)
                data["group_srm_note"] = (
                    "SRM views are summed from per-part projection snapshots of all "
                    "pulled complete tasks; element-level statistics need a merged SRM."
                )
        except (OSError, ValueError, KeyError) as exc:
            data["srm_snapshot_error"] = f"{type(exc).__name__}: {exc}"
        try:
            data["database"] = database_summary(self.database, campaigns)
            data["database"]["combined_sets"] = merge_state(
                self.database, data.get("campaign_group_id") or self.group_id
            )
            self._attach_pixel_source(data, campaigns)
        except sqlite3.Error as exc:
            data["database"] = {"available": False, "reason": f"sqlite: {exc}"}

    def _attach_pixel_source(self, data: dict, campaigns: list[str]) -> None:
        """Per-pixel queries need an SRM: use the newest merged one (the latest
        combined set of complete parts, else the newest merged single part)."""
        root = self.database.parent
        candidates = [Path(entry["output"]) for entry in data["database"]["combined_sets"]]
        candidates += [root / name for name in reversed(campaigns)]
        for candidate in candidates:
            metadata = candidate / "combined_srm_metadata.json"
            if not metadata.is_file():
                continue
            try:
                merged = json.loads(metadata.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if merged.get("layout") != "per_head_csr":
                continue
            for label, entry in (data.get("srm") or {}).items():
                if entry.get("available") and label in merged.get("resolutions", {}):
                    entry["path"] = str(candidate / f"final_srm_{label}.npz")
                    entry["pixel_source"] = candidate.name
            return

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "campaign": self.name,
                "label": self.label,
                "fetched_at": self.fetched_at,
                "error": self.error,
                "progress": self.payload,
            }


def poll_loop(
    caches: list[ProgressCache], interval_s: float, stop: threading.Event
) -> None:
    while not stop.is_set():
        for cache in caches:
            cache.refresh()
        stop.wait(interval_s)


def make_handler(caches: dict[str, ProgressCache], default_name: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, payload: dict) -> None:
            self._send(
                code,
                json.dumps(payload, default=str).encode(),
                "application/json",
            )

        def _select_cache(self, query: dict) -> ProgressCache | None:
            name = (query.get("campaign") or [default_name])[0]
            return caches.get(name)

        def _pixel_query(self) -> None:
            query = parse_qs(urlsplit(self.path).query)
            cache = self._select_cache(query)
            if cache is None:
                self._send_json(404, {"available": False, "error": "unknown campaign"})
                return
            with cache.lock:
                grouped_has_srm = any(
                    (entry or {}).get("path")
                    for entry in (cache.payload.get("srm") or {}).values()
                )
            if len(cache.group_paths) > 1 and not grouped_has_srm:
                self._send_json(
                    409,
                    {
                        "available": False,
                        "error": "pixel queries require a grouped SRM reduction output",
                    },
                )
                return
            ssh_host = cache.ssh_host
            remote_repo_root = cache.remote_repo_root
            try:
                label = query["label"][0]
                crystal = int(query["crystal"][0])
                pixel = int(query["pixel"][0])
            except (KeyError, IndexError, ValueError):
                self._send_json(
                    400,
                    {"available": False, "error": "invalid pixel query"},
                )
                return

            with cache.lock:
                report = cache.payload
            srm = report.get("srm", {}).get(label, {})
            srm_path = srm.get("path")
            if not srm_path:
                self._send_json(
                    404,
                    {"available": False, "error": f"SRM label not found: {label}"},
                )
                return

            if ssh_host:
                # progress.json was fetched over SSH, so srm_path is a path on the
                # remote host, not this machine — the .npz itself was never copied here.
                if not remote_repo_root:
                    self._send_json(
                        501,
                        {
                            "available": False,
                            "error": (
                                "Pixel queries need --remote-repo-root when using "
                                "--ssh-host, so report_campaign_progress.py can be run "
                                "on the remote host against its own filesystem."
                            ),
                        },
                    )
                    return
                remote_srm_dir = posixpath.dirname(srm_path)
                remote_script = posixpath.join(
                    remote_repo_root, "payload", "python", "report_campaign_progress.py"
                )
                remote_cmd = (
                    f"python3 {shlex.quote(remote_script)} --pixel-query "
                    f"--srm-dir {shlex.quote(remote_srm_dir)} "
                    f"--srm-labels {shlex.quote(label)} "
                    f"--crystal {crystal} --pixel {pixel}"
                )
                command = ["ssh", "-C", ssh_host, remote_cmd]
            else:
                command = [
                    sys.executable,
                    str(
                        STATIC_DIR.parent
                        / "payload"
                        / "python"
                        / "report_campaign_progress.py"
                    ),
                    "--pixel-query",
                    "--srm-dir",
                    str(Path(srm_path).parent),
                    "--srm-labels",
                    label,
                    "--crystal",
                    str(crystal),
                    "--pixel",
                    str(pixel),
                ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=True,
                )
                payload = json.loads(completed.stdout)
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
                self._send_json(
                    500,
                    {"available": False, "error": f"{type(exc).__name__}: {exc}"},
                )
                return
            self._send_json(200, payload)

        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path.startswith("/api/campaigns"):
                self._send_json(
                    200,
                    {
                        "default": default_name,
                        "campaigns": [
                            {"name": cache.name, "label": cache.label}
                            for cache in caches.values()
                        ],
                    },
                )
                return
            if self.path.startswith("/api/progress"):
                query = parse_qs(urlsplit(self.path).query)
                cache = self._select_cache(query)
                if cache is None:
                    self._send_json(404, {"error": "unknown campaign"})
                    return
                self._send_json(200, cache.snapshot())
                return
            if self.path.startswith("/api/pixel"):
                self._pixel_query()
                return
            if self.path in ("/", "/index.html"):
                index = STATIC_DIR / "index.html"
                if not index.is_file():
                    self._send(404, b"index.html missing", "text/plain")
                    return
                self._send(200, index.read_bytes(), "text/html; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local SRM progress dashboard.")
    parser.add_argument(
        "--local-json",
        type=Path,
        help="Path to a progress JSON on this machine.",
    )
    parser.add_argument(
        "--ssh-host",
        help="SSH destination, e.g. user@data.bridges2.psc.edu.",
    )
    parser.add_argument(
        "--fetch-command",
        help="Arbitrary shell command whose stdout is the progress JSON.",
    )
    parser.add_argument(
        "--campaign",
        action="append",
        metavar="NAME=PATH",
        help=(
            "Serve several campaigns at once, e.g. "
            "--campaign 'Brain (Expanse)=/data/.../progress.json'. Repeatable; the "
            "dashboard shows a selector. Paths must be local to this machine."
        ),
    )
    parser.add_argument(
        "--campaign-root",
        action="append",
        metavar="NAME=DIR",
        help=(
            "Like --campaign, but points at a directory holding batch_*/progress.json. "
            "The newest batch is re-resolved on every poll, so a new campaign is picked "
            "up without restarting. Repeatable, and can be mixed with --campaign."
        ),
    )
    parser.add_argument(
        "--remote-json",
        help="Path to the progress JSON on the SSH host (with --ssh-host).",
    )
    parser.add_argument(
        "--remote-repo-root",
        help=(
            "Path to this repo's root on the SSH host (with --ssh-host), so pixel-level "
            "queries can run report_campaign_progress.py remotely against the actual "
            ".npz files instead of failing with 'SRM not found' on this machine."
        ),
    )
    parser.add_argument(
        "--database",
        action="append",
        metavar="NAME=PATH",
        help=(
            "Local job database (update_local_slurm_campaign_database.py) for the "
            "campaign NAME (its --campaign/--campaign-root label): task states, "
            "pull/merge status, failures and Slurm timing are added to the dashboard."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (
        args.local_json
        or args.ssh_host
        or args.fetch_command
        or args.campaign
        or args.campaign_root
    ):
        raise SystemExit(
            "one of --local-json, --ssh-host, --fetch-command, --campaign, "
            "or --campaign-root is required"
        )
    if args.ssh_host and not args.remote_json:
        raise SystemExit("--ssh-host requires --remote-json")

    caches: dict[str, ProgressCache] = {}
    if args.campaign or args.campaign_root:
        for entry, is_root in [(item, False) for item in (args.campaign or [])] + [
            (item, True) for item in (args.campaign_root or [])
        ]:
            label, separator, raw_path = entry.partition("=")
            if not separator or not raw_path.strip():
                raise SystemExit(f"--campaign expects NAME=PATH, got: {entry!r}")
            label = label.strip()
            # The name is what travels in query strings; the label is for display.
            name = "".join(
                character if character.isalnum() else "_" for character in label
            ).strip("_")
            if name in caches:
                raise SystemExit(f"duplicate campaign name: {label}")
            target = Path(raw_path.strip()).expanduser()
            caches[name] = ProgressCache(
                None,
                None if is_root else target,
                name=name,
                label=label,
                remote_repo_root=args.remote_repo_root,
                root=target if is_root else None,
            )
    else:
        fetch_command = None
        if args.ssh_host:
            # -C matters: the projection payload is JSON and compresses roughly 10x.
            fetch_command = [
                "ssh",
                "-C",
                args.ssh_host,
                f"cat {shlex.quote(args.remote_json)}",
            ]
        elif args.fetch_command:
            fetch_command = shlex.split(args.fetch_command)
        caches["default"] = ProgressCache(
            fetch_command,
            args.local_json,
            name="default",
            label="campaign",
            ssh_host=args.ssh_host,
            remote_repo_root=args.remote_repo_root,
        )

    labels = {cache.label: cache for cache in caches.values()}
    for entry in args.database or []:
        label, separator, raw_path = entry.partition("=")
        if not separator or label.strip() not in labels:
            raise SystemExit(f"--database expects a campaign label=PATH, got: {entry!r}")
        labels[label.strip()].database = Path(raw_path.strip()).expanduser()

    default_name = next(iter(caches))
    stop = threading.Event()
    thread = threading.Thread(
        target=poll_loop,
        args=(list(caches.values()), args.interval_s, stop),
        daemon=True,
    )
    thread.start()

    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(caches, default_name)
    )
    print(f"SRM monitor on http://{args.host}:{args.port} (poll {args.interval_s}s)")
    for cache in caches.values():
        if cache.fetch_command:
            print(f"  {cache.label}: {' '.join(cache.fetch_command)}")
        elif cache.root is not None:
            print(f"  {cache.label}: newest batch under {cache.root}")
        else:
            print(f"  {cache.label}: local file {cache.local_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
