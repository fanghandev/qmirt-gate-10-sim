#!/usr/bin/env python3
"""Maintain local bookkeeping for a Slurm brain-SPECT campaign."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

DEFAULT_ROOT = Path("/data/fanghan/opengate_sim/data/brain_spect")
DB_NAME = "expanse_slurm_jobs.db"


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def discover_job_id(campaign_dir: Path) -> str:
    """Read the simulation array job ID recorded in the campaign manifest."""
    manifest = read_json(campaign_dir / "campaign_manifest.json")
    slurm_jobs = manifest.get("slurm_jobs")
    if isinstance(slurm_jobs, dict):
        job_id = slurm_jobs.get("array")
        if job_id not in (None, "", "null"):
            return str(job_id)
    for key in ("slurm_job_id", "job_id", "array_job_id"):
        job_id = manifest.get(key)
        if job_id not in (None, "", "null"):
            return str(job_id)
    raise ValueError(
        f"Could not determine Slurm job ID from {campaign_dir / 'campaign_manifest.json'}; "
        "pass --job-id explicitly"
    )


def parse_epoch(value: str) -> int | None:
    value = value.strip()
    if not value or value in {"Unknown", "None", "N/A"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(datetime.fromisoformat(value).timestamp())
        except ValueError:
            return None


def parse_task_id(job_id: str, array_id: str) -> int | None:
    if array_id not in {"", "N/A", "-1", "Unknown"}:
        try:
            return int(array_id)
        except ValueError:
            return None
    if "_" in job_id:
        try:
            return int(job_id.rsplit("_", 1)[1])
        except ValueError:
            return None
    return None


def query_slurm(job_id: str) -> list[tuple]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "expanse",
        "sacct",
        "-j",
        job_id,
        "-X",
        "-n",
        "-P",
        "--format=JobID,State,Submit,Start,End,ExitCode,Eligible",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ssh/sacct timed out") from exc
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "sacct failed")

    rows = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) != 7:
            continue
        array_job_id, state, submit, start, end, exit_code, eligible = fields
        task_id = parse_task_id(array_job_id, "")
        if task_id is None:
            continue
        rows.append(
            (
                job_id,
                task_id,
                state.split()[0],
                parse_epoch(submit),
                parse_epoch(start),
                parse_epoch(end),
                exit_code,
                parse_epoch(eligible),
            )
        )
    return rows


def reconcile_task_ids(rows: list[tuple], campaign_dir: Path) -> list[tuple]:
    """Align sacct array IDs with local task_N directory IDs.

    Slurm arrays are normally zero-based here. A consistently shifted local
    directory set is corrected only when there is no overlap; ambiguous partial
    sets are rejected rather than silently assigned to the wrong task.
    """
    file_ids = {
        int(path.name.removeprefix("task_"))
        for path in campaign_dir.glob("task_*")
        if path.is_dir() and path.name.removeprefix("task_").isdigit()
    }
    if not rows or not file_ids:
        return rows

    slurm_ids = {int(row[1]) for row in rows}
    if file_ids <= slurm_ids:
        return rows

    candidates = [
        offset
        for offset in (-1, 1)
        if {task_id + offset for task_id in slurm_ids} >= file_ids
    ]
    if len(candidates) != 1:
        missing = sorted(file_ids - slurm_ids)
        raise ValueError(
            f"Local task IDs do not match Slurm array IDs for {campaign_dir}: "
            f"file-only IDs={missing[:10]}; pass an explicit corrected dataset "
            "or inspect the campaign index base"
        )

    offset = candidates[0]
    corrected = [(*row[:1], int(row[1]) + offset, *row[2:]) for row in rows]
    print(
        f"Corrected Slurm task IDs by offset {offset:+d} to match local task_N files.",
        flush=True,
    )
    return corrected


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS slurm_jobs (
            job_id TEXT NOT NULL,
            array_task_id INTEGER NOT NULL,
            state TEXT,
            submit_epoch INTEGER,
            start_epoch INTEGER,
            end_epoch INTEGER,
            exit_code TEXT,
            eligible_epoch INTEGER,
            campaign_name TEXT,
            task_name TEXT,
            primaries INTEGER,
            raw_singles INTEGER,
            accepted_singles INTEGER,
            is_pulled INTEGER NOT NULL DEFAULT 0,
            is_merged INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (job_id, array_task_id)
        )
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(slurm_jobs)")}
    for name, definition in {
        "eligible_epoch": "INTEGER",
        "campaign_name": "TEXT",
        "task_name": "TEXT",
        "primaries": "INTEGER",
        "raw_singles": "INTEGER",
        "accepted_singles": "INTEGER",
        "is_pulled": "INTEGER NOT NULL DEFAULT 0",
        "is_merged": "INTEGER NOT NULL DEFAULT 0",
    }.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE slurm_jobs ADD COLUMN {name} {definition}")
    conn.commit()
    return conn


def task_counts(
    task_dir: Path, label: str, progress_entry: dict | None = None
) -> tuple[int, int, int] | None:
    metadata = read_json(task_dir / "combined_srm_metadata.json")
    entry = metadata.get("resolutions", {}).get(label, {})
    if not isinstance(entry, dict):
        return None
    primaries = int(
        entry.get("simulated_primaries", 0)
        or (progress_entry or {}).get("primaries", 0)
        or 0
    )
    raw = int(
        entry.get("raw_events", 0) or (progress_entry or {}).get("raw_singles", 0) or 0
    )
    accepted = int(
        entry.get("accepted_events")
        or entry.get("accumulated_counts")
        or (progress_entry or {}).get("accepted_singles", 0)
        or 0
    )
    return primaries, raw, accepted


def upsert_slurm_rows(
    conn: sqlite3.Connection, campaign: str, rows: list[tuple]
) -> None:
    conn.executemany(
        """
        INSERT INTO slurm_jobs
            (job_id, array_task_id, state, submit_epoch, start_epoch, end_epoch,
             exit_code, eligible_epoch, campaign_name, task_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(job_id, array_task_id) DO UPDATE SET
            state=excluded.state,
            submit_epoch=COALESCE(excluded.submit_epoch, slurm_jobs.submit_epoch),
            start_epoch=COALESCE(excluded.start_epoch, slurm_jobs.start_epoch),
            end_epoch=COALESCE(excluded.end_epoch, slurm_jobs.end_epoch),
            exit_code=excluded.exit_code,
            eligible_epoch=COALESCE(excluded.eligible_epoch, slurm_jobs.eligible_epoch),
            campaign_name=excluded.campaign_name,
            task_name=excluded.task_name
        """,
        [row + (campaign, f"task_{row[1]}") for row in rows],
    )
    conn.commit()


def update_pulled_tasks(
    conn: sqlite3.Connection, campaign_dir: Path, job_id: str, label: str
) -> int:
    updated = 0
    progress = read_json(campaign_dir / "cluster_progress.json")
    progress_by_task = {
        str(item.get("task_id")): item
        for item in progress.get("per_task", [])
        if isinstance(item, dict) and item.get("task_id") is not None
    }
    conn.execute(
        "DELETE FROM slurm_jobs WHERE campaign_name = ? AND job_id != ?",
        (campaign_dir.name, job_id),
    )
    for task_dir in sorted(campaign_dir.glob("task_*")):
        if not task_dir.is_dir() or not task_dir.name.removeprefix("task_").isdigit():
            continue
        task_id = int(task_dir.name.removeprefix("task_"))
        counts = task_counts(
            task_dir,
            label,
            progress_by_task.get(task_dir.name.removeprefix("task_")),
        )
        if counts is None:
            continue
        complete = (task_dir / "TASK_COMPLETE.json").is_file()
        primaries, raw, accepted = counts
        conn.execute(
            """
            INSERT INTO slurm_jobs
                (job_id, array_task_id, state, campaign_name, task_name,
                 primaries, raw_singles, accepted_singles, is_pulled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, array_task_id) DO UPDATE SET
                -- a completed task on disk wins over sacct's record of an earlier
                -- failed attempt (the task was rerun as a separate job)
                state=CASE WHEN excluded.state = 'COMPLETED' THEN 'COMPLETED'
                           ELSE slurm_jobs.state END,
                campaign_name=excluded.campaign_name,
                task_name=excluded.task_name,
                primaries=excluded.primaries,
                raw_singles=excluded.raw_singles,
                accepted_singles=excluded.accepted_singles,
                is_pulled=excluded.is_pulled
            """,
            (
                job_id,
                task_id,
                "COMPLETED" if complete else "PARTIAL",
                campaign_dir.name,
                task_dir.name,
                primaries,
                raw,
                accepted,
                int(complete),
            ),
        )
        updated += 1
    conn.commit()
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument(
        "--job-id",
        default=None,
        help="Slurm array job ID (default: campaign_manifest.json slurm_jobs.array).",
    )
    parser.add_argument(
        "--label", default="1mm", help="Resolution used for task counts."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"Shared database path (default: {DEFAULT_ROOT / DB_NAME}).",
    )
    parser.add_argument(
        "--no-sacct", action="store_true", help="Only scan pulled task files."
    )
    parser.add_argument(
        "--sacct-optional",
        action="store_true",
        help="If sacct over ssh fails (e.g. no open ControlMaster socket for the "
        "2FA login), warn and update from local files only, keeping the stored "
        "Slurm states. For unattended runs (globus_pull_campaign.py --update-db).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    if not campaign_dir.is_dir():
        raise SystemExit(f"Campaign directory not found: {campaign_dir}")
    job_id = args.job_id or discover_job_id(campaign_dir)
    db_path = args.db or DEFAULT_ROOT / DB_NAME
    conn = init_db(db_path)
    try:
        if not args.no_sacct:
            try:
                rows = query_slurm(job_id)
            except RuntimeError as exc:
                if not args.sacct_optional:
                    raise
                print(f"Warning: sacct unavailable ({exc}); updating from local files only.")
            else:
                rows = reconcile_task_ids(rows, campaign_dir)
                upsert_slurm_rows(conn, campaign_dir.name, rows)
                print(f"Updated {len(rows)} Slurm array records from sacct.")
        updated = update_pulled_tasks(conn, campaign_dir, job_id, args.label)
        print(f"Updated {updated} pulled task records from local files.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
