#!/usr/bin/env python3
"""Report Slurm state and SRM counts across brain campaigns."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("/data/fanghan/opengate_sim/data/brain_spect")
DB_NAME = "expanse_slurm_jobs.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=None,
        help="Limit results to this campaign; omit to report all campaigns.",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Limit results to one Slurm array job; omit to report all jobs.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"Shared database path (default: {DEFAULT_ROOT / DB_NAME}).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser.parse_args()


SUMMARY_FIELDS = (
    "task_count",
    "job_count",
    "primaries_total",
    "primaries_avg_task",
    "raw_total",
    "raw_avg_task",
    "accepted_total",
    "accepted_avg_task",
    "pulled_tasks",
    "merged_tasks",
    "completed_tasks",
    "partial_tasks",
    "wall_start_epoch",
    "wall_end_epoch",
    "wall_seconds",
)


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the first version of the shared database."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(slurm_jobs)")}
    if not columns:
        raise SystemExit("Database does not contain a slurm_jobs table.")
    for name, definition in {
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


def query_summaries(
    conn: sqlite3.Connection, campaign_name: str | None, job_id: str | None
) -> tuple[list[dict], list[dict]]:
    clauses = []
    params: list[str] = []
    if campaign_name:
        clauses.append("campaign_name = ?")
        params.append(campaign_name)
    if job_id:
        clauses.append("job_id = ?")
        params.append(job_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    aggregate = """
        COUNT(*) AS task_count,
        COUNT(DISTINCT job_id) AS job_count,
        COALESCE(SUM(primaries), 0) AS primaries_total,
        COALESCE(AVG(primaries), 0) AS primaries_avg_task,
        COALESCE(SUM(raw_singles), 0) AS raw_total,
        COALESCE(AVG(raw_singles), 0) AS raw_avg_task,
        COALESCE(SUM(accepted_singles), 0) AS accepted_total,
        COALESCE(AVG(accepted_singles), 0) AS accepted_avg_task,
        COALESCE(SUM(is_pulled), 0) AS pulled_tasks,
        COALESCE(SUM(is_merged), 0) AS merged_tasks,
        SUM(CASE WHEN state = 'COMPLETED' THEN 1 ELSE 0 END) AS completed_tasks,
        SUM(CASE WHEN state = 'PARTIAL' THEN 1 ELSE 0 END) AS partial_tasks,
        MIN(submit_epoch) AS wall_start_epoch,
        MAX(COALESCE(end_epoch, start_epoch, submit_epoch)) AS wall_end_epoch,
        CASE
            WHEN MIN(submit_epoch) IS NOT NULL
             AND MAX(COALESCE(end_epoch, start_epoch, submit_epoch)) IS NOT NULL
            THEN MAX(COALESCE(end_epoch, start_epoch, submit_epoch))
                 - MIN(submit_epoch)
            ELSE 0
        END AS wall_seconds
    """
    campaign_rows = conn.execute(
        f"SELECT campaign_name, {aggregate} FROM slurm_jobs {where} "
        "GROUP BY campaign_name ORDER BY campaign_name",
        params,
    ).fetchall()
    job_rows = conn.execute(
        f"SELECT campaign_name, job_id, {aggregate} FROM slurm_jobs {where} "
        "GROUP BY campaign_name, job_id ORDER BY campaign_name, job_id",
        params,
    ).fetchall()

    campaign_names = ("campaign_name", *SUMMARY_FIELDS)
    job_names = ("campaign_name", "job_id", *SUMMARY_FIELDS)
    return (
        [dict(zip(campaign_names, row)) for row in campaign_rows],
        [dict(zip(job_names, row)) for row in job_rows],
    )


def print_summary(campaigns: list[dict], jobs: list[dict]) -> None:
    def value(row: dict, key: str) -> str:
        return f"{int(row[key] or 0):,}"

    def name(row: dict) -> str:
        return str(row["campaign_name"] or "<unknown>")

    def duration(seconds: int | float) -> str:
        seconds = max(0, int(seconds or 0))
        days, remainder = divmod(seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        if days:
            return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def timestamp(value: int | None) -> str:
        return datetime.fromtimestamp(value).isoformat(sep=" ") if value else "n/a"

    def print_record(row: dict, label: str, identifier: str) -> None:
        print(f"{label}: {identifier}")
        print(f"  jobs={row['job_count']} tasks={row['task_count']}")
        print(
            f"  totals: primaries={value(row, 'primaries_total')} "
            f"raw={value(row, 'raw_total')} "
            f"accepted={value(row, 'accepted_total')}"
        )
        print(
            f"  averages/task: primaries={row['primaries_avg_task']:,.2f} "
            f"raw={row['raw_avg_task']:,.2f} "
            f"accepted={row['accepted_avg_task']:,.2f}"
        )
        print(
            f"  status: pulled={row['pulled_tasks']} merged={row['merged_tasks']} "
            f"completed={row['completed_tasks']} partial={row['partial_tasks']}"
        )
        print(
            f"  wall time: {duration(row['wall_seconds'])} "
            f"(start={timestamp(row['wall_start_epoch'])} "
            f"end={timestamp(row['wall_end_epoch'])})"
        )

    def total_row(rows: list[dict]) -> dict[str, Any]:
        total: dict[str, Any] = {key: 0 for key in SUMMARY_FIELDS}
        total["campaign_name"] = "ALL CAMPAIGNS"
        for row in rows:
            for key in SUMMARY_FIELDS:
                if key not in {"wall_start_epoch", "wall_end_epoch", "wall_seconds"}:
                    total[key] += row.get(key) or 0
        total["primaries_avg_task"] = (
            total["primaries_total"] / total["task_count"] if total["task_count"] else 0
        )
        total["raw_avg_task"] = (
            total["raw_total"] / total["task_count"] if total["task_count"] else 0
        )
        total["accepted_avg_task"] = (
            total["accepted_total"] / total["task_count"] if total["task_count"] else 0
        )
        starts = [row["wall_start_epoch"] for row in rows if row["wall_start_epoch"]]
        ends = [row["wall_end_epoch"] for row in rows if row["wall_end_epoch"]]
        total["wall_start_epoch"] = min(starts) if starts else None
        total["wall_end_epoch"] = max(ends) if ends else None
        total["wall_seconds"] = (
            total["wall_end_epoch"] - total["wall_start_epoch"]
            if total["wall_start_epoch"] is not None
            and total["wall_end_epoch"] is not None
            else 0
        )
        return total

    print("Campaign summary")
    for row in campaigns:
        print_record(row, "Campaign", name(row))
        print()
    if len(campaigns) > 1:
        print_record(total_row(campaigns), "Total", "ALL CAMPAIGNS")
        print()

    print("\nJob summary")
    for row in jobs:
        print_record(row, "Job", f"{name(row)} / {row['job_id']}")
        print()


def main() -> int:
    args = parse_args()
    campaign_dir = (
        args.campaign_dir.expanduser().resolve() if args.campaign_dir else None
    )
    db_path = args.db or DEFAULT_ROOT / DB_NAME
    if not db_path.is_file():
        raise SystemExit(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        campaigns, jobs = query_summaries(
            conn, campaign_dir.name if campaign_dir else None, args.job_id
        )
    finally:
        conn.close()

    if not campaigns:
        raise SystemExit("No matching records found.")
    if args.json:
        print(json.dumps({"campaigns": campaigns, "jobs": jobs}, indent=2, default=str))
    else:
        print_summary(campaigns, jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
