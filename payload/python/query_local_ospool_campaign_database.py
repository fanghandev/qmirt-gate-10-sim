import argparse
import sqlite3
from pathlib import Path
from typing import Any, List

import polars as pl

LOCAL_DATA_ROOT = Path("/data/fanghan/opengate_sim/data/cardiac_spect")


def query_for_jobs(conn: sqlite3.Connection, cluster_id: int):
    """Query the database for completed jobs for a specific ClusterId."""
    cursor = conn.cursor()
    cursor.execute(
        """
           SELECT ClusterId, ProcId, Cmd, JobStatus, is_pulled, is_merged,
                CASE
                    WHEN QDate IS NULL OR QDate = -1 THEN NULL
                    WHEN typeof(QDate) IN ('integer', 'real')
                        THEN CAST(QDate AS INTEGER)
                    WHEN trim(QDate) <> '' AND trim(QDate) NOT GLOB '*[^0-9]*'
                        THEN CAST(QDate AS INTEGER)
                    ELSE unixepoch(QDate)
                END AS QDate,
                CASE
                    WHEN JobStartDate IS NULL OR JobStartDate = -1 THEN NULL
                    WHEN typeof(JobStartDate) IN ('integer', 'real')
                        THEN CAST(JobStartDate AS INTEGER)
                    WHEN trim(JobStartDate) <> ''
                         AND trim(JobStartDate) NOT GLOB '*[^0-9]*'
                        THEN CAST(JobStartDate AS INTEGER)
                    ELSE unixepoch(JobStartDate)
                END AS JobStartDate,
                CASE
                    WHEN CompletionDate IS NULL OR CompletionDate = -1 THEN NULL
                    WHEN typeof(CompletionDate) IN ('integer', 'real')
                        THEN CAST(CompletionDate AS INTEGER)
                    WHEN trim(CompletionDate) <> ''
                         AND trim(CompletionDate) NOT GLOB '*[^0-9]*'
                        THEN CAST(CompletionDate AS INTEGER)
                    ELSE unixepoch(CompletionDate)
                END AS CompletionDate
        FROM condor_jobs
        WHERE ClusterId = ?
        """,
        (cluster_id,),
    )
    return cursor.fetchall()


def _resolve_db_path(name: str, db_fname: str = "ospool_condor_jobs.db") -> str:
    """Resolves the SQLite database path based on the batch name."""
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("name must be a batch directory name, not a path")

    data_dir = LOCAL_DATA_ROOT / name
    db_path = data_dir / db_fname
    return str(db_path)


def get_parsed_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query OSPool jobs and upsert into SQLite."
    )
    parser.add_argument(
        "-c",
        "--cluster-id",
        dest="cluster_id",
        required=True,
        type=int,
        help="The HTCondor Cluster ID to query.",
    )
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        metavar="BATCH_NAME",
        help="Batch directory name, for example batch_20260909_012610.",
    )
    return parser.parse_args()


def get_parsed_job_status(job_status: List[int]) -> List[str]:
    job_status_mapping = [
        "Unknown",  # Placeholder for unknown status (index 0, will be overridden by actual statuses)
        "Idle",  # 1
        "Running",  # 2
        "Removed by HTCondor",  # 3
        "Completed",  # 4
        "Held",  # 5
        "Transferring Output",  # 6
        "Suspended",  # 7
        "Removed (missing output)",  # 8, project-specific terminal status
    ]
    return [
        job_status_mapping[status]
        if 0 <= status < len(job_status_mapping)
        else f"Unknown({status})"
        for status in job_status
    ]


def analyze_jobs(jobs: List[Any]):
    """Analyze and print the queried jobs."""
    if not jobs:
        print("No jobs found for the specified ClusterId.")
        return

    df = pl.DataFrame(
        jobs,
        schema=[
            "ClusterId",
            "ProcId",
            "Cmd",
            "JobStatus",
            "is_pulled",
            "is_merged",
            "QDate",
            "JobStartDate",
            "CompletionDate",
        ],
        orient="row",
    )
    # Count numbers of finished jobs
    finished_jobs_count = df.filter(pl.col("JobStatus") == 4).height
    pulled_jobs_count = df.filter(
        (pl.col("is_pulled") == 1) & (pl.col("JobStatus") == 4)
    ).height
    merged_jobs_count = df.filter(
        (pl.col("is_merged") == 1) & (pl.col("JobStatus") == 4)
    ).height
    running_jobs_count = df.filter(pl.col("JobStatus") == 2).height
    idle_jobs_count = df.filter(pl.col("JobStatus") == 1).height
    removed_jobs_count = df.filter(pl.col("JobStatus") == 8).height
    held_jobs_count = df.filter(pl.col("JobStatus") == 5).height
    print(
        f"Number of finished jobs: {finished_jobs_count}, Pulled: {pulled_jobs_count}, Merged: {merged_jobs_count}"
    )

    print(f"Number of running jobs: {running_jobs_count}")
    print(f"Number of idle jobs: {idle_jobs_count}")
    print(f"Number of removed jobs: {removed_jobs_count}")
    print(f"Number of held jobs: {held_jobs_count}")

    unique_qdates = df.select(pl.col("QDate").unique()).to_series().to_list()
    print(f"QDate submission: {unique_qdates}")

    # Get Wait time from QDate to JobStartDate
    df = df.with_columns(
        (
            pl.from_epoch("JobStartDate", time_unit="s")
            - pl.from_epoch("QDate", time_unit="s")
        ).alias("WaitTime")
    )

    df = df.with_columns(
        (
            pl.from_epoch("CompletionDate", time_unit="s")
            - pl.from_epoch("JobStartDate", time_unit="s")
        ).alias("RunTime")
    )

    df_n_run = df.filter((pl.col("RunTime") < 0))
    df_n_wait = df.filter((pl.col("WaitTime") < 0))
    n_wait_unique_status = get_parsed_job_status(
        df_n_wait.select(pl.col("JobStatus").unique()).to_series().to_list()
    )
    n_run_unique_status = get_parsed_job_status(
        df_n_run.select(pl.col("JobStatus").unique()).to_series().to_list()
    )
    if df_n_run.height > 0:
        print(
            f"Warning: {df_n_run.height} jobs have negative RunTime. JobStatus: {n_run_unique_status}"
        )

    if df_n_wait.height > 0:
        print(
            f"Warning: {df_n_wait.height} jobs have negative WaitTime. JobStatus: {n_wait_unique_status}"
        )

    completed_jobs_with_timing = df.filter(
        (pl.col("JobStatus") == 4)
        & pl.col("QDate").is_not_null()
        & pl.col("JobStartDate").is_not_null()
        & pl.col("CompletionDate").is_not_null()
    )
    incomplete_timing_count = finished_jobs_count - completed_jobs_with_timing.height
    if incomplete_timing_count:
        print(
            f"Warning: {incomplete_timing_count} completed jobs have incomplete timing data."
        )

    timing_summary = completed_jobs_with_timing.select(
        pl.col("RunTime").min().alias("min_run"),
        pl.col("RunTime").mean().alias("average_run"),
        pl.col("RunTime").max().alias("max_run"),
        pl.col("WaitTime").min().alias("min_wait"),
        pl.col("WaitTime").mean().alias("average_wait"),
        pl.col("WaitTime").max().alias("max_wait"),
    ).row(0, named=True)
    print(
        "Completed run time: "
        f"min {timing_summary['min_run']}; "
        f"average {timing_summary['average_run']}; "
        f"max {timing_summary['max_run']}"
    )
    print(
        "Completed wait time: "
        f"min {timing_summary['min_wait']}; "
        f"average {timing_summary['average_wait']}; "
        f"max {timing_summary['max_wait']}"
    )


def main():
    args = get_parsed_args()

    db_path = _resolve_db_path(args.name)
    conn = sqlite3.connect(db_path)

    jobs = query_for_jobs(conn, args.cluster_id)
    print(f"Queried {len(jobs)} jobs from the database.")
    analyze_jobs(jobs)
    conn.close()


if __name__ == "__main__":
    main()
