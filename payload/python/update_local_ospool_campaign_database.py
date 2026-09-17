#!/usr/bin/env python3
import argparse
import csv
import io
import json
import re
import sqlite3
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

from tqdm.auto import tqdm

LOCAL_DATA_ROOT = Path("/data/fanghan/opengate_sim/data/cardiac_spect")


def _parse_timestamp(value: str) -> int | str | None:
    value = value.strip()
    if value in {"", "undefined"}:
        return None

    try:
        return int(value)
    except ValueError:
        datetime.fromisoformat(value)
        return value


def init_db(db_path: str) -> sqlite3.Connection:
    """Initializes the SQLite database and ensures the table exists."""

    # Create the parent directory if it doesn't exist
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS condor_jobs (
            ClusterId INTEGER,
            ProcId INTEGER,
            Cmd TEXT,
            JobStatus INTEGER,
            QDate INTEGER,
            JobStartDate INTEGER,
            CompletionDate INTEGER,
            is_pulled INTEGER DEFAULT 0,
            is_merged INTEGER DEFAULT 0,
            PRIMARY KEY (ClusterId, ProcId)
        )
    """)
    cursor.execute("""
        UPDATE condor_jobs
        SET QDate = NULLIF(QDate, -1),
            JobStartDate = NULLIF(JobStartDate, -1),
            CompletionDate = NULLIF(CompletionDate, -1)
        WHERE QDate = -1 OR JobStartDate = -1 OR CompletionDate = -1
    """)
    conn.commit()
    return conn


def upsert_jobs_db(conn: sqlite3.Connection, parsed_rows: List[List[Any]]) -> int:
    """Upserts a batch of job rows into the database and returns the count."""
    cursor = conn.cursor()
    upsert_sql = """
        INSERT INTO condor_jobs (ClusterId, ProcId, Cmd, JobStatus, QDate, JobStartDate, CompletionDate)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ClusterId, ProcId) DO UPDATE SET
            JobStatus = excluded.JobStatus,
            QDate = COALESCE(excluded.QDate, condor_jobs.QDate),
            JobStartDate = COALESCE(excluded.JobStartDate, condor_jobs.JobStartDate),
            CompletionDate = COALESCE(excluded.CompletionDate, condor_jobs.CompletionDate)
        WHERE condor_jobs.JobStatus IS NOT 8
          AND (
               excluded.JobStatus IS NOT condor_jobs.JobStatus
               OR (
                   excluded.QDate IS NOT NULL
                   AND excluded.QDate IS NOT condor_jobs.QDate
               )
               OR (
                   excluded.JobStartDate IS NOT NULL
                   AND excluded.JobStartDate IS NOT condor_jobs.JobStartDate
               )
               OR (
                   excluded.CompletionDate IS NOT NULL
                   AND excluded.CompletionDate IS NOT condor_jobs.CompletionDate
               )
          );
    """
    cursor.executemany(upsert_sql, parsed_rows)
    conn.commit()
    return cursor.rowcount


def _resolve_campaign_paths(
    name: str,
    *,
    db_fname: str = "ospool_condor_jobs.db",
    partial_data_subdir: str = "partials/targz",
) -> Tuple[Path, Path, Path]:
    """Resolves the campaign path based on the batch name."""
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("name must be a batch directory name, not a path")

    campaign_dir = LOCAL_DATA_ROOT / name

    # Resolve the database path within the campaign directory
    db_path = campaign_dir / db_fname
    partial_data_path = campaign_dir / partial_data_subdir
    return campaign_dir, db_path, partial_data_path


def get_sim_stats(
    tarball_path: str | Path,
) -> tuple[int | None, int | None] | None:
    """Read SRM arrays and statistics from one tarball into independent objects."""
    if isinstance(tarball_path, str):
        tarball_path = Path(tarball_path)
    if not tarball_path.is_file():
        raise FileNotFoundError(f"The file {tarball_path} does not exist.")
    base_name = tarball_path.name.removesuffix(".tar.gz")
    stats_name = base_name.replace("srm_", "sim_stats_", 1) + "_loop_00000.txt"

    with tarfile.open(tarball_path, "r:gz") as tar:
        try:
            stats_member = tar.getmember(f"{base_name}/stats/{stats_name}")
        except KeyError:
            return None
        if stats_member is not None:
            io_reader = tar.extractfile(stats_member)
            if io_reader is not None:
                with io_reader:
                    parsed_sim_stats = json.loads(io_reader.read().decode("utf-8"))
                    start_time_epoch, stop_time_epoch = parse_sim_stats_timestamp(
                        parsed_sim_stats
                    )
                    return start_time_epoch, stop_time_epoch
        return None  # Return None if stats could not be read


def parse_sim_stats_timestamp(stats_dict: dict):
    time_fmt = "%a %b %d %H:%M:%S %Y"
    try:
        start_dt = datetime.strptime(stats_dict["start_time"]["value"], time_fmt)
        stop_dt = datetime.strptime(stats_dict["stop_time"]["value"], time_fmt)
        return int(start_dt.timestamp()), int(stop_dt.timestamp())

    except (KeyError, TypeError, ValueError):
        return None, None


def _upsert_fs_jobs(
    conn: sqlite3.Connection,
    parsed_rows: list[tuple[int, int, int, int, int]],
) -> int:
    """Insert filesystem-confirmed jobs without replacing complete records."""
    if not parsed_rows:
        return 0

    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT INTO condor_jobs (
            ClusterId, ProcId, JobStatus, QDate, JobStartDate, CompletionDate,
            is_pulled
        )
        VALUES (?, ?, 4, ?, ?, ?, 1)
        ON CONFLICT(ClusterId, ProcId) DO UPDATE SET
            JobStatus = 4,
            QDate = COALESCE(condor_jobs.QDate, excluded.QDate),
            JobStartDate = COALESCE(
                condor_jobs.JobStartDate, excluded.JobStartDate
            ),
            CompletionDate = COALESCE(
                condor_jobs.CompletionDate, excluded.CompletionDate
            ),
            is_pulled = 1
        WHERE condor_jobs.JobStatus IS NOT 4
           OR condor_jobs.QDate IS NULL
           OR condor_jobs.JobStartDate IS NULL
           OR condor_jobs.CompletionDate IS NULL
           OR condor_jobs.is_pulled IS NOT 1
        """,
        parsed_rows,
    )
    conn.commit()
    return cursor.rowcount


def _mark_existing_jobs_pulled(
    conn: sqlite3.Connection, job_keys: list[tuple[int, int]]
) -> int:
    """Mark database jobs as pulled when matching tarballs exist on disk."""
    if not job_keys:
        return 0

    cursor = conn.cursor()
    cursor.executemany(
        """
        UPDATE condor_jobs
        SET is_pulled = 1
        WHERE ClusterId = ? AND ProcId = ? AND is_pulled IS NOT 1
        """,
        job_keys,
    )
    conn.commit()
    return cursor.rowcount


def update_db_from_fs_scan(
    conn: sqlite3.Connection, partial_data_dir: Path, qdate_epoch: int
) -> int:
    """Upsert completion data extracted from pulled simulation tarballs."""
    if not partial_data_dir.is_dir():
        raise NotADirectoryError(
            f"Partial data directory not found: {partial_data_dir}"
        )

    complete_job_keys = set(
        conn.execute(
            """
            SELECT ClusterId, ProcId
            FROM condor_jobs
            WHERE JobStatus = 4
              AND QDate IS NOT NULL
              AND JobStartDate IS NOT NULL
              AND CompletionDate IS NOT NULL
            """
        )
    )
    tarball_fnames = sorted(partial_data_dir.glob("*.tar.gz"))
    # Parse the filenames of the tarballs, e.g. srm_c_14996562_p_3983.tar.gz
    pattern = re.compile(r"srm_c_(?P<ClusterId>\d+)_p_(?P<ProcId>\d+)\.tar\.gz")
    parsed_tarballs: list[tuple[int, int, int, int, int]] = []
    filesystem_job_keys: list[tuple[int, int]] = []
    updated_count = 0
    for tarball_fname in tqdm(tarball_fnames):
        match = pattern.fullmatch(tarball_fname.name)
        if match:
            cluster_id = int(match.group("ClusterId"))
            proc_id = int(match.group("ProcId"))
            filesystem_job_keys.append((cluster_id, proc_id))
            if (cluster_id, proc_id) in complete_job_keys:
                continue

            try:
                stats = get_sim_stats(tarball_fname)
            except (
                OSError,
                tarfile.TarError,
                json.JSONDecodeError,
                UnicodeDecodeError,
            ):
                continue
            if stats is None:
                continue
            start_time, stop_time = stats
            if start_time is None or stop_time is None:
                continue
            parsed_tarballs.append(
                (
                    cluster_id,
                    proc_id,
                    qdate_epoch,
                    start_time,
                    stop_time,
                )
            )
            # For every 500 parsed tarballs, update the database
            if len(parsed_tarballs) == 500:
                updated_count += _upsert_fs_jobs(conn, parsed_tarballs)
                parsed_tarballs.clear()

    # Upsert any remaining parsed tarballs
    if parsed_tarballs:
        updated_count += _upsert_fs_jobs(conn, parsed_tarballs)

    return updated_count + _mark_existing_jobs_pulled(conn, filesystem_job_keys)


def get_parsed_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query OSPool jobs and upsert into SQLite."
    )
    parser.add_argument(
        "-c",
        "--cluster-id",
        dest="cluster_id",
        type=int,
        help="The HTCondor Cluster ID to query (required unless --scanfs).",
    )
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        metavar="BATCH_NAME",
        help="Batch directory name, for example batch_20260909_012610.",
    )
    parser.add_argument(
        "--scanfs",
        action="store_true",
        help="Scan the filesystem for partial data tarballs.",
    )
    parser.add_argument(
        "--qdate-epoch",
        type=int,
        dest="qdate_epoch",
        help="Epoch timestamp to use for QDate when scanning the filesystem.",
    )
    args = parser.parse_args()
    if args.scanfs and args.qdate_epoch is None:
        parser.error("--qdate-epoch is required with --scanfs")
    if not args.scanfs and args.cluster_id is None:
        parser.error("--cluster-id is required unless --scanfs is used")
    return args


def main():
    args = get_parsed_args()
    _, db_path, partial_data_dir = _resolve_campaign_paths(args.name)
    conn = init_db(str(db_path))
    try:
        if args.scanfs:
            count = update_db_from_fs_scan(conn, partial_data_dir, args.qdate_epoch)
            print(f"Successfully upserted {count} filesystem job records.")
            return

        query_current = (
            f"condor_q {args.cluster_id} -af:, ClusterId ProcId Cmd JobStatus QDate JobStartDate CompletionDate ; "
            f"condor_history {args.cluster_id} -af:, ClusterId ProcId Cmd JobStatus QDate JobStartDate CompletionDate"
        )

        print(f"Querying OSPool for Cluster {args.cluster_id}...")
        try:
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "ospool", query_current],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"SSH Command failed: {e.stderr}")
            return

        reader = csv.reader(io.StringIO(result.stdout.strip()))

        parsed_rows = []
        for row in reader:
            if not row or len(row) < 7 or row[0].strip() == "ClusterId":
                continue

            clean_row: List[Any] = [
                int(row[0].strip()),
                int(row[1].strip()),
                row[2].strip(),
                int(row[3].strip()),
                *[col.strip() for col in row[4:7]],
            ]
            for i in range(4, 7):
                clean_row[i] = _parse_timestamp(clean_row[i])
            parsed_rows.append(clean_row)

        if not parsed_rows:
            print("No jobs found or returned by the query.")
            return

        count = upsert_jobs_db(conn, parsed_rows)
        print(f"Successfully upserted {count} job records into '{db_path}'.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
