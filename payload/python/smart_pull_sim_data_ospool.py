import argparse
import os
import re
import sqlite3
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

JobKey = tuple[int, int]
DEFAULT_GROUP_SIZE = 100
LOCAL_DATA_ROOT = Path("/data/fanghan/opengate_sim/data/cardiac_spect")
REMOTE_CAMPAIGN_ROOT = "cardiac_spect_srm"
DEFAULT_FILENAME_PATTERN = re.compile(r"^srm_c_(\d+)_p_(\d+)\.tar\.gz$")


def setup_conn(db_path: str) -> sqlite3.Connection:
    """Sets up and returns a connection to the SQLite database."""
    return sqlite3.connect(db_path)


def get_completed_job_ids(
    conn: sqlite3.Connection, cluster_id: int
) -> tuple[set[JobKey], set[JobKey]]:
    """Return completed jobs that have not been pulled yet."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT ClusterId, ProcId, is_pulled
        FROM condor_jobs
        WHERE JobStatus = 4 AND ClusterId = ?
    """,
        (cluster_id,),
    )
    pulled_jobs = set()
    unpulled_jobs = set()

    # Iterating directly over the cursor avoids loading a large intermediate list into memory
    for cluster, proc, is_pulled in cursor:
        job_key = (cluster, proc)
        if is_pulled:
            pulled_jobs.add(job_key)
        else:
            unpulled_jobs.add(job_key)

    return pulled_jobs, unpulled_jobs


def get_partials_dir(data_dir: str) -> Path:
    """Resolve either a batch directory or its explicit targz directory."""
    path = Path(data_dir)
    return path if path.name == "targz" else path / "partials" / "targz"


def get_campaign_paths(name: str) -> tuple[Path, Path, str]:
    """Derive local data, database, and remote campaign paths from a batch name."""
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("name must be a batch directory name, not a path")

    data_dir = LOCAL_DATA_ROOT / name
    return (
        data_dir,
        data_dir / "ospool_condor_jobs.db",
        f"{REMOTE_CAMPAIGN_ROOT}/{name}",
    )


def pull_completed_unpulled_files(
    conn: sqlite3.Connection,
    data_dir: str,
    remote_campaign: str,
    cluster_id: int,
    *,
    remote_host: str = "ospool",
    dryrun: bool = False,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> int:
    """Pull completed, unpulled tarballs from an OSPool campaign with rsync."""
    if group_size < 1:
        raise ValueError("group_size must be greater than zero")

    pulled_jobs, unpulled_jobs = get_completed_job_ids(conn, cluster_id)
    if not unpulled_jobs:
        return 0

    remote_campaign = remote_campaign.strip("/")
    remote_dir = f"{remote_host}:/ospool/ap40/data/fang.han/{remote_campaign}/"
    local_dir = get_partials_dir(data_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    filenames = sorted(
        f"srm_c_{job_cluster_id}_p_{proc_id}.tar.gz"
        for job_cluster_id, proc_id in unpulled_jobs
    )

    if dryrun:
        return len(filenames)

    marked_count = 0
    for offset in range(0, len(filenames), group_size):
        group_jobs = sorted(unpulled_jobs)[offset : offset + group_size]
        group_filenames = [
            f"srm_c_{job_cluster_id}_p_{proc_id}.tar.gz"
            for job_cluster_id, proc_id in group_jobs
        ]
        subprocess.run(
            [
                "rsync",
                "--info=progress2",
                "--human-readable",
                "--ignore-existing",
                "--files-from=-",
                remote_dir,
                f"{local_dir}/",
            ],
            input="\n".join(group_filenames) + "\n",
            text=True,
            check=True,
        )

        pulled_jobs = {
            job
            for job, filename in zip(group_jobs, group_filenames)
            if (local_dir / filename).is_file()
        }
        marked_count += mark_jobs_as_pulled(conn, pulled_jobs)

    return marked_count


def get_args():
    parser = argparse.ArgumentParser(
        prog="smart_pull_data_ospool",
        description="Fetch completed job IDs for a specific ClusterId from the SQLite database.",
        epilog="Author: Fang Han\nEmail:fanghandev@gmail.com",
    )
    parser.add_argument(
        "-c",
        "--cluster-id",
        type=int,
        metavar="CLUSTER_ID",
        required=True,
        help="The ClusterId to fetch completed jobs for.",
    )
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        metavar="BATCH_NAME",
        help="Batch directory name, for example batch_20260909_012610.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Report work without running rsync or updating the database.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=("pull", "validate"),
        default="validate",
        help="pull completed unpulled files, or validate files already on disk (default: validate).",
    )
    parser.add_argument(
        "--remote-host",
        default="ospool",
        help="SSH host for rsync (default: ospool).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help=f"Number of files per rsync/ database-update group (default: {DEFAULT_GROUP_SIZE}).",
    )
    return parser.parse_args()


def mark_jobs_as_pulled(conn: sqlite3.Connection, jobs: Iterable[JobKey]) -> int:
    """Mark matching jobs as pulled in one database transaction."""
    job_rows = list(jobs)
    if not job_rows:
        return 0

    cursor = conn.cursor()
    cursor.executemany(
        """
        UPDATE condor_jobs
        SET is_pulled = 1
        WHERE ClusterId = ? AND ProcId = ? AND is_pulled = 0
        """,
        job_rows,
    )
    conn.commit()
    return cursor.rowcount


def extract_cluster_proc_from_filename(
    filename: str, *, pattern: str | None = None
) -> JobKey | None:
    filename_pattern = re.compile(pattern) if pattern else DEFAULT_FILENAME_PATTERN
    match = filename_pattern.fullmatch(filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def iter_cluster_procs(partials_dir: str | Path) -> Iterator[JobKey]:
    """Yield job keys encoded by tarball names in ``partials_dir``."""
    if not os.path.isdir(partials_dir):
        return
    with os.scandir(partials_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".tar.gz"):
                cluster_proc = extract_cluster_proc_from_filename(entry.name)
                if cluster_proc:
                    yield cluster_proc


def validate_fs_pulled_file_ids(
    conn: sqlite3.Connection, data_dir: str, cluster_id: int, *, dryrun: bool = False
) -> dict[str, int]:
    """Match completed DB jobs to downloaded files and optionally mark matches."""
    pulled_jobs_db, unpulled_jobs_db = get_completed_job_ids(conn, cluster_id)
    partials_dir = get_partials_dir(data_dir)
    filesystem_job_ids = {
        job_key
        for job_key in iter_cluster_procs(partials_dir)
        if job_key[0] == cluster_id
    }
    # Combine pulled and unpulled set

    matched_pulled_job_ids = pulled_jobs_db & filesystem_job_ids
    matched_unpulled_job_ids = unpulled_jobs_db & filesystem_job_ids

    marked_count = 0 if dryrun else mark_jobs_as_pulled(conn, matched_unpulled_job_ids)
    return {
        "completed": len(pulled_jobs_db | unpulled_jobs_db),
        "files_found": len(filesystem_job_ids),
        "matched pulled": len(matched_pulled_job_ids),
        "matched unpulled": len(matched_unpulled_job_ids),
        "marked": marked_count,
    }


def main():
    args = get_args()
    try:
        data_dir, db_path, remote_campaign = get_campaign_paths(args.name)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    conn = setup_conn(str(db_path))
    try:
        if args.mode == "pull":
            pulled_count = pull_completed_unpulled_files(
                conn,
                str(data_dir),
                remote_campaign,
                args.cluster_id,
                remote_host=args.remote_host,
                dryrun=args.dryrun,
                group_size=args.group_size,
            )
            action = "would pull" if args.dryrun else "pulled and marked"
            print(f"{action} {pulled_count} completed, unpulled files.")
        else:
            summary = validate_fs_pulled_file_ids(
                conn, str(data_dir), args.cluster_id, dryrun=args.dryrun
            )
            action = "would mark" if args.dryrun else "marked"
            print(
                f"Completed: {summary['completed']}; "
                f"files found: {summary['files_found']}; "
                f"matched pulled: {summary['matched pulled']}; "
                f"{action}: {summary['marked'] if not args.dryrun else summary['matched unpulled']}"
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
