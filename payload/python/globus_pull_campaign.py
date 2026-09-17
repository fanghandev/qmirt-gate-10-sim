#!/usr/bin/env python3
"""Pull completed sparse-SRM tasks from a remote campaign to the local workstation.

Runs on the workstation (not the cluster). A task is only pulled once its
``TASK_COMPLETE.json`` marker exists, and only recorded in the ledger once the
local checksums match the marker, so the pull is safe to repeat and resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MARKER_NAME = "TASK_COMPLETE.json"
LEDGER_NAME = "pull_ledger.json"
CLUSTER_PROGRESS_NAME = "cluster_progress.json"
LOCAL_PROGRESS_NAME = "progress.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
# (remote name, local name) for small campaign-level files refreshed every cycle.
# The cluster progress report is renamed so it cannot clobber the locally generated one.
CAMPAIGN_FILES = (
    ("progress.json", CLUSTER_PROGRESS_NAME),
    ("campaign_manifest.json", "campaign_manifest.json"),
    ("geometry_provenance.json", "geometry_provenance.json"),
)
LOOP_STATS_GLOB = "*_sim_stats_loop_*.txt"
FINAL_SRM_NAMES = ("final_srm_1mm.npz", "final_srm_1p5mm.npz", "final_srm_2mm.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="Shell-style KEY=VALUE file with defaults for the options below.",
    )
    parser.add_argument(
        "--source-endpoint", help="Globus collection UUID holding the campaign."
    )
    parser.add_argument(
        "--source-path",
        help="Campaign directory as seen by the source collection (collection-relative).",
    )
    parser.add_argument(
        "--dest-endpoint", help="Globus collection UUID of this workstation."
    )
    parser.add_argument(
        "--dest-path",
        help="Campaign directory on this workstation, as seen by the destination collection.",
    )
    parser.add_argument(
        "--local-path",
        help="POSIX path of --dest-path on this machine (defaults to --dest-path).",
    )
    parser.add_argument(
        "--mount-path",
        help=(
            "Campaign directory reached through a local mount of the cluster filesystem. "
            "When set, listing and progress files are read from the mount instead of Globus, "
            "so only bulk task payloads create Globus transfer tasks."
        ),
    )
    parser.add_argument(
        "--progress-only",
        action="store_true",
        help="Refresh progress from the mount without any Globus transfer. Requires --mount-path.",
    )
    parser.add_argument(
        "--mount-root",
        help=(
            "Parent directory (on a local mount) containing several sibling campaign "
            "directories, e.g. ~/sdsc-expanse/brain_spect_sim. When set, every matching "
            "campaign is harvested directly off the mount (no Globus involved)."
        ),
    )
    parser.add_argument(
        "--campaign-glob",
        default="*",
        help="Glob (relative to --mount-root) selecting campaign directories to harvest.",
    )
    parser.add_argument(
        "--local-root",
        help="Local destination parent directory for --mount-root harvests "
        "(each campaign lands in <local-root>/<campaign-name>).",
    )
    parser.add_argument(
        "--globus-cli", default="globus", help="globus executable to use."
    )
    parser.add_argument(
        "--label", default="qmirt-srm-pull", help="Globus transfer task label."
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Cap on task directories pulled per campaign per run (0 means no cap).",
    )
    parser.add_argument(
        "--purge-after-pull",
        action="store_true",
        help="Delete verified task directories from the source collection.",
    )
    parser.add_argument(
        "--combine-after-pull",
        action="store_true",
        help="Re-run the campaign combine locally when new tasks were verified.",
    )
    parser.add_argument(
        "--report-after-pull",
        action="store_true",
        help="Regenerate the local progress.json the dashboard serves.",
    )
    parser.add_argument(
        "--expected-tasks",
        type=int,
        default=0,
        help="Campaign array size, used for completion percentages in the report.",
    )
    parser.add_argument(
        "--no-srm-stats",
        action="store_true",
        help="Skip SRM matrix statistics in the report (much faster for frequent polling).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be transferred without submitting anything.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, str]:
    config: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, sep, value = line.partition("=")
        if not sep:
            continue
        config[key.strip()] = value.strip().strip('"').strip("'")
    return config


def resolve_settings(args: argparse.Namespace) -> argparse.Namespace:
    config = load_config(args.config) if args.config else {}
    # Mount-root harvests read and write local paths only, no Globus/endpoint config needed.
    if args.mount_root:
        if not args.local_root:
            raise SystemExit("--mount-root requires --local-root")
        return args
    mapping = {
        "source_endpoint": "QMIRT_SRC_ENDPOINT",
        "source_path": "QMIRT_SRC_PATH",
        "dest_endpoint": "QMIRT_DST_ENDPOINT",
        "dest_path": "QMIRT_DST_PATH",
        "local_path": "QMIRT_LOCAL_PATH",
        "mount_path": "QMIRT_MOUNT_PATH",
    }
    for attr, key in mapping.items():
        if getattr(args, attr) is None:
            setattr(args, attr, config.get(key) or os.environ.get(key))

    if args.globus_cli == "globus":
        args.globus_cli = (
            config.get("QMIRT_GLOBUS_CLI")
            or os.environ.get("QMIRT_GLOBUS_CLI")
            or "globus"
        )

    if args.expected_tasks == 0:
        expected = config.get("QMIRT_EXPECTED_TASKS") or os.environ.get(
            "QMIRT_EXPECTED_TASKS"
        )
        if expected:
            args.expected_tasks = int(expected)

    optional = {"local_path", "mount_path"}
    missing = [
        key
        for attr, key in mapping.items()
        if attr not in optional and not getattr(args, attr)
    ]
    if args.progress_only:
        if not args.mount_path:
            raise SystemExit("--progress-only requires --mount-path (QMIRT_MOUNT_PATH)")
        missing = [
            key
            for key in missing
            if key not in {"QMIRT_SRC_ENDPOINT", "QMIRT_DST_ENDPOINT", "QMIRT_SRC_PATH"}
        ]
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")

    if not args.local_path:
        args.local_path = args.dest_path
    if args.source_path:
        args.source_path = args.source_path.rstrip("/")
    if args.dest_path:
        args.dest_path = args.dest_path.rstrip("/")
    return args


def run_globus(cli: str, argv: list[str], capture: bool = True) -> str:
    proc = subprocess.run(
        [cli, *argv],
        check=False,
        capture_output=capture,
        text=True,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"globus {' '.join(argv)} failed: {stderr}")
    return proc.stdout if capture else ""


def globus_ls(cli: str, endpoint: str, path: str) -> list[dict]:
    """List one directory, returning [] when it does not exist."""
    try:
        stdout = run_globus(cli, ["ls", "--format", "json", f"{endpoint}:{path}/"])
    except RuntimeError as exc:
        if "NotFound" in str(exc) or "not found" in str(exc):
            return []
        raise
    return json.loads(stdout).get("DATA", [])


def load_ledger(ledger_path: Path) -> dict:
    if not ledger_path.exists():
        return {"pulled": {}}
    try:
        return json.loads(ledger_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(
            f"Warning: {ledger_path} is not valid JSON; starting a new ledger.",
            file=sys.stderr,
        )
        return {"pulled": {}}


def save_ledger(ledger_path: Path, ledger: dict) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=ledger_path.parent, delete=False, encoding="utf-8"
    ) as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)
        temp_name = handle.name
    os.replace(temp_name, ledger_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_ready_tasks(
    cli: str, endpoint: str, campaign_path: str, already: set[str]
) -> list[str]:
    ready = []
    for entry in globus_ls(cli, endpoint, campaign_path):
        name = entry.get("name", "")
        if (
            entry.get("type") != "dir"
            or not name.startswith("task_")
            or name in already
        ):
            continue
        contents = globus_ls(cli, endpoint, f"{campaign_path}/{name}")
        if any(item.get("name") == MARKER_NAME for item in contents):
            ready.append(name)
    return sorted(ready)


def find_ready_tasks_mount(mount: Path, already: set[str]) -> list[str]:
    """Same selection as find_ready_tasks, but over a local mount and without Globus."""
    ready = []
    for entry in mount.iterdir():
        if not entry.is_dir() or not entry.name.startswith("task_"):
            continue
        if entry.name in already:
            continue
        if (entry / MARKER_NAME).exists():
            ready.append(entry.name)
    return sorted(ready)


def copy_campaign_files_from_mount(mount: Path, local_campaign: Path) -> list[str]:
    """Copy the small campaign-level files straight off the mount, atomically."""
    copied = []
    local_campaign.mkdir(parents=True, exist_ok=True)
    for remote_name, local_name in CAMPAIGN_FILES:
        source = mount / remote_name
        if not source.is_file():
            continue
        destination = local_campaign / local_name
        try:
            with tempfile.NamedTemporaryFile(
                dir=local_campaign, delete=False
            ) as handle:
                temp_name = handle.name
            shutil.copyfile(source, temp_name)
            os.replace(temp_name, destination)
        except OSError as exc:
            print(f"Warning: could not copy {remote_name}: {exc}", file=sys.stderr)
            continue
        copied.append(local_name)
    return copied


def campaign_started(campaign_dir: Path) -> bool:
    """A campaign shard counts as started once its reporter has written progress.json."""
    return (campaign_dir / "progress.json").is_file()


def loops_done_count(task_dir: Path) -> int:
    return len(list((task_dir / "srm_chunks").glob(LOOP_STATS_GLOB)))


def classify_task(task_dir: Path) -> str:
    """Return "complete", "partial", or "in_flight" for one remote task directory.

    "in_flight" tasks are still running (or queued) with zero finished loops yet,
    so there is nothing durable to pull; they are skipped rather than waited on.
    """
    if (task_dir / MARKER_NAME).is_file():
        return "complete"
    if loops_done_count(task_dir) > 0:
        return "partial"
    return "in_flight"


def pull_complete_task(mount_task: Path, local_task: Path) -> bool:
    """Copy one finished task's combined SRMs; returns True once checksums verify."""
    local_task.mkdir(parents=True, exist_ok=True)
    names = [*FINAL_SRM_NAMES, "combined_srm_metadata.json", MARKER_NAME]
    wall_time_name = f"{mount_task.name}_wall_time.txt"
    if (mount_task / wall_time_name).is_file():
        names.append(wall_time_name)
    for name in names:
        source = mount_task / name
        if not source.is_file():
            continue
        with tempfile.NamedTemporaryFile(dir=local_task, delete=False) as handle:
            temp_name = handle.name
        shutil.copyfile(source, temp_name)
        os.replace(temp_name, local_task / name)
    return verify_task(local_task.parent, local_task.name) is not None


def pull_partial_task(mount_task: Path, local_task: Path) -> int:
    """Mirror a still-running task's finished loop chunks; returns loops copied."""
    chunk_source = mount_task / "srm_chunks"
    local_task.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", f"{chunk_source}/", f"{local_task / 'srm_chunks'}/"],
        check=True,
    )
    wall_time_name = f"{mount_task.name}_wall_time.txt"
    source_wall_time = mount_task / wall_time_name
    if source_wall_time.is_file():
        shutil.copyfile(source_wall_time, local_task / wall_time_name)
    return loops_done_count(local_task)


def finalize_simulated_primaries(local_campaign: Path) -> None:
    """Patch simulated_primaries and combine any partial tasks just pulled."""
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "payload" / "python" / "finalize_task_srm.py"),
            str(local_campaign),
            "--apply",
        ],
        check=True,
    )


def harvest_campaign(
    mount_campaign: Path,
    local_campaign: Path,
    *,
    max_tasks: int = 0,
    dry_run: bool = False,
) -> dict:
    """Pull one campaign shard's ready tasks straight off a local mount."""
    ledger_path = local_campaign / LEDGER_NAME
    ledger = load_ledger(ledger_path)
    pulled_state: dict = ledger.setdefault("pulled", {})

    if not dry_run:
        copy_campaign_files_from_mount(mount_campaign, local_campaign)

    task_dirs = sorted(
        (p for p in mount_campaign.glob("task_*") if p.is_dir()),
        key=lambda p: int(p.name.removeprefix("task_")),
    )
    complete_count = partial_count = in_flight_count = 0
    processed = 0
    for task_dir in task_dirs:
        if max_tasks and processed >= max_tasks:
            break
        name = task_dir.name
        state = classify_task(task_dir)
        recorded = pulled_state.get(name)

        if state == "in_flight":
            in_flight_count += 1
            continue

        if state == "complete":
            if recorded and recorded.get("status") == "complete":
                continue
            processed += 1
            if dry_run:
                print(f"  [dry-run] would pull complete {name}")
                complete_count += 1
                continue
            if pull_complete_task(task_dir, local_campaign / name):
                pulled_state[name] = {"status": "complete"}
                complete_count += 1
            else:
                print(f"  {name}: verification failed after pull", file=sys.stderr)
            continue

        # state == "partial": only re-pull once new loops have finished remotely.
        loops_done = loops_done_count(task_dir)
        if recorded and recorded.get("loops_done") == loops_done:
            continue
        processed += 1
        if dry_run:
            print(f"  [dry-run] would refresh partial {name} ({loops_done} loops done)")
            partial_count += 1
            continue
        copied_loops = pull_partial_task(task_dir, local_campaign / name)
        pulled_state[name] = {"status": "partial", "loops_done": copied_loops}
        partial_count += 1

    if not dry_run:
        save_ledger(ledger_path, ledger)

    return {
        "complete": complete_count,
        "partial": partial_count,
        "in_flight_skipped": in_flight_count,
    }


def harvest_root(
    mount_root: Path,
    local_root: Path,
    campaign_glob: str,
    *,
    max_tasks: int = 0,
    dry_run: bool = False,
    combine_after_pull: bool = False,
    report_after_pull: bool = False,
    expected_tasks: int = 0,
    no_srm_stats: bool = False,
) -> dict[str, dict]:
    """Harvest every started campaign shard matching *campaign_glob* under *mount_root*."""
    summary: dict[str, dict] = {}
    for mount_campaign in sorted(
        p for p in mount_root.glob(campaign_glob) if p.is_dir()
    ):
        name = mount_campaign.name
        if not campaign_started(mount_campaign):
            print(f"Skipping {name}: not started (no progress.json)")
            continue

        print(f"== {name} ==")
        local_campaign = local_root / name
        result = harvest_campaign(
            mount_campaign, local_campaign, max_tasks=max_tasks, dry_run=dry_run
        )
        print(
            f"  complete={result['complete']} partial={result['partial']} "
            f"in_flight_skipped={result['in_flight_skipped']}"
        )
        summary[name] = result

        if dry_run:
            continue
        if result["complete"] or result["partial"]:
            print("  finalizing simulated_primaries / partial combines...")
            finalize_simulated_primaries(local_campaign)
        if combine_after_pull and result["complete"]:
            combine_campaign(local_campaign, expected_tasks)
        if report_after_pull:
            report_campaign(local_campaign, expected_tasks, no_srm_stats)
    return summary


def build_batch(tasks: list[str], campaign_files: list[tuple[str, str]]) -> str:
    """Batch lines for one Globus transfer: only each complete task's combined SRMs.

    Raw per-loop srm_chunks are never transferred here; a task only ever appears
    in *tasks* once TASK_COMPLETE.json exists, so the combined files are enough.
    """
    lines = []
    for task in tasks:
        for name in (*FINAL_SRM_NAMES, "combined_srm_metadata.json", MARKER_NAME):
            lines.append(f"{task}/{name} {task}/{name}")
    lines.extend(f"{remote} {local}" for remote, local in campaign_files)
    return "\n".join(lines) + "\n"


def submit_transfer(args: argparse.Namespace, batch: str) -> str:
    proc = subprocess.run(
        [
            args.globus_cli,
            "transfer",
            f"{args.source_endpoint}:{args.source_path}/",
            f"{args.dest_endpoint}:{args.dest_path}/",
            "--batch",
            "-",
            "--sync-level",
            "checksum",
            "--label",
            args.label,
            "--format",
            "json",
        ],
        input=batch,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"globus transfer failed: {(proc.stderr or '').strip()}")
    return json.loads(proc.stdout)["task_id"]


def verify_task(local_campaign: Path, task: str) -> dict | None:
    """Return the marker payload when every local file matches its recorded checksum."""
    marker_path = local_campaign / task / MARKER_NAME
    if not marker_path.exists():
        print(f"  {task}: marker missing locally after transfer", file=sys.stderr)
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for name, meta in marker.get("files", {}).items():
        local_file = local_campaign / task / name
        if not local_file.exists():
            print(f"  {task}: {name} missing locally", file=sys.stderr)
            return None
        actual = sha256_file(local_file)
        if actual != meta.get("sha256"):
            print(f"  {task}: {name} checksum mismatch", file=sys.stderr)
            return None
    return marker


def purge_task(args: argparse.Namespace, task: str) -> None:
    run_globus(
        args.globus_cli,
        [
            "delete",
            "--recursive",
            f"{args.source_endpoint}:{args.source_path}/{task}",
            "--label",
            f"{args.label}-purge",
        ],
    )


def combine_campaign(local_campaign: Path, expected_tasks: int) -> None:
    command = [
        "bash",
        str(REPO_ROOT / "submit_slurm" / "wrapper_campaign_combine.sh"),
        "--campaign-dir",
        str(local_campaign),
        "--input-stage",
        "tasks",
    ]
    if expected_tasks > 0:
        command += ["--expected-tasks", str(expected_tasks)]
    subprocess.run(command, check=True)


def report_campaign(
    local_campaign: Path, expected_tasks: int, skip_srm_stats: bool
) -> None:
    """Rebuild progress.json from local data, keeping the cluster's queue section."""
    output = local_campaign / LOCAL_PROGRESS_NAME
    command = [
        sys.executable,
        str(REPO_ROOT / "payload" / "python" / "report_campaign_progress.py"),
        "--campaign-dir",
        str(local_campaign),
        "--srm-dir",
        str(local_campaign),
        "--output",
        str(output),
    ]
    if expected_tasks > 0:
        command += ["--expected-tasks", str(expected_tasks)]
    if skip_srm_stats:
        command.append("--no-srm-stats")
    subprocess.run(command, check=True)

    # sacct is only reachable from the cluster, so graft its section onto the local report.
    cluster_progress = local_campaign / CLUSTER_PROGRESS_NAME
    if not cluster_progress.exists():
        return
    try:
        cluster = json.loads(cluster_progress.read_text(encoding="utf-8"))
        report = json.loads(output.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: could not merge cluster progress: {exc}", file=sys.stderr)
        return
    for key in ("slurm", "rates", "time"):
        if key in cluster:
            report.setdefault("cluster", {})[key] = cluster[key]
    with tempfile.NamedTemporaryFile(
        "w", dir=output.parent, delete=False, encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
        temp_name = handle.name
    os.replace(temp_name, output)


def main() -> int:
    args = resolve_settings(parse_args())

    if args.mount_root:
        mount_root = Path(args.mount_root).expanduser()
        if not mount_root.is_dir():
            raise SystemExit(f"Mount root is not readable (stale mount?): {mount_root}")
        harvest_root(
            mount_root,
            Path(args.local_root).expanduser(),
            args.campaign_glob,
            max_tasks=args.max_tasks,
            dry_run=args.dry_run,
            combine_after_pull=args.combine_after_pull,
            report_after_pull=args.report_after_pull,
            expected_tasks=args.expected_tasks,
            no_srm_stats=args.no_srm_stats,
        )
        return 0

    local_campaign = Path(args.local_path)
    mount = Path(args.mount_path) if args.mount_path else None
    if mount is not None and not mount.is_dir():
        raise SystemExit(f"Mount path is not readable (stale mount?): {mount}")

    # Progress tracking only touches the mount, so it can poll often without
    # creating a Globus task for every refresh.
    if args.progress_only:
        copied = copy_campaign_files_from_mount(mount, local_campaign)
        print(f"Refreshed from mount: {', '.join(copied) if copied else 'nothing'}")
        report_campaign(local_campaign, args.expected_tasks, args.no_srm_stats)
        return 0

    if shutil.which(args.globus_cli) is None:
        raise SystemExit(f"globus CLI not found: {args.globus_cli}")

    ledger_path = local_campaign / LEDGER_NAME
    ledger = load_ledger(ledger_path)
    already = set(ledger["pulled"])

    if mount is not None:
        tasks = find_ready_tasks_mount(mount, already)
        # Campaign files come off the mount, so Globus only moves bulk payloads.
        campaign_files: list[tuple[str, str]] = []
    else:
        tasks = find_ready_tasks(
            args.globus_cli, args.source_endpoint, args.source_path, already
        )
        remote_names = {
            entry.get("name")
            for entry in globus_ls(
                args.globus_cli, args.source_endpoint, args.source_path
            )
        }
        campaign_files = [pair for pair in CAMPAIGN_FILES if pair[0] in remote_names]

    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    if not tasks and not campaign_files:
        print("No new tasks to transfer.")
        if mount is not None:
            copy_campaign_files_from_mount(mount, local_campaign)
        if args.report_after_pull:
            report_campaign(local_campaign, args.expected_tasks, args.no_srm_stats)
        return 0

    print(f"Ready tasks: {len(tasks)}; campaign files: {len(campaign_files)}")
    batch = build_batch(tasks, campaign_files)
    if args.dry_run:
        print("--- transfer batch preview ---")
        print(batch, end="")
        return 0

    local_campaign.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        (local_campaign / task).mkdir(parents=True, exist_ok=True)
    task_id = submit_transfer(args, batch)
    print(f"Submitted Globus task {task_id}; waiting...")
    run_globus(args.globus_cli, ["task", "wait", task_id], capture=False)

    verified, failed = [], []
    for task in tasks:
        marker = verify_task(local_campaign, task)
        if marker is None:
            failed.append(task)
            continue
        ledger["pulled"][task] = {
            "verified_at": marker.get("completed_at"),
            "globus_task_id": task_id,
            "files": marker.get("files", {}),
        }
        verified.append(task)

    if verified:
        save_ledger(ledger_path, ledger)
    print(f"Verified {len(verified)} task(s); {len(failed)} failed verification.")

    if args.purge_after_pull:
        for task in verified:
            purge_task(args, task)
            print(f"Purged remote {task}")

    if verified:
        print("Checking simulated_primaries...")
        finalize_simulated_primaries(local_campaign)

    if args.combine_after_pull and verified:
        print("Combining campaign locally...")
        combine_campaign(local_campaign, args.expected_tasks)

    if mount is not None:
        copy_campaign_files_from_mount(mount, local_campaign)

    if args.report_after_pull:
        print("Refreshing local progress report...")
        report_campaign(local_campaign, args.expected_tasks, args.no_srm_stats)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
