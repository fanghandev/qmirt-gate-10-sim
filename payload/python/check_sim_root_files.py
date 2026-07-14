#!/usr/bin/env python3
"""Inspect ROOT files in a directory using uproot.

This utility scans a user-provided directory for ROOT files and prints a compact
summary for each file, including discovered TTrees and branch information.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import uproot
except ImportError:  # pragma: no cover - import failure path
    uproot = None


REQUIRED_BRANCHES = (
    "EventPosition_X",
    "EventPosition_Y",
    "EventPosition_Z",
    "PrePosition_X",
    "PrePosition_Y",
    "PrePosition_Z",
)


@dataclass
class TreeSummary:
    """Summary information for one tree in a ROOT file."""

    name: str
    entries: int
    branch_count: int
    sample_branches: list[str]
    branch_integrity: bool


@dataclass
class FileSummary:
    """Summary information for one ROOT file."""

    path: Path
    size_bytes: int
    trees: list[TreeSummary]
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ROOT files in a folder with uproot"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help="Folder that contains ROOT files (mutually exclusive with --file-list)",
    )
    parser.add_argument(
        "--file-list",
        help="Path to a file containing one ROOT file path per line. If provided, files listed here are inspected instead of scanning a folder.",
    )
    parser.add_argument(
        "--output",
        help="Path to write results (CSV or TXT). Defaults to OUTPUT_DIR environment or current dir.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "txt"),
        default="csv",
        help="Output format: csv or txt (default: csv)",
    )
    parser.add_argument(
        "--pattern",
        default="*.root",
        help="Glob pattern used to find ROOT files (default: *.root)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under the target folder",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of files inspected",
    )
    parser.add_argument(
        "--max-branches",
        type=int,
        default=None,
        help="Max number of branch names printed per tree (default: no limit)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use (default: 1, sequential execution)",
    )
    return parser.parse_args()


def find_root_files(folder: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        files = sorted(path for path in folder.rglob(pattern) if path.is_file())
    else:
        files = sorted(path for path in folder.glob(pattern) if path.is_file())
    return files


def inspect_root_file(path: Path, max_branches: int | None) -> FileSummary:
    size_bytes = path.stat().st_size
    try:
        with uproot.open(path) as root_file:
            class_map = root_file.classnames(cycle=False)
            tree_names = [name for name, cls in class_map.items() if cls == "TTree"]

            trees: list[TreeSummary] = []
            for tree_name in tree_names:
                tree = root_file[tree_name]
                branch_names = list(tree.keys())
                branch_set = set(branch_names)
                branch_integrity = all(
                    branch in branch_set for branch in REQUIRED_BRANCHES
                )
                sample_branches = (
                    branch_names
                    if max_branches is None
                    else branch_names[:max_branches]
                )
                trees.append(
                    TreeSummary(
                        name=tree_name,
                        entries=int(tree.num_entries),
                        branch_count=len(branch_names),
                        sample_branches=sample_branches,
                        branch_integrity=branch_integrity,
                    )
                )

            return FileSummary(path=path, size_bytes=size_bytes, trees=trees)
    except Exception as exc:  # pragma: no cover - depends on file contents
        return FileSummary(path=path, size_bytes=size_bytes, trees=[], error=str(exc))


def _sanitize_text(s: str) -> str:
    return s.replace("\n", " ").replace("\r", " ")


def format_size(size_bytes: int) -> str:
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for suffix in suffixes:
        if value < 1024 or suffix == suffixes[-1]:
            if suffix == "B":
                return f"{int(value)} {suffix}"
            return f"{value:.2f} {suffix}"
        value /= 1024
    return f"{size_bytes} B"


def main() -> int:
    if uproot is None:
        print(
            "Error: uproot is not installed. Install it with: pip install uproot",
            file=sys.stderr,
        )
        return 2

    args = parse_args()

    # Decide files to inspect: either from --file-list or by scanning a folder
    files: list[Path] = []
    if args.file_list:
        fl = Path(args.file_list).expanduser().resolve()
        if not fl.exists() or not fl.is_file():
            print(f"Error: file-list not found: {fl}", file=sys.stderr)
            return 2
        for line in fl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            files.append(Path(line).expanduser().resolve())
    else:
        if not args.folder:
            print("Error: either provide a folder or --file-list", file=sys.stderr)
            return 2
        folder = Path(args.folder).expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            print(
                f"Error: folder does not exist or is not a directory: {folder}",
                file=sys.stderr,
            )
            return 2

        files = find_root_files(
            folder=folder, pattern=args.pattern, recursive=args.recursive
        )
    if args.max_files is not None:
        if args.max_files < 1:
            print("Error: --max-files must be >= 1", file=sys.stderr)
            return 2
        files = files[: args.max_files]

    if not files:
        if args.file_list:
            print(f"No files listed in {args.file_list}")
        else:
            print(f"No files found in {folder} matching pattern: {args.pattern}")
        return 1

    ok_count = 0
    err_count = 0
    total_trees = 0

    # Decide output path
    out_path: Path
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_dir = os.environ.get("OUTPUT_DIR", os.getcwd())
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
        task_suffix = f"_{task_id}" if task_id else ""
        fname = f"check_sim_root_files_results_{job_id}{task_suffix}.{args.format}"
        out_path = Path(out_dir) / fname

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file_path",
        "size_bytes",
        "status",
        "error",
        "tree_name",
        "entries",
        "branch_count",
        "key_branch_exist",
    ]

    total_files = len(files)

    try:
        if args.format == "csv":
            fh = out_path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            fh.flush()
        else:
            fh = out_path.open("w", encoding="utf-8")
            fh.write(f"Check run: {datetime.utcnow().isoformat()}Z\n")
            fh.write(f"Files scanned: {total_files}\n\n")
            fh.flush()

        def process_summary(summary: FileSummary, idx: int):
            nonlocal ok_count, err_count, total_trees
            print(f"[{idx}/{total_files}] Checked: {summary.path.name}")
            sys.stdout.flush()

            if summary.error is None:
                ok_count += 1
                total_trees += len(summary.trees)
                if args.format == "csv":
                    if summary.trees:
                        for tree in summary.trees:
                            writer.writerow({
                                "file_path": str(summary.path),
                                "size_bytes": summary.size_bytes,
                                "status": "OK",
                                "error": "",
                                "tree_name": _sanitize_text(tree.name),
                                "entries": tree.entries,
                                "branch_count": tree.branch_count,
                                "key_branch_exist": tree.branch_integrity,
                            })
                    else:
                        writer.writerow({
                            "file_path": str(summary.path),
                            "size_bytes": summary.size_bytes,
                            "status": "OK",
                            "error": "",
                            "tree_name": "",
                            "entries": 0,
                            "branch_count": 0,
                            "key_branch_exist": False,
                        })
                else:  # txt
                    fh.write(f"File: {summary.path}\n")
                    fh.write(f"  Size: {format_size(summary.size_bytes)}\n")
                    fh.write("  Status: OK\n")
                    fh.write(f"  TTrees: {len(summary.trees)}\n")
                    for t in summary.trees:
                        fh.write(f"    - {_sanitize_text(t.name)}\n")
                        fh.write(f"      entries: {t.entries}\n")
                        fh.write(f"      branches: {t.branch_count}\n")
                        fh.write(
                            f"      sample: {', '.join(_sanitize_text(b) for b in t.sample_branches)}\n"
                        )
                    fh.write("\n")
            else:
                err_count += 1
                if args.format == "csv":
                    writer.writerow({
                        "file_path": str(summary.path),
                        "size_bytes": summary.size_bytes,
                        "status": "ERROR",
                        "error": summary.error,
                        "tree_name": "",
                        "entries": 0,
                        "branch_count": 0,
                        "key_branch_exist": False,
                    })
                else:  # txt
                    fh.write(f"File: {summary.path}\n")
                    fh.write(f"  Size: {format_size(summary.size_bytes)}\n")
                    fh.write("  Status: ERROR\n")
                    fh.write(f"  Message: {summary.error}\n\n")
            fh.flush()

        if args.workers > 1:
            print(f"Starting inspection using ProcessPoolExecutor with {args.workers} workers...")
            sys.stdout.flush()
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(inspect_root_file, path, args.max_branches): path
                    for path in files
                }
                for i, future in enumerate(as_completed(futures), 1):
                    summary = future.result()
                    process_summary(summary, i)
        else:
            print("Starting sequential inspection...")
            sys.stdout.flush()
            for i, path in enumerate(files, 1):
                summary = inspect_root_file(path=path, max_branches=args.max_branches)
                process_summary(summary, i)

        if args.format == "txt":
            fh.write("Summary:\n")
            fh.write(f"  OK: {ok_count}, Errors: {err_count}, Total TTrees: {total_trees}\n")
            fh.flush()

        fh.close()

    except Exception as exc:
        print(f"Error writing output file {out_path}: {exc}", file=sys.stderr)
        return 4

    print(f"Finished checking {total_files} files. Results written to: {out_path}")
    return 0 if err_count == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
