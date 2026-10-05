#!/usr/bin/env python3
"""Merge parts of a brain campaign group into one per-head SRM set.

Each part (<root>/<group>_NN, pulled by the qmirt-brain-harvest timer) is checked in
the local job database (pulled COMPLETED tasks vs the part's job_count) and against
the other parts' campaign_manifest.json (FOV, shield, physics, actors), so different
simulations are never added together. Then:
  1. payload/python/merge_brain_campaigns.py merges each part's tasks into per-head
     SRMs (database-aware; marks the tasks is_merged=1), unless they already hold
     exactly those tasks;
  2. campaign_tools/merge_batches.py adds the parts into a new directory,
     <root>/<group>_merged_p01-02 (p<first>-<last>, or p01_03_05 for gaps).

Manual, chosen parts:
  ma opengate && python campaign_tools/merge_brain_group.py \\
      --group brain_288mm_csg_102t_20261005T211647Z --parts 1-2

Automatic (qmirt-brain-harvest.service, after each harvest): every group whose parts
match --auto-prefix; each complete part is merged once, and a new combined set is made
whenever the set of complete parts grows (earlier sets are kept, never deleted).
  python campaign_tools/merge_brain_group.py --auto --auto-prefix brain_288mm_csg_102t_
"""

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/data/fanghan/opengate_sim/data/brain_spect")
DEFAULT_DB = DEFAULT_ROOT / "expanse_slurm_jobs.db"
# campaign_manifest.json fields that must agree between merged parts
MANIFEST_KEYS = ("srm_fov_size_mm", "shield_model", "physics_list", "actor_layout",
                 "campaign_group_id")


def parse_parts(text: str) -> list[int]:
    parts: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if "-" in item:
            lo, hi = (int(v) for v in item.split("-"))
            parts.update(range(lo, hi + 1))
        elif item:
            parts.add(int(item))
    if not parts or min(parts) < 1:
        raise SystemExit("--parts needs positive part numbers, e.g. 1-2 or 1,3,5")
    return sorted(parts)


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def database_tasks(db: Path, campaign: str) -> tuple[set[int], int]:
    """(pulled COMPLETED task ids, rows) for one campaign part."""
    if not db.is_file():
        return set(), 0
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT array_task_id, state, is_pulled FROM slurm_jobs WHERE campaign_name = ?",
            (campaign,),
        ).fetchall()
    return {int(t) for t, state, pulled in rows if pulled and state == "COMPLETED"}, len(rows)


def merged_task_ids(part_dir: Path) -> set[int] | None:
    metadata = read_json(part_dir / "combined_srm_metadata.json") or {}
    ids = metadata.get("merged_task_ids")
    return set(ids) if isinstance(ids, list) else None


def output_name(group: str, parts: list[int]) -> str:
    if parts == list(range(parts[0], parts[-1] + 1)):
        return f"{group}_merged_p{parts[0]:02d}-{parts[-1]:02d}"
    return f"{group}_merged_p{'_'.join(f'{n:02d}' for n in parts)}"


def run(command: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def check_parts(parts: list[int], part_dirs: list[Path], db: Path, expected_tasks: int,
                allow_incomplete: bool) -> list[str]:
    """Problems that block merging these parts (empty list: OK)."""
    problems = []
    manifests = {}
    for n, part_dir in zip(parts, part_dirs):
        if not part_dir.is_dir():
            problems.append(f"part {n}: {part_dir} not pulled yet")
            continue
        done, rows = database_tasks(db, part_dir.name)
        if rows == 0:
            problems.append(f"part {n}: no rows in {db} (has the harvest timer run?)")
        elif len(done) < expected_tasks:
            msg = f"part {n}: {len(done)}/{expected_tasks} tasks pulled and COMPLETED"
            if allow_incomplete:
                print(f"Warning: {msg}; merging what is there (--allow-incomplete)")
            else:
                problems.append(msg)
        manifest = read_json(part_dir / "campaign_manifest.json")
        if manifest is None:
            problems.append(f"part {n}: campaign_manifest.json missing or unreadable")
        else:
            manifests[n] = manifest
    if manifests:
        reference_part = min(manifests)
        reference = {k: manifests[reference_part].get(k) for k in MANIFEST_KEYS}
        for n, manifest in manifests.items():
            actual = {k: manifest.get(k) for k in MANIFEST_KEYS}
            if actual != reference:
                problems.append(f"part {n}: settings {actual} differ from part "
                                f"{reference_part} {reference}")
    return problems


def merge_parts(group: str, parts: list[int], part_dirs: list[Path], args,
                output: Path | None) -> Path | None:
    """Per-part merges, then the combined set (None for a single part)."""
    for n, part_dir in zip(parts, part_dirs):
        done, _ = database_tasks(args.db, part_dir.name)
        if not args.remerge and merged_task_ids(part_dir) == done:
            print(f"part {n}: per-head SRMs already hold its {len(done)} tasks")
            continue
        run([sys.executable, str(REPO_ROOT / "payload" / "python" / "merge_brain_campaigns.py"),
             "--directory", str(part_dir), "--output-dir", str(part_dir), "--db", str(args.db)],
            args.dry_run)
    if len(parts) == 1:
        print(f"One part: its per-head SRMs are in {part_dirs[0]}")
        return None
    output = output or args.root / output_name(group, parts)
    run([sys.executable, str(REPO_ROOT / "campaign_tools" / "merge_batches.py"),
         "--sources", *map(str, part_dirs), "--output-dir", str(output),
         "--workers", str(args.workers), "--database", str(args.db)], args.dry_run)
    return output


def discover_groups(root: Path, prefix: str) -> dict[str, dict[int, Path]]:
    """{campaign_group_id: {part index: part dir}} for pulled parts under *root*."""
    groups: dict[str, dict[int, Path]] = {}
    for part_dir in sorted(root.glob(f"{prefix}*_[0-9][0-9]")):
        if not part_dir.is_dir() or "_merged_" in part_dir.name:
            continue
        manifest = read_json(part_dir / "campaign_manifest.json") or {}
        group = manifest.get("campaign_group_id")
        index = manifest.get("campaign_part_index")
        if not group or not index:
            match = re.match(r"(.+)_(\d{2})$", part_dir.name)
            if not match:
                continue
            group, index = match.group(1), int(match.group(2))
        groups.setdefault(str(group), {})[int(index)] = part_dir
    return groups


def auto(args) -> int:
    """Merge every complete part, and combine whenever the complete set grows."""
    status = 0
    for group, part_map in sorted(discover_groups(args.root, args.auto_prefix).items()):
        complete = []
        for n, part_dir in sorted(part_map.items()):
            manifest = read_json(part_dir / "campaign_manifest.json") or {}
            expected = int(manifest.get("job_count") or args.expected_tasks)
            if not check_parts([n], [part_dir], args.db, expected, False):
                complete.append(n)
        print(f"== {group}: parts pulled {sorted(part_map)}, complete {complete}")
        if not complete:
            continue
        problems = check_parts(complete, [part_map[n] for n in complete], args.db,
                               0, True)  # settings only; completeness checked above
        if problems:
            print("Not merging:\n  " + "\n  ".join(problems), file=sys.stderr)
            status = 1
            continue
        state_path = args.root / f"{group}_merge_state.json"
        state = read_json(state_path) or {"combined": []}
        if len(complete) >= 2 and any(e.get("parts") == complete for e in state["combined"]):
            # combined set is current; still keep the per-part merges current
            for n in complete:
                done, _ = database_tasks(args.db, part_map[n].name)
                if merged_task_ids(part_map[n]) != done:
                    break
            else:
                print(f"   combined set for parts {complete} is up to date")
                continue
        output = args.root / output_name(group, complete)
        if output.exists() and len(complete) >= 2:
            print(f"   {output.name} already exists; not overwriting")
            continue
        try:
            result = merge_parts(group, complete, [part_map[n] for n in complete], args,
                                 None)
        except subprocess.CalledProcessError as exc:
            print(f"Merge failed for {group}: {exc}", file=sys.stderr)
            status = 1
            continue
        if result is not None and not args.dry_run:
            state["combined"].append({"parts": complete, "output": str(result),
                                      "merged_at": time.time()})
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(f"   combined parts {complete} -> {result}")
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", help="campaign_group_id, e.g. brain_288mm_csg_102t_<stamp>")
    ap.add_argument("--parts", help="part numbers: 1-2, 1,3 or 1-10")
    ap.add_argument("--auto", action="store_true",
                    help="merge every complete part of every group matching --auto-prefix")
    ap.add_argument("--auto-prefix", default="brain_288mm_csg_102t_")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--expected-tasks", type=int, default=64,
                    help="tasks per part (--auto reads job_count from each manifest)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="merge parts with fewer pulled COMPLETED tasks than expected")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=4, help="merge_batches.py head workers")
    ap.add_argument("--remerge", action="store_true", help="redo the per-part merges")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.auto:
        return auto(args)
    if not args.group or not args.parts:
        ap.error("--group and --parts are required (or use --auto)")
    parts = parse_parts(args.parts)
    part_dirs = [args.root / f"{args.group}_{n:02d}" for n in parts]
    problems = check_parts(parts, part_dirs, args.db, args.expected_tasks,
                           args.allow_incomplete)
    if problems:
        print("Not merging:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    merge_parts(args.group, parts, part_dirs, args, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
