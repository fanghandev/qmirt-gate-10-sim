#!/usr/bin/env python3
"""Report local pull progress and total simulated events for brain-SPECT campaigns.

Run this anytime on the workstation to see how far each campaign shard has been
pulled (via globus_pull_campaign.py) and the total simulated_primaries collected
so far, without touching the cluster or the sshfs mount.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_LOCAL_ROOT = Path("/data/fanghan/opengate_sim/data/brain_spect")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def task_primaries(task_dir: Path) -> tuple[int | None, bool]:
    """Return (simulated_primaries, complete) from one task's combined_srm_metadata.json."""
    metadata = read_json(task_dir / "combined_srm_metadata.json")
    if not metadata:
        return None, False
    resolutions = metadata.get("resolutions", {})
    if not resolutions:
        return None, False
    # simulated_primaries/complete are the same across resolutions; any one will do.
    entry = next(iter(resolutions.values()))
    return entry.get("simulated_primaries"), bool(entry.get("complete"))


def scan_campaign(campaign_dir: Path) -> dict[str, Any]:
    manifest = read_json(campaign_dir / "campaign_manifest.json") or {}
    job_count = int(manifest.get("job_count") or 0)
    expected_events_total = manifest.get("expected_events_total")

    pulled_complete = pulled_partial = not_pulled = 0
    total_primaries = 0
    for task_dir in sorted(
        (p for p in campaign_dir.glob("task_*") if p.is_dir()),
        key=lambda p: int(p.name.removeprefix("task_")),
    ):
        primaries, complete = task_primaries(task_dir)
        if primaries is None:
            not_pulled += 1
            continue
        total_primaries += primaries
        if complete:
            pulled_complete += 1
        else:
            pulled_partial += 1

    if job_count:
        not_pulled = job_count - pulled_complete - pulled_partial

    return {
        "job_count": job_count,
        "pulled_complete": pulled_complete,
        "pulled_partial": pulled_partial,
        "not_pulled": max(not_pulled, 0),
        "total_primaries": total_primaries,
        "expected_events_total": expected_events_total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-root",
        type=Path,
        default=DEFAULT_LOCAL_ROOT,
        help=f"Parent directory holding pulled campaigns (default: {DEFAULT_LOCAL_ROOT}).",
    )
    parser.add_argument(
        "--campaign-glob",
        default="*",
        help="Glob (relative to --local-root) selecting campaigns to report on.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign_dirs = sorted(
        p for p in args.local_root.glob(args.campaign_glob) if p.is_dir()
    )

    results: dict[str, dict[str, Any]] = {}
    for campaign_dir in campaign_dirs:
        if not (campaign_dir / "campaign_manifest.json").is_file():
            continue
        results[campaign_dir.name] = scan_campaign(campaign_dir)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    grand_total_primaries = 0
    header = f"{'campaign':<40} {'complete':>8} {'partial':>8} {'not pulled':>10} {'simulated_primaries':>20}"
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        print(
            f"{name:<40} {result['pulled_complete']:>8} {result['pulled_partial']:>8} "
            f"{result['not_pulled']:>10} {result['total_primaries']:>20,}"
        )
        grand_total_primaries += result["total_primaries"]

    print("-" * len(header))
    print(f"Total simulated events pulled so far: {grand_total_primaries:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
