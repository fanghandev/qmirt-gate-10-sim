#!/usr/bin/env python3
"""Repair task-level brain-SPECT SRM outputs so sensitivity stays computable.

Handles two situations that leave ``simulated_primaries`` wrong or missing:

1. A task finished all loops and has ``final_srm_*.npz`` +
   ``combined_srm_metadata.json``, but the wrapper never passed
   ``--simulated-primaries`` to the combiner, so the field is stuck at 0.
   This is patched in place from the task's own per-loop stats files;
   the SRM data itself is untouched.
2. A task was killed mid-campaign (SLURM time limit, straggler, etc.) with
   some loops fully done and copied to ``srm_chunks`` but never combined, so
   it has no ``final_srm_*.npz`` at all and contributes nothing. This
   combines just the completed loops (never partially-written ones, since a
   loop's outputs are only copied to ``srm_chunks`` after it finishes) and
   records ``simulated_primaries`` for exactly those loops, with
   ``complete: false`` in the metadata since fewer than the manifest's
   ``num_loops`` were folded in.

Dry-run by default; pass --apply to write changes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMBINE_SCRIPT = REPO_ROOT / "payload" / "python" / "combine_spect_sparse_srm.py"
RESOLUTION_LABELS = ("1mm", "1p5mm", "2mm")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def sum_primaries(chunk_dir: Path) -> tuple[int, int]:
    """Return (simulated_primaries, loops_done) from completed-loop stats files."""
    total = 0
    loops_done = 0
    for stats_path in sorted(chunk_dir.glob("*_sim_stats_loop_*.txt")):
        stats = read_json(stats_path)
        if not stats:
            continue
        loops_done += 1
        total += int(stats.get("events", {}).get("value", 0))
    return total, loops_done


def patch_metadata_primaries(
    metadata_path: Path, simulated_primaries: int, apply: bool
) -> bool:
    metadata = read_json(metadata_path)
    if metadata is None:
        return False
    changed = False
    for label, entry in metadata.get("resolutions", {}).items():
        if entry.get("simulated_primaries") != simulated_primaries:
            entry["simulated_primaries"] = simulated_primaries
            changed = True
    if changed and apply:
        tmp = metadata_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(metadata, indent=2) + "\n")
        tmp.replace(metadata_path)
    return changed


def combine_partial_task(
    task_dir: Path, num_loops_manifest: int, simulated_primaries: int, apply: bool
) -> bool:
    chunk_dir = task_dir / "srm_chunks"
    cmd = [
        sys.executable,
        str(COMBINE_SCRIPT),
        "--input-dir",
        str(chunk_dir),
        "--output-dir",
        str(task_dir),
        "--expected-inputs",
        str(num_loops_manifest),
        "--no-split-per-head",
        "--simulated-primaries",
        str(simulated_primaries),
    ]
    if not apply:
        print(f"  [dry-run] would run: {' '.join(cmd)}")
        return True
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED combining {task_dir}: {result.stderr.strip()}")
        return False
    print(f"  combined {task_dir} ({result.stdout.strip()})")
    return True


def process_campaign(campaign_dir: Path, apply: bool) -> None:
    manifest = read_json(campaign_dir / "campaign_manifest.json") or {}
    num_loops_manifest = int(manifest.get("num_loops") or 1)

    patched = skipped_no_loops = combined = 0
    for task_dir in sorted(
        campaign_dir.glob("task_*"), key=lambda p: int(p.name.removeprefix("task_"))
    ):
        chunk_dir = task_dir / "srm_chunks"
        if not chunk_dir.is_dir():
            continue
        simulated_primaries, loops_done = sum_primaries(chunk_dir)

        final_srms = [
            task_dir / f"final_srm_{label}.npz" for label in RESOLUTION_LABELS
        ]
        metadata_path = task_dir / "combined_srm_metadata.json"
        if any(p.is_file() for p in final_srms) and metadata_path.is_file():
            if patch_metadata_primaries(metadata_path, simulated_primaries, apply):
                print(
                    f"{task_dir.name}: patched simulated_primaries -> {simulated_primaries} "
                    f"({loops_done} loops)"
                )
                patched += 1
            continue

        if loops_done == 0:
            skipped_no_loops += 1
            continue

        print(
            f"{task_dir.name}: no final_srm yet, {loops_done}/{num_loops_manifest} loops "
            f"complete, simulated_primaries={simulated_primaries}"
        )
        if combine_partial_task(
            task_dir, num_loops_manifest, simulated_primaries, apply
        ):
            combined += 1

    print(
        f"\nSummary: patched={patched} combined_partial={combined} "
        f"skipped_no_loops={skipped_no_loops} (apply={apply})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "campaign_dir", type=Path, help="Campaign directory containing task_* dirs."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this flag, only prints what would happen.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    if not campaign_dir.is_dir():
        raise SystemExit(f"Not a directory: {campaign_dir}")
    process_campaign(campaign_dir, args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
