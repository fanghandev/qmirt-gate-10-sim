#!/usr/bin/env python3
"""Merge per-task OpenGATE statistics into one job-level summary."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path


def parse_value(text: str):
    stripped = text.strip()
    if stripped == "":
        return ""
    try:
        number = int(stripped)
        return number
    except ValueError:
        try:
            number = float(stripped)
        except ValueError:
            return stripped
        if number.is_integer():
            return int(number)
        return number


def parse_stats_file(path: Path) -> dict[str, object]:
    stats: dict[str, object] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        stats[key.strip()] = parse_value(value)
    return stats


def merge_stats_files(input_files: list[Path]) -> dict[str, object]:
    merged: dict[str, object] = {
        "task_count": len(input_files),
        "stats_files": [str(path) for path in input_files],
    }
    numeric_totals: dict[str, float] = {}
    other_values: dict[str, object] = {}

    for path in input_files:
        stats = parse_stats_file(path)
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                numeric_totals[key] = numeric_totals.get(key, 0.0) + float(value)
            else:
                other_values.setdefault(key, value)

    for key, value in numeric_totals.items():
        if value.is_integer():
            merged[key] = int(value)
        else:
            merged[key] = value

    for key, value in other_values.items():
        if key not in merged:
            merged[key] = value

    return merged


def write_merged_stats(output_path: Path, merged: dict[str, object]) -> None:
    lines: list[str] = []
    lines.append(f"task_count: {merged.get('task_count', 0)}")
    stats_files = merged.get("stats_files", [])
    lines.append("stats_files:")
    for path in stats_files:
        lines.append(f"  - {path}")

    for key, value in merged.items():
        if key in {"task_count", "stats_files"}:
            continue
        lines.append(f"{key}: {value}")

    output_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge OpenGATE per-task stats files")
    parser.add_argument("--input-glob", required=True, help="Glob pattern for task stats files")
    parser.add_argument("--output", required=True, help="Merged stats output path")
    args = parser.parse_args()

    input_files = [Path(path) for path in sorted(glob.glob(args.input_glob, recursive=True))]
    if not input_files:
        raise FileNotFoundError(f"No stats files matched pattern: {args.input_glob}")

    merged = merge_stats_files(input_files)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_merged_stats(output_path, merged)
    print(f"Merged {len(input_files)} stats files into {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())