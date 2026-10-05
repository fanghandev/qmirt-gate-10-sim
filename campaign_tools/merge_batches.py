#!/usr/bin/env python3
"""Merge completed per-head brain or cardiac SRM campaign outputs."""

import argparse
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from scipy import sparse

METADATA_FILENAME = "combined_srm_metadata.json"
DEFAULT_DATABASE = Path(
    "/data/fanghan/opengate_sim/data/brain_spect/expanse_slurm_jobs.db"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add corresponding per-head brain or cardiac SRMs from multiple campaigns."
    )
    parser.add_argument(
        "-s",
        "--sources",
        nargs="+",
        type=Path,
        help="Campaign directories of one compatible detector type containing per-head SRMs and metadata.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="New directory that will receive the merged SRMs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Concurrent head merges (default: min(4, CPU count)).",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Optional SQLite bookkeeping database used for campaign totals.",
    )
    return parser.parse_args()


def read_metadata(source: Path) -> dict[str, Any]:
    metadata_path = source / METADATA_FILENAME
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    if not metadata_path.is_file():
        raise ValueError(f"Source metadata does not exist: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("layout") != "per_head_csr":
        raise ValueError(f"{source}: expected per_head_csr layout")
    return metadata


def validate_compatibility(
    sources: list[Path], metadata_items: list[dict[str, Any]]
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    reference = metadata_items[0]
    resolutions = list(reference.get("resolutions", {}))
    if not resolutions:
        raise ValueError(f"{sources[0]}: metadata contains no resolutions")

    layouts: dict[str, dict[str, Any]] = {}
    for resolution in resolutions:
        details = reference["resolutions"][resolution]
        layouts[resolution] = {
            key: details.get(key)
            for key in ("layout", "grid_size", "num_heads", "pixels_per_head")
        }

    for source, metadata in zip(sources[1:], metadata_items[1:]):
        if set(metadata.get("resolutions", {})) != set(resolutions):
            raise ValueError(f"{source}: resolution set differs from {sources[0]}")
        for resolution in resolutions:
            details = metadata["resolutions"][resolution]
            actual = {
                key: details.get(key)
                for key in ("layout", "grid_size", "num_heads", "pixels_per_head")
            }
            if actual != layouts[resolution]:
                raise ValueError(
                    f"{source}: incompatible {resolution} layout: "
                    f"expected {layouts[resolution]}, got {actual}"
                )

    return resolutions, layouts


def merge_head(
    sources: list[Path],
    metadata_items: list[dict[str, Any]],
    staging_dir: Path,
    resolution: str,
    head_number: int,
) -> dict[str, Any]:
    filename = f"final_srm_{resolution}_head_{head_number:02d}.npz"
    combined = None
    expected_count = 0

    for source, metadata in zip(sources, metadata_items):
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing input SRM: {path}")
        matrix = sparse.load_npz(path).tocsr()

        head_items = metadata["resolutions"][resolution].get("heads", [])
        if len(head_items) < head_number:
            raise ValueError(
                f"{source}: metadata is missing {resolution} head {head_number}"
            )
        recorded_count = int(head_items[head_number - 1]["accumulated_counts"])
        actual_count = int(matrix.sum())
        if actual_count != recorded_count:
            raise ValueError(
                f"{path}: matrix count {actual_count} differs from metadata "
                f"count {recorded_count}"
            )
        expected_count += recorded_count

        if combined is None:
            combined = matrix
        else:
            if matrix.shape != combined.shape:
                raise ValueError(
                    f"{path}: shape {matrix.shape} differs from {combined.shape}"
                )
            combined = combined + matrix

    assert combined is not None
    combined.sum_duplicates()
    combined.eliminate_zeros()
    actual_count = int(combined.sum())
    if actual_count != expected_count:
        raise ValueError(
            f"{filename}: merged count {actual_count} differs from expected "
            f"count {expected_count}"
        )
    sparse.save_npz(staging_dir / filename, combined)
    return {
        "head": head_number,
        "output": filename,
        "nonzero_entries": int(combined.nnz),
        "accumulated_counts": actual_count,
    }


def summed_integer(metadata_items: list[dict[str, Any]], key: str) -> int:
    return sum(int(metadata.get(key, 0) or 0) for metadata in metadata_items)


def read_database_totals(database: Path, campaign: str) -> dict[str, int] | None:
    if not database.is_file():
        return None
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            """
            SELECT COUNT(DISTINCT array_task_id),
                   COALESCE(SUM(primaries), 0),
                   COALESCE(SUM(raw_singles), 0),
                   COALESCE(SUM(accepted_singles), 0)
            FROM slurm_jobs
            WHERE campaign_name = ?
            """,
            (campaign,),
        ).fetchone()
    if row is None or int(row[0] or 0) == 0:
        return None
    return {
        "input_count": int(row[0] or 0),
        "simulated_primaries": int(row[1] or 0),
        "raw_events": int(row[2] or 0),
        "accepted_events": int(row[3] or 0),
    }


def metadata_resolution_count(metadata: dict[str, Any], resolution: str) -> int:
    """Read a source count from either supported campaign metadata layout."""
    partial_counts = metadata.get("partial_counts")
    if isinstance(partial_counts, dict) and resolution in partial_counts:
        return int(partial_counts[resolution])

    resolution_metadata = metadata.get("resolutions", {}).get(resolution)
    if isinstance(resolution_metadata, dict):
        accumulated_counts = resolution_metadata.get("accumulated_counts")
        if accumulated_counts is not None:
            return int(accumulated_counts)

    raise ValueError(
        f"Metadata has no count for resolution {resolution!r}; expected "
        "partial_counts[resolution] or resolutions[resolution].accumulated_counts"
    )


def metadata_accepted_events(
    metadata: dict[str, Any],
    resolution: str,
    database_total: dict[str, int] | None = None,
) -> int:
    """Read accepted singles from all supported metadata layouts."""
    if database_total is not None and "accepted_events" in database_total:
        return database_total["accepted_events"]
    if metadata.get("accepted_events") is not None:
        return int(metadata["accepted_events"])
    resolution_metadata = metadata.get("resolutions", {}).get(resolution)
    if isinstance(resolution_metadata, dict):
        if resolution_metadata.get("accepted_events") is not None:
            return int(resolution_metadata["accepted_events"])
        if resolution_metadata.get("accumulated_counts") is not None:
            return int(resolution_metadata["accumulated_counts"])
    raise ValueError(
        f"Metadata has no accepted-event total for resolution {resolution!r}; "
        "expected accepted_events or accumulated_counts"
    )


def metadata_total(
    metadata: dict[str, Any],
    resolution: str,
    key: str,
    database_total: dict[str, int] | None = None,
) -> int:
    """Read a campaign total from top-level or per-resolution metadata."""
    if database_total is not None and key in database_total:
        return database_total[key]
    value = metadata.get(key)
    if value is not None:
        return int(value)
    resolution_metadata = metadata.get("resolutions", {}).get(resolution, {})
    if isinstance(resolution_metadata, dict):
        value = resolution_metadata.get(key)
        if value is not None:
            return int(value)
    return 0


def source_resolution_total(
    metadata_items: list[dict[str, Any]],
    resolution: str,
    key: str,
    database_totals: dict[str, dict[str, int] | None],
    sources: list[Path],
) -> int:
    return sum(
        metadata_total(
            metadata,
            resolution,
            key,
            database_totals.get(source.name),
        )
        for metadata, source in zip(metadata_items, sources)
    )


def main() -> None:
    args = parse_args()
    if len(args.sources) < 2:
        raise SystemExit("at least two source campaign directories are required")
    if args.workers < 1:
        raise SystemExit("workers must be positive")

    sources = [source.resolve() for source in args.sources]
    if len(set(sources)) != len(sources):
        raise SystemExit("source campaign directories must be unique")

    output_dir = args.output_dir.resolve()
    print(f"Output directory: {output_dir}")

    if output_dir in sources:
        raise SystemExit("output-dir must differ from every source directory")
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
    print(f"Staging directory: {staging_dir}")
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    if staging_dir.exists():
        raise SystemExit(f"staging directory already exists: {staging_dir}")

    metadata_items = [read_metadata(source) for source in sources]
    database_totals: dict[str, dict[str, int] | None] = {
        source.name: (
            read_database_totals(args.database, source.name) if args.database else None
        )
        for source in sources
    }
    resolutions, layouts = validate_compatibility(sources, metadata_items)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()

    tasks = [
        (resolution, head_number)
        for resolution in resolutions
        for head_number in range(1, int(layouts[resolution]["num_heads"]) + 1)
    ]
    head_metadata: dict[str, list[dict[str, Any]]] = {
        resolution: [] for resolution in resolutions
    }
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                merge_head,
                sources,
                metadata_items,
                staging_dir,
                resolution,
                head_number,
            ): (resolution, head_number)
            for resolution, head_number in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            resolution, head_number = futures[future]
            head_metadata[resolution].append(future.result())
            print(
                f"[{completed}/{len(tasks)}] merged {resolution} head {head_number:02d}",
                flush=True,
            )

    for entries in head_metadata.values():
        entries.sort(key=lambda entry: entry["head"])

    partial_counts = {
        resolution: sum(
            metadata_resolution_count(metadata, resolution)
            for metadata in metadata_items
        )
        for resolution in resolutions
    }
    accepted_events_by_resolution: dict[str, int] = {}
    for resolution, entries in head_metadata.items():
        merged_accepted_singles = sum(
            metadata_accepted_events(
                metadata, resolution, database_totals.get(source.name)
            )
            for metadata, source in zip(metadata_items, sources)
        )
        accepted_events_by_resolution[resolution] = merged_accepted_singles
        output_count = sum(entry["accumulated_counts"] for entry in entries)
        if output_count != partial_counts[resolution]:
            raise ValueError(
                f"{resolution}: output count {output_count} differs from source "
                f"total {partial_counts[resolution]}"
            )
        if output_count != merged_accepted_singles:
            raise ValueError(
                f"{resolution}: merge_head accumulated count {output_count} "
                f"differs from merged metadata accepted_events "
                f"{merged_accepted_singles}"
            )
        print(
            f"{resolution}: accepted singles from merge_head={output_count:,}; "
            f"merged metadata accepted_events={merged_accepted_singles:,}",
            flush=True,
        )

    accepted_totals = set(accepted_events_by_resolution.values())
    if len(accepted_totals) != 1:
        raise ValueError(
            f"Accepted-event totals differ by resolution: "
            f"{accepted_events_by_resolution}"
        )
    merged_accepted_singles = accepted_totals.pop()
    reference_resolution = resolutions[0]
    merged_input_count = source_resolution_total(
        metadata_items,
        reference_resolution,
        "input_count",
        database_totals,
        sources,
    )
    merged_primaries = source_resolution_total(
        metadata_items,
        reference_resolution,
        "simulated_primaries",
        database_totals,
        sources,
    )
    merged_raw_events = source_resolution_total(
        metadata_items,
        reference_resolution,
        "raw_events",
        database_totals,
        sources,
    )

    merged_metadata = {
        "layout": "per_head_csr",
        "input_count": merged_input_count,
        "simulated_primaries": merged_primaries,
        "raw_events": merged_raw_events,
        "accepted_events": merged_accepted_singles,
        "partial_counts": partial_counts,
        "source_campaigns": [
            {
                "directory": str(source),
                "input_count": metadata_total(
                    metadata,
                    reference_resolution,
                    "input_count",
                    database_totals.get(source.name),
                ),
                "simulated_primaries": metadata_total(
                    metadata,
                    reference_resolution,
                    "simulated_primaries",
                    database_totals.get(source.name),
                ),
                "raw_events": metadata_total(
                    metadata,
                    reference_resolution,
                    "raw_events",
                    database_totals.get(source.name),
                ),
                "accepted_events": metadata_accepted_events(
                    metadata,
                    reference_resolution,
                    database_totals.get(source.name),
                ),
            }
            for source, metadata in zip(sources, metadata_items)
        ],
        "resolutions": {
            resolution: {
                **layouts[resolution],
                "outputs": [entry["output"] for entry in head_metadata[resolution]],
                "heads": head_metadata[resolution],
                "input_count": source_resolution_total(
                    metadata_items,
                    resolution,
                    "input_count",
                    database_totals,
                    sources,
                ),
                "simulated_primaries": source_resolution_total(
                    metadata_items,
                    resolution,
                    "simulated_primaries",
                    database_totals,
                    sources,
                ),
                "raw_events": source_resolution_total(
                    metadata_items,
                    resolution,
                    "raw_events",
                    database_totals,
                    sources,
                ),
                "accumulated_counts": sum(
                    entry["accumulated_counts"] for entry in head_metadata[resolution]
                ),
                "accepted_events": merged_accepted_singles,
            }
            for resolution in resolutions
        },
    }
    metadata_path = staging_dir / METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(merged_metadata, indent=2) + "\n", encoding="utf-8"
    )
    staging_dir.replace(output_dir)
    print(f"Merged {len(sources)} campaigns into {output_dir}")


if __name__ == "__main__":
    main()
