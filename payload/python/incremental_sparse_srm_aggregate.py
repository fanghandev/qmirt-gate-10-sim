#!/usr/bin/env python3
"""Incrementally aggregate sparse SRMs through durable deterministic shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--resolutions-mm", default="1,1.5,2")
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--pixels-per-head", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--simulated-primaries", type=int, default=0)
    parser.add_argument("--combiner", type=Path)
    return parser.parse_args()


def label(resolution: str) -> str:
    return f"{resolution.strip().replace('.', 'p')}mm"


def source_fingerprint(path: Path, root: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def shard_for(path: str, count: int) -> int:
    digest = hashlib.sha256(path.encode()).digest()
    return int.from_bytes(digest[:8], "big") % count


def source_key(relative: str, resolutions: list[str]) -> str:
    for resolution in resolutions:
        suffix = f"_{resolution}.npz"
        if relative.endswith(suffix):
            return relative[: -len(suffix)]
    raise ValueError(f"input filename has no configured resolution suffix: {relative}")


def run_combiner(command: list[str]) -> None:
    subprocess.run(command, check=True)


def link_inputs(
    directory: Path, old_output: Path | None, new_paths: list[Path], resolution: str
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if old_output is not None and old_output.is_file():
        (directory / f"state_{resolution}.npz").symlink_to(old_output)
    for index, path in enumerate(new_paths):
        (directory / f"source_{index:07d}_{resolution}.npz").symlink_to(path)


def update_shard(
    shard_index: int,
    new_paths: list[Path],
    shards_dir: Path,
    resolutions: list[str],
    combiner: Path,
    args: argparse.Namespace,
) -> tuple[int, list[Path]]:
    shard_dir = shards_dir / f"shard_{shard_index:03d}"
    update_dir = shard_dir / ".update_inputs"
    shutil.rmtree(update_dir, ignore_errors=True)
    try:
        for resolution in resolutions:
            old_output = shard_dir / f"final_srm_{resolution}.npz"
            matching_paths = [
                path for path in new_paths if path.name.endswith(f"_{resolution}.npz")
            ]
            link_inputs(update_dir, old_output, matching_paths, resolution)
        run_combiner(
            [
                sys.executable,
                str(combiner),
                "--input-dir",
                str(update_dir),
                "--output-dir",
                str(shard_dir),
                "--input-glob",
                "*_{label}.npz",
                "--resolutions-mm",
                args.resolutions_mm,
                "--num-heads",
                str(args.num_heads),
                "--pixels-per-head",
                str(args.pixels_per_head),
                "--no-split-per-head",
            ]
        )
    finally:
        shutil.rmtree(update_dir, ignore_errors=True)
    return shard_index, new_paths


def main() -> int:
    args = parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    resolutions = [
        label(value) for value in args.resolutions_mm.split(",") if value.strip()
    ]
    if not resolutions:
        raise ValueError("--resolutions-mm must not be empty")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    state_dir = output_dir / ".incremental_srm_state"
    shards_dir = state_dir / "shards"
    ledger_path = state_dir / "ledger.json"
    lock_path = state_dir / "lock"
    state_dir.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_handle:
        import fcntl

        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another incremental aggregate owns {output_dir}"
            ) from exc

        combiner = args.combiner or Path(__file__).with_name(
            "combine_spect_sparse_srm.py"
        )
        ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {}
        if ledger.get("shard_count", args.shard_count) != args.shard_count:
            raise ValueError("--shard-count differs from the existing aggregate state")
        consumed: dict[str, dict[str, int | str]] = ledger.get("consumed", {})

        discovered: dict[str, Path] = {}
        for resolution in resolutions:
            pattern = args.input_glob.format(label=resolution)
            for path in sorted(input_dir.glob(pattern)):
                relative = path.relative_to(input_dir).as_posix()
                if relative in discovered:
                    raise ValueError(f"input matches multiple resolutions: {relative}")
                discovered[relative] = path

        new_by_shard: dict[int, list[Path]] = {}
        for relative, path in discovered.items():
            fingerprint = source_fingerprint(path, input_dir)
            previous = consumed.get(relative)
            if previous is not None and previous != fingerprint:
                raise ValueError(f"previously consumed input changed: {relative}")
            if previous is None:
                key = source_key(relative, resolutions)
                new_by_shard.setdefault(shard_for(key, args.shard_count), []).append(
                    path
                )

        completed_shards: list[tuple[int, list[Path]]] = []
        if new_by_shard:
            with ThreadPoolExecutor(
                max_workers=min(args.workers, len(new_by_shard))
            ) as executor:
                futures = [
                    executor.submit(
                        update_shard,
                        shard_index,
                        new_paths,
                        shards_dir,
                        resolutions,
                        combiner,
                        args,
                    )
                    for shard_index, new_paths in sorted(new_by_shard.items())
                ]
                completed_shards = [future.result() for future in futures]
        for _shard_index, new_paths in completed_shards:
            for path in new_paths:
                relative = path.relative_to(input_dir).as_posix()
                consumed[relative] = source_fingerprint(path, input_dir)
        ledger = {
            "schema_version": 1,
            "shard_count": args.shard_count,
            "consumed": consumed,
        }
        temporary_ledger = ledger_path.with_suffix(".json.tmp")
        temporary_ledger.write_text(json.dumps(ledger, indent=2) + "\n")
        temporary_ledger.replace(ledger_path)

        shard_count = len(
            [path for path in shards_dir.glob("shard_*") if path.is_dir()]
        )
        if shard_count == 0:
            raise FileNotFoundError("no matching SRM inputs found")
        run_combiner(
            [
                sys.executable,
                str(combiner),
                "--input-dir",
                str(shards_dir),
                "--output-dir",
                str(output_dir),
                "--input-glob",
                "shard_*/final_srm_{label}.npz",
                "--resolutions-mm",
                args.resolutions_mm,
                "--num-heads",
                str(args.num_heads),
                "--pixels-per-head",
                str(args.pixels_per_head),
                "--expected-inputs",
                str(shard_count),
                "--require-complete",
                "--simulated-primaries",
                str(args.simulated_primaries),
                "--split-per-head",
            ]
        )
        metadata_path = output_dir / "combined_srm_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["incremental"] = {
            "schema_version": 1,
            "state_directory": state_dir.name,
            "shard_count": args.shard_count,
            "consumed_input_files": len(consumed),
            "newly_merged_input_files": sum(
                len(paths) for paths in new_by_shard.values()
            ),
        }
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary_metadata.replace(metadata_path)
        print(
            f"Incremental aggregate: consumed={len(consumed)} newly_merged="
            f"{sum(len(paths) for paths in new_by_shard.values())} shards={shard_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
