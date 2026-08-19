#!/usr/bin/env python3
"""Combine per-loop brain-SPECT sparse SRM files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_resolutions(value: str) -> list[float]:
    resolutions = [float(item) for item in value.split(",") if item.strip()]
    if not resolutions or any(item <= 0 for item in resolutions):
        raise argparse.ArgumentTypeError("resolutions must be positive numbers")
    return resolutions


def resolution_label(resolution_mm: float) -> str:
    return f"{resolution_mm:g}".replace(".", "p") + "mm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-loop sparse brain-SPECT SRM files."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resolutions-mm",
        type=parse_resolutions,
        default=[1.0, 1.5, 2.0],
        help="Comma-separated voxel sizes, for example 1,1.5,2.",
    )
    parser.add_argument("--expected-loops", type=int, default=0)
    return parser.parse_args()


def combine_resolution(input_dir: Path, resolution_mm: float) -> tuple[dict, list[str]]:
    label = resolution_label(resolution_mm)
    paths = sorted(input_dir.glob(f"srm_{label}_loop_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No {label} chunk SRMs found in {input_dir}")

    combined: defaultdict[tuple[int, int, int, int, int], int] = defaultdict(int)
    reference: dict[str, object] | None = None
    source_names = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            coords = np.asarray(data["coords"], dtype=np.int32)
            counts = np.asarray(data["counts"], dtype=np.int64)
            if coords.ndim != 2 or coords.shape[1] != 5:
                raise ValueError(f"Invalid coordinate shape in {path}: {coords.shape}")
            if coords.shape[0] != counts.shape[0]:
                raise ValueError(f"coords/counts length mismatch in {path}")

            metadata = {
                "voxel_size_mm": float(
                    np.asarray(data["voxel_size_mm"]).reshape(-1)[0]
                ),
                "grid_size": int(np.asarray(data["grid_size"]).reshape(-1)[0]),
                "hist_range": np.asarray(data["hist_range"]).reshape(-1).tolist(),
                "energy_min_kev": float(
                    np.asarray(data["energy_min_kev"]).reshape(-1)[0]
                ),
                "energy_max_kev": float(
                    np.asarray(data["energy_max_kev"]).reshape(-1)[0]
                ),
            }
            if reference is None:
                reference = metadata
            elif metadata != reference:
                raise ValueError(f"Metadata mismatch in {path}")

            for coordinate, count in zip(coords, counts):
                combined[tuple(int(value) for value in coordinate)] += int(count)
            source_names.append(path.name)

    assert reference is not None
    return {
        "coords": np.asarray(list(combined), dtype=np.int32).reshape(-1, 5),
        "counts": np.asarray(list(combined.values()), dtype=np.int64),
        "metadata": reference,
        "chunk_count": len(paths),
        "source_names": source_names,
    }, source_names


def main() -> int:
    args = parse_args()
    resolutions_mm = list(dict.fromkeys(args.resolutions_mm))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"resolutions": {}, "total_chunk_count": 0}

    for resolution_mm in resolutions_mm:
        result, _ = combine_resolution(args.input_dir, resolution_mm)
        if args.expected_loops and result["chunk_count"] != args.expected_loops:
            raise ValueError(
                f"Expected {args.expected_loops} {resolution_label(resolution_mm)} chunks, "
                f"found {result['chunk_count']}"
            )
        output_path = (
            args.output_dir / f"final_srm_{resolution_label(resolution_mm)}.npz"
        )
        metadata = result["metadata"]
        np.savez_compressed(
            output_path,
            coords=result["coords"],
            counts=result["counts"],
            voxel_size_mm=np.asarray([metadata["voxel_size_mm"]], dtype=np.float32),
            grid_size=np.asarray([metadata["grid_size"]], dtype=np.int32),
            hist_range=np.asarray(metadata["hist_range"], dtype=np.float32),
            energy_min_kev=np.asarray([metadata["energy_min_kev"]], dtype=np.float32),
            energy_max_kev=np.asarray([metadata["energy_max_kev"]], dtype=np.float32),
            chunk_count=np.asarray([result["chunk_count"]], dtype=np.int64),
            accumulated_counts=np.asarray(
                [int(result["counts"].sum())], dtype=np.int64
            ),
            source_files=np.asarray(result["source_names"], dtype=str),
        )
        summary["resolutions"][resolution_label(resolution_mm)] = {
            "output": output_path.name,
            "chunk_count": result["chunk_count"],
            "nonzero_entries": int(result["counts"].size),
            "accumulated_counts": int(result["counts"].sum()),
        }
        summary["total_chunk_count"] = max(
            summary["total_chunk_count"], result["chunk_count"]
        )

    (args.output_dir / "combined_srm_metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Wrote {len(resolutions_mm)} combined SRMs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
