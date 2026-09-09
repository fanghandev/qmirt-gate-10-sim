#!/usr/bin/env python3
"""Combine per-loop brain-SPECT sparse SRM files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, save_npz


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
    parser.add_argument(
        "--input-glob",
        default="srm_{label}_loop_*.npz",
        help="Glob relative to --input-dir; '{label}' expands to the resolution label.",
    )
    parser.add_argument(
        "--expected-inputs",
        type=int,
        default=0,
        help="Inputs expected per resolution; 0 disables the comparison.",
    )
    parser.add_argument(
        "--min-inputs",
        type=int,
        default=1,
        help="Fail if fewer than this many inputs are found.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail instead of warning when fewer than --expected-inputs are found.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Select inputs[shard_index::shard_count] for tree reduction.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Number of shards the inputs are split across.",
    )
    parser.add_argument(
        "--split-per-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one CSR SRM per detector head instead of a single combined file. "
        "Use --no-split-per-head for intermediate combines whose output is fed "
        "back into another combine.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=73,
        help="Number of detector heads expected when splitting.",
    )
    parser.add_argument(
        "--pixels-per-head",
        type=int,
        default=625,
        help="Detector pixels per head; becomes the row count of each per-head SRM.",
    )
    return parser.parse_args()


def merge_sparse(
    coords_a: np.ndarray,
    counts_a: np.ndarray,
    coords_b: np.ndarray,
    counts_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.concatenate((coords_a, coords_b))
    counts = np.concatenate((counts_a, counts_b))
    if coords.shape[0] == 0:
        return coords.reshape(-1, 5), counts

    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    inverse = np.ravel(inverse)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    # Segment boundaries let reduceat sum in int64 ; bincount would force float64.
    boundaries = np.flatnonzero(
        np.concatenate(([True], sorted_inverse[1:] != sorted_inverse[:-1]))
    )
    summed = np.add.reduceat(counts[order], boundaries)
    return unique_coords, summed.astype(np.int64, copy=False)


def combine_resolution(
    input_dir: Path,
    resolution_mm: float,
    input_glob: str,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict:
    label = resolution_label(resolution_mm)
    paths = sorted(input_dir.glob(input_glob.format(label=label)))
    if shard_count > 1:
        paths = paths[shard_index::shard_count]
    if not paths:
        raise FileNotFoundError(f"No {label} input SRMs found in {input_dir}")

    # Fold one input at a time so peak memory tracks the union, not the sum, of inputs.
    acc_coords = np.empty((0, 5), dtype=np.int32)
    acc_counts = np.empty((0,), dtype=np.int64)
    reference: dict[str, object] | None = None
    source_names = []
    source_chunk_count = 0
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

            acc_coords, acc_counts = merge_sparse(
                acc_coords, acc_counts, coords, counts
            )
            source_names.append(path.relative_to(input_dir).as_posix())
            # Inputs may themselves be combined SRMs, so accumulate their provenance.
            if "chunk_count" in data.files:
                source_chunk_count += int(
                    np.asarray(data["chunk_count"]).reshape(-1)[0]
                )
            else:
                source_chunk_count += 1

    assert reference is not None
    return {
        "coords": acc_coords.astype(np.int32, copy=False).reshape(-1, 5),
        "counts": acc_counts,
        "metadata": reference,
        "input_count": len(paths),
        "chunk_count": source_chunk_count,
        "source_names": source_names,
    }


def write_per_head_srms(
    output_dir: Path,
    label: str,
    coords: np.ndarray,
    counts: np.ndarray,
    grid_size: int,
    num_heads: int,
    pixels_per_head: int,
) -> list[dict]:
    """Split the combined SRM into one CSR matrix per head, shape (pixels, voxels).

    Files land in *output_dir* under their final names; the caller is expected to
    stage them and rename them into place so readers never see a half-updated set.
    """
    head_ids = coords[:, 0].astype(np.int64, copy=False)
    pixel_ids = coords[:, 1].astype(np.int64, copy=False)
    if head_ids.size:
        if head_ids.min() < 0 or head_ids.max() >= num_heads:
            raise ValueError(
                f"Head IDs out of range for --num-heads {num_heads}: "
                f"[{head_ids.min()}, {head_ids.max()}]"
            )
        if pixel_ids.min() < 0 or pixel_ids.max() >= pixels_per_head:
            raise ValueError(
                f"Pixel IDs out of range for --pixels-per-head {pixels_per_head}: "
                f"[{pixel_ids.min()}, {pixel_ids.max()}]"
            )

    num_voxels = grid_size**3
    voxel_ids = (
        coords[:, 2].astype(np.int64, copy=False) * grid_size * grid_size
        + coords[:, 3].astype(np.int64, copy=False) * grid_size
        + coords[:, 4].astype(np.int64, copy=False)
    )
    # One sort instead of a full scan per head.
    order = np.argsort(head_ids, kind="stable")
    starts = np.searchsorted(head_ids[order], np.arange(num_heads), side="left")
    ends = np.searchsorted(head_ids[order], np.arange(num_heads), side="right")

    entries = []
    for head_index in range(num_heads):
        rows = order[starts[head_index] : ends[head_index]]
        matrix = coo_matrix(
            (counts[rows], (pixel_ids[rows], voxel_ids[rows])),
            shape=(pixels_per_head, num_voxels),
            dtype=np.int64,
        ).tocsr()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        output_path = output_dir / f"final_srm_{label}_head_{head_index + 1:02d}.npz"
        save_npz(output_path, matrix)
        entries.append(
            {
                "head": head_index + 1,
                "output": output_path.name,
                "nonzero_entries": int(matrix.nnz),
                "accumulated_counts": int(matrix.sum()),
            }
        )
    return entries


def main() -> int:
    args = parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, --shard-count)")
    resolutions_mm = list(dict.fromkeys(args.resolutions_mm))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"resolutions": {}, "total_chunk_count": 0}
    # The dashboard re-reads this directory every minute, so per-head files are
    # built here and renamed into place only once all of them exist.
    staging_dir = args.output_dir / ".srm_staging"
    if args.split_per_head:
        shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir()

    for resolution_mm in resolutions_mm:
        label = resolution_label(resolution_mm)
        result = combine_resolution(
            args.input_dir,
            resolution_mm,
            args.input_glob,
            args.shard_index,
            args.shard_count,
        )
        found = result["input_count"]
        if found < args.min_inputs:
            raise ValueError(
                f"Found {found} {label} inputs, below the required minimum of "
                f"{args.min_inputs}"
            )
        if args.expected_inputs and found != args.expected_inputs:
            message = f"Expected {args.expected_inputs} {label} inputs, found {found}"
            if args.require_complete or found > args.expected_inputs:
                raise ValueError(message)
            print(f"Warning: {message}; combining the available subset.")
        output_path = args.output_dir / f"final_srm_{label}.npz"
        metadata = result["metadata"]
        common_summary = {
            "chunk_count": result["chunk_count"],
            "input_count": found,
            "expected_input_count": args.expected_inputs,
            "complete": not args.expected_inputs or found == args.expected_inputs,
            "nonzero_entries": int(result["counts"].size),
            "accumulated_counts": int(result["counts"].sum()),
            "source_files": result["source_names"],
        }
        if args.split_per_head:
            head_entries = write_per_head_srms(
                staging_dir,
                label,
                result["coords"],
                result["counts"],
                metadata["grid_size"],
                args.num_heads,
                args.pixels_per_head,
            )
            summary["resolutions"][label] = {
                # save_npz cannot carry extra keys, so the grid/energy metadata
                # the per-head CSR files need lives here instead.
                "layout": "per_head_csr",
                "outputs": [entry["output"] for entry in head_entries],
                "heads": head_entries,
                "num_heads": args.num_heads,
                "pixels_per_head": args.pixels_per_head,
                "voxel_size_mm": metadata["voxel_size_mm"],
                "grid_size": metadata["grid_size"],
                "hist_range": metadata["hist_range"],
                "energy_min_kev": metadata["energy_min_kev"],
                "energy_max_kev": metadata["energy_max_kev"],
                **common_summary,
            }
        else:
            np.savez_compressed(
                output_path,
                coords=result["coords"],
                counts=result["counts"],
                voxel_size_mm=np.asarray([metadata["voxel_size_mm"]], dtype=np.float32),
                grid_size=np.asarray([metadata["grid_size"]], dtype=np.int32),
                hist_range=np.asarray(metadata["hist_range"], dtype=np.float32),
                energy_min_kev=np.asarray(
                    [metadata["energy_min_kev"]], dtype=np.float32
                ),
                energy_max_kev=np.asarray(
                    [metadata["energy_max_kev"]], dtype=np.float32
                ),
                chunk_count=np.asarray([result["chunk_count"]], dtype=np.int64),
                input_count=np.asarray([found], dtype=np.int64),
                expected_input_count=np.asarray([args.expected_inputs], dtype=np.int64),
                accumulated_counts=np.asarray(
                    [int(result["counts"].sum())], dtype=np.int64
                ),
                source_files=np.asarray(result["source_names"], dtype=str),
            )
            summary["resolutions"][label] = {
                "layout": "combined_coords",
                "output": output_path.name,
                **common_summary,
            }
        summary["total_chunk_count"] = max(
            summary["total_chunk_count"], result["chunk_count"]
        )

    if args.split_per_head:
        for staged in sorted(staging_dir.glob("*.npz")):
            staged.replace(args.output_dir / staged.name)
        staging_dir.rmdir()
        for label in (resolution_label(item) for item in resolutions_mm):
            stale = args.output_dir / f"final_srm_{label}.npz"
            stale.unlink(missing_ok=True)

    metadata_path = args.output_dir / "combined_srm_metadata.json"
    metadata_tmp = metadata_path.with_suffix(".json.tmp")
    metadata_tmp.write_text(json.dumps(summary, indent=2) + "\n")
    metadata_tmp.replace(metadata_path)
    if args.split_per_head:
        print(
            f"Wrote {len(resolutions_mm) * args.num_heads} per-head SRMs to "
            f"{args.output_dir}"
        )
    else:
        print(f"Wrote {len(resolutions_mm)} combined SRMs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
