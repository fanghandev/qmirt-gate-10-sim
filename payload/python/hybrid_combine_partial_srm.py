import json
import shutil
import sqlite3
import tarfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from scipy.sparse import coo_matrix, csr_matrix, save_npz
from tqdm import tqdm

# ospool jobs are keyed by (ClusterId, ProcId); slurm tasks are keyed by their
# "task_N" directory name.
JobKey = tuple[int, int] | str
DEFAULT_GROUP_SIZE = 100
DEFAULT_RESOLUTIONS = ("1mm", "1p5mm", "2mm")
LOCAL_DATA_ROOTS = {
    "ospool": Path("/data/fanghan/opengate_sim/data/cardiac_spect"),
    "slurm": Path("/data/fanghan/opengate_sim/data/brain_spect"),
}
# Detector head counts for the per-head CSR split (25x25 = 625 pixels per head, both scanners).
NUM_HEADS = {"ospool": 80, "slurm": 73}
PIXELS_PER_HEAD = 625
# Brain tasks are already fully combined per task (5 loops folded in) with tens of
# millions of nonzero entries each, far heavier than one raw ospool job tarball. Rather
# than batching several such tasks through one GPU merge, brain tasks are folded in
# one at a time, per detector head (see process_brain_jobs_per_head).
GROUP_SIZE_DEFAULTS = {"ospool": DEFAULT_GROUP_SIZE, "slurm": 5}


def show_tar_contents(tarball_path: str) -> None:
    path = Path(tarball_path)
    if not path.is_file():
        raise FileNotFoundError(f"The file {tarball_path} does not exist.")
    with tarfile.open(path, "r:gz") as tar:
        for name in tar.getnames():
            print(name)


def parse_stats_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    try:
        return int(value)
    except ValueError:
        try:
            number = float(value)
        except ValueError:
            return value
        return int(number) if number.is_integer() else number


def parse_text_stats(text: str) -> dict[str, Any]:
    """Parse OpenGATE's ``key: value`` simulation statistics format."""
    stats: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            stats[key.strip()] = parse_stats_value(value)
    return stats


def get_tar_contents(
    tarball_path: str | Path,
    resolutions: tuple[str, ...] = DEFAULT_RESOLUTIONS,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Read SRM arrays and statistics from one tarball into independent objects."""
    if isinstance(tarball_path, str):
        tarball_path = Path(tarball_path)
    if not tarball_path.is_file():
        raise FileNotFoundError(f"The file {tarball_path} does not exist.")
    base_name = tarball_path.name.removesuffix(".tar.gz")
    metadata_name = base_name.replace("srm_", "srm_metadata_", 1) + "_loop_00000.json"
    stats_name = base_name.replace("srm_", "sim_stats_", 1) + "_loop_00000.txt"
    srms: dict[str, dict[str, np.ndarray]] = {}
    stats: dict[str, Any] = {}

    with tarfile.open(tarball_path, "r:gz") as tar:
        members = {member.name: member for member in tar.getmembers()}

        for resolution in resolutions:
            srm_name = f"{base_name}_loop_00000_{resolution}.npz"
            member = members.get(f"{base_name}/{srm_name}")
            if member is None:
                continue
            io_reader = tar.extractfile(member)
            if io_reader is None:
                continue
            with io_reader as stream, np.load(stream, allow_pickle=False) as data:
                srms[resolution] = {
                    key: np.array(data[key], copy=True) for key in data.files
                }

        metadata_member = members.get(f"{base_name}/stats/{metadata_name}")
        if metadata_member is not None:
            io_reader = tar.extractfile(metadata_member)
            if io_reader is not None:
                with io_reader:
                    stats[metadata_name] = json.load(io_reader)

        stats_member = members.get(f"{base_name}/stats/{stats_name}")
        if stats_member is not None:
            io_reader = tar.extractfile(stats_member)
            if io_reader is not None:
                with io_reader:
                    stats[stats_name] = parse_text_stats(
                        io_reader.read().decode("utf-8")
                    )

    return srms, stats


def get_task_dir_contents(
    task_dir: Path,
    resolutions: tuple[str, ...] = DEFAULT_RESOLUTIONS,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Read a Slurm brain-SPECT task's already-combined per-resolution SRMs.

    Unlike ospool jobs, each task's ``final_srm_{label}.npz`` already holds the
    coords/counts merged across that task's loops, so no tarball unpacking is
    needed here.
    """
    if not task_dir.is_dir():
        raise FileNotFoundError(f"The directory {task_dir} does not exist.")

    srms: dict[str, dict[str, np.ndarray]] = {}
    for resolution in resolutions:
        srm_path = task_dir / f"final_srm_{resolution}.npz"
        if not srm_path.is_file():
            continue
        with np.load(srm_path, allow_pickle=False) as data:
            srms[resolution] = {
                key: np.array(data[key], copy=True) for key in data.files
            }

    stats: dict[str, Any] = {}
    metadata_path = task_dir / "combined_srm_metadata.json"
    if metadata_path.is_file():
        stats[f"{task_dir.name}_combined_srm_metadata.json"] = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    return srms, stats


def scalar_array(data: dict[str, np.ndarray], key: str) -> int | float | None:
    """Return a scalar NPZ field as a native Python value."""
    if key not in data:
        return None
    value = np.asarray(data[key]).reshape(-1)[0]
    return value.item()


def resolution_metadata(
    resolution: str, data: dict[str, np.ndarray], stats: dict[str, Any]
) -> dict[str, Any]:
    """Build the metadata shape consumed by the campaign dashboard."""
    metadata = {
        "voxel_size_mm": scalar_array(data, "voxel_size_mm"),
        "grid_size": scalar_array(data, "grid_size"),
        "hist_range": np.asarray(data["hist_range"]).reshape(-1).tolist()
        if "hist_range" in data
        else None,
        "energy_min_kev": scalar_array(data, "energy_min_kev"),
        "energy_max_kev": scalar_array(data, "energy_max_kev"),
        "raw_events": scalar_array(data, "raw_events") or 0,
        "accepted_events": scalar_array(data, "accepted_events") or 0,
        "nonzero_entries": int(np.asarray(data["counts"]).size)
        if "counts" in data
        else 0,
        "accumulated_counts": int(np.asarray(data["counts"]).sum())
        if "counts" in data
        else 0,
    }
    metadata["resolution"] = resolution
    metadata["simulation_stats"] = stats
    return metadata


def update_campaign_metadata(
    output_path: Path,
    summaries: dict[str, dict[str, Any]],
    processed_jobs: int,
    *,
    complete: bool = False,
) -> None:
    """Atomically publish dashboard-readable progress metadata."""
    payload = {
        "schema_version": 1,
        "complete": complete,
        "processed_jobs": processed_jobs,
        "resolutions": summaries,
    }
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)


def accumulate_metadata(
    summaries: dict[str, dict[str, Any]],
    srms: dict[str, dict[str, np.ndarray]],
    stats: dict[str, Any],
) -> None:
    """Accumulate one job's resolution metadata into the dashboard summary."""
    for resolution, data in srms.items():
        current = resolution_metadata(resolution, data, stats)
        previous = summaries.get(resolution)
        if previous is None:
            current["input_count"] = 1
            current["source_jobs"] = []
            summaries[resolution] = current
            continue

        for key in (
            "voxel_size_mm",
            "grid_size",
            "hist_range",
            "energy_min_kev",
            "energy_max_kev",
        ):
            if previous[key] != current[key]:
                raise ValueError(
                    f"Inconsistent {key} for resolution {resolution}: "
                    f"{previous[key]} != {current[key]}"
                )
        previous["input_count"] += 1
        previous["raw_events"] += current["raw_events"]
        previous["accepted_events"] += current["accepted_events"]
        previous["nonzero_entries"] += current["nonzero_entries"]
        previous["accumulated_counts"] += current["accumulated_counts"]


def _sparse_tensor(data: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    """Create and coalesce one 5D SRM sparse tensor on ``device``."""
    coords = np.asarray(data["coords"], dtype=np.int64)
    counts = np.asarray(data["counts"], dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 5:
        raise ValueError(f"Expected coords with shape (N, 5), got {coords.shape}")
    if counts.ndim != 1 or counts.shape[0] != coords.shape[0]:
        raise ValueError("coords and counts must contain the same number of entries")

    grid_size = int(scalar_array(data, "grid_size") or 0)
    if grid_size < 1:
        raise ValueError("SRM grid_size must be positive")
    detector_shape = (
        max(int(coords[:, 0].max()) + 1, 1) if coords.size else 1,
        max(int(coords[:, 1].max()) + 1, 1) if coords.size else 1,
    )
    indices = torch.as_tensor(coords.T, dtype=torch.int64, device=device)
    values = torch.as_tensor(counts, dtype=torch.int64, device=device)
    with torch.sparse.check_sparse_tensor_invariants(False):
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=(*detector_shape, grid_size, grid_size, grid_size),
            device=device,
            check_invariants=False,
            is_coalesced=False,
        ).coalesce()


def _merge_sparse_tensors(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Merge two coalesced sparse tensors, summing duplicate coordinates."""
    size = tuple(max(a, b) for a, b in zip(left.shape, right.shape))
    indices = torch.cat((left.indices(), right.indices()), dim=1)
    values = torch.cat((left.values(), right.values()))
    with torch.sparse.check_sparse_tensor_invariants(False):
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=size,
            device=left.device,
            check_invariants=False,
            is_coalesced=False,
        ).coalesce()


def merge_srm_jobs_torch(
    job_srms: Iterable[dict[str, dict[str, np.ndarray]]],
    *,
    device: str = "cuda",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Merge SRMs from a job group with Torch sparse COO tensors."""
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")

    accumulators = _merge_srm_job_tensors(job_srms, requested_device)

    merged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for resolution, sparse in accumulators.items():
        merged[resolution] = (
            sparse.indices().T.cpu().numpy(),
            sparse.values().cpu().numpy(),
        )
    return merged


def _merge_srm_job_tensors(
    job_srms: Iterable[dict[str, dict[str, np.ndarray]]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    accumulators: dict[str, torch.Tensor] = {}
    for srms in job_srms:
        for resolution, data in srms.items():
            current = _sparse_tensor(data, device)
            previous = accumulators.get(resolution)
            accumulators[resolution] = (
                current
                if previous is None
                else _merge_sparse_tensors(previous, current)
            )
    return accumulators


def write_per_head_srms(
    output_dir: Path,
    label: str,
    coords: np.ndarray,
    counts: np.ndarray,
    grid_size: int,
    num_heads: int,
    pixels_per_head: int,
) -> list[dict[str, Any]]:
    """Split one resolution's combined SRM into one CSR matrix per detector head."""
    head_ids = coords[:, 0].astype(np.int64, copy=False)
    pixel_ids = coords[:, 1].astype(np.int64, copy=False)
    if head_ids.size:
        if head_ids.min() < 0 or head_ids.max() >= num_heads:
            raise ValueError(
                f"Head IDs out of range for num_heads={num_heads}: "
                f"[{head_ids.min()}, {head_ids.max()}]"
            )
        if pixel_ids.min() < 0 or pixel_ids.max() >= pixels_per_head:
            raise ValueError(
                f"Pixel IDs out of range for pixels_per_head={pixels_per_head}: "
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


def write_split_outputs(
    output_dir: Path,
    merged: dict[str, tuple[np.ndarray, np.ndarray]],
    metadata: dict[str, dict[str, Any]],
    *,
    num_heads: int,
    pixels_per_head: int = PIXELS_PER_HEAD,
) -> None:
    """Write per-head CSR SRMs instead of one giant combined file per resolution.

    Staged in a scratch directory and renamed into place once every head file for
    a resolution exists, so the dashboard never reads a half-written set.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / ".srm_staging"
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir()

    for resolution, (coords, counts) in merged.items():
        entry = metadata[resolution]
        head_entries = write_per_head_srms(
            staging_dir,
            resolution,
            coords,
            counts,
            int(entry["grid_size"]),
            num_heads,
            pixels_per_head,
        )
        entry["layout"] = "per_head_csr"
        entry["outputs"] = [item["output"] for item in head_entries]
        entry["heads"] = head_entries
        entry["num_heads"] = num_heads
        entry["pixels_per_head"] = pixels_per_head
        entry["nonzero_entries"] = sum(item["nonzero_entries"] for item in head_entries)
        entry["accumulated_counts"] = sum(
            item["accumulated_counts"] for item in head_entries
        )
        # A stale single-file output from an earlier run would otherwise shadow the split.
        (output_dir / f"final_srm_{resolution}.npz").unlink(missing_ok=True)

    for staged in staging_dir.glob("*.npz"):
        staged.replace(output_dir / staged.name)
    staging_dir.rmdir()


def _accumulate_head_chunks(
    data: dict[str, np.ndarray],
    res_accumulators: dict[int, csr_matrix],
    num_heads: int,
    pixels_per_head: int,
    device: torch.device,
) -> None:
    """Dedupe and fold one task's one-resolution SRM into per-head CSR accumulators.

    Each head's chunk is tiny (~1/num_heads of the task), so GPU coalesce never
    holds more than a small slice in memory; the running total lives on the CPU
    as a CSR matrix (its ``+`` already sums duplicate entries).
    """
    coords = np.asarray(data["coords"], dtype=np.int64)
    counts = np.asarray(data["counts"], dtype=np.int64)
    grid_size = int(scalar_array(data, "grid_size") or 0)
    num_voxels = grid_size**3
    head_ids = coords[:, 0]
    pixel_ids = coords[:, 1]
    if head_ids.size:
        if head_ids.min() < 0 or head_ids.max() >= num_heads:
            raise ValueError(
                f"Head IDs out of range for num_heads={num_heads}: "
                f"[{head_ids.min()}, {head_ids.max()}]"
            )
        if pixel_ids.min() < 0 or pixel_ids.max() >= pixels_per_head:
            raise ValueError(
                f"Pixel IDs out of range for pixels_per_head={pixels_per_head}: "
                f"[{pixel_ids.min()}, {pixel_ids.max()}]"
            )
    voxel_ids = (
        coords[:, 2] * grid_size * grid_size + coords[:, 3] * grid_size + coords[:, 4]
    )
    order = np.argsort(head_ids, kind="stable")
    sorted_heads = head_ids[order]
    starts = np.searchsorted(sorted_heads, np.arange(num_heads), side="left")
    ends = np.searchsorted(sorted_heads, np.arange(num_heads), side="right")

    for head_index in range(num_heads):
        rows = order[starts[head_index] : ends[head_index]]
        if rows.size == 0:
            continue
        indices = torch.as_tensor(
            np.stack([pixel_ids[rows], voxel_ids[rows]]),
            dtype=torch.int64,
            device=device,
        )
        values = torch.as_tensor(counts[rows], dtype=torch.int64, device=device)
        with torch.sparse.check_sparse_tensor_invariants(False):
            tensor = torch.sparse_coo_tensor(
                indices,
                values,
                size=(pixels_per_head, num_voxels),
                device=device,
                check_invariants=False,
                is_coalesced=False,
            ).coalesce()
        new_matrix = coo_matrix(
            (tensor.values().cpu().numpy(), tensor.indices().cpu().numpy()),
            shape=(pixels_per_head, num_voxels),
        ).tocsr()
        del tensor, indices, values
        previous = res_accumulators.get(head_index)
        res_accumulators[head_index] = (
            new_matrix if previous is None else previous + new_matrix
        )


def write_head_accumulators(
    output_dir: Path,
    accumulators: dict[str, dict[int, csr_matrix]],
    grid_metadata: dict[str, dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    *,
    num_heads: int,
    pixels_per_head: int,
) -> None:
    """Write the current running per-head totals, staged then renamed into place."""
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / ".srm_staging"
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir()

    for resolution, heads in accumulators.items():
        entry = metadata[resolution]
        num_voxels = grid_metadata[resolution]["grid_size"] ** 3
        outputs = []
        nonzero_total = 0
        accumulated_total = 0
        for head_index in range(num_heads):
            matrix = heads.get(head_index)
            if matrix is None:
                matrix = csr_matrix((pixels_per_head, num_voxels), dtype=np.int64)
            else:
                matrix.sum_duplicates()
                matrix.eliminate_zeros()
            output_path = (
                staging_dir / f"final_srm_{resolution}_head_{head_index + 1:02d}.npz"
            )
            save_npz(output_path, matrix)
            outputs.append(output_path.name)
            nonzero_total += int(matrix.nnz)
            accumulated_total += int(matrix.sum())
        entry["layout"] = "per_head_csr"
        entry["outputs"] = outputs
        entry["num_heads"] = num_heads
        entry["pixels_per_head"] = pixels_per_head
        entry["nonzero_entries"] = nonzero_total
        entry["accumulated_counts"] = accumulated_total
        (output_dir / f"final_srm_{resolution}.npz").unlink(missing_ok=True)

    for staged in staging_dir.glob("*.npz"):
        staged.replace(output_dir / staged.name)
    staging_dir.rmdir()


def process_brain_jobs_per_head(
    job_ids: Iterable[str],
    partial_data_dir: Path,
    *,
    progress_log: Path | None = None,
    metadata_path: Path | None = None,
    output_dir: Path | None = None,
    device: str = "cuda",
    write_every: int = 1,
    num_heads: int = NUM_HEADS["slurm"],
    pixels_per_head: int = PIXELS_PER_HEAD,
    resolutions: tuple[str, ...] = DEFAULT_RESOLUTIONS,
) -> None:
    """Merge brain-SPECT task SRMs one task at a time, per detector head.

    Avoids ever materializing one huge cross-head/cross-task coordinate array;
    only a single head's chunk of a single task is ever built as a GPU tensor.
    """
    ordered_jobs = sorted(job_ids, key=job_sort_key)
    total = len(ordered_jobs)
    if write_every < 1:
        raise ValueError("write_every must be greater than zero")
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")

    metadata: dict[str, dict[str, Any]] = {}
    grid_metadata: dict[str, dict[str, Any]] = {}
    accumulators: dict[str, dict[int, csr_matrix]] = {
        resolution: {} for resolution in resolutions
    }

    log_file = progress_log.open("w", encoding="utf-8") if progress_log else None
    try:
        progress = tqdm(
            total=total,
            desc="Merging brain tasks (per-head)",
            unit="task",
            disable=log_file is not None,
        )
        try:
            for index, job_key in enumerate(ordered_jobs, start=1):
                assert isinstance(job_key, str)
                srms, stats = get_task_dir_contents(
                    partial_data_dir / job_key, resolutions
                )
                accumulate_metadata(metadata, srms, stats)
                for resolution, data in srms.items():
                    grid_metadata.setdefault(
                        resolution,
                        {"grid_size": int(scalar_array(data, "grid_size") or 0)},
                    )
                    _accumulate_head_chunks(
                        data,
                        accumulators[resolution],
                        num_heads,
                        pixels_per_head,
                        requested_device,
                    )
                if requested_device.type == "cuda":
                    torch.cuda.empty_cache()
                progress.update(1)

                # Keep the dashboard metadata's processed_jobs count aligned with what
                # write_head_accumulators actually flushed, not ahead of it.
                if output_dir is not None and (
                    index % write_every == 0 or index == total
                ):
                    write_head_accumulators(
                        output_dir,
                        accumulators,
                        grid_metadata,
                        metadata,
                        num_heads=num_heads,
                        pixels_per_head=pixels_per_head,
                    )
                    if metadata_path is not None:
                        update_campaign_metadata(
                            metadata_path, metadata, index, complete=index == total
                        )
                if log_file is not None:
                    log_file.write(f"{index}/{total} completed: {job_label(job_key)}\n")
                    log_file.flush()
        finally:
            progress.close()
    finally:
        if log_file is not None:
            log_file.close()


def merge_cached_groups_cpu(
    cached_groups: Iterable[Path],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Merge cached GPU shards on CPU with a Polars coordinate group-by."""
    coordinate_columns = ["crystal", "pixel", "x", "y", "z"]
    frames_by_resolution: dict[str, list[pl.DataFrame]] = {}
    for cache_path in cached_groups:
        resolution = cache_path.stem.rsplit("_", 1)[-1]
        with np.load(cache_path, allow_pickle=False) as cached:
            coords = np.asarray(cached["coords"], dtype=np.int64)
            counts = np.asarray(cached["counts"], dtype=np.int64)
        if coords.size == 0:
            continue
        frames_by_resolution.setdefault(resolution, []).append(
            pl.DataFrame(
                {
                    column: coords[:, index]
                    for index, column in enumerate(coordinate_columns)
                }
                | {"counts": counts}
            )
        )

    merged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for resolution, frames in frames_by_resolution.items():
        combined = (
            pl.concat(frames, rechunk=False)
            .group_by(coordinate_columns)
            .agg(pl.col("counts").sum())
            .sort(coordinate_columns)
        )
        merged[resolution] = (
            combined.select(coordinate_columns).to_numpy().astype(np.int64, copy=False),
            combined.get_column("counts").to_numpy().astype(np.int64, copy=False),
        )
    return merged


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Efficiently combine partial SRM tarballs"
    )

    parser.add_argument(
        "-c",
        "--cluster_id",
        type=int,
        metavar="CLUSTER_ID",
        default=None,
        help="The ClusterId to process (required when --source ospool).",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        metavar="BATCH_NAME",
        required=True,
        help="Campaign batch name, e.g. batch_20260909_012610.",
    )
    parser.add_argument(
        "--source",
        choices=["ospool", "slurm"],
        default="ospool",
        help="Campaign origin: OSPool/HTCondor cardiac tarballs (default) or "
        "SDSC Expanse Slurm brain-SPECT task directories.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["batch", "interactive"],
        metavar="MODE",
        default="interactive",
        help="Mode of operation: batch or interactive.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for sparse merging (default: cuda; falls back to CPU).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="ospool: number of tarballs loaded per Torch merge group. "
        "slurm: number of tasks between writes of the running per-head totals. "
        f"(default: {GROUP_SIZE_DEFAULTS['ospool']} for --source ospool, "
        f"{GROUP_SIZE_DEFAULTS['slurm']} for --source slurm).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for coalesced group caches (default: /tmp/hybrid_combine_<batch>).",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Detector heads for the per-head SRM split "
        "(default: 80 for --source ospool, 73 for --source slurm).",
    )
    parser.add_argument(
        "--pixels-per-head",
        type=int,
        default=PIXELS_PER_HEAD,
        help=f"Detector pixels per head (default: {PIXELS_PER_HEAD}).",
    )
    args = parser.parse_args()
    if args.source == "ospool" and args.cluster_id is None:
        parser.error("--cluster_id is required when --source ospool")
    if args.group_size is None:
        args.group_size = GROUP_SIZE_DEFAULTS[args.source]
    return args


# def accumulate_stats(stats_list: list[dict[str, Any]]) -> dict[str, Any]:


def get_campaign_paths(name: str, source: str) -> tuple[Path, Path | None]:
    """Derive the local campaign data directory (and ospool db, if any)."""
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("name must be a batch directory name, not a path")

    data_dir = LOCAL_DATA_ROOTS[source] / name
    if source == "ospool":
        return data_dir, data_dir / "ospool_condor_jobs.db"
    return data_dir, None


def get_unmerged_job_ids(conn: sqlite3.Connection, cluster_id: int) -> set[JobKey]:
    """Return unconsumed (completed and pulled but not processed) jobs for a given cluster."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT ClusterId, ProcId
        FROM condor_jobs
        WHERE JobStatus = 4 AND is_pulled = 1 AND is_merged = 0 AND ClusterId = ?
    """,
        (cluster_id,),
    )
    unmerged_job_ids = set()

    # Iterating directly over the cursor avoids loading a large intermediate list into memory
    for cluster, proc in cursor:
        job_key = (cluster, proc)
        unmerged_job_ids.add(job_key)

    return unmerged_job_ids


def get_unmerged_task_dirs(campaign_dir: Path) -> list[str]:
    """Return names of Slurm task_N dirs that finished (have TASK_COMPLETE.json).

    Unlike the ospool db, there's no is_merged flag to consult, so every task
    with the completion marker is reprocessed each run.
    """
    return sorted(
        entry.name
        for entry in campaign_dir.glob("task_*")
        if entry.is_dir() and (entry / "TASK_COMPLETE.json").is_file()
    )


def job_label(job_key: JobKey) -> str:
    """Human-readable source filename/dirname for one job, for progress logs."""
    if isinstance(job_key, tuple):
        cluster_id, proc_id = job_key
        return f"srm_c_{cluster_id}_p_{proc_id}.tar.gz"
    return job_key


def job_sort_key(job_key: JobKey) -> tuple:
    if isinstance(job_key, tuple):
        return (0, job_key)
    return (1, int(job_key.rsplit("_", 1)[-1]))


def read_and_merge(job_key: JobKey, partial_data_dir: Path, source: str):
    if source == "ospool":
        assert isinstance(job_key, tuple)
        cluster_id, proc_id = job_key
        partial_data_fname = f"srm_c_{cluster_id}_p_{proc_id}.tar.gz"
        return get_tar_contents(partial_data_dir / partial_data_fname)
    assert isinstance(job_key, str)
    return get_task_dir_contents(partial_data_dir / job_key)


def process_jobs(
    job_ids: Iterable[JobKey],
    partial_data_dir: Path,
    *,
    source: str = "ospool",
    progress_log: Path | None = None,
    metadata_path: Path | None = None,
    output_dir: Path | None = None,
    device: str = "cuda",
    group_size: int = DEFAULT_GROUP_SIZE,
    cache_dir: Path | None = None,
    num_heads: int | None = None,
    pixels_per_head: int = PIXELS_PER_HEAD,
) -> None:
    """Process jobs and write durable newline-delimited progress when requested."""
    ordered_jobs = sorted(job_ids, key=job_sort_key)
    total = len(ordered_jobs)
    if group_size < 1:
        raise ValueError("group_size must be greater than zero")
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    metadata: dict[str, dict[str, Any]] = {}
    if cache_dir is None:
        cache_dir = partial_data_dir.parent / "torch_merge_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old_cache in cache_dir.glob("group_*.npz"):
        old_cache.unlink()
    cached_groups: list[Path] = []
    log_file = progress_log.open("w", encoding="utf-8") if progress_log else None
    try:
        read_progress = tqdm(
            total=total,
            desc="Reading partial SRMs",
            unit="job",
            disable=log_file is not None,
        )
        merge_progress = tqdm(
            total=(total + group_size - 1) // group_size,
            desc="Merging groups",
            unit="group",
            disable=log_file is not None,
        )
        try:
            for group_index, offset in enumerate(range(0, total, group_size)):
                group_srms = []
                for job_key in ordered_jobs[offset : offset + group_size]:
                    srms, stats = read_and_merge(job_key, partial_data_dir, source)
                    group_srms.append(srms)
                    accumulate_metadata(metadata, srms, stats)
                    read_progress.update(1)

                group_accumulators = _merge_srm_job_tensors(
                    group_srms, requested_device
                )
                for resolution, current in group_accumulators.items():
                    cache_path = cache_dir / f"group_{group_index:06d}_{resolution}.npz"
                    np.savez_compressed(
                        cache_path,
                        coords=current.indices().T.cpu().numpy(),
                        counts=current.values().cpu().numpy(),
                        grid_size=np.asarray([current.shape[-1]], dtype=np.int32),
                    )
                    cached_groups.append(cache_path)
                del group_accumulators, group_srms
                if requested_device.type == "cuda":
                    torch.cuda.empty_cache()

                completed = offset + len(ordered_jobs[offset : offset + group_size])
                # Re-fold every cached group so far and re-split per head; cheap relative
                # to a GPU pass, and it means results are usable after every group instead
                # of only once the whole campaign has been read.
                if output_dir is not None and num_heads is not None:
                    merged = merge_cached_groups_cpu(cached_groups)
                    write_split_outputs(
                        output_dir,
                        merged,
                        metadata,
                        num_heads=num_heads,
                        pixels_per_head=pixels_per_head,
                    )
                    del merged
                merge_progress.update(1)
                if metadata_path is not None:
                    update_campaign_metadata(
                        metadata_path, metadata, completed, complete=completed == total
                    )
                if log_file is not None:
                    for job_number, job_key in enumerate(
                        ordered_jobs[offset:completed], start=offset + 1
                    ):
                        log_file.write(
                            f"{job_number}/{total} completed: {job_label(job_key)}\n"
                        )
                    log_file.flush()
        finally:
            read_progress.close()
            merge_progress.close()
    finally:
        if log_file is not None:
            log_file.close()


def main():
    args = get_args()
    data_dir, db_path = get_campaign_paths(args.name, args.source)
    cache_dir = args.cache_dir or Path("/tmp") / f"hybrid_combine_{args.name}"

    progress_log = Path("progress.log") if args.mode != "interactive" else None
    num_heads = args.num_heads or NUM_HEADS[args.source]

    if args.source == "ospool":
        assert db_path is not None
        partial_data_dir = data_dir / "partials" / "targz"
        with sqlite3.connect(db_path) as conn:
            unmerged_job_ids: list[JobKey] = list(
                get_unmerged_job_ids(conn, cluster_id=args.cluster_id)
            )
        unmerged_job_ids = unmerged_job_ids[:500]
        process_jobs(
            unmerged_job_ids,
            partial_data_dir,
            source=args.source,
            progress_log=progress_log,
            metadata_path=data_dir / "combined_srm_metadata.json",
            output_dir=data_dir,
            device=args.device,
            group_size=args.group_size,
            cache_dir=cache_dir,
            num_heads=num_heads,
            pixels_per_head=args.pixels_per_head,
        )
    else:
        partial_data_dir = data_dir
        unmerged_task_ids = get_unmerged_task_dirs(data_dir)
        process_brain_jobs_per_head(
            unmerged_task_ids,
            partial_data_dir,
            progress_log=progress_log,
            metadata_path=data_dir / "combined_srm_metadata.json",
            output_dir=data_dir,
            device=args.device,
            write_every=args.group_size,
            num_heads=num_heads,
            pixels_per_head=args.pixels_per_head,
        )


if __name__ == "__main__":
    main()
