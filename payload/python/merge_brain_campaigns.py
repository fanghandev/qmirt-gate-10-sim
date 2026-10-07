import json
import os
import re
import sqlite3

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz


def scan_brain_campaigns(directory):
    valid_per_task_files = [
        "combined_srm_metadata.json",
        "final_srm_1mm.npz",
        "final_srm_1p5mm.npz",
        "final_srm_2mm.npz",
        "TASK_COMPLETE.json",
    ]
    pulled_task_ids = []
    valid_task_ids = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            match = re.match(r"task_(\d+)", entry.name)
            if match:
                task_id = int(match.groups()[0])
                pulled_task_ids.append(task_id)
                valid_file_counter = 0
                for sub_entry in os.scandir(entry.path):
                    if sub_entry.name in valid_per_task_files:
                        valid_file_counter += 1
                if valid_file_counter == len(valid_per_task_files):
                    valid_task_ids.append(task_id)

    return pulled_task_ids, valid_task_ids


def database_task_ids(db_path: str, campaign: str) -> set[int] | None:
    """Tasks the local job database records as pulled and complete for *campaign*.

    None when the database has no rows for the campaign (fall back to the file scan).
    """
    if not os.path.isfile(db_path):
        return None
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT array_task_id, state, is_pulled FROM slurm_jobs WHERE campaign_name = ?",
            (campaign,),
        ).fetchall()
    if not rows:
        return None
    return {int(task) for task, state, pulled in rows if pulled and state == "COMPLETED"}


def mark_tasks_merged(db_path: str, campaign: str, task_ids: list[int]) -> None:
    """Record which tasks the per-head SRMs now contain (and that the others are not)."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE slurm_jobs SET is_merged = 0 WHERE campaign_name = ?", (campaign,))
        conn.executemany(
            "UPDATE slurm_jobs SET is_merged = 1 WHERE campaign_name = ? AND array_task_id = ?",
            [(campaign, task_id) for task_id in task_ids],
        )


def accumulate_per_head(
    task_ids: list[int],
    resolution_stem: str,
    data_dir: str,
    *,
    n_heads: int,
    n_pixels: int,
    grid_size: int,
    tasks_per_batch: int = 4,
) -> list[csr_matrix]:
    """Sum the tasks' sparse SRMs into one int64 CSR matrix per head.

    Tasks are read a few at a time and their entries added head by head, so memory
    follows the merged matrices (unique pixel-voxel pairs), not the sum of every
    task's entries: a 288 mm part is 64 tasks x ~54M entries at 1 mm, which the
    earlier all-tasks-in-one-table merge could not hold.
    """
    from tqdm import tqdm

    num_voxels = grid_size**3
    shape = (n_pixels, num_voxels)
    heads: list[csr_matrix] = [csr_matrix(shape, dtype=np.int64) for _ in range(n_heads)]
    srm_fname = f"final_srm_{resolution_stem}.npz"
    progress = tqdm(total=len(task_ids), desc=f"{resolution_stem} tasks", unit="task",
                    dynamic_ncols=True, mininterval=1.0)
    for start in range(0, len(task_ids), tasks_per_batch):
        pieces: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [[] for _ in range(n_heads)]
        for task_id in task_ids[start : start + tasks_per_batch]:
            with np.load(os.path.join(data_dir, f"task_{task_id}", srm_fname)) as data:
                coords = data["coords"]
                counts = data["counts"].astype(np.int64, copy=False)
                if int(data["grid_size"][0]) != grid_size:
                    raise ValueError(f"task_{task_id}: grid size differs from {grid_size}")
            head = coords[:, 0].astype(np.int64)
            pixel = coords[:, 1].astype(np.int64)
            voxel = (
                coords[:, 2].astype(np.int64) * grid_size**2
                + coords[:, 3].astype(np.int64) * grid_size
                + coords[:, 4].astype(np.int64)
            )
            del coords
            order = np.argsort(head, kind="stable")
            bounds = np.searchsorted(head[order], np.arange(n_heads + 1))
            for h in range(n_heads):
                idx = order[bounds[h] : bounds[h + 1]]
                if idx.size:
                    pieces[h].append((pixel[idx], voxel[idx], counts[idx]))
            del head, pixel, voxel, counts, order
            progress.update(1)
        for h in range(n_heads):
            if not pieces[h]:
                continue
            rows = np.concatenate([piece[0] for piece in pieces[h]])
            cols = np.concatenate([piece[1] for piece in pieces[h]])
            vals = np.concatenate([piece[2] for piece in pieces[h]])
            pieces[h] = []
            batch = coo_matrix((vals, (rows, cols)), shape=shape, dtype=np.int64).tocsr()
            batch.sum_duplicates()
            heads[h] = heads[h] + batch
    progress.close()
    return heads


def write_per_head_srms(
    heads: list[csr_matrix],
    output_dir: str,
    resolution_stem: str,
) -> list[dict]:
    """Write one int64 CSR SRM per head for one resolution."""
    output_path = os.path.abspath(output_dir)
    os.makedirs(output_path, exist_ok=True)
    head_metadata = []
    for head_id, matrix in enumerate(heads):
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        filename = f"final_srm_{resolution_stem}_head_{head_id + 1:02d}.npz"
        save_npz(os.path.join(output_path, filename), matrix)
        head_metadata.append(
            {
                "head": head_id + 1,
                "output": filename,
                "nonzero_entries": int(matrix.nnz),
                "accumulated_counts": int(matrix.sum()),
            }
        )
    return head_metadata


def task_counts_total(task_ids: list[int], data_dir: str, resolution_stem: str) -> int:
    """Sum of the tasks' counts, to check the merge lost nothing."""
    total = 0
    for task_id in task_ids:
        with np.load(os.path.join(data_dir, f"task_{task_id}", f"final_srm_{resolution_stem}.npz")) as data:
            total += int(data["counts"].sum(dtype=np.int64))
    return total


def merge_resolution_metadata(
    task_ids: list[int],
    data_dir: str,
    resolution_stem: str,
    head_metadata: list[dict],
    *,
    n_heads: int,
    n_pixels: int,
    grid_size: int,
) -> dict:
    """Aggregate task metadata into the campaign's per-head layout."""
    task_entries = []
    for task_id in task_ids:
        metadata_path = os.path.join(
            data_dir, f"task_{task_id}", "combined_srm_metadata.json"
        )
        with open(metadata_path) as handle:
            metadata = json.load(handle)
        entry = metadata.get("resolutions", {}).get(resolution_stem)
        if entry is None:
            raise KeyError(f"{metadata_path} has no {resolution_stem} resolution")
        task_entries.append(entry)

    reference = task_entries[0] if task_entries else {}
    for entry in task_entries[1:]:
        for key in (
            "voxel_size_mm",
            "grid_size",
            "hist_range",
            "energy_min_kev",
            "energy_max_kev",
        ):
            if entry.get(key) != reference.get(key):
                raise ValueError(f"Metadata mismatch for {resolution_stem}: {key}")

    def summed(key: str) -> int:
        return sum(int(entry.get(key, 0) or 0) for entry in task_entries)

    source_files = []
    for task_id, entry in zip(task_ids, task_entries):
        source_files.extend(
            f"task_{task_id}/{source}" for source in entry.get("source_files", [])
        )

    return {
        "layout": "per_head_csr",
        "outputs": [entry["output"] for entry in head_metadata],
        "heads": head_metadata,
        "num_heads": n_heads,
        "pixels_per_head": n_pixels,
        "voxel_size_mm": reference.get("voxel_size_mm", 1.0),
        "grid_size": reference.get("grid_size", grid_size),
        "hist_range": reference.get("hist_range", [-105.0, 105.0]),
        "energy_min_kev": reference.get("energy_min_kev", 126.0),
        "energy_max_kev": reference.get("energy_max_kev", 154.0),
        "chunk_count": summed("chunk_count"),
        "input_count": summed("input_count"),
        "expected_input_count": summed("expected_input_count"),
        "complete": all(entry.get("complete", True) for entry in task_entries),
        "nonzero_entries": sum(entry["nonzero_entries"] for entry in head_metadata),
        "accumulated_counts": sum(
            entry["accumulated_counts"] for entry in head_metadata
        ),
        "accepted_events": sum(entry["accumulated_counts"] for entry in head_metadata),
        "simulated_primaries": summed("simulated_primaries"),
        "source_files": source_files,
    }


def task_grid_size(task_ids: list[int], data_dir: str, resolution_stem: str) -> int:
    """Grid size shared by every task's SRM; refuse to mix FOVs or voxel sizes."""
    grid_sizes = set()
    for task_id in task_ids:
        srm_path = os.path.join(
            data_dir, f"task_{task_id}", f"final_srm_{resolution_stem}.npz"
        )
        with np.load(srm_path) as data:
            grid_sizes.add(int(data["grid_size"][0]))
    if len(grid_sizes) != 1:
        raise ValueError(
            f"Tasks disagree on the {resolution_stem} grid size: {sorted(grid_sizes)}"
        )
    return grid_sizes.pop()


def get_parser_args():
    import argparse

    parser = argparse.ArgumentParser(description="Scan brain campaign directories")
    parser.add_argument("-d", "--directory", help="Directory to scan")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory for per-resolution, per-head SRMs (defaults to --directory)",
    )
    parser.add_argument("--heads", type=int, default=73, help="Number of heads")
    parser.add_argument(
        "--pixels", type=int, default=625, help="Number of pixels per head"
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Local job database (update_local_slurm_campaign_database.py). When it has "
        "rows for this campaign, only tasks it records as pulled and COMPLETED are "
        "merged, and they are marked is_merged=1 afterwards.",
    )
    return parser.parse_args()


def main():

    args = get_parser_args()
    directory = args.directory
    output_dir = args.output_dir or directory
    pulled_task_ids, valid_task_ids = scan_brain_campaigns(directory)
    # sort the task IDs
    valid_task_ids.sort()
    print(f"Find {len(pulled_task_ids)} pulled task IDs")
    print(f"Find {len(valid_task_ids)} valid task IDs")
    campaign = os.path.basename(os.path.normpath(directory))
    db_ids = database_task_ids(args.db, campaign) if args.db else None
    if db_ids is not None:
        skipped = sorted(set(valid_task_ids) - db_ids)
        missing = sorted(db_ids - set(valid_task_ids))
        valid_task_ids = [task_id for task_id in valid_task_ids if task_id in db_ids]
        print(f"Database {args.db}: {len(db_ids)} pulled COMPLETED tasks; merging {len(valid_task_ids)}")
        if skipped:
            print(f"  not COMPLETED/pulled in the database, skipped: {skipped[:20]}")
        if missing:
            print(f"  COMPLETED in the database but files incomplete, skipped: {missing[:20]}")
    elif args.db:
        print(f"Database {args.db} has no rows for {campaign}; using the file scan")
    if not valid_task_ids:
        raise SystemExit(f"No mergeable tasks in {directory}")

    resolution_name_stems = ["1mm", "1p5mm", "2mm"]
    grid_sizes = [
        task_grid_size(valid_task_ids, directory, stem) for stem in resolution_name_stems
    ]
    print(f"Grid sizes: {dict(zip(resolution_name_stems, grid_sizes))}")
    metadata = {
        "layout": "per_head_csr",
        "input_count": 0,
        "simulated_primaries": 0,
        "raw_events": 0,
        "accepted_events": 0,
        "partial_counts": {},
        "resolutions": {},
    }
    for resolution_id, (resolution_stem, grid_size) in enumerate(
        zip(resolution_name_stems, grid_sizes)
    ):
        heads = accumulate_per_head(
            valid_task_ids,
            resolution_stem,
            directory,
            n_heads=args.heads,
            n_pixels=args.pixels,
            grid_size=grid_size,
        )
        head_metadata = write_per_head_srms(heads, output_dir, resolution_stem)
        del heads
        merged_counts = sum(entry["accumulated_counts"] for entry in head_metadata)
        expected_counts = task_counts_total(valid_task_ids, directory, resolution_stem)
        if merged_counts != expected_counts:
            raise ValueError(
                f"{resolution_stem}: merged counts {merged_counts} != task total {expected_counts}"
            )
        metadata["resolutions"][resolution_stem] = merge_resolution_metadata(
            valid_task_ids,
            directory,
            resolution_stem,
            head_metadata,
            n_heads=args.heads,
            n_pixels=args.pixels,
            grid_size=grid_size,
        )
        resolution_metadata = metadata["resolutions"][resolution_stem]
        metadata["partial_counts"][resolution_stem] = int(
            resolution_metadata["accumulated_counts"]
        )
        metadata["accepted_events"] = int(resolution_metadata["accepted_events"])
        metadata["simulated_primaries"] = int(
            resolution_metadata["simulated_primaries"]
        )
        metadata["input_count"] = len(valid_task_ids)
        resolution_metadata["input_count"] = len(valid_task_ids)
        print(f"Wrote {args.heads} {resolution_stem} per-head SRMs to {output_dir}")

    metadata["merged_task_ids"] = valid_task_ids
    with open(os.path.join(output_dir, "combined_srm_metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    if db_ids is not None:
        mark_tasks_merged(args.db, campaign, valid_task_ids)
        print(f"Marked {len(valid_task_ids)} tasks is_merged=1 in {args.db}")


if __name__ == "__main__":
    main()
