import json
import os
import re

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, save_npz


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


def create_per_resolution_srm_pl_df(
    task_ids: list[int],
    resolution_id: int,
    data_dir: str,
    *,
    n_pixels: int,
):
    from tqdm import tqdm

    resolution_name_stems = ["1mm", "1p5mm", "2mm"]
    srm_fname = f"final_srm_{resolution_name_stems[resolution_id]}.npz"
    global_df = None
    chunk_size = 20  # Process 20 tasks at a time

    progress = tqdm(
        total=len(task_ids),
        desc=f"{resolution_name_stems[resolution_id]} tasks",
        unit="task",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=True,
    )
    # Split task_ids into chunks of 20 while keeping one stable screen bar.
    for i in range(0, len(task_ids), chunk_size):
        batch_task_ids = task_ids[i : i + chunk_size]
        batch_df_list = []

        for task_id in batch_task_ids:
            # 1. Load and process exactly as before
            file_path = os.path.join(data_dir, f"task_{task_id}", srm_fname)
            with np.load(file_path) as data:
                coords, values, grid_size = (
                    data["coords"],
                    data["counts"],
                    int(data["grid_size"][0]),
                )

            p_id = coords[:, 0] * n_pixels + coords[:, 1]
            v_id = (
                coords[:, 2] * (grid_size**2) + coords[:, 3] * grid_size + coords[:, 4]
            )

            # Local reduction
            df_chunk = pl.DataFrame({"p_id": p_id, "v_id": v_id, "value": values})
            batch_df_list.append(
                df_chunk.group_by(["p_id", "v_id"]).agg(pl.col("value").sum())
            )
            progress.update(1)

        # 2. Batch Reduction
        batch_df = (
            pl.concat(batch_df_list)
            .group_by(["p_id", "v_id"])
            .agg(pl.col("value").sum())
        )

        # 3. Merge into the Global DataFrame
        if global_df is None:
            global_df = batch_df
        else:
            global_df = (
                pl.concat([global_df, batch_df])
                .group_by(["p_id", "v_id"])
                .agg(pl.col("value").sum())
            )

    progress.close()

    # Derive h_id at the very end
    if global_df is None:
        return pl.DataFrame(
            {
                "p_id": pl.Series([], dtype=pl.Int64),
                "v_id": pl.Series([], dtype=pl.Int64),
                "value": pl.Series([], dtype=pl.Int64),
                "h_id": pl.Series([], dtype=pl.Int64),
            }
        )
    return global_df.with_columns((pl.col("p_id") // n_pixels).alias("h_id"))


def write_per_head_srms(
    global_df: pl.DataFrame,
    output_dir: str,
    resolution_stem: str,
    *,
    n_heads: int,
    n_pixels: int,
    grid_size: int,
) -> list[dict]:
    """Write one int64 CSR SRM per head for one resolution."""
    if n_heads <= 0 or n_pixels <= 0 or grid_size <= 0:
        raise ValueError("n_heads, n_pixels, and grid_size must be positive")

    output_path = os.path.abspath(output_dir)
    os.makedirs(output_path, exist_ok=True)
    num_voxels = grid_size**3
    head_metadata = []
    for head_id in range(n_heads):
        head_rows = global_df.filter(pl.col("h_id") == head_id)
        matrix = coo_matrix(
            (
                head_rows["value"].to_numpy(),
                (
                    (head_rows["p_id"] - head_id * n_pixels).to_numpy(),
                    head_rows["v_id"].to_numpy(),
                ),
            ),
            shape=(n_pixels, num_voxels),
            dtype=np.int64,
        ).tocsr()
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
        "simulated_primaries": summed("simulated_primaries"),
        "source_files": source_files,
    }


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

    resolution_name_stems = ["1mm", "1p5mm", "2mm"]
    grid_sizes = [210, 140, 105]
    metadata = {"layout": "per_head_csr", "resolutions": {}}
    for resolution_id, (resolution_stem, grid_size) in enumerate(
        zip(resolution_name_stems, grid_sizes)
    ):
        global_df = create_per_resolution_srm_pl_df(
            valid_task_ids, resolution_id, directory, n_pixels=args.pixels
        )
        head_metadata = write_per_head_srms(
            global_df,
            output_dir,
            resolution_stem,
            n_heads=args.heads,
            n_pixels=args.pixels,
            grid_size=grid_size,
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
        print(f"Wrote {args.heads} {resolution_stem} per-head SRMs to {output_dir}")

    with open(os.path.join(output_dir, "combined_srm_metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
