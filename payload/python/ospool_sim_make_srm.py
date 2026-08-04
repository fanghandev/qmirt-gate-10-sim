import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import uproot

file_list_txt_file = Path(
    "/home/fanghan/Work/MGB-HMS/RPIL/QMIRT/ospool_sim_data/files_list.txt"
)
primary_energy = 0.14
energy_tolerance = 0.001
energy_min = primary_energy * (1 - energy_tolerance)
energy_max = primary_energy * (1 + energy_tolerance)
hist_min = -105.0
hist_max = 105.0
hist_bins = 140
bin_width = (hist_max - hist_min) / hist_bins

total_events = 0
total_sim_time = 0.0
sparse_histograms = defaultdict(int)
with open(file_list_txt_file, "r") as f:
    file_list = f.read().splitlines()
    for file in file_list:
        with (
            open(file, "rb") as file_obj,
            tarfile.open(fileobj=file_obj, mode="r:gz") as tar,
        ):
            for member_name in tar.getnames():
                # Check if the file is a .txt file
                if member_name.endswith(".txt"):
                    # Extract the file as a file-like object
                    extracted_stats_file = tar.extractfile(member_name)

                    if extracted_stats_file is not None:
                        # Read the binary data and decode it to a standard string
                        text_content = extracted_stats_file.read().decode("utf-8")
                        stats_dict = json.loads(text_content)
                        total_events += float(stats_dict["events"]["value"])
                        total_sim_time += (
                            stats_dict["sim_stop_time"]["value"]
                            - stats_dict["sim_start_time"]["value"]
                        )
                elif member_name.endswith(".root"):
                    print(f"Processing ROOT file: {member_name}")
                    extracted_root_file = tar.extractfile(member_name)
                    if extracted_root_file is not None:
                        with uproot.open(extracted_root_file) as root_file:
                            tree_names = [
                                name
                                for name, class_name in root_file.classnames(
                                    cycle=False
                                ).items()
                                if class_name == "TTree"
                            ]

                            branches = [
                                "TotalEnergyDeposit",
                                "EventPosition_X",
                                "EventPosition_Y",
                                "EventPosition_Z",
                                "PreStepUniqueVolumeID",
                            ]

                            for tree_name in tree_names:
                                tree = root_file[tree_name]
                                if tree.num_entries == 0:
                                    continue

                                data = tree.arrays(branches, library="np")
                                df = pl.DataFrame(data).with_columns(
                                    pl.col("PreStepUniqueVolumeID")
                                    .str.split("_")
                                    .list.get(1)
                                    .cast(pl.Int32)
                                    .alias("CrystalID"),
                                    pl.col("PreStepUniqueVolumeID")
                                    .str.split("_")
                                    .list.get(-1)
                                    .cast(pl.Int32)
                                    .alias("PixelID"),
                                )

                                df = df.filter(
                                    (pl.col("TotalEnergyDeposit") >= energy_min)
                                    & (pl.col("TotalEnergyDeposit") <= energy_max)
                                )
                                if df.is_empty():
                                    continue

                                crystal_ids = df["CrystalID"].to_numpy()
                                pixel_ids = df["PixelID"].to_numpy()
                                event_x = df["EventPosition_X"].to_numpy()
                                event_y = df["EventPosition_Y"].to_numpy()
                                event_z = df["EventPosition_Z"].to_numpy()

                                valid_mask = (
                                    (event_x >= hist_min)
                                    & (event_x < hist_max)
                                    & (event_y >= hist_min)
                                    & (event_y < hist_max)
                                    & (event_z >= hist_min)
                                    & (event_z < hist_max)
                                )
                                if not np.any(valid_mask):
                                    continue

                                crystal_ids = crystal_ids[valid_mask].astype(np.int32)
                                pixel_ids = pixel_ids[valid_mask].astype(np.int32)
                                event_x = event_x[valid_mask]
                                event_y = event_y[valid_mask]
                                event_z = event_z[valid_mask]

                                x_bin = np.floor(
                                    (event_x - hist_min) / bin_width
                                ).astype(np.int32)
                                y_bin = np.floor(
                                    (event_y - hist_min) / bin_width
                                ).astype(np.int32)
                                z_bin = np.floor(
                                    (event_z - hist_min) / bin_width
                                ).astype(np.int32)

                                coords = np.column_stack(
                                    (crystal_ids, pixel_ids, x_bin, y_bin, z_bin)
                                )
                                unique_coords, counts = np.unique(
                                    coords, axis=0, return_counts=True
                                )
                                for coord, count in zip(unique_coords, counts):
                                    sparse_histograms[
                                        tuple(int(value) for value in coord)
                                    ] += int(count)
if sparse_histograms:
    sparse_histogram_path = Path("sparse_3d_histograms.npz")
    coords = np.array(list(sparse_histograms.keys()), dtype=np.int32)
    counts = np.array(list(sparse_histograms.values()), dtype=np.int64)
    np.savez_compressed(
        sparse_histogram_path,
        coords=coords,
        counts=counts,
        hist_bins=np.array([hist_bins], dtype=np.int32),
        hist_range=np.array([hist_min, hist_max], dtype=np.float32),
        energy_min=np.array([energy_min], dtype=np.float32),
        energy_max=np.array([energy_max], dtype=np.float32),
    )

print(f"Sparse histogram bins: {len(sparse_histograms):,d}")
print(f"Total simulation time: {total_sim_time:.2f} seconds")
