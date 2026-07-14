#!/usr/bin/env python3
"""
Generate System Response Matrix (SRM) for a SINGLE head in sparse CSR format.
Designed to be run as an independent task within a SLURM Job Array.
Features a '--max-files' constraint that limits file processing while 
re-calculating accurate subset metadata for emitted primary gammas.
"""
import argparse
import sys
import time
import csv
from pathlib import Path

import numpy as np
import uproot
from scipy.sparse import coo_matrix, csr_matrix, save_npz

NUM_PIXELS_PER_HEAD = 625   # 25×25 grid

def process_single_head(head_idx, files, total_emitted_events, energy_min, energy_max, grid_size, voxel_size, output_dir):
    num_voxels = grid_size ** 3
    half_size  = grid_size * voxel_size * 0.5
    origin_mm  = np.array([-half_size, -half_size, -half_size], dtype=np.float32)

    local_srm = csr_matrix((NUM_PIXELS_PER_HEAD, num_voxels), dtype=np.float32)
    total_raw = 0
    total_valid = 0
    tree_name = f"Pixel_{head_idx + 1}_Singles"
    
    print(f"Starting processing for Head {head_idx} across {len(files)} subset files...")
    start_time = time.time()

    for fpath in files:
        if not fpath.exists():
            print(f"Warning: Validated file missing at runtime, skipping -> {fpath}", file=sys.stderr)
            continue
            
        try:
            with uproot.open(fpath) as f:
                if tree_name not in f:
                    continue
                tree = f[tree_name]
                if tree.num_entries == 0:
                    continue

                total_raw += tree.num_entries
                data = tree.arrays([
                    "EventPosition_X", "EventPosition_Y", "EventPosition_Z",
                    "PreStepUniqueVolumeID", "TotalEnergyDeposit",
                ], library="np")

                edep = data["TotalEnergyDeposit"]
                energy_mask = (edep >= energy_min) & (edep <= energy_max)
                if not np.any(energy_mask):
                    continue

                src_x = data["EventPosition_X"][energy_mask]
                src_y = data["EventPosition_Y"][energy_mask]
                src_z = data["EventPosition_Z"][energy_mask]
                vol_ids = data["PreStepUniqueVolumeID"][energy_mask]

                pixel_ids = np.array([int(s.rsplit("_", 1)[-1]) for s in vol_ids], dtype=np.int32)

                ix = np.floor((src_x - origin_mm[0]) / voxel_size).astype(np.int32)
                iy = np.floor((src_y - origin_mm[1]) / voxel_size).astype(np.int32)
                iz = np.floor((src_z - origin_mm[2]) / voxel_size).astype(np.int32)

                valid_vox = ((ix >= 0) & (ix < grid_size) &
                             (iy >= 0) & (iy < grid_size) &
                             (iz >= 0) & (iz < grid_size))
                valid_det = (pixel_ids >= 0) & (pixel_ids < NUM_PIXELS_PER_HEAD)
                valid_mask = valid_vox & valid_det

                n_valid = int(np.sum(valid_mask))
                if n_valid > 0:
                    total_valid += n_valid
                    final_voxels = (ix[valid_mask] * grid_size * grid_size
                                    + iy[valid_mask] * grid_size
                                    + iz[valid_mask])
                    final_pixels = pixel_ids[valid_mask]
                    ones = np.ones(n_valid, dtype=np.float32)
                    
                    coo = coo_matrix(
                        (ones, (final_pixels, final_voxels)),
                        shape=(NUM_PIXELS_PER_HEAD, num_voxels),
                    )
                    local_srm += coo.tocsr()

        except Exception as e:
            print(f"Error processing Head {head_idx} in file {fpath.name}: {e}", file=sys.stderr)

    local_srm.eliminate_zeros()
    output_path = Path(output_dir) / f"srm_head_{head_idx}.npz"
    
    # Save sparse arrays alongside our customized subset tracking value
    np.savez_compressed(
        output_path,
        data=local_srm.data,
        indices=local_srm.indices,
        indptr=local_srm.indptr,
        shape=local_srm.shape,
        total_emitted_primaries=np.array(total_emitted_events, dtype=np.int64)
    )
    
    elapsed = time.time() - start_time
    print(f"Head {head_idx} finished. Saved matrix + Prior subset info ({total_emitted_events} emitted gammas) to {output_path} ({elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Cluster-optimized single-head SRM generator with file subset scaling options.")
    parser.add_argument("--head-idx",    type=int, required=True, help="0-indexed head ID to process (0 to 79)")
    parser.add_argument("--summary-csv", type=str, required=True, help="Path to 'root_files_validation_summary_*.csv'")
    parser.add_argument("--output-dir",  type=str, default="srm_sparse_output", help="Output directory.")
    parser.add_argument("--max-files",   type=int, default=0, help="Maximum number of intact files to read. 0 reads all.")
    parser.add_argument("--energy-min",  type=float, default=0.126)
    parser.add_argument("--energy-max",  type=float, default=0.154)
    parser.add_argument("--grid-size",   type=int, default=90)
    parser.add_argument("--voxel-size",  type=float, default=2.0)
    args = parser.parse_args()

    summary_path = Path(args.summary_csv).resolve()
    if not summary_path.exists():
        print(f"Error: Validation summary file '{summary_path}' not found.", file=sys.stderr)
        sys.exit(1)

    all_valid_entries = []

    # Read all clean rows into memory using DictReader
    with open(summary_path, "r", encoding="utf-8") as f:
        # Ignore comment indicators
        filtered_rows = (row for row in f if row.strip() and not row.startswith("#"))
        reader = csv.DictReader(filtered_rows)
        
        for row in reader:
            fpath_str = row.get("file_path", "").strip()
            is_intact = row.get("all_trees_intact", "").strip().upper() == "TRUE"
            
            if fpath_str and is_intact:
                # Safely pull per-file emission numbers directly from the specific row
                try:
                    emitted_events = int(row.get("emitted_primary_events", 0))
                except ValueError:
                    emitted_events = 0
                
                all_valid_entries.append((Path(fpath_str), emitted_events))

    # Apply the absolute subset limitation slice if requested
    if args.max_files > 0 and args.max_files < len(all_valid_entries):
        selected_entries = all_valid_entries[:args.max_files]
        is_subset = True
    else:
        selected_entries = all_valid_entries
        is_subset = False

    # Extract target paths and calculate the tailored metadata sum
    files = [entry[0] for entry in selected_entries]
    total_emitted_primaries = sum(entry[1] for entry in selected_entries)

    if args.head_idx == 0:
        print(f"[Summary Diagnostics] Total Valid Files Found: {len(all_valid_entries)}")
        if is_subset:
            print(f"[Summary Diagnostics] Subset Constraint Triggered! Limited to the first {args.max_files} files.")
        print(f"[Summary Diagnostics] Calculated Emitted Primaries for this subset: {total_emitted_primaries}")

    if not files:
        print(f"Error: No valid data files selected.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    process_single_head(
        head_idx=args.head_idx,
        files=files,
        total_emitted_events=total_emitted_primaries,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
        grid_size=args.grid_size,
        voxel_size=args.voxel_size,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
