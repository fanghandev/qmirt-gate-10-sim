#!/usr/bin/env python3
"""
1. Merge parallel SLURM CSV check pieces into a single consolidated CSV file.
2. Parse the combined records to evaluate the complete structural health of each ROOT file.
3. Locate companion simulation performance text files from an explicitly provided folder.
4. Uses strict pattern matching to target ONLY chunk files like '..._<job_id>_<chunk_num>.csv'
"""
import argparse
import glob
import re
import csv
from pathlib import Path

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def extract_job_id(dir_path):
    match = re.search(r'(?<!\d)\d{7}(?!\d)', str(dir_path))
    if match:
        return match.group(0)
    return Path(dir_path).name if Path(dir_path).name else "cluster"

def find_stats_file(root_fpath_str, stats_dir_path):
    filename = Path(root_fpath_str).name
    match = re.search(r'_a_(\d+)_j_(\d+)\.root$', filename)
    if not match:
        return None
        
    batch_id = match.group(1)
    task_id = match.group(2)
    
    stats_filename = f"a_{batch_id}_j_{task_id}_sim_stats.txt"
    return Path(stats_dir_path) / stats_filename

def parse_emitted_events(stats_fpath):
    if not stats_fpath or not stats_fpath.exists():
        return 0
    try:
        with open(stats_fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("events:"):
                    return int(line.split(":")[1].strip())
    except Exception:
        pass
    return 0

def merge_csv_files(input_dir_path, job_id, output_file):
    """
    Builds a strict glob pattern targeting chunk variations such as
    check_sim_root_files_results_2363241_0.csv up to _9.csv (or higher).
    """
    # Pattern looks for any file ending in _<job_id>_<digits>.csv
    # This prevents picking up merged output files which don't have a final trailing chunk number
    strict_pattern = str(Path(input_dir_path) / f"*_{job_id}_[0-9]*.csv")
    
    csv_files = sorted(glob.glob(strict_pattern), key=natural_sort_key)
    if not csv_files:
        print(f"Error: No matching chunk files found with pattern: {strict_pattern}")
        return False

    print(f"Found {len(csv_files)} partition chunk files to merge.")
    header_written = False
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_path in csv_files:
            # Safety backup check
            if Path(file_path).resolve() == Path(output_file).resolve():
                continue
                
            with open(file_path, "r", encoding="utf-8") as infile:
                lines = infile.readlines()
                if not lines:
                    continue
                first_line_idx = 0
                while first_line_idx < len(lines) and not lines[first_line_idx].strip():
                    first_line_idx += 1
                if first_line_idx >= len(lines):
                    continue
                if not header_written:
                    outfile.write(lines[first_line_idx])
                    header_written = True
                for line in lines[first_line_idx + 1:]:
                    cleaned = line.strip()
                    if not cleaned or cleaned.startswith("file_path,") or cleaned.startswith("..."):
                        continue
                    outfile.write(line)
    return True

def generate_validation_summary(merged_csv, summary_csv, stats_dir):
    file_registry = {}
    expected_trees = {f"Pixel_{i}_Singles" for i in range(1, 81)}
    file_recorded_events = {}

    print("Parsing merged results to verify all 80 detector head trees...")
    with open(merged_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fpath = row.get("file_path", "").strip()
            if not fpath:
                continue
                
            if fpath not in file_registry:
                file_registry[fpath] = {"is_valid": True, "seen_trees": set(), "reason": "OK"}
                file_recorded_events[fpath] = 0
            
            status = row.get("status", "").strip().upper()
            branch_ok = row.get("key_branch_exist", "").strip().upper() == "TRUE"
            
            if status != "OK" or not branch_ok:
                file_registry[fpath]["is_valid"] = False
                file_registry[fpath]["reason"] = "Corrupted/Missing Branches"
                continue

            tree_name = row.get("tree_name", "").strip()
            if tree_name and tree_name in expected_trees:
                file_registry[fpath]["seen_trees"].add(tree_name)
                try:
                    file_recorded_events[fpath] += int(row.get("entries", 0))
                except ValueError:
                    pass

    total_files = 0
    valid_files = 0
    missing_stats_count = 0
    global_recorded_events = 0
    global_emitted_events = 0
    
    file_emitted_events = {}
    stats_dir_path = Path(stats_dir).resolve()

    for fpath, data in file_registry.items():
        if data["is_valid"]:
            missing_count = len(expected_trees - data["seen_trees"])
            if missing_count > 0:
                data["is_valid"] = False
                data["reason"] = f"Incomplete Structure (Missing {missing_count} trees)"
            else:
                stats_fpath = find_stats_file(fpath, stats_dir_path)
                
                if stats_fpath and stats_fpath.exists():
                    emitted = parse_emitted_events(stats_fpath)
                    file_emitted_events[fpath] = emitted
                    global_emitted_events += emitted
                    valid_files += 1
                    global_recorded_events += file_recorded_events[fpath]
                else:
                    missing_stats_count += 1
                    data["is_valid"] = False
                    data["reason"] = "Missing stats configuration log tracking file"
        
        if fpath not in file_emitted_events:
            file_emitted_events[fpath] = 0
        total_files += 1

    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_path", 
            "all_trees_intact", 
            "trees_found_count", 
            "notes", 
            "recorded_detected_events", 
            "emitted_primary_events"
        ])
        
        for fpath, data in sorted(file_registry.items(), key=lambda x: natural_sort_key(x[0])):
            writer.writerow([
                fpath, 
                str(data["is_valid"]), 
                len(data["seen_trees"]), 
                data["reason"],
                file_recorded_events[fpath] if data["is_valid"] else 0,
                file_emitted_events[fpath] if data["is_valid"] else 0
            ])
            
        writer.writerow([])
        writer.writerow(["# METADATA", "TOTAL_RECORDED_DETECTIONS", global_recorded_events, "Recorded counts from trees", ""])
        writer.writerow(["# METADATA", "TOTAL_EMITTED_PRIMARIES", global_emitted_events, "True emitted counts from explicit stats folder", ""])

    print(f"Summary diagnostics compiled: {valid_files}/{total_files} ROOT files are 100% verified.")
    if missing_stats_count > 0:
        print(f"Warning: {missing_stats_count} files marked invalid due to missing stats files.")
    print(f"Global total detected events:  {global_recorded_events}")
    print(f"Global total emitted gammas:   {global_emitted_events}")
    print(f"Summary saved directly to: {summary_csv}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate parallel cluster logs with explicit verification mappings.")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing the target check result partition CSV parts.")
    parser.add_argument("--stats-dir", type=str, required=True,
                        help="Directory where simulation text reports (*_sim_stats.txt) are strictly contained.")
    parser.add_argument("--merged-name", type=str, default=None)
    parser.add_argument("--summary-name", type=str, default=None)
    args = parser.parse_args()

    input_dir_path = Path(args.input_dir)
    job_id = extract_job_id(input_dir_path)
    print(f"Identified Job ID Context: {job_id}")

    merged_filename = args.merged_name if args.merged_name else f"merged_check_sim_root_results_{job_id}.csv"
    summary_filename = args.summary_name if args.summary_name else f"root_files_validation_summary_{job_id}.csv"

    merged_output = input_dir_path / merged_filename
    summary_output = input_dir_path / summary_filename

    # Pass the job ID into the search utility block to filter down glob outputs strictly
    if merge_csv_files(input_dir_path, job_id, merged_output):
        print(f"Merged output successfully compiled to: {merged_output}")
        print("-" * 60)
        generate_validation_summary(merged_output, summary_output, args.stats_dir)

if __name__ == "__main__":
    main()
