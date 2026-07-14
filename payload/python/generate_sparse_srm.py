#!/usr/bin/env python3
"""
Generate System Response Matrix (SRM) in sparse CSR format, split by head ID.

Architecture
============
  - 1 Reader process  : opens each ROOT file once, reads all 80 head trees
                        sequentially, applies the energy filter, parses pixel
                        IDs, then writes the filtered arrays into SharedMemory
                        blocks and signals 80 persistent HeadWorker processes.

  - 80 HeadWorker processes : each owns one detector head for the entire run.
                        Attaches to SharedMemory (zero-copy), runs voxel
                        mapping, accumulates a per-head CSR matrix, then
                        signals completion back to the reader.

  - Main process      : shows a two-row Rich progress display
                        (files processed + aggregate event stats).

Synchronisation
===============
  The reader acts as a per-file barrier: it signals all 80 head-workers, waits
  for all 80 completion signals, unlinks SharedMemory, then moves to the next
  file.  Heads with no data for a given file receive an n=0 sentinel and
  immediately re-signal without touching SharedMemory.

Finalisation
============
  Because each HeadWorker accumulates across ALL files, the final step is a
  trivial rename (tmp_head_i/srm.npz → srm_head_i.npz) — no merging needed.
"""
import argparse
import sys
import time
import shutil
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import uproot
from scipy.sparse import coo_matrix, csr_matrix, save_npz
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn,
    MofNCompleteColumn, SpinnerColumn,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_HEADS          = 80
NUM_PIXELS_PER_HEAD = 625   # 25×25 grid


# ---------------------------------------------------------------------------
# SharedMemory helpers
# ---------------------------------------------------------------------------

def _create_shm(size):
    """Allocate a SharedMemory block without resource tracker registration.

    Python's resource tracker auto-registers every SharedMemory created with
    create=True and also sets _unlink=True, so close() auto-unlinks at GC
    time.  When the reader explicitly manages the lifetime (one unlink() per
    file), both behaviours produce spurious 'leaked shared_memory' / 'No such
    file or directory' warnings at process shutdown.

    Patching resource_tracker.register to a no-op for the duration of the
    constructor prevents registration entirely.  Setting _unlink=False stops
    close() from auto-unlinking.  The reader then owns one explicit unlink()
    followed by close() per block — clean, single-ownership, no warnings.
    """
    import multiprocessing.resource_tracker as _rt
    _orig = _rt.register
    _rt.register = lambda *a, **kw: None   # disable for this allocation
    try:
        shm = SharedMemory(create=True, size=max(size, 1))
        shm._unlink = False   # prevent close() / __del__ from auto-unlinking
    finally:
        _rt.register = _orig              # always restore
    return shm


# ---------------------------------------------------------------------------
# HeadWorker — persistent, handles one head for all files
# ---------------------------------------------------------------------------

def head_worker_func(head_idx, per_head_queue, completion_queue,
                     grid_size, voxel_size, output_dir):
    """
    Persistent process for detector head `head_idx`.

    Message protocol (per_head_queue):
      - (shm_x, shm_y, shm_z, shm_pid, n)  → process this file's data
      - (None,  None,  None,  None,    0 )  → no data for this file (skip)
      - None                                → shutdown

    Signals to completion_queue:
      - (head_idx, n_valid : int)   after each file
      - (head_idx, "saved")         after saving SRM on shutdown
    """
    num_voxels = grid_size ** 3
    half_size  = grid_size * voxel_size * 0.5
    origin_mm  = np.array([-half_size, -half_size, -half_size], dtype=np.float32)

    local_srm = csr_matrix((NUM_PIXELS_PER_HEAD, num_voxels), dtype=np.float32)

    while True:
        msg = per_head_queue.get()

        # ---- Shutdown sentinel ----
        if msg is None:
            tmp_dir = Path(output_dir) / f"tmp_head_{head_idx}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            local_srm.eliminate_zeros()
            save_npz(tmp_dir / "srm.npz", local_srm)
            completion_queue.put((head_idx, "saved"))
            break

        shm_name_x, shm_name_y, shm_name_z, shm_name_pid, n = msg

        # ---- No data for this head in this file ----
        if n == 0:
            completion_queue.put((head_idx, 0))
            continue

        # ---- Attach to SharedMemory (zero-copy numpy views) ----
        # Note: create=False attachments are NOT registered with the resource
        # tracker in Python 3.12+, so no _untrack_shm() call is needed here.
        shm_x   = SharedMemory(name=shm_name_x)
        shm_y   = SharedMemory(name=shm_name_y)
        shm_z   = SharedMemory(name=shm_name_z)
        shm_pid = SharedMemory(name=shm_name_pid)

        src_x          = np.ndarray((n,), dtype=np.float32, buffer=shm_x.buf)
        src_y          = np.ndarray((n,), dtype=np.float32, buffer=shm_y.buf)
        src_z          = np.ndarray((n,), dtype=np.float32, buffer=shm_z.buf)
        local_pixel_id = np.ndarray((n,), dtype=np.int32,   buffer=shm_pid.buf)

        # ---- Voxel mapping ----
        ix = np.floor((src_x - origin_mm[0]) / voxel_size).astype(np.int32)
        iy = np.floor((src_y - origin_mm[1]) / voxel_size).astype(np.int32)
        iz = np.floor((src_z - origin_mm[2]) / voxel_size).astype(np.int32)

        valid_vox  = ((ix >= 0) & (ix < grid_size) &
                      (iy >= 0) & (iy < grid_size) &
                      (iz >= 0) & (iz < grid_size))
        valid_det  = (local_pixel_id >= 0) & (local_pixel_id < NUM_PIXELS_PER_HEAD)
        valid_mask = valid_vox & valid_det

        n_valid = int(np.sum(valid_mask))

        if n_valid > 0:
            final_voxels = (ix[valid_mask] * grid_size * grid_size
                            + iy[valid_mask] * grid_size
                            + iz[valid_mask])
            final_pixels = local_pixel_id[valid_mask]
            ones = np.ones(n_valid, dtype=np.float32)
            coo  = coo_matrix(
                (ones, (final_pixels, final_voxels)),
                shape=(NUM_PIXELS_PER_HEAD, num_voxels),
            )
            local_srm += coo.tocsr()

        # ---- Detach (reader owns + will unlink) ----
        shm_x.close()
        shm_y.close()
        shm_z.close()
        shm_pid.close()

        completion_queue.put((head_idx, n_valid))


# ---------------------------------------------------------------------------
# Reader — reads files sequentially, distributes via SharedMemory
# ---------------------------------------------------------------------------

def reader_process_func(file_task_queue, done_queue, per_head_queues,
                        completion_queue, energy_min, energy_max):
    """
    Reads one ROOT file at a time:
      1. Open file, iterate over all 80 trees.
      2. Apply energy filter + parse pixel IDs in reader (shared work).
      3. Write filtered arrays into SharedMemory; record SHM names.
      4. Signal all 80 head-workers simultaneously via per_head_queues.
      5. Block on completion_queue until all 80 head-workers report done.
      6. Unlink SharedMemory, report stats to main via done_queue.
      7. Repeat until None sentinel received.
    On shutdown: forwards None to all head-workers, waits for "saved" signals.
    """
    while True:
        fpath = file_task_queue.get()

        # ---- Shutdown ----
        if fpath is None:
            for q in per_head_queues:
                q.put(None)
            saved = 0
            while saved < NUM_HEADS:
                _, status = completion_queue.get()
                if status == "saved":
                    saved += 1
            done_queue.put("done")
            break

        # ---- Read file and populate SharedMemory ----
        file_raw   = 0
        shm_blocks = {}   # head_idx -> [shm_x, shm_y, shm_z, shm_pid]
        head_msgs  = {}   # head_idx -> msg tuple (or absent → n=0 sentinel)

        try:
            with uproot.open(fpath) as f:
                for head_idx in range(NUM_HEADS):
                    tree_name = f"Pixel_{head_idx + 1}_Singles"
                    if tree_name not in f:
                        continue

                    tree = f[tree_name]
                    if tree.num_entries == 0:
                        continue

                    file_raw += tree.num_entries

                    data = tree.arrays([
                        "EventPosition_X", "EventPosition_Y", "EventPosition_Z",
                        "PreStepUniqueVolumeID", "TotalEnergyDeposit",
                    ], library="np")

                    # Energy filter — done once in reader for this head
                    edep        = data["TotalEnergyDeposit"]
                    energy_mask = (edep >= energy_min) & (edep <= energy_max)
                    if not np.any(energy_mask):
                        continue

                    # Filtered, contiguous arrays
                    src_x = np.ascontiguousarray(
                        data["EventPosition_X"][energy_mask], dtype=np.float32)
                    src_y = np.ascontiguousarray(
                        data["EventPosition_Y"][energy_mask], dtype=np.float32)
                    src_z = np.ascontiguousarray(
                        data["EventPosition_Z"][energy_mask], dtype=np.float32)

                    # Parse pixel IDs in reader (string → int32)
                    vol_ids   = data["PreStepUniqueVolumeID"][energy_mask]
                    pixel_ids = np.ascontiguousarray(
                        np.array([int(s.rsplit("_", 1)[-1]) for s in vol_ids],
                                 dtype=np.int32))

                    n = len(src_x)

                    # Allocate untracked SharedMemory (reader manages lifecycle)
                    shm_x   = _create_shm(src_x.nbytes)
                    shm_y   = _create_shm(src_y.nbytes)
                    shm_z   = _create_shm(src_z.nbytes)
                    shm_pid = _create_shm(pixel_ids.nbytes)

                    np.ndarray((n,), dtype=np.float32, buffer=shm_x.buf)[:] = src_x
                    np.ndarray((n,), dtype=np.float32, buffer=shm_y.buf)[:] = src_y
                    np.ndarray((n,), dtype=np.float32, buffer=shm_z.buf)[:] = src_z
                    np.ndarray((n,), dtype=np.int32,   buffer=shm_pid.buf)[:] = pixel_ids

                    shm_blocks[head_idx] = [shm_x, shm_y, shm_z, shm_pid]
                    head_msgs[head_idx]  = (shm_x.name, shm_y.name,
                                            shm_z.name, shm_pid.name, n)

        except Exception as e:
            print(f"\n[Reader] Error processing {fpath}: {e}", file=sys.stderr)

        # ---- Signal all 80 head-workers simultaneously ----
        _EMPTY = (None, None, None, None, 0)
        for head_idx in range(NUM_HEADS):
            per_head_queues[head_idx].put(head_msgs.get(head_idx, _EMPTY))

        # ---- Wait for all 80 to finish (barrier) ----
        file_valid = 0
        finished   = 0
        while finished < NUM_HEADS:
            _, n_valid = completion_queue.get()
            if isinstance(n_valid, int):
                file_valid += n_valid
            finished += 1

        # ---- Unlink SharedMemory (reader is sole owner) ----
        for blocks in shm_blocks.values():
            for shm in blocks:
                shm.unlink()   # free OS resource
                shm.close()    # close fd/mmap (_unlink=False → no double-unlink)

        done_queue.put(("file", file_raw, file_valid))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate System Response Matrix (SRM) in sparse CSR format split by head ID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir",  type=str,   default=".",
                        help="Directory containing the simulation ROOT files.")
    parser.add_argument("--output-dir", type=str,   default="srm_sparse_output",
                        help="Directory to save the resulting .npz files.")
    parser.add_argument("--file-list",  type=str,   default="",
                        help="Path to a text file containing the specific list of ROOT files to process.")
    parser.add_argument("--energy-min", type=float, default=0.126,
                        help="Minimum energy window value in MeV.")
    parser.add_argument("--energy-max", type=float, default=0.154,
                        help="Maximum energy window value in MeV.")
    parser.add_argument("--grid-size",  type=int,   default=90,
                        help="Grid dimension N for NxNxN voxels. "
                             "Default 90 → FOV -90 to +90 mm at 2.0 mm voxel size.")
    parser.add_argument("--voxel-size", type=float, default=2.0,
                        help="Voxel size in mm.")

    args = parser.parse_args()

    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.file_list:
        file_list_path = Path(args.file_list).resolve()
        if not file_list_path.exists():
            print(f"Error: File list {file_list_path} not found.", file=sys.stderr)
            sys.exit(1)
        with open(file_list_path, "r") as f:
            files = [Path(line.strip()).resolve() for line in f if line.strip()]
    else:
        files = sorted(list(input_dir.glob("pixel_singles_*.root")))

    if not files:
        print(f"No ROOT files found in {input_dir}.", file=sys.stderr)
        sys.exit(1)

    num_files = len(files)
    print(f"Found {num_files} files to process.")
    print(f"Output directory: {output_dir}")
    print(f"Architecture: 1 reader + {NUM_HEADS} head-workers (SharedMemory, zero-copy)")
    print(f"Energy Window: [{args.energy_min}, {args.energy_max}] MeV")
    print(f"Source Voxel Grid: {args.grid_size}×{args.grid_size}×{args.grid_size} "
          f"(voxel size = {args.voxel_size} mm, "
          f"FOV = ±{args.grid_size * args.voxel_size * 0.5:.0f} mm)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Queues ----
    file_task_queue  = multiprocessing.Queue()
    done_queue       = multiprocessing.Queue()
    per_head_queues  = [multiprocessing.Queue() for _ in range(NUM_HEADS)]
    completion_queue = multiprocessing.Queue()

    for fpath in files:
        file_task_queue.put(fpath)
    file_task_queue.put(None)   # single sentinel for the single reader

    # ---- Spawn 80 persistent head-workers ----
    head_workers = []
    for head_idx in range(NUM_HEADS):
        p = multiprocessing.Process(
            target=head_worker_func,
            args=(head_idx, per_head_queues[head_idx], completion_queue,
                  args.grid_size, args.voxel_size, str(output_dir)),
            daemon=True,
        )
        p.start()
        head_workers.append(p)

    # ---- Spawn reader ----
    start_time = time.time()
    reader = multiprocessing.Process(
        target=reader_process_func,
        args=(file_task_queue, done_queue, per_head_queues, completion_queue,
              args.energy_min, args.energy_max),
        daemon=True,
    )
    reader.start()

    # ---- Progress display ----
    def _fmt(n):
        """Human-readable count with K/M suffix."""
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f} M"
        if n >= 1_000:
            return f"{n / 1_000:.1f} K"
        return str(n)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        file_task  = progress.add_task("[cyan]  Files processed",   total=num_files)
        event_task = progress.add_task("[green] Events (all heads)", total=None)

        total_raw   = 0
        total_valid = 0

        reader_running = True
        while reader_running:
            try:
                msg = done_queue.get(timeout=0.5)
                if msg == "done":
                    reader_running = False
                elif msg[0] == "file":
                    _, file_raw, file_valid = msg
                    total_raw   += file_raw
                    total_valid += file_valid
                    elapsed     = time.time() - start_time
                    rate        = total_raw / elapsed if elapsed > 0 else 0
                    efficiency  = (100.0 * total_valid / total_raw
                                   if total_raw > 0 else 0.0)
                    progress.update(file_task, advance=1)
                    progress.update(
                        event_task,
                        description=(
                            f"[green] Events — read: {_fmt(total_raw)}"
                            f"  valid: {_fmt(total_valid)}"
                            f"  ({efficiency:.1f}%)"
                            f"  {_fmt(int(rate))}/s"
                        ),
                    )
            except Exception:
                if not reader.is_alive():
                    reader_running = False

    reader.join()
    for p in head_workers:
        p.join()

    # ---- Finalise: trivial rename, no merging needed ----
    print("\nFinalising head SRM files...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("[yellow]Finalising heads", total=NUM_HEADS)
        for head_idx in range(NUM_HEADS):
            src = output_dir / f"tmp_head_{head_idx}" / "srm.npz"
            dst = output_dir / f"srm_head_{head_idx}.npz"
            if src.exists():
                src.rename(dst)
            shutil.rmtree(output_dir / f"tmp_head_{head_idx}", ignore_errors=True)
            progress.update(task_id, advance=1)

    total_time = time.time() - start_time
    print(f"\nSRM generation finished in {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
