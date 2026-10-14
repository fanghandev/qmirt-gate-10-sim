"""MLEM with the whole system matrix resident on a small GPU, in compact form.

The brain SRM has 1.5e9 non-zeros. As int32 indices + float32 values it needs
~11.2 GiB, which does not fit a 12 GB card next to anything else, so
SparseMLEMReconstructor falls back to the CPU. Here the matrix is stored once, as
CSR with int32 column indices and 1-byte values, ~7 GiB, and no transpose:

- Values are Monte Carlo counts. Integers up to 255 are stored exactly as uint8;
  otherwise float16 (exact for integers up to 2048, ~3 significant digits beyond).
- Forward projection: row chunks of the CSR, values cast to float32 into a reused
  buffer, multiplied with cuSPARSE.
- Backprojection: the same row chunks, scattered into the image with index_add_
  (atomic adds over ~9e6 voxels, little contention). PyTorch's CSC mat-vec would
  read the CSR as H^T without a copy, but it converts internally and needs several
  times the matrix's memory.
- Sensitivity H^T 1 is computed exactly on the CPU before upload.

The update is the same as SparseMLEMReconstructor's: voxels with zero sensitivity
are never updated. min_sensitivity > 0 also freezes voxels below that sensitivity,
which removes the hot speckles at the FOV boundary.
"""

import time

import numpy as np
import scipy.sparse as sp
import torch
from tqdm.auto import tqdm


def compact_matrix_bytes(nnz: int, rows: int) -> int:
    """GPU bytes for the compact matrix: int32 column index + 1-byte value per
    non-zero, plus row pointers."""
    return 5 * nnz + 8 * (rows + 1)


class CompactGpuMLEM:
    def __init__(
        self,
        csr_scipy: sp.csr_matrix,
        measured,
        *,
        device: torch.device | str = "cuda",
        chunk_nnz: int = 100_000_000,
        epsilon: float = 1e-12,
        min_sensitivity: float = 0.0,
    ):
        if not (sp.issparse(csr_scipy) and csr_scipy.format == "csr"):
            raise TypeError("csr_scipy must be a scipy.sparse CSR matrix")
        self.device = torch.device(device)
        self.shape = tuple(int(n) for n in csr_scipy.shape)
        self.epsilon = float(epsilon)
        rows, voxels = self.shape
        nnz = int(csr_scipy.nnz)
        if nnz >= 2**31:
            raise ValueError("int32 indices limit the matrix to < 2^31 non-zeros")

        measured = np.asarray(measured, dtype=np.float32).reshape(-1)
        if measured.size != rows:
            raise ValueError(f"measured has {measured.size} values, matrix has {rows} rows")

        data = csr_scipy.data
        integral = bool(np.all(data == np.round(data)))
        if integral and data.min() >= 0 and data.max() <= 255:
            self.value_dtype = torch.uint8
        else:
            self.value_dtype = torch.float16
            if not (integral and np.abs(data).max() <= 2048):
                print("Warning: values are not small integers; float16 rounds them")

        # Exact sensitivity on the CPU, then the voxels MLEM is allowed to update
        sensitivity = np.asarray(csr_scipy.sum(axis=0), dtype=np.float64).reshape(-1)
        self.sens = torch.from_numpy(sensitivity.astype(np.float32)).to(self.device)
        self.update_mask = self.sens > max(float(min_sensitivity), 0.0)

        # Row chunks of at most chunk_nnz non-zeros (a single row may exceed it)
        indptr = np.asarray(csr_scipy.indptr, dtype=np.int64)
        bounds = [0]
        while bounds[-1] < rows:
            start = bounds[-1]
            stop = int(np.searchsorted(indptr, indptr[start] + chunk_nnz, side="right")) - 1
            bounds.append(min(rows, max(stop, start + 1)))
        self.chunks = []
        for r0, r1 in zip(bounds[:-1], bounds[1:]):
            a, b = int(indptr[r0]), int(indptr[r1])
            crow = torch.from_numpy((indptr[r0 : r1 + 1] - a).astype(np.int32))
            lengths = torch.from_numpy(np.diff(indptr[r0 : r1 + 1]))
            self.chunks.append(
                (r0, r1, a, b, crow.to(self.device), lengths.to(self.device))
            )
        max_chunk = max(b - a for _, _, a, b, _, _ in self.chunks)

        t0 = time.perf_counter()
        self.cols = torch.from_numpy(
            np.asarray(csr_scipy.indices, dtype=np.int32)
        ).to(self.device)
        if self.value_dtype == torch.uint8:
            values = data.astype(np.uint8)
        else:
            values = data.astype(np.float16)
        self.vals = torch.from_numpy(values).to(self.device)
        del values
        self.measured = torch.from_numpy(measured).to(self.device)
        self.value_buffer = torch.empty(max_chunk, dtype=torch.float32, device=self.device)
        self.fp = torch.empty(rows, dtype=torch.float32, device=self.device)
        self.bp = torch.empty(voxels, dtype=torch.float32, device=self.device)
        print(
            f"CompactGpuMLEM: {nnz:,} non-zeros as int32 + {self.value_dtype}, "
            f"{len(self.chunks)} chunks, uploaded in {time.perf_counter() - t0:.1f} s, "
            f"GPU allocated {torch.cuda.memory_allocated(self.device) / 2**30:.2f} GiB"
            if self.device.type == "cuda"
            else f"CompactGpuMLEM on {self.device}"
        )

    def _chunk_values(self, a: int, b: int) -> torch.Tensor:
        buffer = self.value_buffer[: b - a]
        buffer.copy_(self.vals[a:b])
        return buffer

    def forward_project(self, image: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        for r0, r1, a, b, crow, _ in self.chunks:
            block = torch.sparse_csr_tensor(
                crow,
                self.cols[a:b],
                self._chunk_values(a, b),
                size=(r1 - r0, self.shape[1]),
            )
            out[r0:r1] = torch.mv(block, image)
        return out

    def back_project(self, projection: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        out.zero_()
        for r0, r1, a, b, _, lengths in self.chunks:
            local_rows = torch.repeat_interleave(
                torch.arange(r1 - r0, device=self.device, dtype=torch.int32),
                lengths,
                output_size=b - a,
            )
            contributions = self._chunk_values(a, b)
            contributions.mul_(projection[r0:r1][local_rows])
            del local_rows
            out.index_add_(0, self.cols[a:b], contributions)
        return out

    def _update_estimate(self, estimate: torch.Tensor) -> None:
        self.forward_project(estimate, self.fp)
        self.fp.add_(self.epsilon)
        ratio = self.measured / self.fp
        self.back_project(ratio, self.bp)
        mask = self.update_mask
        estimate[mask] *= self.bp[mask] / self.sens[mask]

    def get_sensitivity_map(self) -> torch.Tensor:
        return self.sens

    def reconstruct(self, iterations: list[int], initial_estimate=None):
        """Same contract as SparseMLEMReconstructor.reconstruct: estimates (on the
        CPU) after each listed iteration count."""
        if initial_estimate is None:
            estimate = torch.ones(self.shape[1], device=self.device)
        else:
            estimate = torch.as_tensor(
                initial_estimate, dtype=torch.float32, device=self.device
            ).clone()
            if estimate.numel() != self.shape[1]:
                raise ValueError("initial_estimate size does not match the image")
        wanted = set(iterations)
        estimates = []
        for iteration in tqdm(range(1, max(iterations, default=0) + 1), desc="MLEM (compact GPU)"):
            self._update_estimate(estimate)
            if iteration in wanted:
                estimates.append(estimate.detach().cpu().clone())
        return estimates

    def destroy(self):
        for name in ("cols", "vals", "value_buffer", "fp", "bp", "sens", "update_mask", "measured", "chunks"):
            if hasattr(self, name):
                delattr(self, name)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class StreamingCompactMLEM:
    """MLEM for system matrices larger than the GPU (the 1.5 mm and 1 mm brain SRMs, 2.2e9 and
    4.6e9 non-zeros): the matrix is kept in pinned host memory as one CSR block per detector
    module (int32 columns + float16 values); blocks that fit in gpu_budget_bytes stay on the
    GPU, the others are copied in every iteration on a side stream, overlapped with compute.

    One pass per iteration: the MLEM ratio y / (H x) of a row needs only that row's forward
    projection, so each block is forward- and back-projected in turn and crosses PCIe once.

    blocks: iterable of (row0, scipy CSR block) in row order, consumed once (a generator keeps
    only one block in host memory besides the compact copy).
    support: boolean (voxels,) mask of the voxels MLEM may update (and sensitivity > 0)."""

    def __init__(self, blocks, measured, *, support, device="cuda", gpu_budget_bytes: float = 7.0 * 2**30,
                 epsilon: float = 1e-12):
        self.device = torch.device(device)
        self.epsilon = float(epsilon)
        sensitivity = None
        self.blocks, used, streamed_max, rows = [], 0, 0, 0
        t0 = time.perf_counter()
        for row0, b in blocks:
            if sensitivity is None:
                voxels = int(b.shape[1])
                sensitivity = np.zeros(voxels, np.float64)
            if row0 != rows:
                raise ValueError(f"block at row {row0}, expected {rows}")
            rows += b.shape[0]
            sensitivity += np.bincount(b.indices, weights=b.data, minlength=voxels)
            cols = torch.from_numpy(np.asarray(b.indices, dtype=np.int32))
            vals = torch.from_numpy(np.asarray(b.data, dtype=np.float16))
            nbytes = 6 * b.nnz
            entry = {"row0": row0, "rows": b.shape[0], "nnz": int(b.nnz),
                     "crow": torch.from_numpy(np.asarray(b.indptr, dtype=np.int32)).to(self.device),
                     "lengths": torch.from_numpy(np.diff(b.indptr).astype(np.int64)).to(self.device)}
            if used + nbytes <= gpu_budget_bytes:
                entry["cols"], entry["vals"] = cols.to(self.device), vals.to(self.device)
                used += nbytes
            else:
                entry["cols_host"], entry["vals_host"] = cols.pin_memory(), vals.pin_memory()
                streamed_max = max(streamed_max, b.nnz)
            self.blocks.append(entry)
        self.shape = (rows, voxels)
        measured = np.asarray(measured, dtype=np.float32).reshape(-1)
        if measured.size != rows:
            raise ValueError(f"measured has {measured.size} values, matrix has {rows} rows")
        self.sens = torch.from_numpy(sensitivity.astype(np.float32)).to(self.device)
        self.update_mask = (self.sens > 0) & torch.as_tensor(support, device=self.device)
        self.measured = torch.from_numpy(measured).to(self.device)
        self.staging = [(torch.empty(streamed_max, dtype=torch.int32, device=self.device),
                         torch.empty(streamed_max, dtype=torch.float16, device=self.device)) for _ in range(2)]
        self.copy_stream = torch.cuda.Stream(self.device)
        self.bp = torch.empty(voxels, dtype=torch.float32, device=self.device)
        n_streamed = sum("cols_host" in e for e in self.blocks)
        total = sum(e["nnz"] for e in self.blocks)
        print(f"StreamingCompactMLEM: {total:,} non-zeros in {len(self.blocks)} blocks, {len(self.blocks) - n_streamed} "
              f"resident ({used / 2**30:.1f} GiB), {n_streamed} streamed; prepared in {time.perf_counter() - t0:.0f} s, "
              f"GPU allocated {torch.cuda.memory_allocated(self.device) / 2**30:.2f} GiB")

    def _block_pass(self, e, cols, vals, estimate):
        nnz = e["nnz"]
        v = vals[:nnz].float()
        block = torch.sparse_csr_tensor(e["crow"], cols[:nnz], v, size=(e["rows"], self.shape[1]))
        ratio = self.measured[e["row0"]:e["row0"] + e["rows"]] / (torch.mv(block, estimate) + self.epsilon)
        local_rows = torch.repeat_interleave(torch.arange(e["rows"], device=self.device, dtype=torch.int32),
                                             e["lengths"], output_size=nnz)
        v.mul_(ratio[local_rows])
        self.bp.index_add_(0, cols[:nnz], v)

    def _update_estimate(self, estimate: torch.Tensor) -> None:
        self.bp.zero_()
        compute = torch.cuda.current_stream(self.device)
        streamed = [e for e in self.blocks if "cols_host" in e]
        done = [None, None]  # compute finished with staging buffer k

        def issue(i):
            e, (cols, vals) = streamed[i], self.staging[i % 2]
            with torch.cuda.stream(self.copy_stream):
                if done[i % 2] is not None:
                    self.copy_stream.wait_event(done[i % 2])
                cols[:e["nnz"]].copy_(e["cols_host"], non_blocking=True)
                vals[:e["nnz"]].copy_(e["vals_host"], non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(self.copy_stream)
            return ready

        ready = issue(0) if streamed else None
        for e in self.blocks:  # resident blocks while the first streamed block is copied
            if "cols" in e:
                self._block_pass(e, e["cols"], e["vals"], estimate)
        for i, e in enumerate(streamed):
            compute.wait_event(ready)
            if i + 1 < len(streamed):
                nxt = issue(i + 1)
            cols, vals = self.staging[i % 2]
            self._block_pass(e, cols, vals, estimate)
            done[i % 2] = torch.cuda.Event()
            done[i % 2].record(compute)
            if i + 1 < len(streamed):
                ready = nxt
        mask = self.update_mask
        estimate[mask] *= self.bp[mask] / self.sens[mask]

    def get_sensitivity_map(self) -> torch.Tensor:
        return self.sens

    def reconstruct(self, iterations: list[int], initial_estimate=None):
        estimate = (torch.ones(self.shape[1], device=self.device) if initial_estimate is None else
                    torch.as_tensor(initial_estimate, dtype=torch.float32, device=self.device).clone())
        wanted, estimates = set(iterations), []
        for iteration in tqdm(range(1, max(iterations, default=0) + 1), desc="MLEM (streaming)"):
            self._update_estimate(estimate)
            if iteration in wanted:
                estimates.append(estimate.detach().cpu().clone())
        return estimates

    def destroy(self):
        self.blocks, self.staging = [], []
        for name in ("sens", "update_mask", "measured", "bp"):
            if hasattr(self, name):
                delattr(self, name)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
