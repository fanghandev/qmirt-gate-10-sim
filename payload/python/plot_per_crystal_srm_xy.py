#!/usr/bin/env python3
"""Plot per-crystal cardiac SPECT SRM maps in the XY plane from sparse histogram data.

The script consumes the ``sparse_3d_histograms.npz`` file produced by
``ospool_sim_make_srm.py``. The file stores sparse coordinates as
``(CrystalID, PixelID, x_bin, y_bin, z_bin)`` along with counts. This script
collapses the sparse 3D representation onto the XY plane for each crystal and
writes one PNG per crystal.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-crystal SRM maps in the XY plane from sparse histograms."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("sparse_3d_histograms.npz"),
        help="Path to the sparse histogram NPZ file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cardiac_spect/srm_xy_maps"),
        help="Directory for generated PNG files.",
    )
    parser.add_argument(
        "--crystal-id",
        type=int,
        action="append",
        default=None,
        help="Plot only the selected crystal ID. Repeat for multiple crystals.",
    )
    parser.add_argument(
        "--wrl-path",
        type=Path,
        default=Path("results/cardiac_spect/scanner_and_phantom_geometry.wrl"),
        help="Optional WRL file used to overlay the crystal outline.",
    )
    return parser.parse_args()


def load_sparse_histogram(
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray, int, float, float, int | None]:
    data = np.load(npz_path)
    coords = np.asarray(data["coords"], dtype=np.int32)
    counts = np.asarray(data["counts"], dtype=np.int64)
    hist_bins = int(np.asarray(data["hist_bins"]).reshape(-1)[0])
    hist_range = np.asarray(data["hist_range"], dtype=np.float32).reshape(-1)
    if hist_range.size != 2:
        raise ValueError(f"Expected hist_range to have 2 values, got {hist_range!r}")
    hist_min = float(hist_range[0])
    hist_max = float(hist_range[1])
    if coords.ndim != 2 or coords.shape[1] not in (4, 5):
        raise ValueError(
            f"Expected coords to have shape (N, 4) or (N, 5), got {coords.shape!r}"
        )
    if coords.shape[0] != counts.shape[0]:
        raise ValueError("coords and counts must have the same number of rows")
    crystal_id = None
    if "crystal_id" in data:
        crystal_id = int(np.asarray(data["crystal_id"]).reshape(-1)[0])
    return coords, counts, hist_bins, hist_min, hist_max, crystal_id


def _polygon_bounds(polygons: list[list[list[float]]]):
    xs: list[float] = []
    ys: list[float] = []
    for poly in polygons:
        for x, y in poly:
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def load_wrl_polygons(wrl_path: Path) -> dict[int, list[list[list[float]]]]:
    polygons_by_crystal: dict[int, list[list[list[float]]]] = defaultdict(list)
    if not wrl_path.exists():
        return {}

    wrl_content = wrl_path.read_text()
    solid_blocks = re.split(r"#---------- SOLID:\s+", wrl_content)

    for block in solid_blocks[1:]:
        first_newline = block.find("\n")
        solid_name = block[:first_newline].strip()
        match = re.match(r"(Collimator|DetectorCrystal)_(\d+):0", solid_name)
        if not match:
            continue

        crystal_id = int(match.group(2))
        points_match = re.search(r"point\s*\[(.*?)\]", block, re.DOTALL)
        indices_match = re.search(r"coordIndex\s*\[(.*?)\]", block, re.DOTALL)
        if not points_match or not indices_match:
            continue

        points: list[list[float]] = []
        for line in points_match.group(1).split(","):
            nums = line.strip().split()
            if len(nums) == 3:
                points.append([float(nums[0]), float(nums[1])])

        polygon: list[int] = []
        for index in (int(num) for num in re.findall(r"-?\d+", indices_match.group(1))):
            if index == -1:
                if polygon:
                    polygons_by_crystal[crystal_id].append([points[i] for i in polygon])
                polygon = []
            else:
                polygon.append(index)

    return dict(polygons_by_crystal)


def build_xy_maps(
    coords: np.ndarray,
    counts: np.ndarray,
    hist_bins: int,
    crystal_ids: list[int] | None,
) -> dict[int, np.ndarray]:
    if coords.shape[1] == 5:
        crystal_col = coords[:, 0]
        x_bin = coords[:, 2]
        y_bin = coords[:, 3]
    else:
        crystal_col = None
        x_bin = coords[:, 1]
        y_bin = coords[:, 2]

    if crystal_ids is None:
        if crystal_col is None:
            selected = np.array([1], dtype=np.int32)
        else:
            selected = np.unique(crystal_col)
    else:
        selected = np.asarray(sorted(set(crystal_ids)), dtype=np.int32)

    xy_maps: dict[int, np.ndarray] = {}
    for crystal_id in selected:
        if crystal_col is None:
            mask = np.ones(coords.shape[0], dtype=bool)
        else:
            mask = crystal_col == crystal_id
            if not np.any(mask):
                continue

        image = np.zeros((hist_bins, hist_bins), dtype=np.float64)
        np.add.at(image, (y_bin[mask], x_bin[mask]), counts[mask])
        xy_maps[int(crystal_id)] = image

    return xy_maps


def plot_crystal_map(
    crystal_id: int,
    image: np.ndarray,
    hist_bins: int,
    hist_min: float,
    hist_max: float,
    output_dir: Path,
    polygons_by_crystal: dict[int, list[list[list[float]]]],
) -> None:
    edges = np.linspace(hist_min, hist_max, hist_bins + 1)
    fig, ax = plt.subplots(figsize=(8, 8))
    mesh = ax.pcolormesh(edges, edges, image, cmap="viridis", shading="auto")
    fig.colorbar(mesh, ax=ax, label="Counts")
    ax.set_xlabel("EventPosition_X (mm)")
    ax.set_ylabel("EventPosition_Y (mm)")
    ax.set_title(f"Cardiac SPECT per-crystal SRM XY map - CrystalID {crystal_id}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(hist_min, hist_max)
    ax.set_ylim(hist_min, hist_max)

    for poly in polygons_by_crystal.get(crystal_id, []):
        xs = [point[0] for point in poly] + [poly[0][0]]
        ys = [point[1] for point in poly] + [poly[0][1]]
        ax.plot(xs, ys, color="red", linewidth=0.5, alpha=0.6)

    output_path = output_dir / f"per_crystal_srm_xy_crystal_{crystal_id:03d}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = args.input
    input_files = [input_path]
    if input_path.is_dir():
        input_files = sorted(
            path for path in input_path.glob("*.npz") if path.is_file()
        )

    polygons_by_crystal = load_wrl_polygons(args.wrl_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plotted = 0
    for npz_path in input_files:
        coords, counts, hist_bins, hist_min, hist_max, crystal_id_meta = (
            load_sparse_histogram(npz_path)
        )
        crystal_ids = args.crystal_id
        if crystal_ids is None and crystal_id_meta is not None:
            crystal_ids = [crystal_id_meta]

        xy_maps = build_xy_maps(coords, counts, hist_bins, crystal_ids)
        if not xy_maps:
            continue

        for crystal_id, image in sorted(xy_maps.items()):
            effective_crystal_id = (
                crystal_id_meta if crystal_id_meta is not None else crystal_id
            )
            plot_crystal_map(
                effective_crystal_id,
                image,
                hist_bins,
                hist_min,
                hist_max,
                args.output_dir,
                polygons_by_crystal,
            )
            print(
                f"Saved CrystalID {effective_crystal_id} XY map to "
                f"{args.output_dir / f'per_crystal_srm_xy_crystal_{effective_crystal_id:03d}.png'}"
            )
            plotted += 1

    if not plotted:
        print("No crystals matched the input sparse histogram data.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
