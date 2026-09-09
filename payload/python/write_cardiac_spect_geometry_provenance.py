#!/usr/bin/env python3
"""Write a peer-reportable snapshot of the cardiac-SPECT geometry inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gate_sim_cardiac_spect_boolean import (
    _resolve_xlsx_path,
    get_dc_spect_geometry_config,
)

import qmirt


def file_record(path: Path, root: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fov-size-mm", type=float, required=True)
    parser.add_argument("--fov-shape", choices=("box", "sphere"), default="sphere")
    parser.add_argument(
        "--with-shielding", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    persistent_data_dir = qmirt.utils.filesystem.search_dir_up(
        "persistent_data", __file__
    )
    cardiac_data_dir = persistent_data_dir / "cardiac_spect"
    spreadsheet_path = _resolve_xlsx_path(persistent_data_dir, None)
    shielding_path = cardiac_data_dir / "stl" / "dc_spect_shielding_combined.stl"
    config = get_dc_spect_geometry_config(spreadsheet_path, cardiac_data_dir / "stl")
    collimator_keys = (
        "collimator_body_length_mm_np",
        "collimator_hole_coords_mm_np",
        "collimator_body_translation_mm",
        "collimator_body_inner_top_mm_np",
        "collimator_body_inner_bottom_mm_np",
        "collimator_body_outer_top_mm_np",
        "collimator_body_outer_bottom_mm_np",
        "collimator_guide_length_mm_np",
        "collimator_guide_translation_mm",
        "collimator_guide_inner_top_mm_np",
        "collimator_guide_outer_top_mm_np",
        "collimator_guide_inner_bottom_mm_np",
        "collimator_guide_outer_bottom_mm_np",
        "collimator_wall_thickness_mm",
    )
    provenance = {
        "schema_version": 1,
        "fov": {"shape": args.fov_shape, "size_mm": args.fov_size_mm},
        "shielding": {
            "enabled": args.with_shielding,
            "material": "Lead",
            "source": file_record(shielding_path, persistent_data_dir),
        },
        "detector": {
            "module_count": 80,
            "crystal": {
                "size_mm": json_value(config["detector_crystal_size_mm"]),
                "n_pixels": json_value(config["n_pixels"]),
                "pixel_size_mm": json_value(config["pixel_size_mm"]),
            },
            "crystal_translations_mm": json_value(
                config["detector_crystal_translation_mm"]
            ),
            "configuration_source": file_record(spreadsheet_path, persistent_data_dir),
        },
        "pinhole_pattern": {
            "mapping_mode": "spreadsheet-coordinates",
            "point_cloud": file_record(spreadsheet_path, persistent_data_dir),
            "collimator_variants": [{"w_pinhole": 2.3, "count": 80}],
            "head_configuration": {
                key: json_value(config[key]) for key in collimator_keys
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_path.write_text(json.dumps(provenance, indent=2) + "\n")
    temporary_path.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
