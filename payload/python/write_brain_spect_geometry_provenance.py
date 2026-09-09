#!/usr/bin/env python3
"""Write a peer-reportable snapshot of the brain-SPECT geometry inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gate_sim_brain_spect_boolean import (
    get_geometry_base_definition,
    get_geometry_definitions,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fov-size-mm", type=float, required=True)
    parser.add_argument("--fov-shape", choices=("box", "sphere"), default="sphere")
    parser.add_argument("--mapping-mode", default="sequential")
    parser.add_argument(
        "--with-shielding", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = (
        qmirt.utils.filesystem.search_dir_up("persistent_data", __file__)
        / "brain_spect"
    )
    base_definitions = [get_geometry_base_definition(index) for index in range(4)]
    shielding_path = Path(base_definitions[0]["shielding file path"])
    point_cloud_path = (
        data_dir / "csv" / "BrainSPECT_Point_Cloud.007.25mmx0.556mm_pinhole.csv"
    )
    detector_module_path = data_dir / "stl" / "BrainSPECT_Module.008.25mm_at_0.556.STL"
    transforms = get_geometry_definitions()

    provenance = {
        "schema_version": 1,
        "fov": {"shape": args.fov_shape, "size_mm": args.fov_size_mm},
        "shielding": {
            "enabled": args.with_shielding,
            "material": "Lead",
            "source": file_record(shielding_path, data_dir),
        },
        "detector": {
            "module_count": transforms.height,
            "crystal_center_radius_mm": 179.61,
            "crystal": base_definitions[0]["crystal definition"],
            "module_mesh": file_record(detector_module_path, data_dir),
        },
        "pinhole_pattern": {
            "mapping_mode": args.mapping_mode,
            "point_cloud": file_record(point_cloud_path, data_dir),
            "collimator_variants": [
                definition["collimator definition"] for definition in base_definitions
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_path.write_text(json.dumps(provenance, indent=2) + "\n")
    temporary_path.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
