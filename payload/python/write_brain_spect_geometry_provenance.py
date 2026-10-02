#!/usr/bin/env python3
"""Write a peer-reportable snapshot of the brain-SPECT geometry inputs.

Schema 2 adds ``analytic_geometry``: every module's resolved pose plus the
Geant4 primitives of each part, so consumers can rebuild exact outlines without
Gate, the simulation script, or a VRML export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gate_sim_brain_spect_boolean import (
    get_collimator_center,
    get_collimator_outer_primitives,
    get_geometry_base_definition,
    get_geometry_definitions,
    get_head_rotation_matrix,
    get_shielding_rotation_matrix,
    map_crystal_id,
)

import qmirt

N_COLLIMATOR_VARIANTS = 4


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


def build_analytic_geometry(mapping_mode: str) -> dict:
    """Resolved per-module poses and part primitives, in Gate's world frame."""
    transforms = get_geometry_definitions()
    n_modules = transforms.height
    base_definitions = [
        get_geometry_base_definition(index) for index in range(N_COLLIMATOR_VARIANTS)
    ]

    # Rings are the distinct elevations; get_geometry_definitions already rounds
    # them and sorts modules by (elevation, azimuth), which is the SRM head order.
    elevation = transforms["elevation"].to_numpy()
    ring_elevations, ring_index = np.unique(elevation, return_inverse=True)

    if mapping_mode == "random":
        # Drawn per simulation task, so it cannot be reconstructed here
        collimator_variant = None
        collimator_center = None
    else:
        collimator_variant = [
            map_crystal_id(index, n_modules, mapping_mode) for index in range(n_modules)
        ]
        collimator_center = [
            get_collimator_center(
                base_definitions[collimator_variant[index]], transforms, index
            ).tolist()
            for index in range(n_modules)
        ]

    crystal = base_definitions[0]["crystal definition"]
    return {
        "frame": (
            "Gate world frame, mm and radians. A part's world points are "
            "center_mm[i] + rotation[i] @ (local_point + primitive.offset_mm), with "
            "rotation[i] a row-major 3x3 matrix. Module index i is SRM head i "
            "(Gate DetectorCrystal_{i+1} / Collimator_{i+1})."
        ),
        "primitive_types": {
            "box": "size_mm = full edge lengths along local x, y, z",
            "trd": "Geant4 G4Trd half-lengths: dx1/dy1 at z=-dz, dx2/dy2 at z=+dz",
        },
        "modules": {
            "count": n_modules,
            "ring_index": ring_index.tolist(),
            "ring_elevation_rad": ring_elevations.tolist(),
            "elevation_rad": elevation.tolist(),
            "azimuth_rad": transforms["azimuth"].to_list(),
            "pinhole_mm": transforms.select(
                "Pinhole_x", "Pinhole_y", "Pinhole_z"
            ).to_numpy().tolist(),
            "rotation": [
                get_head_rotation_matrix(transforms, index).ravel().tolist()
                for index in range(n_modules)
            ],
        },
        "parts": {
            "crystal": {
                "variants": [
                    [
                        {
                            "type": "box",
                            "size_mm": list(crystal["size_mm"]),
                            "offset_mm": [0.0, 0.0, 0.0],
                        }
                    ]
                ],
                "variant_index": [0] * n_modules,
                "center_mm": transforms.select(
                    "Crystal_x", "Crystal_y", "Crystal_z"
                ).to_numpy().tolist(),
                "n_pixels": list(crystal["n_pixels"]),
            },
            "collimator": {
                "variants": [
                    get_collimator_outer_primitives(definition)
                    for definition in base_definitions
                ],
                "variant_index": collimator_variant,
                "center_mm": collimator_center,
            },
        },
    }


def shield_model_record(model: str, pieces_dir: str | None, data_dir: Path) -> dict:
    """What the simulation actually placed: the STL, STL pieces or the CSG tiles."""
    record: dict = {"model": model}
    if model == "pieces":
        pieces = Path(pieces_dir)
        record["pieces_manifest"] = file_record(pieces / "manifest.json", data_dir)
    elif model == "csg":
        from brain_spect_shield_csg import SHIELD_CSG

        record["csg"] = {
            "layout": "tiles",
            "parameters": SHIELD_CSG,
            "builder": "payload/python/brain_spect_shield_csg.py",
            "fitted_to": "the 'source' STL; see dev/python/brain_shield_csg.md",
        }
    return record


def build_provenance(
    fov_shape: str,
    fov_size_mm: float,
    mapping_mode: str,
    with_shielding: bool,
    shield_model: str = "stl",
    shield_pieces_dir: str | None = None,
) -> dict:
    data_dir = (
        qmirt.utils.filesystem.search_dir_up("persistent_data", __file__)
        / "brain_spect"
    )
    base_definitions = [
        get_geometry_base_definition(index) for index in range(N_COLLIMATOR_VARIANTS)
    ]
    shielding_path = Path(base_definitions[0]["shielding file path"])
    point_cloud_path = (
        data_dir / "csv" / "BrainSPECT_Point_Cloud.007.25mmx0.556mm_pinhole.csv"
    )
    detector_module_path = data_dir / "stl" / "BrainSPECT_Module.008.25mm_at_0.556.STL"
    analytic_geometry = build_analytic_geometry(mapping_mode)

    return {
        "schema_version": 2,
        "fov": {"shape": fov_shape, "size_mm": fov_size_mm},
        "shielding": {
            "enabled": with_shielding,
            "material": "Lead",
            **shield_model_record(shield_model, shield_pieces_dir, data_dir),
            # the reference STL (the CSG and the pieces are derived from it)
            "source": file_record(shielding_path, data_dir),
            # STL vertices map to world as rotation @ v (origin_at_cog = False)
            "placement": {
                "rotation": get_shielding_rotation_matrix().ravel().tolist(),
                "translation_mm": [0.0, 0.0, 0.0],
                "origin_at_cog": False,
            },
        },
        "detector": {
            "module_count": analytic_geometry["modules"]["count"],
            "crystal_center_radius_mm": 179.61,
            "crystal": base_definitions[0]["crystal definition"],
            "module_mesh": file_record(detector_module_path, data_dir),
        },
        "pinhole_pattern": {
            "mapping_mode": mapping_mode,
            "point_cloud": file_record(point_cloud_path, data_dir),
            "collimator_variants": [
                definition["collimator definition"] for definition in base_definitions
            ],
        },
        "analytic_geometry": analytic_geometry,
    }


def check_sources_unchanged(existing: dict, provenance: dict) -> None:
    """Refuse to upgrade a file whose recorded inputs differ from today's files,
    since the resolved poses would then describe a different geometry."""
    pairs = {
        "point cloud": (
            existing["pinhole_pattern"]["point_cloud"],
            provenance["pinhole_pattern"]["point_cloud"],
        ),
        "shielding": (
            existing["shielding"]["source"],
            provenance["shielding"]["source"],
        ),
    }
    for label, (old, new) in pairs.items():
        if old["sha256"] != new["sha256"]:
            raise SystemExit(
                f"{label} changed since this provenance was written: "
                f"{old['path']} {old['sha256'][:12]} -> {new['path']} {new['sha256'][:12]}"
            )
    if existing["pinhole_pattern"]["collimator_variants"] != (
        provenance["pinhole_pattern"]["collimator_variants"]
    ):
        raise SystemExit("collimator variant definitions changed; refusing to upgrade")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--from-existing",
        action="store_true",
        help="Upgrade the file at --output in place, reusing its FOV, shielding and "
        "mapping settings after checking its recorded input hashes still match.",
    )
    parser.add_argument("--fov-size-mm", type=float)
    parser.add_argument("--fov-shape", choices=("box", "sphere"), default="sphere")
    parser.add_argument("--mapping-mode", default="sequential")
    parser.add_argument(
        "--with-shielding", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--shield-model", choices=("stl", "pieces", "csg"), default="stl")
    parser.add_argument("--shield-pieces-dir", default=None)
    args = parser.parse_args()
    if not args.from_existing and args.fov_size_mm is None:
        parser.error("--fov-size-mm is required unless --from-existing is given")
    return args


def main() -> int:
    args = parse_args()
    if args.from_existing:
        existing = json.loads(args.output.read_text())
        provenance = build_provenance(
            existing["fov"]["shape"],
            existing["fov"]["size_mm"],
            existing["pinhole_pattern"]["mapping_mode"],
            existing["shielding"]["enabled"],
            existing["shielding"].get("model", "stl"),
        )
        check_sources_unchanged(existing, provenance)
    else:
        provenance = build_provenance(
            args.fov_shape,
            args.fov_size_mm,
            args.mapping_mode,
            args.with_shielding,
            args.shield_model,
            args.shield_pieces_dir,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_path.write_text(json.dumps(provenance, indent=2) + "\n")
    temporary_path.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
