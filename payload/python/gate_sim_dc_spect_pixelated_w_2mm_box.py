import argparse
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import opengate as gate
import pandas as pd
from opengate.geometry.volumes import subtract_volumes, unite_volumes
from scipy.spatial.transform import Rotation

import qmirt


def _parse_activity_value_and_unit(
    activity_value: str | list[str],
) -> dict[str, float | str]:
    _ACTIVITY_VALUE_RE = re.compile(
        r"^\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?P<unit>[A-Za-zµμ]+)?\s*$"
    )
    if isinstance(activity_value, list):
        text = " ".join(activity_value).strip()
    else:
        text = activity_value.strip()

    match = _ACTIVITY_VALUE_RE.match(text)
    if not match:
        raise argparse.ArgumentTypeError(
            "source activity must be a number optionally followed by a unit, such as '1.2e10 Bq', '10 mCi', or '0.01 Ci'"
        )

    value = float(match.group("value"))
    unit = (match.group("unit") or "Bq").replace("µ", "u").replace("μ", "u")
    unit_lower = unit.lower()
    canonical_unit = {
        "bq": "Bq",
        "kbq": "kBq",
        "mbq": "MBq",
        "gbq": "GBq",
        "tbq": "TBq",
        "ci": "Ci",
        "mci": "mCi",
        "uci": "uCi",
        "nci": "nCi",
        "pci": "pCi",
    }[unit_lower]

    return {"value": value, "unit": canonical_unit}


def _activity_to_gate_units(activity: dict[str, float | str]) -> float:
    unit = str(activity["unit"])
    if not hasattr(gate.g4_units, unit):
        raise ValueError(f"Unsupported activity unit: {unit}")
    return float(activity["value"]) * getattr(gate.g4_units, unit)


def _activity_to_bq(activity: dict[str, float | str]) -> float:
    value = float(activity["value"])
    unit = str(activity["unit"]).replace("µ", "u").replace("μ", "u").lower()
    unit_scale_to_bq = {
        "bq": 1.0,
        "kbq": 1e3,
        "mbq": 1e6,
        "gbq": 1e9,
        "tbq": 1e12,
        "ci": 3.7e10,
        "mci": 3.7e7,
        "uci": 3.7e4,
        "nci": 3.7e1,
        "pci": 3.7e-2,
    }
    if unit not in unit_scale_to_bq:
        raise ValueError(f"Unsupported activity unit: {activity['unit']}")
    return value * unit_scale_to_bq[unit]


# Helper function: Generate a triangular mesh array of cold rods within a 60-degree sector
def add_rod_sector(sim, mother_name, sector_index, rod_radius_mm, spacing_mm):
    """
    sector_index: 0 to 5, representing the six 60-degree sectors
    rod_radius_mm: Radius of the cold rods in this sector
    spacing_mm: Center-to-center spacing of the rods (typically 2x the diameter)
    """
    cm = gate.g4_units.cm
    mm = gate.g4_units.mm
    rod_height = 8.8 * cm
    z_offset_rods = -4.65 * cm

    # Base rotation angle (each sector spans 60 degrees)
    theta = np.deg2rad(sector_index * 60)
    # Rotation matrix to map the 0-degree reference sector to its target position
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    # Generate triangular grid points within the 60-degree sector (simplified high-density generation strategy)
    # In practice, adjust 'rows' to control the number of rod layers
    rows = int((10.0 * cm) / (spacing_mm * mm))
    rod_count = 0

    for row in range(1, rows):
        # Increment the number of rods per row
        for col in range(row):
            # Local coordinate system for the 0-degree reference sector (X is the central axis of the sector)
            local_x = row * spacing_mm * mm * np.cos(np.deg2rad(30))
            # Y-coordinates are distributed symmetrically across the central axis based on the column index
            local_y = (col - (row - 1) / 2.0) * spacing_mm * mm

            # Discard if the coordinate exceeds the inner radius of the main cylinder (leaving a marginal gap)
            if np.sqrt(local_x**2 + local_y**2) + rod_radius_mm * mm > 9.8 * cm:
                continue

            # Apply the rotation matrix to map coordinates to the global XY plane
            global_xy = rot_matrix.dot(np.array([local_x, local_y]))

            rod = sim.add_volume("TubsVolume", f"ColdRod_S{sector_index}_{rod_count}")
            rod.mother = mother_name
            rod.material = "G4_PLEXIGLASS"
            rod.rmin = 0
            rod.rmax = rod_radius_mm * mm
            rod.dz = rod_height * 0.5
            rod.translation = [global_xy[0], global_xy[1], z_offset_rods]
            rod.color = [0.8, 0.8, 0.8, 1]

            rod_count += 1


def add_Jaszczak_phantom(sim: gate.Simulation):
    # ========================================================
    # 1. Define Mother Volume - Filled with radioactive water solution
    # ========================================================
    # Use Geant4's built-in NIST material database to avoid loading external db files

    cm = gate.g4_units.cm
    mm = gate.g4_units.mm
    phantom = sim.add_volume("TubsVolume", "Jaszczak_Phantom")
    phantom.mother = "world"
    phantom.material = "G4_WATER"
    phantom.rmin = 0 * cm
    phantom.rmax = 10.2 * cm
    phantom.dz = 18.6 * cm * 0.5  # Total cylinder height
    # Set display color: RGBA (translucent blue)
    phantom.color = [0, 0, 1, 0.2]

    # ========================================================
    # 2. Construct upper section Cold Spheres
    # ========================================================
    # Cold sphere diameters (mm): 31.8, 25.4, 19.1, 15.9, 12.7, 9.5
    sphere_radii_mm = [15.9, 12.7, 9.55, 7.95, 6.35, 4.75]
    sphere_angles_deg = [0, 60, 120, 180, 240, 300]
    sphere_placement_radius = 5.72 * cm
    z_offset_spheres = 4.65 * cm  # Z-axis offset for the upper section

    for i, (r, angle) in enumerate(zip(sphere_radii_mm, sphere_angles_deg)):
        sph = sim.add_volume("Sphere", f"ColdSphere_{i}")
        sph.mother = (
            "Jaszczak_Phantom"  # CSG: Placed directly in water as a daughter volume
        )
        sph.material = "G4_PLEXIGLASS"  # Acrylic (PMMA) material
        sph.rmin = 0
        sph.rmax = r * mm

        # Calculate XY coordinates directly in Python
        x = sphere_placement_radius * np.cos(np.deg2rad(angle))
        y = sphere_placement_radius * np.sin(np.deg2rad(angle))
        sph.translation = [x, y, z_offset_spheres]
        sph.color = [1, 1, 1, 0.8]  # Opaque white

    # ========================================================
    # 3. Construct lower section Cold Rods array
    # ========================================================

    # Cold rod radius specifications for the 6 sectors (mm): 6.35, 5.55, 4.75, 3.95, 3.2, 2.4
    rod_radii_mm = [6.35, 5.55, 4.75, 3.95, 3.2, 2.4]

    # Loop to generate cold rods for all 6 sectors
    for sector, r in enumerate(rod_radii_mm):
        # Center-to-center spacing is typically 2x the rod diameter
        spacing = r * 4.0
        add_rod_sector(sim, "Jaszczak_Phantom", sector, r, spacing)


def add_background_source(
    sim: gate.Simulation,
    phantom_name: str = "Jaszczak_Phantom",
    activity: dict[str, float | str] | None = None,
    source_type: str = "Gamma-140",
):
    """
    Adds a background radioactive source to the specified phantom volume.
    Leverages GATE's 'confine' feature combined with the CSG mother-daughter
    hierarchy to automatically exclude radioactive emissions from the cold rods and spheres.

    Args:
        sim: The opengate simulation object.
        phantom_name: Name of the mother volume (water cylinder).
        activity_mCi: Total source activity in mCi.
        source_type: 'Gamma-140', 'Tc-99m', or 'Co-57'.
    """
    # ========================================================
    # Add Background Radioactive Source
    # ========================================================
    source = sim.add_source("GenericSource", f"{source_type}_Background")

    # 1. Particle Type Definition based on selected source type
    if source_type.upper() == "GAMMA-140":
        # Pure 140 keV monoenergetic gamma (Fastest simulation speed)
        # Skips all atomic de-excitations, X-rays, and Auger electrons.
        # Note: True Tc-99m photopeak is 140.5 keV, adjusted to 140.0 keV per request.
        source.particle = "gamma"
        source.energy.type = "mono"
        source.energy.mono = 140.0 * gate.g4_units.keV

    elif source_type.upper() == "TC-99M":
        # Full Tc-99m metastable decay cascade.
        # Simulates the isomeric transition including internal conversion and X-rays.
        # GATE/Geant4 requires specifying the excitation energy (142.6836 keV)
        # to correctly identify the metastable state (Tc-99m) instead of the ground state (Tc-99).
        source.particle = "ion 43 99"
        # source.energy.type = "mono"
        # source.energy.mono = 0 * gate.keV
        # Depending on the specific opengate-python version, excitation energy for isomers
        # is typically passed via the ion property or appending to the string.
        # e.g., source.particle = 'ion 43 99 0 142.6836' (Z, A, Q, E_ex in keV)
        source.particle = "ion 43 99 0 142.6836"

    elif source_type.upper() == "CO-57":
        # Cobalt-57 full radioactive decay (Z=27, A=57)
        # Includes the 122 keV and 136 keV gammas, plus Fe X-rays.
        source.particle = "ion 27 57"
        # source.energy.type = "mono"
        # source.energy.mono = 0 * gate.g4_units.keV

    else:
        raise ValueError(
            "Unsupported source type. Please choose 'Gamma-140', 'Tc-99m', or 'Co-57'."
        )

    # 2. Spatial Distribution Setting
    # Define a cylindrical emission region identical to the main water cavity dimensions
    source.position.type = "cylinder"
    source.position.radius = 10.2 * gate.g4_units.cm
    source.position.dz = 18.6 * gate.g4_units.cm
    source.position.translation = [
        0,
        0,
        0,
    ]  # Aligned with the center of the phantom cavity

    # 3. Core Constraint: Cold Spot Exclusion
    # Strictly confine photon emission to the volume physically named by 'phantom_name'.
    # Due to the mother-daughter CSG hierarchy, daughters (acrylic rods/spheres) are excluded automatically.
    source.position.confine = phantom_name

    # 4. Activity Setting
    if activity is None:
        activity = {"value": 10.0, "unit": "mCi"}
    source.activity = _activity_to_gate_units(activity)
    return source


def add_point_source(
    sim: gate.Simulation, energy_keV: float = 140.0, name: str = "PointSource", *, args
):

    source = gate.sources.generic.GenericSource(name=name)
    source.particle = "gamma"
    source.energy.type = "mono"
    source.activity = _activity_to_gate_units(args.source_activity)
    source.energy.mono = energy_keV * gate.g4_units.keV
    source.position.type = "point"
    source.position.point = [0, 0, 0]  # unit is mm
    sim.add_source(source, name=name)


def generate_unique_seed(job_array_id: str, job_array_task_id: str) -> int:
    seed_string = f"gate_sim_{job_array_id}_{job_array_task_id}"
    # Also add timestamp to ensure uniqueness across different runs, if needed
    seed_string += f"_{os.times()}"
    return int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)


def get_dc_spect_geometry_config(
    xlsx_path: Path, stl_dir: Path, *, n_pixels=(25, 25, 1)
) -> dict:
    n_heads = 80
    collimator_hole_size_mm = 2.3  # unit is mm
    collimator_wall_thickness_mm = 2.0  # unit is mm
    collimator_guide_length_mm = 3.0
    detector_crystal_size_mm = [50.0, 50.0, 10.0]  # unit is mm
    n_pixels = np.array(n_pixels)

    pixel_size_mm = detector_crystal_size_mm / n_pixels

    shielding_file_path = (
        stl_dir / "dc_spect_shielding_combined.stl" if stl_dir else None
    )

    df_coords = pd.read_excel(
        xlsx_path, sheet_name="Coordinates"
    )  # read the "Coordinates" sheet
    df_coords.columns = df_coords.iloc[0]
    df_coords = df_coords[1:]  # remove the first row which is now the header
    df_coords = df_coords.reset_index(
        drop=True
    )  # reset the index after removing the first row
    df_coords = df_coords.apply(
        pd.to_numeric, errors="coerce"
    )  # convert all columns to numeric, coercing errors to NaN
    df_coords.columns.name = "Coordinates Sheet"

    collimator_body_length_mm_np = df_coords["length of collimator"].values
    collimator_hole_coords_mm = df_coords[
        [
            "x coordinate value at center of hole",
            "y coordinate value at center of hole",
            "z coordinate value at center of hole",
        ]
    ].values

    # Fail early if the spreadsheet contains non-numeric values in required fields.
    if (
        np.isnan(collimator_body_length_mm_np).any()
        or np.isnan(collimator_hole_coords_mm).any()
    ):
        raise ValueError(
            "Invalid geometry spreadsheet values detected (NaN) in required columns. "
            "Please check the Coordinates sheet numeric fields."
        )

    if not isinstance(collimator_body_length_mm_np, np.ndarray):
        collimator_body_length_mm_np = np.asarray(collimator_body_length_mm_np)

    if not isinstance(collimator_hole_coords_mm, np.ndarray):
        collimator_hole_coords_mm_np = np.asarray(collimator_hole_coords_mm)
    else:
        collimator_hole_coords_mm_np = collimator_hole_coords_mm

    hole_fov_center_distance_mm_np = np.linalg.norm(
        collimator_hole_coords_mm_np, axis=1
    )
    azimuthal_angle_deg = (
        np.arctan2(
            collimator_hole_coords_mm_np[:, 1], collimator_hole_coords_mm_np[:, 0]
        )
        * 180
        / np.pi
    )
    hole_fov_center_dist_xy_mm_np = np.linalg.norm(
        collimator_hole_coords_mm_np[:, :2], axis=1
    )
    polar_angle_deg = (
        np.arctan2(collimator_hole_coords_mm_np[:, 2], hole_fov_center_dist_xy_mm_np)
        * 180
        / np.pi
    )
    collimator_body_center_dist_mm_np = (
        hole_fov_center_distance_mm_np + collimator_body_length_mm_np * 0.5
    )
    collimator_body_translation_mm = collimator_body_center_dist_mm_np.reshape(
        -1, 1
    ) * np.column_stack(
        (
            np.cos(np.radians(polar_angle_deg))
            * np.cos(np.radians(azimuthal_angle_deg)),
            np.cos(np.radians(polar_angle_deg))
            * np.sin(np.radians(azimuthal_angle_deg)),
            np.sin(np.radians(polar_angle_deg)),
        )
    )
    detector_crystal_center_dist_mm_np = (
        hole_fov_center_distance_mm_np
        + collimator_body_length_mm_np
        + detector_crystal_size_mm[2] * 0.5
    )
    detector_crystal_translation_mm = detector_crystal_center_dist_mm_np.reshape(
        -1, 1
    ) * np.column_stack(
        (
            np.cos(np.radians(polar_angle_deg))
            * np.cos(np.radians(azimuthal_angle_deg)),
            np.cos(np.radians(polar_angle_deg))
            * np.sin(np.radians(azimuthal_angle_deg)),
            np.sin(np.radians(polar_angle_deg)),
        )
    )

    collimator_wall_thickness_mm_np = np.full((n_heads,), collimator_wall_thickness_mm)
    collimator_body_inner_top_mm_np = np.full((n_heads,), detector_crystal_size_mm[0])
    collimator_body_inner_bottom_mm_np = np.full((n_heads,), collimator_hole_size_mm)
    collimator_body_outer_top_mm_np = (
        collimator_body_inner_top_mm_np + collimator_wall_thickness_mm_np * 2
    )
    collimator_body_outer_bottom_mm_np = (
        collimator_body_inner_bottom_mm_np + collimator_wall_thickness_mm_np * 2
    )

    collimator_guide_exit_angle_rad = np.arctan2(
        (collimator_body_inner_top_mm_np + collimator_body_inner_bottom_mm_np) * 0.5,
        collimator_body_length_mm_np,
    )

    collimator_guide_length_mm_np = np.full((n_heads,), collimator_guide_length_mm)
    collimator_guide_distance_mm_np = (
        hole_fov_center_distance_mm_np - collimator_guide_length_mm_np
    )
    collimator_guide_translation_mm = collimator_guide_distance_mm_np.reshape(
        -1, 1
    ) * np.column_stack(
        (
            np.cos(np.radians(polar_angle_deg))
            * np.cos(np.radians(azimuthal_angle_deg)),
            np.cos(np.radians(polar_angle_deg))
            * np.sin(np.radians(azimuthal_angle_deg)),
            np.sin(np.radians(polar_angle_deg)),
        )
    )

    collimator_guide_inner_top_mm_np = np.full((n_heads,), collimator_hole_size_mm)
    collimator_guide_outer_top_mm_np = (
        collimator_guide_inner_top_mm_np + collimator_wall_thickness_mm_np * 2
    )
    collimator_guide_inner_bottom_mm_np = (
        collimator_guide_inner_top_mm_np
        + np.tan(collimator_guide_exit_angle_rad) * collimator_guide_length_mm_np * 2
    )
    collimator_guide_outer_bottom_mm_np = (
        collimator_guide_inner_bottom_mm_np + collimator_wall_thickness_mm_np * 2
    )

    return {
        "collimator_body_length_mm_np": collimator_body_length_mm_np,
        "collimator_hole_coords_mm_np": collimator_hole_coords_mm_np,
        "collimator_body_translation_mm": collimator_body_translation_mm,
        "collimator_body_inner_top_mm_np": collimator_body_inner_top_mm_np,
        "collimator_body_inner_bottom_mm_np": collimator_body_inner_bottom_mm_np,
        "collimator_body_outer_top_mm_np": collimator_body_outer_top_mm_np,
        "collimator_body_outer_bottom_mm_np": collimator_body_outer_bottom_mm_np,
        "collimator_guide_length_mm_np": collimator_guide_length_mm_np,
        "collimator_guide_translation_mm": collimator_guide_translation_mm,
        "collimator_guide_inner_top_mm_np": collimator_guide_inner_top_mm_np,
        "collimator_guide_outer_top_mm_np": collimator_guide_outer_top_mm_np,
        "collimator_guide_inner_bottom_mm_np": collimator_guide_inner_bottom_mm_np,
        "collimator_guide_outer_bottom_mm_np": collimator_guide_outer_bottom_mm_np,
        "collimator_wall_thickness_mm": collimator_wall_thickness_mm,
        "detector_crystal_size_mm": detector_crystal_size_mm,
        "pixel_size_mm": pixel_size_mm,
        "n_pixels": n_pixels,
        "detector_crystal_translation_mm": detector_crystal_translation_mm,
        "azimuthal_angle_deg": azimuthal_angle_deg,
        "polar_angle_deg": polar_angle_deg,
        "shielding_file_path": str(shielding_file_path),
    }


def construct_collimator_box_gate_geometry(config: dict, id: int):
    collimator_body_inner = gate.geometry.volumes.TrdVolume(  # type: ignore
        name=f"CollimatorBody_{id + 1}"
    )
    collimator_body_outer = gate.geometry.volumes.TrdVolume(  # type: ignore
        name=f"CollimatorBody_outer_{id + 1}"
    )
    collimator_body_outer.dx1 = config["collimator_body_outer_top_mm_np"][id] * 0.5
    collimator_body_outer.dy1 = config["collimator_body_outer_top_mm_np"][id] * 0.5
    collimator_body_outer.dx2 = config["collimator_body_outer_bottom_mm_np"][id] * 0.5
    collimator_body_outer.dy2 = config["collimator_body_outer_bottom_mm_np"][id] * 0.5
    collimator_body_outer.dz = config["collimator_body_length_mm_np"][id] * 0.5
    collimator_body_inner.dx1 = config["collimator_body_inner_top_mm_np"][id] * 0.5
    collimator_body_inner.dy1 = config["collimator_body_inner_top_mm_np"][id] * 0.5
    collimator_body_inner.dx2 = config["collimator_body_inner_bottom_mm_np"][id] * 0.5
    collimator_body_inner.dy2 = config["collimator_body_inner_bottom_mm_np"][id] * 0.5
    collimator_body_inner.dz = config["collimator_body_length_mm_np"][id] * 0.5 + 0.1

    # Add a box with 1 mm thickness around the crystal.
    # Use unite_volumes to create the box by subtracting the inner box from the outer box
    # The outer box height is 11 mm and inner box height 10 mm

    collimator_box_outer = gate.geometry.volumes.BoxVolume(  # type: ignore
        name=f"CollimatorBox_outer_{id + 1}"
    )
    collimator_box_inner = gate.geometry.volumes.BoxVolume(  # type: ignore
        name=f"CollimatorBox_inner_{id + 1}"
    )
    box_outer_size_mm = np.full(
        (3,),
        config["collimator_body_outer_top_mm_np"][id]
        + 2.0
        + config["collimator_wall_thickness_mm"] * 2.0,
    )
    # add 2 mm to ensure the box fully covers the crystal
    box_outer_size_mm[2] = config["detector_crystal_size_mm"][2] + 4.0
    box_inner_size_mm = np.full(
        (3,), config["collimator_body_outer_top_mm_np"][id] + 2.0
    )

    box_inner_size_mm[2] = config["detector_crystal_size_mm"][2] + 4.0
    collimator_box_outer.size = box_outer_size_mm
    collimator_box_inner.size = box_inner_size_mm
    collimator_box = subtract_volumes(
        collimator_box_outer,
        collimator_box_inner,
        translation=[0, 0, -2.0],
    )

    # unite the box with the collimator body
    collimator_body_step_1 = unite_volumes(
        collimator_body_outer,
        collimator_box,
        translation=[
            0,
            0,
            -config["collimator_body_length_mm_np"][id] * 0.5
            - box_outer_size_mm[2] * 0.5
            + 2.0,
        ],
        new_name=f"CollimatorBody_step1_{id + 1}",
    )

    collimator_body = subtract_volumes(collimator_body_step_1, collimator_body_inner)
    # Collimator Guide
    collimator_guide_inner = gate.geometry.volumes.TrdVolume(  # type: ignore
        name=f"CollimatorGuide_{id + 1}"
    )
    collimator_guide_outer = gate.geometry.volumes.TrdVolume(  # type: ignore
        name=f"CollimatorGuide_outer_{id + 1}"
    )
    collimator_guide_outer.dx1 = config["collimator_guide_outer_top_mm_np"][id] * 0.5
    collimator_guide_outer.dy1 = config["collimator_guide_outer_top_mm_np"][id] * 0.5
    collimator_guide_outer.dx2 = config["collimator_guide_outer_bottom_mm_np"][id] * 0.5
    collimator_guide_outer.dy2 = config["collimator_guide_outer_bottom_mm_np"][id] * 0.5
    collimator_guide_outer.dz = config["collimator_guide_length_mm_np"][id] * 0.5
    collimator_guide_inner.dx1 = config["collimator_guide_inner_top_mm_np"][id] * 0.5
    collimator_guide_inner.dy1 = config["collimator_guide_inner_top_mm_np"][id] * 0.5
    collimator_guide_inner.dx2 = config["collimator_guide_inner_bottom_mm_np"][id] * 0.5
    collimator_guide_inner.dy2 = config["collimator_guide_inner_bottom_mm_np"][id] * 0.5
    collimator_guide_inner.dz = (
        config["collimator_guide_length_mm_np"][id] * 0.5 + 0.1
    )  # add a small extra length to ensure the subtraction works correctly
    collimator_guide = subtract_volumes(collimator_guide_outer, collimator_guide_inner)

    collimator = unite_volumes(
        collimator_body,
        collimator_guide,
        new_name=f"Collimator_{id + 1}",
        translation=[
            0,
            0,
            config["collimator_body_length_mm_np"][id] * 0.5
            + config["collimator_guide_length_mm_np"][id] * 0.5,
        ],
    )

    return collimator


def get_head_rotation_matrix(config: dict, id: int):

    # First rotate around x axis by 90 degrees
    rx_0 = Rotation.from_euler("x", -90, degrees=True).as_matrix()
    rz_0 = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    # Then rotate around z axis by the azimuthal angle
    rz_1 = Rotation.from_euler(
        "z", config["azimuthal_angle_deg"][id], degrees=True
    ).as_matrix()
    # Then rotate around y axis by the polar angle
    rx_1 = Rotation.from_euler(
        "x", -config["polar_angle_deg"][id], degrees=True
    ).as_matrix()
    r = rz_1 @ rz_0 @ rx_1 @ rx_0
    return r


def add_collimator_to_gate_sim(sim: gate.Simulation, config: dict, id: int):
    collimator = construct_collimator_box_gate_geometry(config, id)
    sim.volume_manager.add_volume(collimator)
    collimator.mother = "world"
    collimator.translation = config["collimator_body_translation_mm"][id]
    r = get_head_rotation_matrix(config, id)
    collimator.rotation = r
    collimator.material = "Tungsten"


def add_crystal_box(sim: gate.Simulation, name: str):
    mm = gate.g4_units.mm
    crystal_box = sim.add_volume("Box", name=name)
    crystal_box.size = [50.5 * mm, 50.5 * mm, 12.0 * mm]  # unit is mm
    crystal_box.material = "Air"
    return crystal_box


def add_pixelated_detector_to_gate_sim(sim: gate.Simulation, config: dict, id: int):
    r = get_head_rotation_matrix(config, id)
    crystal_box = add_crystal_box(sim, name=f"DetectorCrystal_{id + 1}")
    crystal_box.size = config["detector_crystal_size_mm"]
    crystal_box.translation = config["detector_crystal_translation_mm"][id]
    crystal_box.rotation = r

    detector_pixel = sim.add_volume("Box", name=f"pixel_{id + 1}")
    detector_pixel.size = config["pixel_size_mm"]
    detector_pixel.mother = crystal_box.name
    pixel_repeater = gate.geometry.volumes.RepeatParametrisedVolume(
        repeated_volume=detector_pixel
    )
    pixel_repeater.linear_repeat = config["n_pixels"]
    pixel_repeater.translation = config["pixel_size_mm"]
    sim.volume_manager.add_volume(pixel_repeater)
    detector_pixel.material = "CsI"


def add_shielding_to_gate_sim(sim: gate.Simulation, config: dict):

    shielding = gate.geometry.volumes.TesselatedVolume(name="Shielding")
    # Make sure the shielding file path is valid before proceeding
    shielding_file_path = Path(config["shielding_file_path"])
    if not shielding_file_path.exists():
        raise FileNotFoundError(
            f"Shielding STL file not found at: {shielding_file_path}"
        )

    shielding.mother = "world"

    shielding.file_name = Path(config["shielding_file_path"]).as_posix()
    shielding.origin_at_cog = False
    sim.add_volume(shielding)
    rz = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    shielding.rotation = rz
    shielding.material = "Lead"


def _configure_wrl_export(
    sim: gate.Simulation, *, force_phantom_wireframe: bool = False
):
    sim.user_info.visu = True
    sim.user_info.visu_type = "vrml_file_only"

    sim.visu_commands_vrml = ["/vis/open VRML2FILE", "/vis/drawVolume"]
    sim.visu_commands_vrml.append("/vis/geometry/set/visibility world 0 false")

    if force_phantom_wireframe:
        sim.visu_commands_vrml.append(
            "/vis/geometry/set/forceWireframe Jaszczak_Phantom 0 true"
        )


def _apply_debug_geometry_settings(sim: gate.Simulation, args):
    if not args.debug_geometry:
        return

    print(
        "Geometry debug mode enabled: dumping volume tree and enabling verbose G4 output."
    )
    print(f"check_volumes_overlap: {sim.check_volumes_overlap}")
    print(sim.volume_manager.dump_volume_tree())
    sim.g4_verbose = True
    sim.g4_verbose_level = 2


def _finalize_wrl_export(sim: gate.Simulation, visu_filename: Path | str):
    sim.visu_commands_vrml.append("/vis/viewer/flush")
    sim.user_info.visu_filename = str(Path(visu_filename).resolve())
    print(f"Geometry stored in {sim.user_info.visu_filename}")
    sim.run()


def _add_scanner_geometry(
    sim: gate.Simulation, config: dict, *, include_shielding: bool
):
    for i in range(80):
        print(f"Adding geometry for head {i + 1}...")
        add_collimator_to_gate_sim(sim, config, id=i)
        add_pixelated_detector_to_gate_sim(sim, config, id=i)

    if include_shielding:
        add_shielding_to_gate_sim(sim, config)


def _add_phantom_geometry(sim: gate.Simulation, config: dict):
    print("Adding Jaszczak phantom geometry...")
    add_Jaszczak_phantom(sim)


def save_geometry_to_wrl(
    config: dict,
    persist_data_dir: Path,
    args,
    export_target: str = "scanner",
):
    """Export scanner, phantom, or combined geometry to a VRML file."""
    sim = gate.Simulation()
    sim.volume_manager.add_material_database(persist_data_dir / "GateMaterials.db")
    wrl_output_dir = Path(args.output_dir).resolve()
    wrl_output_dir.mkdir(parents=True, exist_ok=True)

    export_target = export_target.lower()
    if export_target not in {"scanner", "phantom", "both"}:
        raise ValueError("export_target must be one of: scanner, phantom, both")

    include_shielding = bool(getattr(args, "with_shielding", False))

    if export_target == "scanner":
        _add_scanner_geometry(sim, config, include_shielding=include_shielding)
        _configure_wrl_export(sim)
        _apply_debug_geometry_settings(sim, args)
        print("Storing scanner geometry to WRL without running the simulation...")
        _finalize_wrl_export(
            sim,
            wrl_output_dir / "dc_spect_geometry_2mm_box.wrl",
        )
        return

    if export_target == "phantom":
        _add_phantom_geometry(sim, config)
    else:
        _add_scanner_geometry(sim, config=config, include_shielding=include_shielding)
        _add_phantom_geometry(sim, config)
    _configure_wrl_export(sim, force_phantom_wireframe=True)

    _apply_debug_geometry_settings(sim, args)

    if export_target == "phantom":
        filename = "jaszczak_phantom_only.wrl"
    else:
        filename = "scanner_and_phantom_geometry.wrl"

    print("Storing geometry into wrl file only without running the simulation...")
    _finalize_wrl_export(sim, wrl_output_dir / filename)


def save_simulation_geometry_to_wrl(config: dict, persist_data_dir: Path, args):
    save_geometry_to_wrl(
        config,
        persist_data_dir,
        args,
        export_target=(
            args.geometry_only if isinstance(args.geometry_only, str) else "scanner"
        ),
    )


def render_wrl_to_html(wrl_path: Path, html_output_path: Path):
    import pyvista as pv

    print(f"Rendering WRL geometry from {wrl_path} to HTML for visualization...")
    wrl_path = Path(wrl_path)
    if not wrl_path.exists():
        raise FileNotFoundError(f"WRL file not found at: {wrl_path}")

    original_solid_names = extract_solid_names_from_wrl(wrl_path)

    detector_mesh = load_wrl_as_mesh(wrl_path)

    print(f"\nTotal SOLID objects in WRL: {len(original_solid_names)}")

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(detector_mesh, color="lightblue", show_edges=True, opacity=0.7)

    plotter.export_html(html_output_path)


def extract_solid_names_from_wrl(wrl_path):
    """Extract SOLID names from WRL file comments"""
    solid_names = []
    try:
        with open(wrl_path, "r") as f:
            for line in f:
                if "#---------- SOLID:" in line:
                    # Extract the name after "SOLID: "
                    name = line.split("#---------- SOLID:")[1].strip()
                    solid_names.append(name)
    except Exception as e:
        print(f"Error reading WRL file: {e}")
    return solid_names


def load_wrl_as_mesh(wrl_path):
    import logging

    import pyvista as pv
    import vtk

    vtk.vtkObject.GlobalWarningDisplayOff()
    logging.disable(logging.CRITICAL)

    importer = vtk.vtkVRMLImporter()
    importer.SetFileName(wrl_path)
    importer.Update()

    append_filter = vtk.vtkAppendPolyData()

    renderer = importer.GetRenderer()
    actors = renderer.GetActors()
    actors.InitTraversal()

    for i in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        if actor and actor.GetMapper():
            poly_data = actor.GetMapper().GetInput()
            if poly_data:
                append_filter.AddInputData(poly_data)

    append_filter.Update()

    mesh = pv.wrap(append_filter.GetOutput())

    return mesh


def add_box_source(
    sim: gate.Simulation, energy_keV: float = 140.0, name: str = "BoxSource", *, args
):

    source = gate.sources.generic.GenericSource(name=name)
    source.particle = "gamma"
    source.energy.type = "mono"
    source.activity = _activity_to_gate_units(args.source_activity)
    source.energy.mono = energy_keV * gate.g4_units.keV
    source.position.type = "box"
    source.position.size = [210, 210, 210]  # unit is mms
    sim.add_source(source, name=name)


def configure_chunked_run_timing(sim: gate.Simulation, args):
    if args.chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be > 0")
    if args.num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")
    activity_value = float(args.source_activity["value"])
    if activity_value <= 0:
        raise ValueError("source activity value must be > 0")

    sec = gate.g4_units.s
    interval_duration = args.chunk_duration_s * sec
    sim.run_timing_intervals = [
        [i * interval_duration, (i + 1) * interval_duration]
        for i in range(args.num_chunks)
    ]

    expected_events_per_chunk_per_thread = (
        _activity_to_bq(args.source_activity) * args.chunk_duration_s
    )
    expected_events_per_chunk = expected_events_per_chunk_per_thread * int(
        args.num_threads
    )
    expected_total_events = expected_events_per_chunk * int(args.num_chunks)

    print(f"Chunk duration (s): {args.chunk_duration_s}")
    print(f"Number of chunks: {args.num_chunks}")
    print(f"Number of threads: {args.num_threads}")
    print(
        "Source activity: "
        f"{args.source_activity['value']} {args.source_activity['unit']}"
    )
    print(
        "Expected primaries/chunk/thread (mean): "
        f"{expected_events_per_chunk_per_thread:.3e}"
    )
    print(
        f"Expected primaries/chunk all threads (mean): {expected_events_per_chunk:.3e}"
    )
    print(f"Expected primaries total all chunks (mean): {expected_total_events:.3e}")

    if expected_events_per_chunk >= args.eventid_warn_threshold:
        print(
            "WARNING: Expected events/chunk is high relative to 32-bit EventID range. "
            "Reduce activity or chunk_duration_s to lower overflow risk."
        )


def add_dc_spect_geometry(sim: gate.Simulation, config: dict):

    # Add a box at the center, and attach the source to it
    # The box is used to define the source position,
    # but is not part of the simulation geometry
    # The size of the box is 220 mm x 220 mm x 220 mm
    source_box = sim.add_volume("Box", name="SourceBox")
    source_box.size = [220, 220, 220]  # unit is mm
    source_box.material = "Air"

    for i in range(80):
        # Add the i_th collimator
        add_collimator_to_gate_sim(sim, config, id=i)
        # Add the i_th pixelated detector crystal
        add_pixelated_detector_to_gate_sim(sim, config, id=i)
    # add_shielding_to_gate_sim(sim, config)  # Add the shielding as an example


def run_simulation(config: dict, persist_data_dir: Path, args):
    output_dir = Path(args.output_dir).resolve()
    print("Resolved output directory: ", output_dir)
    print(
        "Slurm context: "
        f"job_array_id={args.job_array_id}, job_array_task_id={args.job_array_task_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    job_array_id = args.job_array_id
    job_array_task_id = args.job_array_task_id

    unique_seed = generate_unique_seed(str(job_array_id), str(job_array_task_id))
    print(f"Using random seed: {unique_seed}")

    sim = gate.Simulation(progress_bar=True, output_dir=output_dir)
    sim.random_seed = unique_seed
    sim.volume_manager.add_material_database(persist_data_dir / "GateMaterials.db")

    simulation_mode = getattr(args, "mode", "box")
    print(f"Simulation mode: {simulation_mode}")

    # Add Geometry to the simulation.
    # The phantom mode keeps the center source box out of the world volume to avoid overlaps.
    if simulation_mode == "box":
        _add_scanner_geometry(
            sim, config, include_shielding=bool(getattr(args, "with_shielding", False))
        )
    elif simulation_mode == "jaszczak":
        _add_scanner_geometry(
            sim, config, include_shielding=bool(getattr(args, "with_shielding", False))
        )
        _add_phantom_geometry(sim, config)
    else:
        raise ValueError("simulation_mode must be one of: 'box', 'jaszczak'")

    if args.with_shielding:
        print("Simulate with lead shielding = True")
    else:
        print("Simulate with lead shielding = False")

    # Add Source to the simulation
    if simulation_mode == "box":
        add_box_source(sim, energy_keV=140.0, args=args)
    else:
        add_background_source(
            sim,
            phantom_name="Jaszczak_Phantom",
            activity=args.source_activity,
            source_type=getattr(args, "phantom_source_type", "Tc-99m"),
        )

    sim.number_of_threads = int(args.num_threads)
    configure_chunked_run_timing(sim, args)
    # In activity mode, expected event count is stochastic and controlled by
    # activity * run_timing_intervals.
    print(f"Number of threads: {sim.number_of_threads}")

    output_stem = f"a_{job_array_id}_j_{job_array_task_id}"
    add_actors(sim, output_dir, output_stem)
    add_stats_actor(sim, output_dir, output_stem)
    sim.run()


def add_stats_actor(sim: gate.Simulation, output_dir: Path, output_stem: str):
    stats_actor = sim.add_actor("SimulationStatisticsActor", "Stats")  # type: ignore
    stats_path = output_dir / f"{output_stem}_sim_stats.txt"
    # GATE will automatically write to this file after sim.run() finishes
    stats_actor.output_filename = str(stats_path)


def add_actors(sim: gate.Simulation, output_dir: Path, output_stem: str):
    pixel_array_name = [f"pixel_{i + 1}" for i in range(80)]

    # Keep hits in-memory only as input to the singles chain.
    for i in range(80):
        pixel_hits_actor: gate.actors.digitizers.DigitizerHitsCollectionActor = (
            sim.add_actor("DigitizerHitsCollectionActor", f"PixelHits_{i + 1}")
        )
        pixel_hits_actor.attached_to = pixel_array_name[i]
        pixel_hits_actor.output_filename = ""
        pixel_hits_actor.attributes = [
            "RunID",
            # "ThreadID",
            "EventID",
            # "TrackID",
            "TotalEnergyDeposit",
            "PostPosition",
            "PrePosition",
            "EventPosition",
            "GlobalTime",
            "PreStepUniqueVolumeID",
            "PreStepUniqueVolumeIDAsInt",
        ]
        pixel_readout_actor = sim.add_actor(
            "DigitizerReadoutActor", f"Pixel_{i + 1}_Singles"
        )
        pixel_readout_actor.input_digi_collection = pixel_hits_actor.name
        # pixel_readout_actor.group_volume = pixel_array_name[i]
        pixel_readout_actor.discretize_volume = pixel_array_name[i]
        pixel_readout_actor.policy = "EnergyWeightedCentroidPosition"
        pixel_readout_actor.output_filename = (
            output_dir / f"pixel_singles_{output_stem}.root"
        )


def _resolve_xlsx_path(persistent_data_dir: Path, xlsx_path: str | None) -> Path:
    if xlsx_path is not None:
        candidate = Path(xlsx_path)
    else:
        candidate = (
            persistent_data_dir
            / "cardiac_spect"
            / "spreadsheet"
            / "MDSL.excel80M10RFR.cut-plate.010.150roi.2.30pin.105ellipse.xlsx"
        )
    if not candidate.exists():
        raise FileNotFoundError(
            f"Geometry configuration xlsx file not found at {candidate}"
        )
    return candidate


def _resolve_job_array_ids(job_array_id: str | None, job_array_task_id: str | None):
    resolved_job_array_id = (
        job_array_id
        or os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or "local"
    )
    resolved_job_array_task_id = (
        job_array_task_id
        or os.environ.get("SLURM_ARRAY_TASK_ID")
        or os.environ.get("SLURM_PROCID")
        or "0"
    )
    return resolved_job_array_id, resolved_job_array_task_id


def parse_args(args=None):
    """
    Parses command line arguments.
    Accepts an optional list of arguments for easy unit testing.
    """

    parser = argparse.ArgumentParser(description="Simulation Runner")

    parser.add_argument(
        "-s",
        "--source-activity",
        nargs="+",
        default=["1e6", "Bq"],
        metavar=("VALUE", "UNIT"),
        help=(
            "Source activity as a value with an optional unit, for example: "
            "1.2e10 Bq, 10 mCi, 0.01 Ci, or 1e6."
        ),
    )
    parser.add_argument(
        "-d",
        "--chunk-duration-s",
        type=float,
        default=1.0,
        help="Duration of each run chunk in seconds.",
    )
    parser.add_argument(
        "-c",
        "--num-chunks",
        type=int,
        default=1,
        help="Number of run timing intervals (chunks).",
    )
    parser.add_argument(
        "--eventid-warn-threshold",
        type=float,
        default=1.5e9,
        help="Warn if expected events per chunk exceed this threshold.",
    )
    parser.add_argument(
        "--debug-geometry",
        action="store_true",
        help="Dump the geometry tree and enable verbose Geant4 output before running.",
    )
    parser.add_argument(
        "-t", "--num-threads", type=int, default=1, help="Number of threads requested."
    )
    parser.add_argument(
        "--xlsx-path",
        type=str,
        default=None,
        help="Path to the geometry configuration xlsx file",
    )
    parser.add_argument(
        "-g",
        "--geometry-only",
        nargs="?",
        const="scanner",
        default=False,
        type=str,
        choices=["scanner", "phantom", "both"],
        help=(
            "Store geometry to WRL only without running the simulation. "
            "Optionally choose scanner, phantom, or both; default is scanner."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store the simulation output files",
    )
    parser.add_argument(
        "-j",
        "--job-array-id",
        type=str,
        default=None,
        help="SLURM_ARRAY_JOB_ID used for naming output files.",
    )
    parser.add_argument(
        "-k",
        "--job-array-task-id",
        type=str,
        default=None,
        help="SLURM_ARRAY_TASK_ID used for naming output files.",
    )
    parser.add_argument(
        "--with-shielding",
        action="store_true",
        help="Include shielding in the simulation.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="box",
        choices=["box", "jaszczak"],
        help=(
            "Select the simulation setup: box for SRM generation or "
            "jaszczak for phantom acquisition."
        ),
    )
    parser.add_argument(
        "--phantom-source-type",
        type=str,
        default="Tc-99m",
        choices=["Gamma-140", "Tc-99m", "Co-57"],
        help="Radioisotope used for the Jaszczak phantom background source.",
    )
    parsed_args = parser.parse_args(args)
    parsed_args.source_activity = _parse_activity_value_and_unit(
        parsed_args.source_activity
    )
    return parsed_args


def main():
    args = parse_args()

    persistent_data_dir = qmirt.utils.filesystem.search_dir_up(
        "persistent_data", __file__
    )
    resolved_job_array_id, resolved_job_array_task_id = _resolve_job_array_ids(
        args.job_array_id, args.job_array_task_id
    )

    # Update args directly to pass downstream
    args.job_array_id = resolved_job_array_id
    args.job_array_task_id = resolved_job_array_task_id

    if args.geometry_only:
        config = get_dc_spect_geometry_config(
            _resolve_xlsx_path(persistent_data_dir, args.xlsx_path),
            stl_dir=persistent_data_dir / "cardiac_spect" / "stl",
            n_pixels=(1, 1, 1),
        )
        save_geometry_to_wrl(
            config,
            persistent_data_dir,
            args,
            export_target=args.geometry_only,
        )
    else:
        config = get_dc_spect_geometry_config(
            _resolve_xlsx_path(persistent_data_dir, args.xlsx_path),
            stl_dir=persistent_data_dir / "cardiac_spect" / "stl",
        )
        run_simulation(config, persistent_data_dir, args)


if __name__ == "__main__":
    main()
