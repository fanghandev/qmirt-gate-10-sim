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


def _parse_activity_to_bq(
    activity: str | list[str],
) -> float:
    _ACTIVITY_VALUE_RE = re.compile(
        r"^\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?P<unit>[A-Za-zµμ]+)?\s*$"
    )
    if isinstance(activity, list):
        text = " ".join(activity)
    else:
        text = activity

    match = _ACTIVITY_VALUE_RE.match(text)
    if not match:
        raise argparse.ArgumentTypeError(
            "source activity must be a number optionally followed by a unit, such as '1.2e10 Bq', '10 mCi', or '0.01 Ci'"
        )

    value = float(match.group("value"))
    unit = (match.group("unit") or "Bq").replace("µ", "u").replace("μ", "u").lower()
    unit_scale_to_bq = {
        "bq": 1.0,
        "kbq": 1e3,
        "mbq": 1e6,
        "gbq": 1e9,
        "tbq": 1e12,
        "ci": 3.7e10,
        "mci": 3.7e7,
        "uci": 3.7e4,
    }
    if unit not in unit_scale_to_bq:
        raise ValueError(f"Unsupported activity unit: {unit}")
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
    args,
    *,
    phantom_name: str = "Jaszczak_Phantom",
):
    """Add a monoenergetic gamma source confined to the Jaszczak phantom."""
    source_type = str(getattr(args, "source_type", "Gamma-140")).upper()
    if source_type not in {"GAMMA-140", "GAMMA"}:
        raise ValueError(
            "Only monoenergetic gamma sources are supported for the cardiac script."
        )

    source = sim.add_source("GenericSource", "Gamma_Background")
    source.particle = "gamma"
    source.energy.type = "mono"
    source.energy.mono = 140.0 * gate.g4_units.keV

    source.position.type = "cylinder"
    source.position.radius = 10.2 * gate.g4_units.cm
    source.position.dz = 18.6 * gate.g4_units.cm
    source.position.translation = [0, 0, 0]
    source.position.confine = phantom_name

    source_activity_bq = getattr(args, "source_activity_bq", None)
    if source_activity_bq is None and hasattr(args, "source_activity"):
        source_activity_bq = _parse_activity_to_bq(args.source_activity)
    if source_activity_bq is None:
        raise ValueError("A positive source activity is required.")
    source.activity = source_activity_bq * gate.g4_units.Bq
    print(
        f"Background gamma source added to '{phantom_name}' with activity {source.activity:.2e} Bq."
    )
    return source


def add_point_source(
    sim: gate.Simulation, energy_keV: float = 140.0, name: str = "PointSource", *, args
):

    source = gate.sources.generic.GenericSource(name=name)
    source.particle = "gamma"
    source.energy.type = "mono"
    source.activity = _parse_activity_to_bq(args.source_activity) * gate.g4_units.Bq
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


def _make_shell_trd(
    *,
    name: str,
    top_outer_mm: float,
    bottom_outer_mm: float,
    top_inner_mm: float,
    bottom_inner_mm: float,
    length_mm: float,
):
    outer = gate.geometry.volumes.TrdVolume(  # type: ignore
        name=f"{name}_outer"
    )
    inner = gate.geometry.volumes.TrdVolume(  # type: ignore
        name=name
    )
    # Keep the same top/bottom convention as qmirt.utils.simulation.make_gate_shell_trd
    # so the boolean and non-boolean geometries are consistent.
    outer.dx2 = bottom_outer_mm * 0.5
    outer.dy2 = bottom_outer_mm * 0.5
    outer.dx1 = top_outer_mm * 0.5
    outer.dy1 = top_outer_mm * 0.5
    outer.dz = length_mm * 0.5
    inner.dx2 = bottom_inner_mm * 0.5
    inner.dy2 = bottom_inner_mm * 0.5
    inner.dx1 = top_inner_mm * 0.5
    inner.dy1 = top_inner_mm * 0.5
    inner.dz = length_mm * 0.5
    return outer, inner


def _make_shell_box(
    *,
    name: str,
    outer_size_mm: np.ndarray,
    inner_size_mm: np.ndarray,
    inner_translation_mm: list[float] | None = None,
):
    outer = gate.geometry.volumes.BoxVolume(  # type: ignore
        name=f"{name}_outer"
    )
    inner = gate.geometry.volumes.BoxVolume(  # type: ignore
        name=f"{name}_inner"
    )
    outer.size = outer_size_mm
    inner.size = inner_size_mm
    return subtract_volumes(
        outer,
        inner,
        translation=inner_translation_mm or [0, 0, 0],
    )


def construct_collimator_boolean_gate_geometry(config: dict, id: int):
    body_name = f"CollimatorBody_{id + 1}"
    guide_name = f"CollimatorGuide_{id + 1}"

    collimator_body_shell, collimator_body_cavity = _make_shell_trd(
        name=body_name,
        top_outer_mm=config["collimator_body_outer_top_mm_np"][id],
        bottom_outer_mm=config["collimator_body_outer_bottom_mm_np"][id],
        top_inner_mm=config["collimator_body_inner_top_mm_np"][id],
        bottom_inner_mm=config["collimator_body_inner_bottom_mm_np"][id],
        length_mm=config["collimator_body_length_mm_np"][id],
    )

    box_outer_size_mm = np.full(
        (3,),
        config["collimator_body_outer_top_mm_np"][id]
        + 2.0
        + config["collimator_wall_thickness_mm"] * 2.0,
    )
    box_outer_size_mm[2] = config["detector_crystal_size_mm"][2] + 4.0
    box_inner_size_mm = np.full(
        (3,), config["collimator_body_outer_top_mm_np"][id] + 2.0
    )

    box_inner_size_mm[2] = config["detector_crystal_size_mm"][2] + 4.0
    collimator_box_shell = _make_shell_box(
        name=f"CollimatorBox_{id + 1}",
        outer_size_mm=box_outer_size_mm,
        inner_size_mm=box_inner_size_mm,
        inner_translation_mm=[0, 0, 0],
    )

    collimator_body_step_1 = unite_volumes(
        collimator_body_shell,
        collimator_box_shell,
        translation=[
            0,
            0,
            -config["collimator_body_length_mm_np"][id] * 0.5
            - box_outer_size_mm[2] * 0.5,
        ],
        new_name=f"CollimatorBody_step1_{id + 1}",
    )

    collimator_body = subtract_volumes(collimator_body_step_1, collimator_body_cavity)
    collimator_guide_shell, collimator_guide_cavity = _make_shell_trd(
        name=guide_name,
        top_outer_mm=config["collimator_guide_outer_top_mm_np"][id],
        bottom_outer_mm=config["collimator_guide_outer_bottom_mm_np"][id],
        top_inner_mm=config["collimator_guide_inner_top_mm_np"][id],
        bottom_inner_mm=config["collimator_guide_inner_bottom_mm_np"][id],
        length_mm=config["collimator_guide_length_mm_np"][id],
    )
    collimator_guide = subtract_volumes(collimator_guide_shell, collimator_guide_cavity)

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
    # Then rotate around z axis by the azimuthal angle
    rz_1 = Rotation.from_euler(
        "z", config["azimuthal_angle_deg"][id] + 90, degrees=True
    ).as_matrix()
    # Then rotate around x axis by the polar angle using the same convention
    # as qmirt.utils.simulation.get_head_rotation_matrix.
    rx_1 = Rotation.from_euler(
        "x", -config["polar_angle_deg"][id], degrees=True
    ).as_matrix()
    r = rz_1 @ rx_1 @ rx_0
    return r


def add_collimator_to_gate_sim(sim: gate.Simulation, config: dict, id: int):
    collimator = construct_collimator_boolean_gate_geometry(config, id)
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


def _add_scanner_geometry(sim: gate.Simulation, config: dict, *, args):
    for i in range(80):
        print(f"Adding geometry for head {i + 1}...")
        add_collimator_to_gate_sim(sim, config, id=i)
        add_pixelated_detector_to_gate_sim(sim, config, id=i)

    if args.with_shielding:
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
    if export_target == "scanner":
        _add_scanner_geometry(
            sim,
            config,
            args=args,
        )
        _configure_wrl_export(sim)
        add_fov_volume_to_gate_sim(sim, shape=args.fov_shape, size_mm=args.fov_size_mm)
        _apply_debug_geometry_settings(sim, args)
        print("Storing scanner geometry to WRL without running the simulation...")
        _finalize_wrl_export(
            sim,
            wrl_output_dir / "dc_spect_boolean_geometry.wrl",
        )
        return

    if export_target == "phantom":
        _add_phantom_geometry(sim, config)
    else:
        _add_scanner_geometry(sim, config=config, args=args)
        add_fov_volume_to_gate_sim(sim, shape=args.fov_shape, size_mm=args.fov_size_mm)
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


def add_fov_volume_to_gate_sim(
    sim: gate.Simulation, shape: str = "sphere", size_mm: float = 210.0
):
    shape_name = str(shape).lower()
    if shape_name == "box":
        fov_volume = sim.add_volume("Box", name="FOVBox")
        size_value = float(size_mm)
        fov_volume.size = [size_value, size_value, size_value]
        fov_volume.mother = "world"
        fov_volume.material = "Air"
        return fov_volume
    if shape_name == "sphere":
        fov_volume = sim.add_volume("Sphere", name="FOVSphere")
        fov_volume.rmin = 0.0
        fov_volume.rmax = float(size_mm) * 0.5 * gate.g4_units.mm
        fov_volume.mother = "world"
        fov_volume.material = "Air"
        return fov_volume
    raise ValueError(f"Unsupported FOV shape: {shape!r}. Use 'box' or 'sphere'.")


def add_volume_source(
    sim: gate.Simulation,
    energy_keV: float = 140.0,
    name: str = "VolumeSource",
    *,
    args,
    fov_shape: str = "sphere",
    fov_size_mm: float = 210.0,
):
    source = gate.sources.generic.GenericSource(name=name)
    source.particle = "gamma"
    source.energy.type = "mono"
    source_activity_bq = getattr(args, "source_activity_bq", None)
    if source_activity_bq is None and hasattr(args, "source_activity"):
        source_activity_bq = _parse_activity_to_bq(args.source_activity)
    if source_activity_bq is None:
        raise ValueError("A positive source activity is required.")
    source.activity = source_activity_bq * gate.g4_units.Bq
    source.energy.mono = energy_keV * gate.g4_units.keV

    fov_shape_name = str(fov_shape).lower()
    fov_size = float(fov_size_mm)
    if fov_shape_name == "box":
        source.position.type = "box"
        source.position.size = [fov_size, fov_size, fov_size]
        source_obj = sim.add_source(source, name=name)
        source_obj.attached_to = "FOVBox"
        return source_obj
    if fov_shape_name == "sphere":
        source.position.type = "sphere"
        source.position.radius = fov_size * 0.5 * gate.g4_units.mm
        source_obj = sim.add_source(source, name=name)
        source_obj.attached_to = "FOVSphere"
        return source_obj
    raise ValueError(f"Unsupported FOV shape: {fov_shape!r}. Use 'box' or 'sphere'.")


def configure_chunked_run_timing(sim: gate.Simulation, args):
    if args.chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be > 0")
    if args.num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")

    source_activity_bq = getattr(args, "source_activity_bq", None)
    if source_activity_bq is None and hasattr(args, "source_activity"):
        source_activity_bq = _parse_activity_to_bq(args.source_activity)
    if source_activity_bq is None or source_activity_bq <= 0:
        raise ValueError("source activity value must be > 0")

    sec = gate.g4_units.s
    interval_duration = args.chunk_duration_s * sec
    sim.run_timing_intervals = [
        [i * interval_duration, (i + 1) * interval_duration]
        for i in range(args.num_chunks)
    ]

    expected_events_per_chunk_per_thread = source_activity_bq * args.chunk_duration_s
    expected_events_per_chunk = expected_events_per_chunk_per_thread * int(
        args.num_threads
    )
    expected_total_events = expected_events_per_chunk * int(args.num_chunks)

    print(f"Chunk duration (s): {args.chunk_duration_s}")
    print(f"Number of chunks: {args.num_chunks}")
    print(f"Number of threads: {args.num_threads}")
    print(f"Source activity: {source_activity_bq:.3e} Bq")
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

    simulation_mode = getattr(args, "mode", "srm-sim")
    if simulation_mode == "geometry-only":
        raise ValueError(
            "geometry-only is not a simulation mode; use the geometry-only flag instead."
        )
    print(f"Simulation mode: {simulation_mode}")

    # Add Geometry to the simulation.
    if simulation_mode == "srm-sim":
        _add_scanner_geometry(sim, config, args=args)
    elif simulation_mode == "jaszczak":
        _add_scanner_geometry(sim, config, args=args)
        _add_phantom_geometry(sim, config)
    else:
        raise ValueError("simulation_mode must be one of: 'srm-sim' or 'jaszczak'")

    if args.with_shielding:
        print("Simulate with lead shielding = True")
    else:
        print("Simulate with lead shielding = False")

    # Add Source to the simulation
    if simulation_mode == "srm-sim":
        add_fov_volume_to_gate_sim(sim, shape=args.fov_shape, size_mm=args.fov_size_mm)
        add_volume_source(
            sim,
            energy_keV=140.0,
            name="VolumeSource",
            args=args,
            fov_shape=args.fov_shape,
            fov_size_mm=args.fov_size_mm,
        )
    elif simulation_mode == "jaszczak":
        add_background_source(sim, args, phantom_name="Jaszczak_Phantom")

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
        "--source-activity-bq",
        type=float,
        default=1e6,
        help="Source activity in Becquerels.",
    )
    parser.add_argument(
        "--source-activity",
        nargs="+",
        default=None,
        metavar=("VALUE", "UNIT"),
        help=(
            "Legacy source activity in value/unit form, for example: "
            "1.2e10 Bq, 0.01 Ci, or 1e6."
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
        type=int,
        default=1.5e9,
        help="Warn if expected events per chunk exceed this threshold.",
    )
    parser.add_argument(
        "--debug-geometry",
        action="store_true",
        help="Dump the geometry tree and enable verbose Geant4 output before running.",
    )
    parser.add_argument(
        "--execution-environment",
        type=str,
        default="auto",
        choices=["auto", "ospool", "slurm", "local"],
        help=(
            "Select the runtime environment so the script can resolve default job IDs "
            "and thread counts for OSPool, SLURM, or local runs."
        ),
    )
    parser.add_argument(
        "-n", "--num-threads", type=int, default=1, help="Number of threads requested."
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
        default="srm-sim",
        choices=["srm-sim", "geometry-only", "jaszczak"],
        help=(
            "Select the execution mode: 'srm-sim' for the source-in-FOV scan, "
            "'jaszczak' for phantom acquisition, or 'geometry-only' for WRL export."
        ),
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="Gamma",
        choices=["Gamma"],
        help="Only monoenergetic gamma emission is supported for the cardiac script.",
    )
    parser.add_argument(
        "--fov-shape",
        type=str,
        choices=["box", "sphere"],
        default="sphere",
        help="Geometry used for the source FOV region: 'box' or 'sphere'.",
    )
    parser.add_argument(
        "--fov-size-mm",
        type=float,
        default=210.0,
        help="FOV size in mm. For box it's the side length; for sphere it's the radius.",
    )
    parsed_args = parser.parse_args(args)
    return parsed_args


def main():
    args = parse_args()
    if args.source_activity is not None:
        args.source_activity_bq = _parse_activity_to_bq(args.source_activity)

    persistent_data_dir = qmirt.utils.filesystem.search_dir_up(
        "persistent_data", __file__
    )
    resolved_job_array_id, resolved_job_array_task_id = _resolve_job_array_ids(
        args.job_array_id, args.job_array_task_id
    )

    args.job_array_id = resolved_job_array_id
    args.job_array_task_id = resolved_job_array_task_id

    if args.mode == "geometry-only" or args.geometry_only:
        config = get_dc_spect_geometry_config(
            _resolve_xlsx_path(persistent_data_dir, args.xlsx_path),
            stl_dir=persistent_data_dir / "cardiac_spect" / "stl",
            n_pixels=(1, 1, 1),
        )
        save_geometry_to_wrl(
            config,
            persistent_data_dir,
            args,
            export_target=(args.geometry_only if args.geometry_only else "scanner"),
        )
        return

    config = get_dc_spect_geometry_config(
        _resolve_xlsx_path(persistent_data_dir, args.xlsx_path),
        stl_dir=persistent_data_dir / "cardiac_spect" / "stl",
    )
    if args.mode not in {"srm-sim", "jaszczak"}:
        raise ValueError(f"Unsupported mode: {args.mode!r}")
    run_simulation(config, persistent_data_dir, args)


if __name__ == "__main__":
    main()
