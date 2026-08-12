"""Shared simulation helpers for GATE scripts."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import opengate as gate
from opengate.geometry.volumes import BoxVolume, RepeatParametrisedVolume, TrdVolume
from opengate.sources.generic import GenericSource
from scipy.spatial.transform import Rotation


def _first_env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def parse_activity_to_bq(activity: str | list[str]) -> float:
    activity_value_re = re.compile(
        r"^\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?P<unit>[A-Za-zµμ]+)?\s*$"
    )
    text = " ".join(activity) if isinstance(activity, list) else activity
    match = activity_value_re.match(text)
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


def generate_unique_seed(job_array_id: str, job_array_task_id: str) -> int:
    seed_string = f"gate_sim_{job_array_id}_{job_array_task_id}"
    seed_string += f"_{os.times()}"
    return int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)


def detect_execution_environment(explicit_environment: str | None = None) -> str:
    if explicit_environment and explicit_environment != "auto":
        return explicit_environment

    if _first_env_value("SLURM_ARRAY_JOB_ID", "SLURM_JOB_ID"):
        return "slurm"

    if _first_env_value(
        "_CONDOR_JOB_AD",
        "_CONDOR_MACHINE_AD",
        "_CONDOR_SLOT",
        "CONDOR_CLUSTER_ID",
        "CONDOR_PROC_ID",
        "CLUSTER_ID",
        "PROC_ID",
        "JOB_CLUSTER_ID",
        "JOB_PROC_ID",
        "ClusterId",
        "ProcId",
    ):
        return "ospool"

    return "local"


def resolve_job_and_task_ids(
    job_array_id: str | None,
    job_array_task_id: str | None,
) -> tuple[str, str]:
    resolved_job_array_id = (
        job_array_id
        or _first_env_value(
            "SLURM_ARRAY_JOB_ID",
            "SLURM_JOB_ID",
            "CONDOR_CLUSTER_ID",
            "CLUSTER_ID",
            "JOB_CLUSTER_ID",
            "ClusterId",
        )
        or "local"
    )
    resolved_job_array_task_id = (
        job_array_task_id
        or _first_env_value(
            "SLURM_ARRAY_TASK_ID",
            "SLURM_PROCID",
            "CONDOR_PROC_ID",
            "PROC_ID",
            "JOB_PROC_ID",
            "ProcId",
        )
        or "0"
    )
    return resolved_job_array_id, resolved_job_array_task_id


def resolve_num_threads(
    requested_num_threads: int | None,
    execution_environment: str,
    *,
    local_thread_cap: int = 128,
) -> int:
    if execution_environment == "ospool":
        return 1

    if requested_num_threads is not None:
        return int(requested_num_threads)

    if execution_environment == "slurm":
        slurm_cpus_per_task = _first_env_value("SLURM_CPUS_PER_TASK")
        if slurm_cpus_per_task:
            try:
                value = int(slurm_cpus_per_task)
                if value > 0:
                    return value
            except ValueError:
                pass

        slurm_ntasks = _first_env_value("SLURM_NTASKS")
        if slurm_ntasks:
            try:
                value = int(slurm_ntasks)
                if value > 0:
                    return value
            except ValueError:
                pass

        return 1

    cpu_count = os.cpu_count() or 1
    return max(1, min(local_thread_cap, cpu_count))


def resolve_simulation_runtime_context(
    job_array_id: str | None,
    job_array_task_id: str | None,
    num_threads: int | None,
    execution_environment: str | None = None,
    *,
    local_thread_cap: int = 128,
) -> tuple[str, str, int, str]:
    resolved_execution_environment = detect_execution_environment(execution_environment)
    resolved_job_array_id, resolved_job_array_task_id = resolve_job_and_task_ids(
        job_array_id,
        job_array_task_id,
    )
    resolved_num_threads = resolve_num_threads(
        num_threads,
        resolved_execution_environment,
        local_thread_cap=local_thread_cap,
    )
    return (
        resolved_job_array_id,
        resolved_job_array_task_id,
        resolved_num_threads,
        resolved_execution_environment,
    )


# Generate a triangular mesh array of cold rods within a 60-degree sector
def _add_rod_sector(sim, mother_name, sector_index, rod_radius_mm, spacing_mm):
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
    if phantom is None:
        raise RuntimeError("Failed to create Jaszczak_Phantom volume.")
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
        if sph is None:
            raise RuntimeError(f"Failed to create ColdSphere_{i} volume.")
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
        _add_rod_sector(sim, "Jaszczak_Phantom", sector, r, spacing)


def add_background_source(
    sim: gate.Simulation,
    args,
    *,
    phantom_name: str = "Jaszczak_Phantom",
):
    """
    Adds a background radioactive source to the specified phantom volume.
    Leverages GATE's 'confine' feature combined with the CSG mother-daughter
    hierarchy to automatically exclude radioactive emissions from the cold rods and spheres.

    Args:
        sim: The opengate simulation object.
        phantom_name: Name of the mother volume (water cylinder).
        args: Command-line arguments containing source type and activity.
    """
    # ========================================================
    # Add Background Radioactive Source
    # ========================================================
    source_type = args.source_type
    source = GenericSource(name=f"{source_type}_Background")
    sim.add_source(source, name=source.name)
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
        # Depending on the specific opengate-python version, excitation energy for isomers
        # is typically passed via the ion property or appending to the string.
        # e.g., source.particle = 'ion 43 99 0 142.6836' (Z, A, Q, E_ex in keV)
        source.particle = "ion 43 99 0 142.6836"

    elif source_type.upper() == "CO-57":
        # Cobalt-57 full radioactive decay (Z=27, A=57)
        # Includes the 122 keV and 136 keV gammas, plus Fe X-rays.
        source.particle = "ion 27 57"

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
    source.activity = (
        parse_activity_to_bq(args.source_activity) * gate.g4_units.Bq
    )  # Convert to Bq for GATE
    print(
        f"Background source '{source_type}' added to '{phantom_name}' with activity {source.activity:.2e} Bq."
    )
    return source


def add_point_source(
    sim: gate.Simulation, energy_keV: float = 140.0, name: str = "PointSource", *, args
):

    source = GenericSource(name=name)
    source.particle = "gamma"
    source.energy.type = "mono"
    source.activity = parse_activity_to_bq(args.source_activity) * gate.g4_units.Bq
    source.energy.mono = energy_keV * gate.g4_units.keV
    source.position.type = "point"
    source.position.point = [0, 0, 0]  # unit is mm
    sim.add_source(source, name=name)


def add_box_source(
    sim: gate.Simulation, energy_keV: float = 140.0, name: str = "BoxSource", *, args
):

    source = GenericSource(name=name)
    source.particle = "gamma"
    source.energy.type = "mono"
    source.activity = parse_activity_to_bq(args.source_activity) * gate.g4_units.Bq
    source.energy.mono = energy_keV * gate.g4_units.keV
    source.position.type = "box"
    source.position.size = [210, 210, 210]  # unit is mms
    sim.add_source(source, name=name)


def configure_chunked_run_timing(sim: gate.Simulation, args):
    if args.chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be > 0")
    if args.num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")
    source_activity_bq = parse_activity_to_bq(args.source_activity)
    if source_activity_bq <= 0:
        raise ValueError("source activity value must be > 0")

    sec = gate.g4_units.s
    interval_duration = args.chunk_duration_s * sec
    sim.run_timing_intervals = [
        [i * interval_duration, (i + 1) * interval_duration]
        for i in range(args.num_chunks)
    ]

    # Runtime policies vary by environment; OSPool stays single-threaded while
    # SLURM and local runs can use more threads when requested or available.
    expected_events_per_chunk_per_thread = source_activity_bq * args.chunk_duration_s
    expected_events_per_chunk = expected_events_per_chunk_per_thread * int(
        args.num_threads
    )
    expected_total_events = expected_events_per_chunk * int(args.num_chunks)

    print(f"Chunk duration (s): {args.chunk_duration_s}")
    print(f"Number of chunks: {args.num_chunks}")
    print(f"Number of threads: {args.num_threads}")
    print(f"Source activity: {source_activity_bq:.3e} Bq ({args.source_activity})")
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
        "x", config["polar_angle_deg"][id] + 180, degrees=True
    ).as_matrix()
    r = rz_1 @ rz_0 @ rx_1 @ rx_0
    return r


id = 0


def make_gate_shell_trd(
    sim: gate.Simulation,
    *,
    name: str,
    mother: str,
    top_inner_mm: float,
    top_outer_mm: float,
    bottom_inner_mm: float,
    bottom_outer_mm: float,
    length_mm: float,
    translation_mm: list[float],
    material: str,
    inner_material: str = "Air",
):
    shell = TrdVolume(
        name=f"{name}_outer",
        mother=mother,
        dx1=bottom_outer_mm * 0.5,
        dy1=bottom_outer_mm * 0.5,
        dx2=top_outer_mm * 0.5,
        dy2=top_outer_mm * 0.5,
        dz=length_mm * 0.5,
        translation=translation_mm,
        material=material,
    )
    sim.add_volume(shell, name=shell.name)

    cavity = TrdVolume(
        name=f"{name}_cavity",
        mother=shell.name,
        dx1=bottom_inner_mm * 0.5,
        dy1=bottom_inner_mm * 0.5,
        dx2=top_inner_mm * 0.5,
        dy2=top_inner_mm * 0.5,
        dz=length_mm * 0.5,
        translation=[0, 0, 0],
        material=inner_material,
    )
    sim.add_volume(cavity, name=cavity.name)
    return shell, cavity


def make_gate_shell_box(
    sim: gate.Simulation,
    *,
    name: str,
    mother: str,
    outer_size_mm: np.ndarray,
    inner_size_mm: np.ndarray,
    spacing_size_mm: np.ndarray,
    translation_mm: list[float] | np.ndarray,
    inner_shift_mm: list[float] | np.ndarray = [0, 0, 0],
    material: str,
    inner_material: str = "Air",
):
    shell = BoxVolume(
        name=f"{name}_outer",
        mother=mother,
        size=outer_size_mm,
        translation=translation_mm,
        material=material,
    )
    sim.add_volume(shell, name=shell.name)
    opening_box_size = np.array(
        [
            inner_size_mm[0],
            inner_size_mm[1],
            outer_size_mm[2] - inner_size_mm[2] - spacing_size_mm[2],
        ]
    )
    opening_box = BoxVolume(
        name=f"{name}_opening",
        mother=shell.name,
        size=opening_box_size,
        translation=[0, 0, (inner_size_mm[2] + spacing_size_mm[2]) * 0.5],
        material=inner_material,
    )
    sim.add_volume(opening_box, name=opening_box.name)
    cavity = BoxVolume(
        name=f"{name}_cavity",
        mother=shell.name,
        size=inner_size_mm,
        translation=inner_shift_mm,
        material=inner_material,
    )
    sim.add_volume(cavity, name=cavity.name)
    return shell, cavity


def add_crystal_box(sim: gate.Simulation, *, mother: str, name: str):
    mm = gate.g4_units.mm
    crystal_box = BoxVolume(
        name=name,
        mother=mother,
        size=[50.05 * mm, 50.05 * mm, 12.0 * mm],
        material="Air",
    )
    sim.add_volume(crystal_box, name=crystal_box.name)
    return crystal_box


def add_pixelated_detector_to_gate_sim(
    sim: gate.Simulation,
    *,
    mother: str,
    id: int,
    translation_mm: list[float] | np.ndarray,
    pixel_size_mm: np.ndarray,
    pixel_count: np.ndarray,
):

    detector_pixel = BoxVolume(
        name=f"DetectorPixel_{id}",
        mother=mother,
        size=list(pixel_size_mm),
        translation=translation_mm,
        material="CsI",
    )
    sim.add_volume(detector_pixel, name=detector_pixel.name)

    pixel_repeater = RepeatParametrisedVolume(repeated_volume=detector_pixel)
    pixel_repeater.linear_repeat = list(pixel_count)
    pixel_repeater.translation = list(pixel_size_mm)
    sim.volume_manager.add_volume(pixel_repeater)
    pixel_repeater.material = "CsI"


def configure_wrl_export(
    sim: gate.Simulation,
):
    sim.user_info.visu = True
    sim.user_info.visu_type = "vrml_file_only"

    sim.visu_commands_vrml = ["/vis/open VRML2FILE", "/vis/drawVolume"]
    sim.visu_commands_vrml.append("/vis/geometry/set/visibility world 0 false")


def apply_wrl_export(
    sim: gate.Simulation,
    *,
    visu_filename: Path | str,
    debug_geometry: bool = False,
    force_phantom_wireframe: bool = False,
):

    if force_phantom_wireframe:
        sim.visu_commands_vrml.append(
            "/vis/geometry/set/forceWireframe Jaszczak_Phantom 0 true"
        )

    if not debug_geometry:
        return

    print(
        "Geometry debug mode enabled: dumping volume tree and enabling verbose G4 output."
    )
    print(f"check_volumes_overlap: {sim.check_volumes_overlap}")
    print(sim.volume_manager.dump_volume_tree())
    sim.g4_verbose = True
    sim.g4_verbose_level = 2
    sim.visu_commands_vrml.append("/vis/viewer/flush")
    sim.user_info.visu_filename = str(Path(visu_filename).resolve())
    sim.run(start_new_process=True)


def create_pyramid_detector_module(
    sim: gate.Simulation,
    *,
    id: int,
    geometry_config_df,
    crystal_size_mm: np.ndarray = np.array([50.0, 50.0, 10.0]),
    pixel_count: np.ndarray = np.array([1, 1, 1]),
):
    body, _ = make_gate_shell_trd(
        sim,
        name="CollimatorBody_" + str(id),
        mother="world",
        top_inner_mm=geometry_config_df["body_inner_top_corrected (mm)"][id],
        top_outer_mm=geometry_config_df["body_outer_top (mm)"][id],
        bottom_inner_mm=geometry_config_df["body_inner_bottom (mm)"][id],
        bottom_outer_mm=geometry_config_df["body_outer_bottom (mm)"][id],
        length_mm=geometry_config_df["body_l_corrected (mm)"][id],
        translation_mm=(geometry_config_df["body_l_corrected (mm)"][id] * 0.5)
        * np.array([0, 0, 1]),
        material="Tungsten",
        inner_material="Air",
    )

    box_shell, box_cavity = make_gate_shell_box(
        sim,
        name="ShieldingBox_" + str(id),
        mother="world",
        outer_size_mm=np.array(
            [
                geometry_config_df["box_outer_size_x (mm)"][id],
                geometry_config_df["box_outer_size_y (mm)"][id],
                geometry_config_df["box_outer_size_z (mm)"][id],
            ]
        ),
        inner_size_mm=np.array(
            [
                geometry_config_df["box_inner_size_x (mm)"][id],
                geometry_config_df["box_inner_size_y (mm)"][id],
                geometry_config_df["box_inner_size_z (mm)"][id],
            ]
        ),
        spacing_size_mm=np.array([0, 0, geometry_config_df["wall_thickness (mm)"][id]]),
        translation_mm=np.array(
            [
                0,
                0,
                geometry_config_df["body_l_corrected (mm)"][id]
                + geometry_config_df["box_outer_size_z (mm)"][id] * 0.5,
            ]
        ),
        inner_shift_mm=np.array(
            [
                0,
                0,
                -(
                    geometry_config_df["box_outer_size_z (mm)"][id]
                    - geometry_config_df["box_inner_size_z (mm)"][id]
                )
                * 0.5
                + geometry_config_df["wall_thickness (mm)"][id],
            ]
        ),
        material="Tungsten",
        inner_material="Air",
    )

    add_pixelated_detector_to_gate_sim(
        sim,
        mother=box_cavity.name,
        id=id,
        translation_mm=np.array(
            [
                0,
                0,
                (crystal_size_mm[2] - geometry_config_df["box_inner_size_z (mm)"][id]),
            ]
        ),
        pixel_count=pixel_count,
        pixel_size_mm=crystal_size_mm / pixel_count,
    )
    frustum_cavity = TrdVolume(
        name=f"box_f_cavity_{id}",
        mother=box_shell.name,
        dx1=geometry_config_df["body_inner_top_corrected (mm)"][id] * 0.5,
        dy1=geometry_config_df["body_inner_top_corrected (mm)"][id] * 0.5,
        dx2=geometry_config_df["body_inner_top (mm)"][id] * 0.5,
        dy2=geometry_config_df["body_inner_top (mm)"][id] * 0.5,
        dz=geometry_config_df["wall_thickness (mm)"][id] * 0.5,
        translation=[
            0,
            0,
            (
                geometry_config_df["wall_thickness (mm)"][id]
                - geometry_config_df["box_outer_size_z (mm)"][id]
            )
            * 0.5,
        ],
        material="Air",
    )

    sim.add_volume(frustum_cavity, name=frustum_cavity.name)

    guide, _ = make_gate_shell_trd(
        sim,
        name="CollimatorGuide_" + str(id),
        mother="world",
        top_inner_mm=geometry_config_df["guide_inner_top (mm)"][id],
        top_outer_mm=geometry_config_df["guide_outer_top (mm)"][id],
        bottom_inner_mm=geometry_config_df["guide_inner_bottom (mm)"][id],
        bottom_outer_mm=geometry_config_df["guide_outer_bottom (mm)"][id],
        length_mm=geometry_config_df["guide_l (mm)"][id],
        translation_mm=-geometry_config_df["guide_l (mm)"][id]
        / 2
        * np.array([0, 0, 1]),
        material="Tungsten",
        inner_material="Air",
    )
    return guide
