import argparse
import os
from pathlib import Path

import numpy as np
import opengate as gate
import polars as pl
from opengate.geometry.volumes import (
    RepeatParametrisedVolume,
    TesselatedVolume,
)
from qmirt.utils import (
    generate_unique_seed,
    parse_activity_to_bq,
    resolve_simulation_runtime_context,
)
from scipy.spatial.transform import Rotation

import qmirt


def get_geometry_base_definition(id: int = 0):
    data_dir = (
        qmirt.utils.filesystem.search_dir_up("persistent_data", __file__)
        / "brain_spect"
    )
    stl_dir = data_dir / "stl"
    stl_filename = "BrainFrame.008.Lead_Shield.STL"
    w_pinhole_array = np.array([0.556, 0.797, 1.215, 2.007])
    h_nozzle_array = np.array([5.04, 5.02, 5.00, 4.96])
    l_top_array = np.array(
        [
            10.74,
            11.00,
            11.46,
            12.33,
        ]
    )

    collimator_definition = {
        "l_top": l_top_array[id],
        "h_nozzle": h_nozzle_array[id],
        "w_pinhole": w_pinhole_array[id],
        "w_wall": 2.03,
        "l_bottom_inner": 50.0,
        "l_bottom_outer": 56.064,
        "h_body": 25.0,
        "h_box": 23.5,
    }
    crystal_definition = {"size_mm": [50.0, 50.0, 10.0], "n_pixels": [25, 25, 1]}
    geometry_base_definition = {
        "collimator definition": collimator_definition,
        "crystal definition": crystal_definition,
        "shielding file path": str(stl_dir / stl_filename),
    }
    return geometry_base_definition


def get_geometry_definitions():
    data_dir = (
        qmirt.utils.filesystem.search_dir_up("persistent_data", __file__)
        / "brain_spect"
    )
    csv_dir = data_dir / "csv"
    csv_filename = "BrainSPECT_Point_Cloud.007.25mmx0.556mm_pinhole.csv"

    csv_pl_df = pl.read_csv(csv_dir / csv_filename)
    csv_pl_df = csv_pl_df.with_columns(
        Pinhole_y=pl.col("Pinhole_z"),
        Pinhole_z=pl.col("Pinhole_y"),
        Crystal_y=pl.col("Crystal_z"),
        Crystal_z=pl.col("Crystal_y"),
    )
    elevation = np.arctan2(
        csv_pl_df["Pinhole_z"],
        np.sqrt(csv_pl_df["Pinhole_x"] ** 2 + csv_pl_df["Pinhole_y"] ** 2),
    )

    # convert azimuthal angle to 0 to 2pi range
    azimuth = (
        np.arctan2(csv_pl_df["Pinhole_y"], csv_pl_df["Pinhole_x"]) + 2 * np.pi
    ) % (2 * np.pi)
    csv_pl_df = csv_pl_df.with_columns(
        pl.Series("elevation", elevation), pl.Series("azimuth", azimuth)
    )
    # Round elevation and azimuth to 2 decimal places
    csv_pl_df = csv_pl_df.with_columns(
        pl.col("elevation").round(6), pl.col("azimuth").round(6)
    )
    azimuth_minus_half_pi = np.array(csv_pl_df["azimuth"]) - 0.5 * np.pi
    azimuth_minus_half_pi = np.where(
        azimuth_minus_half_pi < 0,
        azimuth_minus_half_pi + 2 * np.pi,
        azimuth_minus_half_pi,
    )
    csv_pl_df = csv_pl_df.with_columns(
        pl.Series("azimuth_minus_half_pi", azimuth_minus_half_pi)
    )
    csv_pl_df = csv_pl_df.sort(["elevation", "azimuth_minus_half_pi"])

    crystal_center_r = 179.61
    corrected_crystal_x = (
        crystal_center_r * np.cos(csv_pl_df["elevation"]) * np.cos(csv_pl_df["azimuth"])
    )
    corrected_crystal_y = (
        crystal_center_r * np.cos(csv_pl_df["elevation"]) * np.sin(csv_pl_df["azimuth"])
    )
    corrected_crystal_z = crystal_center_r * np.sin(csv_pl_df["elevation"])
    geometry_transformation_dataframe = csv_pl_df.with_columns(
        pl.Series("Crystal_x", corrected_crystal_x),
        pl.Series("Crystal_y", corrected_crystal_y),
        pl.Series("Crystal_z", corrected_crystal_z),
    )

    return geometry_transformation_dataframe


def _make_shell_trd(
    sim: gate.Simulation,
    *,
    name: str,
    mother: str,
    top_mm: float,
    bottom_mm: float,
    length_mm: float,
    translation_mm: list[float],
    material: str,
    inner_material: str = "Air",
    inner_length_margin_mm: float = 0.1,
):
    shell = sim.add_volume("TrdVolume", name=f"{name}_outer")
    shell.mother = mother
    shell.dx1 = top_mm * 0.5
    shell.dy1 = top_mm * 0.5
    shell.dx2 = bottom_mm * 0.5
    shell.dy2 = bottom_mm * 0.5
    shell.dz = length_mm * 0.5
    shell.translation = translation_mm
    shell.material = material

    cavity = sim.add_volume("TrdVolume", name=f"{name}_cavity")
    cavity.mother = shell.name
    cavity.dx1 = top_mm * 0.5
    cavity.dy1 = top_mm * 0.5
    cavity.dx2 = bottom_mm * 0.5
    cavity.dy2 = bottom_mm * 0.5
    cavity.dz = length_mm * 0.5 + inner_length_margin_mm
    cavity.translation = [0.0, 0.0, 0.0]
    cavity.material = inner_material
    return shell


def _make_shell_box(
    sim: gate.Simulation,
    *,
    name: str,
    mother: str,
    outer_size_mm: list[float] | np.ndarray,
    inner_size_mm: list[float] | np.ndarray,
    translation_mm: list[float],
    material: str,
    inner_material: str = "Air",
):
    shell = sim.add_volume("Box", name=f"{name}_outer")
    shell.mother = mother
    shell.size = outer_size_mm
    shell.translation = translation_mm
    shell.material = material

    cavity = sim.add_volume("Box", name=f"{name}_cavity")
    cavity.mother = shell.name
    cavity.size = inner_size_mm
    cavity.translation = [0.0, 0.0, 0.0]
    cavity.material = inner_material
    return shell


def construct_collimator_no_boolean_gate_geometry(
    sim: gate.Simulation, config: dict, id: int
):
    collimator_definition = config["collimator definition"]

    box_outer = _make_shell_box(
        sim,
        name=f"CollimatorBox_{id + 1}",
        mother="world",
        outer_size_mm=[
            collimator_definition["l_bottom_outer"],
            collimator_definition["l_bottom_outer"],
            collimator_definition["h_box"],
        ],
        inner_size_mm=[
            collimator_definition["l_bottom_outer"]
            - 2.0 * collimator_definition["w_wall"],
            collimator_definition["l_bottom_outer"]
            - 2.0 * collimator_definition["w_wall"],
            collimator_definition["h_box"] + 2.0,
        ],
        translation_mm=[
            0.0,
            0.0,
            -0.5
            * (collimator_definition["h_nozzle"] + collimator_definition["h_body"]),
        ],
        material="Tungsten",
    )

    frustum_outer = _make_shell_trd(
        sim,
        name=f"CollimatorFrustum_{id + 1}",
        mother="world",
        top_mm=collimator_definition["l_bottom_outer"],
        bottom_mm=collimator_definition["l_top"],
        length_mm=collimator_definition["h_nozzle"] + collimator_definition["h_body"],
        translation_mm=[0.0, 0.0, 0.5 * collimator_definition["h_box"]],
        material="Tungsten",
    )

    return [box_outer, frustum_outer]


def construct_collimator_geometry(sim: gate.Simulation, config: dict, id: int):
    return construct_collimator_no_boolean_gate_geometry(sim, config, id)


def get_head_rotation_matrix(pl_df: pl.DataFrame, id: int):
    azimuth = pl_df.item(id, "azimuth")
    elevation = pl_df.item(id, "elevation")
    # 1. Define the initial base rotations (in degrees)
    r_base_x = Rotation.from_euler("x", -90, degrees=True)
    r_base_z = Rotation.from_euler("z", 90, degrees=True)
    # 2. Define the azimuth and elevation rotations (in degrees)
    r_dyn_z = Rotation.from_euler("z", azimuth, degrees=False)
    r_dyn_x = Rotation.from_euler("x", -elevation, degrees=False)
    r_total = r_dyn_z * r_base_z * r_dyn_x * r_base_x
    # Return the final resulting matrix
    return r_total.as_matrix()


def add_collimator_to_gate_sim(
    sim: gate.Simulation, config: dict, pl_df: pl.DataFrame, id: int
):
    collimator_volumes = construct_collimator_geometry(sim, config, id)
    for volume in collimator_volumes:
        sim.volume_manager.add_volume(volume)

    collimator = collimator_volumes[0]
    collimator.mother = "world"

    # Extract the pinhole locations directly from the DataFrame
    px = pl_df.item(id, "Pinhole_x")
    py = pl_df.item(id, "Pinhole_y")
    pz = pl_df.item(id, "Pinhole_z")

    # Assign the translation as a standard 3-element list
    collimator.translation = [px, py, pz]

    r = get_head_rotation_matrix(pl_df, id)
    collimator.rotation = r

    # 3. Define the local offset of the inherent center relative to the pinhole
    z_offset = (
        config["collimator definition"]["h_nozzle"]
        - config["collimator definition"]["h_body"]
    ) * 0.5
    local_offset_vector = np.array([0.0, 0.0, z_offset])

    # 4. Rotate the local offset into global space
    global_offset_vector = r @ local_offset_vector

    # 5. Apply the final corrected translation
    collimator.translation = [
        px + global_offset_vector[0],
        py + global_offset_vector[1],
        pz + global_offset_vector[2],
    ]
    collimator.material = "Tungsten"


def add_crystal_box(sim: gate.Simulation, name: str):
    mm = gate.g4_units.mm
    crystal_box = sim.add_volume("Box", name=name)
    crystal_box.size = [50.5 * mm, 50.5 * mm, 12.0 * mm]  # unit is mm
    crystal_box.material = "Air"
    return crystal_box


def add_pixelated_detector_to_gate_sim(
    sim: gate.Simulation, config: dict, pl_df: pl.DataFrame, id: int
):
    r = get_head_rotation_matrix(pl_df, id)
    crystal_box = add_crystal_box(sim, name=f"DetectorCrystal_{id + 1}")
    crystal_box.size = config["crystal definition"]["size_mm"]
    px = pl_df.item(id, "Crystal_x")
    py = pl_df.item(id, "Crystal_y")
    pz = pl_df.item(id, "Crystal_z")
    crystal_box.translation = [px, py, pz]
    crystal_box.rotation = r

    n_pixels = config["crystal definition"]["n_pixels"]
    pixel_size_mm = np.array(config["crystal definition"]["size_mm"]) / np.array(
        n_pixels
    )
    config["crystal definition"]["pixel_size_mm"] = pixel_size_mm.tolist()
    detector_pixel = sim.add_volume("Box", name=f"pixel_{id + 1}")
    detector_pixel.size = pixel_size_mm
    detector_pixel.mother = crystal_box.name
    pixel_repeater = RepeatParametrisedVolume(repeated_volume=detector_pixel)
    pixel_repeater.linear_repeat = n_pixels
    pixel_repeater.translation = pixel_size_mm
    sim.volume_manager.add_volume(pixel_repeater)
    detector_pixel.material = "CsI"


def add_shielding_to_gate_sim(sim: gate.Simulation, config: dict):

    shielding = TesselatedVolume(name="Shielding")
    # Make sure the shielding file path is valid before proceeding
    shielding_file_path = Path(config["shielding file path"])
    if not shielding_file_path.exists():
        raise FileNotFoundError(
            f"Shielding STL file not found at: {shielding_file_path}"
        )

    shielding.mother = "world"

    shielding.file_name = Path(config["shielding file path"]).as_posix()
    shielding.origin_at_cog = False
    sim.add_volume(shielding)
    rx = Rotation.from_euler("x", -90, degrees=True).as_matrix()
    rz = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    shielding.rotation = rx @ rz
    shielding.material = "Lead"


def map_crystal_id(id: int, n_crystals: int, mode: str) -> int:
    """
    Maps the crystal ID to a new ID based on the number of crystals.
    This function can be customized to implement any specific mapping logic mode.
    The goal is to map the crystal ID to 0,1,2,3, because we selected 4 collimator
    geometry parameter sets.

    Args:
        id: The original crystal ID.
        n_crystals: The total number of crystals.
        mode: The mapping mode to use. Can be 'sequential', 'reverse', 'random'

    Returns:
        int: The new mapped crystal ID.
    """
    mapped_id = 0
    match mode:
        case "sequential":
            mapped_id = id % 4
        case "reverse":
            mapped_id = (n_crystals - 1 - id) % 4
        case "random":
            mapped_id = np.random.randint(0, 4)
    return mapped_id


def add_geometry_to_gate_sim(sim: gate.Simulation, pl_df: pl.DataFrame, args):
    n_crystals = pl_df.shape[0]
    for id in range(n_crystals):
        mapped_id = map_crystal_id(id, n_crystals, args.mapping_mode)
        config = get_geometry_base_definition(mapped_id)
        if args.geometry_only:
            config["crystal definition"]["n_pixels"] = [1, 1, 1]
        add_collimator_to_gate_sim(sim, config, pl_df, id)
        add_pixelated_detector_to_gate_sim(sim, config, pl_df, id)
    if getattr(args, "with_shielding", False):
        add_shielding_to_gate_sim(sim, config)
    add_fov_volume_to_gate_sim(sim, shape=args.fov_shape, size_mm=args.fov_size_mm)


def _apply_debug_geometry_settings(sim: gate.Simulation, args):
    if not getattr(args, "debug_geometry", False):
        return

    print(
        "Geometry debug mode enabled: dumping volume tree and enabling verbose G4 output."
    )
    print(f"check_volumes_overlap: {sim.check_volumes_overlap}")
    print(sim.volume_manager.dump_volume_tree())
    sim.g4_verbose = True
    sim.g4_verbose_level = 2


def run_simulation_with_geometry_only(args):

    output_dir = Path(args.output_dir).resolve()
    print("Output directory: ", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = gate.Simulation()
    persist_data_dir = qmirt.utils.filesystem.search_dir_up("persistent_data", __file__)
    sim.volume_manager.add_material_database(persist_data_dir / "GateMaterials.db")
    geometry_transformation_dataframe = get_geometry_definitions()

    # Add Geometry to the simulation
    add_geometry_to_gate_sim(sim, geometry_transformation_dataframe, args)
    _apply_debug_geometry_settings(sim, args)

    sim.user_info.visu = True
    sim.user_info.visu_type = "vrml_file_only"
    sim.visu_commands_vrml = ["/vis/open VRML2FILE", "/vis/drawVolume"]
    sim.visu_commands_vrml.append("/vis/geometry/set/visibility world 0 false")
    sim.visu_commands_vrml.append("/vis/viewer/flush")
    print("Storing geometry into wrl file only without running the simulation...")
    sim.user_info.visu_filename = str(output_dir / "brain_spect_geometry.wrl")
    sim.run(start_new_process=True)
    print(f"Geometry stored in:\n  {sim.user_info.visu_filename}")


def add_fov_volume_to_gate_sim(
    sim: gate.Simulation, shape: str = "box", size_mm: float = 150.0
):
    shape_name = str(shape).lower()
    if shape_name == "box":
        source_volume = sim.add_volume("Box", name="FOVBox")
        size_value = float(size_mm)
        source_volume.size = [size_value, size_value, size_value]
        source_volume.mother = "world"
        source_volume.material = "Air"
        return source_volume
    if shape_name == "sphere":
        source_volume = sim.add_volume("Sphere", name="FOVSphere")
        source_volume.rmax = float(size_mm) * gate.g4_units.mm
        source_volume.mother = "world"
        source_volume.material = "Air"
        return source_volume
    raise ValueError(f"Unsupported FOV shape: {shape!r}. Use 'box' or 'sphere'.")


def add_fov_box_to_gate_sim(sim: gate.Simulation, size_mm: float = 150.0):
    return add_fov_volume_to_gate_sim(sim, shape="box", size_mm=size_mm)


def add_fov_sphere_to_gate_sim(sim: gate.Simulation, size_mm: float = 150.0):
    return add_fov_volume_to_gate_sim(sim, shape="sphere", size_mm=size_mm)


def add_volume_source(
    sim: gate.Simulation,
    energy_keV: float = 140.0,
    name: str = "BoxSource",
    *,
    args,
    fov_shape: str = "box",
    fov_size_mm: float = 150.0,
):
    source = gate.sources.generic.GenericSource(name=name)
    source.particle = "gamma"
    source.energy.type = "mono"
    source.activity = parse_activity_to_bq(args.source_activity) * gate.g4_units.Bq
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
        source.position.radius = fov_size * gate.g4_units.mm
        source_obj = sim.add_source(source, name=name)
        source_obj.attached_to = "FOVSphere"
        return source_obj
    raise ValueError(f"Unsupported FOV shape: {fov_shape!r}. Use 'box' or 'sphere'.")


def add_point_source(
    sim: gate.Simulation, energy_keV: float = 140.0, name: str = "PointSource", *, args
):
    source = gate.sources.generic.GenericSource(name=name)
    source.particle = "gamma"
    source.energy.type = "mono"
    source.activity = parse_activity_to_bq(args.source_activity) * gate.g4_units.Bq
    source.energy.mono = energy_keV * gate.g4_units.keV
    source.position.type = "point"
    source.position.point = [0, 0, 0]
    sim.add_source(source, name=name)


def add_stats_actor(sim: gate.Simulation, output_dir: Path, output_stem: str):
    stats_actor = sim.add_actor("SimulationStatisticsActor", "Stats")  # type: ignore
    stats_path = output_dir / f"{output_stem}_sim_stats.txt"
    # GATE will automatically write to this file after sim.run() finishes
    stats_actor.output_filename = str(stats_path)


def add_actors(
    sim: gate.Simulation, n_crystals: int, output_dir: Path, output_stem: str
):
    pixel_array_name = [f"pixel_{i + 1}" for i in range(n_crystals)]

    # Keep hits in-memory only as input to the singles chain.
    for i in range(n_crystals):
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


def configure_chunked_run_timing(sim: gate.Simulation, args):
    if args.chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be > 0")
    if args.num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")
    source_activity_bq = parse_activity_to_bq(args.source_activity)
    if source_activity_bq <= 0:
        raise ValueError("source_activity must be > 0")

    sec = gate.g4_units.s
    interval_duration = args.chunk_duration_s * sec
    sim.run_timing_intervals = [
        [i * interval_duration, (i + 1) * interval_duration]
        for i in range(args.num_chunks)
    ]

    # Runtime policies vary by environment; OSPool stays single-threaded while
    # SLURM and local runs can use more threads when requested or available.
    expected_events_per_chunk_per_thread = source_activity_bq * args.chunk_duration_s
    expected_events_per_chunk = expected_events_per_chunk_per_thread * args.num_threads
    expected_events_per_thread = expected_events_per_chunk_per_thread * args.num_chunks
    expected_events_total = expected_events_per_thread * args.num_threads

    print(f"Chunk duration (s): {args.chunk_duration_s}")
    print(f"Number of chunks: {args.num_chunks}")
    print(f"Number of threads: {args.num_threads}")
    print(f"Source activity (Bq): {source_activity_bq:.3e} ({args.source_activity})")
    print(
        "Expected primaries per chunk per thread: "
        f"{expected_events_per_chunk_per_thread:.3e}"
    )
    print(f"Expected primaries per thread: {expected_events_per_thread:.3e}")
    print(
        f"Expected primaries total all chunks all threads: {expected_events_total:.3e}"
    )

    if expected_events_per_chunk >= args.eventid_warn_threshold:
        print(
            "WARNING: Expected events per chunk is high relative to 32-bit EventID range. "
            "Reduce activity or chunk_duration_s to lower overflow risk."
        )


def run_simulation(
    args,
):
    output_dir = Path(args.output_dir).resolve()
    print("Output directory: ", output_dir)
    print(f"Execution environment: {args.execution_environment}")
    print(
        "Runtime context: "
        f"job_array_id={args.job_array_id}, job_array_task_id={args.job_array_task_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    persist_data_dir = qmirt.utils.filesystem.search_dir_up("persistent_data", __file__)
    geometry_transformation_dataframe = get_geometry_definitions()
    n_crystals = geometry_transformation_dataframe.shape[0]

    job_array_id = args.job_array_id
    job_array_task_id = args.job_array_task_id

    unique_seed = generate_unique_seed(str(job_array_id), str(job_array_task_id))
    print(f"Using random seed: {unique_seed}")

    sim = gate.Simulation(progress_bar=True, output_dir=output_dir)
    sim.random_seed = unique_seed
    sim.volume_manager.add_material_database(persist_data_dir / "GateMaterials.db")
    print(f"Using GateMaterials.db from {persist_data_dir}")
    # Add Geometry to the simulation
    add_geometry_to_gate_sim(sim, geometry_transformation_dataframe, args)
    _apply_debug_geometry_settings(sim, args)

    source_type = str(args.source_type).lower()
    effective_fov_shape = str(args.fov_shape).lower()
    if source_type in {"box", "sphere"}:
        assert effective_fov_shape == source_type, (
            f"source-type '{source_type}' is incompatible with fov-shape '{effective_fov_shape}'. They must match for volume sources."
        )

    if source_type in {"box", "sphere"}:
        add_volume_source(
            sim,
            energy_keV=140.0,
            args=args,
            fov_shape=effective_fov_shape,
            fov_size_mm=args.fov_size_mm,
        )
    elif source_type == "point":
        add_point_source(sim, energy_keV=140.0, args=args)
    else:
        raise ValueError("source-type must be one of: 'box', 'sphere', 'point'")

    sim.number_of_threads = int(args.num_threads)
    configure_chunked_run_timing(sim, args)
    # In activity mode, expected event count is stochastic and controlled by
    # activity * run_timing_intervals.
    print(f"Number of threads: {sim.number_of_threads}")

    output_stem = f"a_{job_array_id}_j_{job_array_task_id}"
    add_actors(
        sim,
        n_crystals,
        output_dir,
        output_stem,
    )
    add_stats_actor(sim, output_dir, output_stem)
    sim.run()


def parse_arguments():
    def _parse_job_id(value: str) -> str:
        return value

    parser = argparse.ArgumentParser(
        description="Run a GATE simulation with Brain SPECT geometry."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store simulation outputs.",
    )
    parser.add_argument(
        "-j",
        "--job-array-id",
        type=_parse_job_id,
        default=None,
        help="SLURM_ARRAY_JOB_ID used for naming output files.",
    )
    parser.add_argument(
        "-k",
        "--job-array-task-id",
        type=_parse_job_id,
        default=None,
        help="SLURM_ARRAY_TASK_ID used for naming output files.",
    )
    parser.add_argument(
        "-d",
        "--chunk-duration-s",
        type=float,
        default=1.0,
        help="Duration of each chunk in seconds.",
    )
    parser.add_argument(
        "-c", "--num-chunks", type=int, default=10, help="Number of chunks to simulate."
    )
    parser.add_argument(
        "-t",
        "--num-threads",
        type=int,
        default=None,
        help=(
            "Number of threads requested. Defaults to SLURM_CPUS_PER_TASK on SLURM, "
            "1 on OSPool, and up to 128 on a local machine."
        ),
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
        "-s",
        "--source-activity",
        nargs="+",
        default=["1e6", "Bq"],
        metavar=("VALUE", "UNIT"),
        help=(
            "Source activity as a value with an optional unit, for example: "
            "1.2e10 Bq, 0.01 Ci, or 1e6."
        ),
    )
    parser.add_argument(
        "--debug-geometry",
        action="store_true",
        help="Dump the geometry tree and enable verbose Geant4 output before running.",
    )
    parser.add_argument(
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
        "--eventid-warn-threshold",
        type=float,
        default=1.5e9,
        help="Threshold for expected events per chunk to warn about EventID overflow.",
    )
    parser.add_argument(
        "--with-shielding",
        action="store_true",
        help="Include shielding in the simulation.",
    )
    parser.add_argument(
        "-m",
        "--mapping-mode",
        type=str,
        choices=["sequential", "reverse", "random"],
        default="sequential",
        help="Mapping mode for crystal IDs to collimator configurations.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="geometry-only",
        choices=["box", "sphere", "geometry-only"],
        help="Compatibility alias for the source FOV shape or geometry-only export mode.",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="box",
        choices=["box", "sphere", "point"],
        help="Select the source type used by the brain simulation: box, sphere, or point.",
    )
    parser.add_argument(
        "--fov-shape",
        type=str,
        default="box",
        choices=["box", "sphere"],
        help="FOV shape for the volume source: box or sphere.",
    )
    parser.add_argument(
        "--fov-size-mm",
        type=float,
        default=150.0,
        help="FOV size in mm. For box, this is the side length; for sphere, it is the radius.",
    )

    return parser.parse_args()


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


def main():
    args = parse_arguments()
    if args.mode is not None:
        mode_name = str(args.mode).lower()
        if mode_name in {"box", "sphere"}:
            args.source_type = mode_name
            args.fov_shape = mode_name
        (
            resolved_job_array_id,
            resolved_job_array_task_id,
            resolved_num_threads,
            resolved_execution_environment,
        ) = resolve_simulation_runtime_context(
            args.job_array_id,
            args.job_array_task_id,
            args.num_threads,
            args.execution_environment,
        )
    args.job_array_id = resolved_job_array_id
    args.job_array_task_id = resolved_job_array_task_id
    args.num_threads = resolved_num_threads
    args.execution_environment = resolved_execution_environment
    if args.mode == "geometry-only":
        run_simulation_with_geometry_only(args)
    else:
        run_simulation(
            args,
        )


if __name__ == "__main__":
    main()
