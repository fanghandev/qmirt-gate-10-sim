from pathlib import Path

import numpy as np
import opengate as gate
import polars as pl
from opengate.geometry.volumes import subtract_volumes, unite_volumes
from qmirt_utility.utils import find_project_root
from scipy.spatial.transform import Rotation


def get_geometry_definitions():
    project_root = find_project_root()
    data_dir = project_root / "persistent_data" / "brain_spect"
    csv_dir = data_dir / "csv"
    csv_filename = "BrainSPECT_Point_Cloud.007.25mmx0.556mm_pinhole.csv"
    stl_dir = data_dir / "stl"
    stl_filename = "BrainFrame.008.Lead_Shield.STL"

    collimator_definition = {
        "l_top": 10.74,
        "h_nozzle": 5.04,
        "w_pinhole": 0.556,
        "w_wall": 2.03,
        "l_bottom_inner": 50.0,
        "l_bottom_outer": 56.064,
        "h_body": 25.0,
        "h_box": 23.5,
    }

    crystal_definition = {"size_mm": [50.0, 50.0, 10.0], "n_pixels": [25, 25, 1]}

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
    # elevation = (-elevation + 2 * np.pi) % (2 * np.pi)  # convert elevation angle to 0 to 2pi range
    # convert azimuthal angle to 0 to 2pi range
    azimuth = (
        np.arctan2(csv_pl_df["Pinhole_y"], csv_pl_df["Pinhole_x"]) + 2 * np.pi
    ) % (2 * np.pi)
    # azimuth = np.arctan2(csv_pl_df['Pinhole_y'], csv_pl_df['Pinhole_x'])
    csv_pl_df = csv_pl_df.with_columns(
        pl.Series("elevation", elevation), pl.Series("azimuth", azimuth)
    )
    # Round elevation and azimuth to 2 decimal places
    csv_pl_df = csv_pl_df.with_columns(
        pl.col("elevation").round(6), pl.col("azimuth").round(6)
    )

    csv_pl_df = csv_pl_df.sort(["elevation", "azimuth"])

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
    crystal_definition["n crystals"] = geometry_transformation_dataframe.shape[0]
    geometry_base_definition = {
        "collimator definition": collimator_definition,
        "crystal definition": crystal_definition,
        "shielding file path": str(stl_dir / stl_filename),
    }
    return geometry_base_definition, geometry_transformation_dataframe


def construct_collimator_geometry(config: dict, id: int):
    frustum_a = gate.geometry.volumes.TrdVolume(
        name=f"Frustum_A_{id + 1}",
        dx1=config["collimator definition"]["l_bottom_outer"] * 0.5,
        dy1=config["collimator definition"]["l_bottom_outer"] * 0.5,
        dx2=config["collimator definition"]["l_top"] * 0.5,
        dy2=config["collimator definition"]["l_top"] * 0.5,
        dz=(
            config["collimator definition"]["h_nozzle"]
            + config["collimator definition"]["h_body"]
        )
        * 0.5,
    )

    # We need to make the frustum_b an frustum_c slightly longer
    # to ensure clean subtration
    delta_z = 0.1
    delta_xy_b = (
        (
            config["collimator definition"]["l_top"]
            - config["collimator definition"]["w_pinhole"]
        )
        / config["collimator definition"]["h_nozzle"]
        * delta_z
        * 0.5
    )
    delta_xy_c = (
        (
            config["collimator definition"]["w_pinhole"]
            - config["collimator definition"]["l_bottom_inner"]
        )
        / config["collimator definition"]["h_body"]
        * delta_z
        * 0.5
    )
    frustum_b = gate.geometry.volumes.TrdVolume(
        name=f"Frustum_B_{id + 1}",
        dx1=config["collimator definition"]["w_pinhole"] * 0.5 - delta_xy_b,
        dy1=config["collimator definition"]["w_pinhole"] * 0.5 - delta_xy_b,
        dx2=config["collimator definition"]["l_top"] * 0.5 + delta_xy_b,
        dy2=config["collimator definition"]["l_top"] * 0.5 + delta_xy_b,
        dz=config["collimator definition"]["h_nozzle"] * 0.5 + delta_z,
    )

    frustum_c = gate.geometry.volumes.TrdVolume(
        name=f"Frustum_C_{id + 1}",
        dx1=config["collimator definition"]["l_bottom_inner"] * 0.5 - delta_xy_c,
        dy1=config["collimator definition"]["l_bottom_inner"] * 0.5 - delta_xy_c,
        dx2=config["collimator definition"]["w_pinhole"] * 0.5 + delta_xy_c,
        dy2=config["collimator definition"]["w_pinhole"] * 0.5 + delta_xy_c,
        dz=config["collimator definition"]["h_body"] * 0.5 + delta_z,
    )

    box_a = gate.geometry.volumes.BoxVolume(
        name=f"Box_A_{id + 1}",
        size=[
            config["collimator definition"]["l_bottom_outer"],
            config["collimator definition"]["l_bottom_outer"],
            config["collimator definition"]["h_box"],
        ],
    )
    box_b = gate.geometry.volumes.BoxVolume(
        name=f"Box_B_{id + 1}",
        size=[
            config["collimator definition"]["l_bottom_outer"]
            - 2 * config["collimator definition"]["w_wall"],
            config["collimator definition"]["l_bottom_outer"]
            - 2 * config["collimator definition"]["w_wall"],
            config["collimator definition"]["h_box"] + 2.0,
        ],  # Extend the inner box by 2 units in height to ensure a clean cut
    )

    # Create the hollow box by subtracting box_b from box_a
    hollow_box = subtract_volumes(box_a, box_b)

    # move hollow_box down by h_body + h_box/2
    hollow_frustum = frustum_a
    hollow_frustum = subtract_volumes(
        hollow_frustum,
        frustum_b,
        translation=[0, 0, config["collimator definition"]["h_body"] * 0.5],
    )
    hollow_frustum = subtract_volumes(
        hollow_frustum,
        frustum_c,
        translation=[0, 0, -config["collimator definition"]["h_nozzle"] * 0.5],
    )
    z_shift_union = -0.5 * (
        config["collimator definition"]["h_nozzle"]
        + config["collimator definition"]["h_body"]
        + config["collimator definition"]["h_box"]
    )

    # 2. Unite the volumes using the correct relative shift
    collimator = unite_volumes(
        hollow_frustum,
        hollow_box,
        new_name=f"Collimator_{id + 1}",
        translation=[0, 0, z_shift_union],
    )
    return collimator


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
    collimator = construct_collimator_geometry(config, id)
    sim.volume_manager.add_volume(collimator)
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

    collimator.name = f"Collimator_{id + 1}"
    # collimator.material = "Tungsten"


def add_crystal_box(sim: gate.Simulation, name: str):
    mm = gate.g4_units.mm
    crystal_box = sim.add_volume("Box", name=name)
    crystal_box.size = [50.5 * mm, 50.5 * mm, 12.0 * mm]  # unit is mm
    # crystal_box.material = "Air"
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
    pixel_repeater = gate.geometry.volumes.RepeatParametrisedVolume(
        repeated_volume=detector_pixel
    )
    pixel_repeater.linear_repeat = n_pixels
    pixel_repeater.translation = pixel_size_mm
    # pixel_repeater.rotation = r
    sim.volume_manager.add_volume(pixel_repeater)


def add_shielding_to_gate_sim(sim: gate.Simulation, config: dict):

    shielding = gate.geometry.volumes.TesselatedVolume(name="Shielding")
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


def add_geometry_to_gate_sim(sim: gate.Simulation, config: dict, pl_df: pl.DataFrame):
    for id in range(config["crystal definition"]["n crystals"]):
        add_collimator_to_gate_sim(sim, config, pl_df, id)
        add_pixelated_detector_to_gate_sim(sim, config, pl_df, id)
    add_shielding_to_gate_sim(sim, config)


def run_simulation_with_geometry_only(
    geometry_base_definition, geometry_transformation_dataframe
):
    sim = gate.Simulation()
    add_geometry_to_gate_sim(
        sim, geometry_base_definition, geometry_transformation_dataframe
    )
    sim.user_info.visu = True
    sim.user_info.visu_type = "vrml_file_only"
    sim.visu_commands_vrml = ["/vis/open VRML2FILE", "/vis/drawVolume"]
    sim.visu_commands_vrml.append("/vis/geometry/set/visibility world 0 false")
    sim.visu_commands_vrml.append("/vis/viewer/flush")
    print("Storing geometry into wrl file only without running the simulation...")
    sim.user_info.visu_filename = "brain_spect_geometry.wrl"
    sim.run(start_new_process=True)
    print(f"Geometry stored in {sim.user_info.visu_filename}")


def generate_unique_seed(job_array_id: str, job_array_task_id: str) -> int:
    from hashlib import md5
    from os import times

    seed_string = f"gate_sim_{job_array_id}_{job_array_task_id}"
    # Also add timestamp to ensure uniqueness across different runs, if needed
    seed_string += f"_{times()}"
    return int(md5(seed_string.encode()).hexdigest()[:8], 16)


def add_box_source(
    sim: gate.Simulation, energy_keV: float = 140.0, name: str = "BoxSource", *, args
):

    source = gate.sources.generic.GenericSource(name=name)
    source.particle = "gamma"
    source.energy.type = "mono"
    source.activity = args.source_activity_bq * gate.g4_units.Bq
    source.energy.mono = energy_keV * gate.g4_units.keV
    source.position.type = "box"
    source.position.size = [210, 210, 210]  # unit is mms
    sim.add_source(source, name=name)


def add_stats_actor(sim: gate.Simulation, output_dir: Path, output_stem: str):
    stats_actor = sim.add_actor("SimulationStatisticsActor", "Stats")  # type: ignore
    stats_path = output_dir / f"{output_stem}_sim_stats.txt"
    # GATE will automatically write to this file after sim.run() finishes
    stats_actor.output_filename = str(stats_path)


def add_actors(sim: gate.Simulation, output_dir: Path, config: dict, output_stem: str):
    pixel_array_name = [
        f"pixel_{i + 1}" for i in range(config["crystal definition"]["n crystals"])
    ]

    # Keep hits in-memory only as input to the singles chain.
    for i in range(config["crystal definition"]["n crystals"]):
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
    if args.source_activity_bq <= 0:
        raise ValueError("source_activity_bq must be > 0")

    sec = gate.g4_units.s
    interval_duration = args.chunk_duration_s * sec
    sim.run_timing_intervals = [
        [i * interval_duration, (i + 1) * interval_duration]
        for i in range(args.num_chunks)
    ]

    # For this SLURM workflow we enforce one thread per task.
    expected_events_per_chunk_per_thread = (
        args.source_activity_bq * args.chunk_duration_s
    )
    expected_events_per_chunk = expected_events_per_chunk_per_thread * args.num_threads
    expected_events_per_thread = expected_events_per_chunk_per_thread * args.num_chunks
    expected_events_total = expected_events_per_thread * args.num_threads

    print(f"Chunk duration (s): {args.chunk_duration_s}")
    print(f"Number of chunks: {args.num_chunks}")
    print(f"Number of threads: {args.num_threads}")
    print(f"Source activity (Bq): {args.source_activity_bq}")
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
    persist_data_dir: Path,
    geometry_base_definition: dict,
    geometry_transformation_dataframe: pl.DataFrame,
    args,
):
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

    # Add Geometry to the simulation
    add_geometry_to_gate_sim(
        sim, geometry_base_definition, geometry_transformation_dataframe
    )

    # Add Source to the simulation
    add_box_source(sim, energy_keV=140.0, args=args)

    sim.number_of_threads = int(args.num_threads)
    configure_chunked_run_timing(sim, args)
    # In activity mode, expected event count is stochastic and controlled by
    # activity * run_timing_intervals.
    print(f"Number of threads: {sim.number_of_threads}")

    output_stem = f"a_{job_array_id}_j_{job_array_task_id}"
    add_actors(sim, output_dir, geometry_base_definition, output_stem)
    add_stats_actor(sim, output_dir, output_stem)
    sim.run()


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a GATE simulation with Brain SPECT geometry."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store simulation outputs.",
    )
    parser.add_argument(
        "-j", "--job_array_id", type=int, required=True, help="SLURM job array ID."
    )
    parser.add_argument(
        "-k",
        "--job_array_task_id",
        type=int,
        required=True,
        help="SLURM job array task ID.",
    )
    parser.add_argument(
        "--chunk_duration_s",
        type=float,
        default=1.0,
        help="Duration of each chunk in seconds.",
    )
    parser.add_argument(
        "-n", "--num_chunks", type=int, default=10, help="Number of chunks to simulate."
    )
    parser.add_argument(
        "-t", "--num_threads", type=int, default=1, help="Number of threads to use."
    )
    parser.add_argument(
        "-s",
        "--source_activity_bq",
        type=float,
        default=1e6,
        help="Activity of the source in Becquerels.",
    )
    parser.add_argument(
        "--geometry_only", action="store_true", help="Run geometry-only simulation."
    )
    parser.add_argument(
        "--eventid_warn_threshold",
        type=int,
        default=1.5e9,
        help="Threshold for expected events per chunk to warn about EventID overflow.",
    )

    return parser.parse_args()


def main():
    geometry_base_definition, geometry_transformation_dataframe = (
        get_geometry_definitions()
    )
    args = parse_arguments()
    if args.geometry_only:
        run_simulation_with_geometry_only(
            geometry_base_definition, geometry_transformation_dataframe
        )
    else:
        persist_data_dir = find_project_root() / "persistent_data"
        run_simulation(
            persist_data_dir,
            geometry_base_definition,
            geometry_transformation_dataframe,
            args,
        )


if __name__ == "__main__":
    main()
