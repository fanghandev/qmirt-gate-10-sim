import hashlib
import os
import pandas as pd
from pathlib import Path
import numpy as np
import opengate as gate
from opengate.geometry.volumes import unite_volumes, subtract_volumes
from scipy.spatial.transform import Rotation

def generate_unique_seed(job_array_id: str, job_array_task_id: str) -> int:
    seed_string = f"gate_sim_{job_array_id}_{job_array_task_id}"
    return int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)



def get_dc_spect_geometry_config(xlsx_path: Path):

    n_heads = 80
    collimator_hole_size_mm = 2.3  # unit is mm
    collimator_wall_thickness_mm = 2.0  # unit is mm
    collimator_guide_length_mm = 3.0
    detector_crystal_size_mm = [50.0, 50.0, 10.0]  # unit is mm

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
    if np.isnan(collimator_body_length_mm_np).any() or np.isnan(
        collimator_hole_coords_mm
    ).any():
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
    azmuthal_angle_deg = (
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
            * np.cos(np.radians(azmuthal_angle_deg)),
            np.cos(np.radians(polar_angle_deg))
            * np.sin(np.radians(azmuthal_angle_deg)),
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
            * np.cos(np.radians(azmuthal_angle_deg)),
            np.cos(np.radians(polar_angle_deg))
            * np.sin(np.radians(azmuthal_angle_deg)),
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
            * np.cos(np.radians(azmuthal_angle_deg)),
            np.cos(np.radians(polar_angle_deg))
            * np.sin(np.radians(azmuthal_angle_deg)),
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
        "detector_crystal_size_mm": detector_crystal_size_mm,
        "detector_crystal_translation_mm": detector_crystal_translation_mm,
        "azmuthal_angle_deg": azmuthal_angle_deg,
        "polar_angle_deg": polar_angle_deg,
    }


def add_dc_spect_geometry(sim: gate.Simulation, config: dict, persist_data_dir: Path):

    # Load materials
    stl_dir = persist_data_dir / "stl"
    sim.volume_manager.add_material_database(persist_data_dir / "GateMaterials.db")

    collimator_body_length_mm_np = config["collimator_body_length_mm_np"]
    collimator_body_translation_mm = config["collimator_body_translation_mm"]
    collimator_body_inner_top_mm_np = config["collimator_body_inner_top_mm_np"]
    collimator_body_inner_bottom_mm_np = config["collimator_body_inner_bottom_mm_np"]
    collimator_body_outer_top_mm_np = config["collimator_body_outer_top_mm_np"]
    collimator_body_outer_bottom_mm_np = config["collimator_body_outer_bottom_mm_np"]
    collimator_guide_length_mm_np = config["collimator_guide_length_mm_np"]
    collimator_guide_inner_top_mm_np = config["collimator_guide_inner_top_mm_np"]
    collimator_guide_outer_top_mm_np = config["collimator_guide_outer_top_mm_np"]
    collimator_guide_inner_bottom_mm_np = config["collimator_guide_inner_bottom_mm_np"]
    collimator_guide_outer_bottom_mm_np = config["collimator_guide_outer_bottom_mm_np"]
    azmuthal_angle_deg = config["azmuthal_angle_deg"]
    polar_angle_deg = config["polar_angle_deg"]
    detector_crystal_size_mm = config["detector_crystal_size_mm"]
    detector_crystal_translation_mm = config["detector_crystal_translation_mm"]

    for i in range(80):

        collimator_body_inner = gate.geometry.volumes.TrdVolume(  # type: ignore
            name=f"CollimatorBody_{i+1}"
        )
        collimator_body_outer = gate.geometry.volumes.TrdVolume(  # type: ignore
            name=f"CollimatorBody_outer_{i+1}"
        )
        collimator_body_outer.dx1 = collimator_body_outer_top_mm_np[i] * 0.5
        collimator_body_outer.dy1 = collimator_body_outer_top_mm_np[i] * 0.5
        collimator_body_outer.dx2 = collimator_body_outer_bottom_mm_np[i] * 0.5
        collimator_body_outer.dy2 = collimator_body_outer_bottom_mm_np[i] * 0.5
        collimator_body_outer.dz = collimator_body_length_mm_np[i] * 0.5
        collimator_body_inner.dx1 = collimator_body_inner_top_mm_np[i] * 0.5
        collimator_body_inner.dy1 = collimator_body_inner_top_mm_np[i] * 0.5
        collimator_body_inner.dx2 = collimator_body_inner_bottom_mm_np[i] * 0.5
        collimator_body_inner.dy2 = collimator_body_inner_bottom_mm_np[i] * 0.5
        collimator_body_inner.dz = collimator_body_length_mm_np[i] * 0.5 + 0.1
        collimator_body = subtract_volumes(collimator_body_outer, collimator_body_inner)

        # Collimator Guide
        collimator_guide_inner = gate.geometry.volumes.TrdVolume(  # type: ignore
            name=f"CollimatorGuide_{i+1}"
        )
        collimator_guide_outer = gate.geometry.volumes.TrdVolume(  # type: ignore
            name=f"CollimatorGuide_outer_{i+1}"
        )
        collimator_guide_outer.dx1 = collimator_guide_outer_top_mm_np[i] * 0.5
        collimator_guide_outer.dy1 = collimator_guide_outer_top_mm_np[i] * 0.5
        collimator_guide_outer.dx2 = collimator_guide_outer_bottom_mm_np[i] * 0.5
        collimator_guide_outer.dy2 = collimator_guide_outer_bottom_mm_np[i] * 0.5
        collimator_guide_outer.dz = collimator_guide_length_mm_np[i] * 0.5
        collimator_guide_inner.dx1 = collimator_guide_inner_top_mm_np[i] * 0.5
        collimator_guide_inner.dy1 = collimator_guide_inner_top_mm_np[i] * 0.5
        collimator_guide_inner.dx2 = collimator_guide_inner_bottom_mm_np[i] * 0.5
        collimator_guide_inner.dy2 = collimator_guide_inner_bottom_mm_np[i] * 0.5
        collimator_guide_inner.dz = (
            collimator_guide_length_mm_np[i] * 0.5 + 0.1
        )  # add a small extra length to ensure the subtraction works correctly
        collimator_guide = subtract_volumes(
            collimator_guide_outer, collimator_guide_inner
        )

        collimator = unite_volumes(
            collimator_body,
            collimator_guide,
            new_name=f"Collimator_{i+1}",
            translation=[
                0,
                0,
                collimator_body_length_mm_np[i] * 0.5
                + collimator_guide_length_mm_np[i] * 0.5,
            ],
        )
        sim.add_volume(collimator, name=f"Collimator_{i+1}")
        collimator.mother = "world"
        collimator.translation = collimator_body_translation_mm[i]
        # First rotate around x axis by 90 degrees
        rx_0 = Rotation.from_euler("x", -90, degrees=True).as_matrix()
        rz_0 = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        # Then rotate around z axis by the azmuthal angle
        rz_1 = Rotation.from_euler("z", azmuthal_angle_deg[i], degrees=True).as_matrix()

        # # Then rotate around y axis by the polar angle
        rx_1 = Rotation.from_euler("x", -polar_angle_deg[i], degrees=True).as_matrix()
        r = rz_1 @ rz_0 @ rx_1 @ rx_0
        collimator.rotation = r
        collimator.material = "Tungsten"
        # Add in detector crystal volume
        detector_crystal = gate.geometry.volumes.BoxVolume(  # type: ignore
            name=f"DetectorCrystal_{i+1}"
        )
        detector_crystal.size = detector_crystal_size_mm
        detector_crystal.mother = "world"
        detector_crystal.translation = detector_crystal_translation_mm[i]

        sim.add_volume(detector_crystal, name=f"DetectorCrystal_{i+1}")
        detector_crystal.rotation = r
        detector_crystal.material = "CsI"

    shielding = gate.geometry.volumes.TesselatedVolume(name="Shielding")
    shielding.mother = "world"
    shielding.file_name = str((stl_dir / "dc_spect_shielding_combined.stl").as_posix())
    shielding.origin_at_cog = False
    sim.add_volume(shielding)
    rz = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    shielding.rotation = rz
    shielding.material = "Lead"


def add_box_source(
    sim: gate.Simulation,
    energy_keV: float = 140.0,
    name: str = "BoxSource",
    *,
    args
):

    source = gate.sources.generic.GenericSource(name=name)
    source.particle = "gamma"
    source.energy.type = "mono"
    source.activity = args.source_activity_bq * gate.g4_units.Bq
    source.energy.mono = energy_keV * gate.g4_units.keV
    source.position.type = "box"
    source.position.size = [170.0, 170.0, 170.0]  # unit is mms
    sim.add_source(source, name=name)


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
    expected_events_per_chunk_per_thread = args.source_activity_bq * args.chunk_duration_s
    expected_events_per_chunk = expected_events_per_chunk_per_thread * 1
    expected_total_events = expected_events_per_chunk * args.num_chunks

    print(f"Chunk duration (s): {args.chunk_duration_s}")
    print(f"Number of chunks: {args.num_chunks}")
    print("Number of threads: 1 (enforced single-thread mode)")
    print(f"Source activity (Bq): {args.source_activity_bq}")
    print(
        "Expected primaries/chunk/thread (mean): "
        f"{expected_events_per_chunk_per_thread:.3e}"
    )
    print(f"Expected primaries/chunk all threads (mean): {expected_events_per_chunk:.3e}")
    print(f"Expected primaries total all chunks (mean): {expected_total_events:.3e}")

    if expected_events_per_chunk >= args.eventid_warn_threshold:
        print(
            "WARNING: Expected events/chunk is high relative to 32-bit EventID range. "
            "Reduce activity or chunk_duration_s to lower overflow risk."
        )


def run_simulation(config: dict, persist_data_dir: Path, args):
    output_dir = Path(args.output_dir).resolve()
    print("Resolved output directory: ",output_dir)
    print(
        "Slurm context: "
        f"job_array_id={args.job_array_id}, job_array_task_id={args.job_array_task_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    job_array_id = args.job_array_id
    job_array_task_id = args.job_array_task_id

    unique_seed = generate_unique_seed(
        str(job_array_id), str(job_array_task_id)
    )
    print(f"Using random seed: {unique_seed}")

    sim = gate.Simulation(progress_bar=False, output_dir=output_dir)
    sim.random_seed = unique_seed

    # Add Geometry to the simulation
    add_dc_spect_geometry(sim, config, persist_data_dir)

    if args.debug_geometry:
        print("Geometry debug mode enabled: dumping volume tree and enabling verbose G4 output.")
        print(f"check_volumes_overlap: {sim.check_volumes_overlap}")
        print(sim.volume_manager.dump_volume_tree())
        sim.g4_verbose = True
        sim.g4_verbose_level = 2

    # Add Source to the simulation
    add_box_source(sim, energy_keV=140.0, args=args)

    if args.num_threads != 1:
        print(
            f"Warning: num_threads={args.num_threads} requested, forcing to 1 for SLURM core-based mode."
        )
    sim.number_of_threads = int(args.num_threads)
    configure_chunked_run_timing(sim, args)
    # In activity mode, expected event count is stochastic and controlled by
    # activity * run_timing_intervals.
    print(f"Number of threads: {sim.number_of_threads}")

    detector_crystals = [f"DetectorCrystal_{i+1}" for i in range(80)]

    # Keep hits in-memory only as input to the singles chain.
    detector_hits_actor: gate.actors.digitizers.DigitizerHitsCollectionActor = sim.add_actor(
        "DigitizerHitsCollectionActor", "DetectorHitsActor"
    )
    detector_hits_actor.attached_to = detector_crystals
    detector_hits_actor.output_filename = ""
    detector_hits_actor.attributes = [
        "RunID",
        "ThreadID",
        "EventID",
        "TrackID",
        "TotalEnergyDeposit",
        "PostPosition",
        "PostPositionLocal",
        "PrePosition",
        "PrePositionLocal",
        "EventPosition",
        "GlobalTime",
        "PreStepUniqueVolumeID",
        "PreStepUniqueVolumeIDAsInt",
        "PostStepUniqueVolumeIDAsInt",
        "TrackVolumeName",
    ]

    singles_adder_actor: gate.actors.digitizers.DigitizerAdderActor = sim.add_actor(
        "DigitizerAdderActor", "DetectorSinglesAdderActor"
    )
    singles_adder_actor.attached_to = detector_crystals
    singles_adder_actor.input_digi_collection = detector_hits_actor.name
    singles_adder_actor.policy = "EnergyWinnerPosition"
    singles_adder_actor.output_filename = ""

    # Apply 10% Gaussian energy resolution at 140 keV to singles energy.
    singles_blur_actor: gate.actors.digitizers.DigitizerBlurringActor = sim.add_actor(
        "DigitizerBlurringActor", "DetectorSinglesBlurActor"
    )
    singles_blur_actor.attached_to = detector_crystals
    singles_blur_actor.input_digi_collection = singles_adder_actor.name
    singles_blur_actor.blur_attribute = "TotalEnergyDeposit"
    singles_blur_actor.blur_method = "Gaussian"
    singles_blur_actor.blur_fwhm = 0.10 * 140.0 * gate.g4_units.keV
    
    output_stem = f"a_{job_array_id}_j_{job_array_task_id}"
    singles_root_filename = f"{output_stem}.root"
    singles_root_file_path = output_dir / singles_root_filename
    print(singles_root_file_path)

    singles_blur_actor.output_filename = (
        singles_root_file_path
    )
    # Add the Simulation statistics actor to record the number of events
    stats_actor = sim.add_actor("SimulationStatisticsActor", "Stats")  # type: ignore
    sim.run()
    simulation_stats = stats_actor.user_output["stats"].get_processed_output()
    stats_path = output_dir / f"{output_stem}_sim_stats.txt"
    with open(stats_path, "w") as f:
        for key, value in simulation_stats.items():
            f.write(f"{key}: {value['value']}\n")

def save_simulation_geometry_to_wrl(config: dict, persist_data_dir: Path, args):
    sim = gate.Simulation()
    add_dc_spect_geometry(sim, config, persist_data_dir)
    if args.debug_geometry:
        print("Geometry debug mode enabled: dumping volume tree and enabling verbose G4 output.")
        print(f"check_volumes_overlap: {sim.check_volumes_overlap}")
        print(sim.volume_manager.dump_volume_tree())
        sim.g4_verbose = True
        sim.g4_verbose_level = 2
    sim.user_info.visu = True
    sim.user_info.visu_type = "vrml_file_only"
    print("Storing geometry into wrl file only without running the simulation...")
    sim.user_info.visu_filename = str(
        (persist_data_dir.parent / "dev" / "collimator_geometry.wrl").resolve()
    )
    print(f"Geometry stored in {sim.user_info.visu_filename}")
    sim.run()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run DC-SPECT simulation")
    parser.add_argument(
        "--source-activity-bq",
        type=float,
        default=1e6,
        required=False,
        help="Source activity in Bq used with run timing intervals.",
    )
    parser.add_argument(
        "--chunk-duration-s",
        type=float,
        default=1.0,
        required=False,
        help="Duration of each run chunk in seconds.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        required=False,
        help="Number of run timing intervals (chunks).",
    )
    parser.add_argument(
        "--eventid-warn-threshold",
        type=float,
        default=1.5e9,
        required=False,
        help="Warn if expected events per chunk exceed this threshold.",
    )
    parser.add_argument(
        "--debug-geometry",
        action="store_true",
        help="Dump the geometry tree and enable verbose Geant4 output before running.",
    )
    parser.add_argument(
        "-t",
        "--num-threads",
        type=int,
        default=1,
        required=False,
        help="Number of threads requested (this SLURM script enforces 1).",
    )
    parser.add_argument(
        "-x",
        "--xlsx-path",
        type=str,
        default=None,
        required=False,
        help="Path to the geometry configuration xlsx file",
    )
    # Add an option to store the geometry into wrl file only without running the simulation, for debugging purposes
    parser.add_argument(
        "-g",
        "--geometry-only",
        action="store_true",
        help="Store geometry to WRL only without running the simulation",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=".",
        required=False,
        help="Directory to store the simulation output files",
    )
    parser.add_argument(
        "--job-array-id",
        type=str,
        default=None,
        required=False,
        help="SLURM_ARRAY_JOB_ID used for naming output files.",
    )
    parser.add_argument(
        "--job-array-task-id",
        type=str,
        default=None,
        required=False,
        help="SLURM_ARRAY_TASK_ID used for naming output files.",
    )

    args = parser.parse_args()

    if args.job_array_id is None:
        args.job_array_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"
    if args.job_array_task_id is None:
        args.job_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID") or os.environ.get("SLURM_PROCID") or "0"

    # This script is located in slurm/ at the repository root.
    base_dir = Path(__file__).resolve().parents[3]
    persistent_data_dir = base_dir / "persistent_data"

    if args.xlsx_path is not None:
        xlsx_path = Path(args.xlsx_path)
    else:
        xlsx_path = (
            persistent_data_dir
            / "spreadsheet"
            / "MDSL.excel80M10RFR.cut-plate.010.150roi.2.30pin.105ellipse.xlsx"
        )
    # Make sure the xlsx file exists
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Geometry configuration xlsx file not found at {xlsx_path}"
        )

    config = get_dc_spect_geometry_config(xlsx_path)
    # Confirm config has been loaded correctly by printing out the keys and shapes of the numpy arrays
    for key, value in config.items():
        if isinstance(value, np.ndarray):
            assert (
                value.shape[0] == 80
            ), f"Expected numpy array of shape (80,) for key {key}, but got shape {value.shape}"
            continue
        elif isinstance(value, list):
            assert (
                len(value) == 3
            ), f"Expected list of length 3 for key {key}, but got length {len(value)}"
        else:
            raise ValueError(
                f"Config value loaded unsuccessfully for key {key}, expected numpy array or list but got {type(value)}"
            )



    if args.geometry_only:
        save_simulation_geometry_to_wrl(config, persistent_data_dir, args)
        exit(0)
    else:  # run the simulation
        run_simulation(config, persistent_data_dir, args)
