import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import opengate as gate
import pandas as pd
from opengate.geometry.volumes import subtract_volumes, unite_volumes
from scipy.spatial.transform import Rotation
# import uproot  # removed: using in‑memory actor output
from scipy.sparse import coo_matrix, csr_matrix, save_npz


def generate_unique_seed(job_array_id: str, job_array_task_id: str) -> int:
    seed_string = f"gate_sim_{job_array_id}_{job_array_task_id}"
    seed_string += f"_{os.times()}"
    return int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)


def get_dc_spect_geometry_config(xlsx_path: Path):
    n_heads = 80
    collimator_hole_size_mm = 2.3  # unit is mm
    collimator_wall_thickness_mm = 2.0  # unit is mm
    collimator_guide_length_mm = 3.0
    detector_crystal_size_mm = [50.0, 50.0, 10.0]  # unit is mm

    df_coords = pd.read_excel(xlsx_path, sheet_name="Coordinates")
    df_coords.columns = df_coords.iloc[0]
    df_coords = df_coords[1:]
    df_coords = df_coords.reset_index(drop=True)
    df_coords = df_coords.apply(pd.to_numeric, errors="coerce")
    df_coords.columns.name = "Coordinates Sheet"

    collimator_body_length_mm_np = df_coords["length of collimator"].values
    collimator_hole_coords_mm = df_coords[
        [
            "x coordinate value at center of hole",
            "y coordinate value at center of hole",
            "z coordinate value at center of hole",
        ]
    ].values

    if np.isnan(collimator_body_length_mm_np).any() or np.isnan(collimator_hole_coords_mm).any():
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

    hole_fov_center_distance_mm_np = np.linalg.norm(collimator_hole_coords_mm_np, axis=1)
    azmuthal_angle_deg = (
        np.arctan2(collimator_hole_coords_mm_np[:, 1], collimator_hole_coords_mm_np[:, 0])
        * 180
        / np.pi
    )
    hole_fov_center_dist_xy_mm_np = np.linalg.norm(collimator_hole_coords_mm_np[:, :2], axis=1)
    polar_angle_deg = (
        np.arctan2(collimator_hole_coords_mm_np[:, 2], hole_fov_center_dist_xy_mm_np)
        * 180
        / np.pi
    )
    collimator_body_center_dist_mm_np = (
        hole_fov_center_distance_mm_np + collimator_body_length_mm_np * 0.5
    )
    collimator_body_translation_mm = collimator_body_center_dist_mm_np.reshape(-1, 1) * np.column_stack(
        (
            np.cos(np.radians(polar_angle_deg)) * np.cos(np.radians(azmuthal_angle_deg)),
            np.cos(np.radians(polar_angle_deg)) * np.sin(np.radians(azmuthal_angle_deg)),
            np.sin(np.radians(polar_angle_deg)),
        )
    )
    detector_crystal_center_dist_mm_np = (
        hole_fov_center_distance_mm_np + collimator_body_length_mm_np + detector_crystal_size_mm[2] * 0.5
    )
    detector_crystal_translation_mm = detector_crystal_center_dist_mm_np.reshape(-1, 1) * np.column_stack(
        (
            np.cos(np.radians(polar_angle_deg)) * np.cos(np.radians(azmuthal_angle_deg)),
            np.cos(np.radians(polar_angle_deg)) * np.sin(np.radians(azmuthal_angle_deg)),
            np.sin(np.radians(polar_angle_deg)),
        )
    )

    collimator_wall_thickness_mm_np = np.full((80,), collimator_wall_thickness_mm)
    collimator_body_inner_top_mm_np = np.full((80,), detector_crystal_size_mm[0])
    collimator_body_inner_bottom_mm_np = np.full((80,), collimator_hole_size_mm)
    collimator_body_outer_top_mm_np = collimator_body_inner_top_mm_np + collimator_wall_thickness_mm_np * 2
    collimator_body_outer_bottom_mm_np = collimator_body_inner_bottom_mm_np + collimator_wall_thickness_mm_np * 2

    collimator_guide_exit_angle_rad = np.arctan2(
        (collimator_body_inner_top_mm_np + collimator_body_inner_bottom_mm_np) * 0.5,
        collimator_body_length_mm_np,
    )

    collimator_guide_length_mm_np = np.full((80,), collimator_guide_length_mm)
    collimator_guide_distance_mm_np = hole_fov_center_distance_mm_np - collimator_guide_length_mm_np
    collimator_guide_translation_mm = collimator_guide_distance_mm_np.reshape(-1, 1) * np.column_stack(
        (
            np.cos(np.radians(polar_angle_deg)) * np.cos(np.radians(azmuthal_angle_deg)),
            np.cos(np.radians(polar_angle_deg)) * np.sin(np.radians(azmuthal_angle_deg)),
            np.sin(np.radians(polar_angle_deg)),
        )
    )

    collimator_guide_inner_top_mm_np = np.full((80,), collimator_hole_size_mm)
    collimator_guide_outer_top_mm_np = collimator_guide_inner_top_mm_np + collimator_wall_thickness_mm_np * 2
    collimator_guide_inner_bottom_mm_np = (
        collimator_guide_inner_top_mm_np + np.tan(collimator_guide_exit_angle_rad) * collimator_guide_length_mm_np * 2
    )
    collimator_guide_outer_bottom_mm_np = collimator_guide_inner_bottom_mm_np + collimator_wall_thickness_mm_np * 2

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


def add_crystal_box(sim: gate.Simulation, name: str):
    mm = gate.g4_units.mm
    crystal_box = sim.add_volume("Box", name=name)
    crystal_box.size = [52.0 * mm, 52.0 * mm, 12.0 * mm]
    crystal_box.material = "Air"
    return crystal_box


def add_dc_spect_geometry(sim: gate.Simulation, config: dict, persist_data_dir: Path):
    stl_dir = persist_data_dir / "stl"
    sim.volume_manager.add_material_database(persist_data_dir / "GateMaterials.db")

    collimator_body_length_mm_np = config["collimator_body_length_mm_np"]
    collimator_body_translation_mm = config["collimator_body_translation_mm"]
    collimator_body_inner_top_mm_np = config["collimator_body_inner_top_mm_np"]
    collimator_body_inner_bottom_mm_np = config["collimator_body_inner_bottom_mm_np"]
    collimator_body_outer_top_mm_np = config["collimator_body_outer_top_mm_np"]
    collimator_body_outer_bottom_mm_np = config["collimator_body_outer_bottom_mm_np"]
    collimator_guide_length_mm_np = config["collimator_guide_length_mm_np"]
    collimator_guide_translation_mm = config["collimator_guide_translation_mm"]
    collimator_guide_inner_top_mm_np = config["collimator_guide_inner_top_mm_np"]
    collimator_guide_outer_top_mm_np = config["collimator_guide_outer_top_mm_np"]
    collimator_guide_inner_bottom_mm_np = config["collimator_guide_inner_bottom_mm_np"]
    collimator_guide_outer_bottom_mm_np = config["collimator_guide_outer_bottom_mm_np"]
    azmuthal_angle_deg = config["azmuthal_angle_deg"]
    polar_angle_deg = config["polar_angle_deg"]
    detector_crystal_size_mm = config["detector_crystal_size_mm"]
    detector_crystal_translation_mm = config["detector_crystal_translation_mm"]

    n_pixels = np.array([25, 25, 1])
    pixel_size_mm = detector_crystal_size_mm / n_pixels

    for i in range(80):
        collimator_body_inner = gate.geometry.volumes.TrdVolume(name=f"CollimatorBody_{i + 1}")
        collimator_body_outer = gate.geometry.volumes.TrdVolume(name=f"CollimatorBody_outer_{i + 1}")
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

        collimator_guide_inner = gate.geometry.volumes.TrdVolume(name=f"CollimatorGuide_{i + 1}")
        collimator_guide_outer = gate.geometry.volumes.TrdVolume(name=f"CollimatorGuide_outer_{i + 1}")
        collimator_guide_outer.dx1 = collimator_guide_outer_top_mm_np[i] * 0.5
        collimator_guide_outer.dy1 = collimator_guide_outer_top_mm_np[i] * 0.5
        collimator_guide_outer.dx2 = collimator_guide_outer_bottom_mm_np[i] * 0.5
        collimator_guide_outer.dy2 = collimator_guide_outer_bottom_mm_np[i] * 0.5
        collimator_guide_outer.dz = collimator_guide_length_mm_np[i] * 0.5
        collimator_guide_inner.dx1 = collimator_guide_inner_top_mm_np[i] * 0.5
        collimator_guide_inner.dy1 = collimator_guide_inner_top_mm_np[i] * 0.5
        collimator_guide_inner.dx2 = collimator_guide_inner_bottom_mm_np[i] * 0.5
        collimator_guide_inner.dy2 = collimator_guide_inner_bottom_mm_np[i] * 0.5
        collimator_guide_inner.dz = collimator_guide_length_mm_np[i] * 0.5 + 0.1
        collimator_guide = subtract_volumes(collimator_guide_outer, collimator_guide_inner)

        collimator = unite_volumes(
            collimator_body,
            collimator_guide,
            new_name=f"Collimator_{i + 1}",
            translation=[
                0,
                0,
                collimator_body_length_mm_np[i] * 0.5 + collimator_guide_length_mm_np[i] * 0.5,
            ],
        )
        sim.add_volume(collimator, name=f"Collimator_{i + 1}")
        collimator.mother = "world"
        collimator.translation = collimator_body_translation_mm[i]

        rx_0 = Rotation.from_euler("x", -90, degrees=True).as_matrix()
        rz_0 = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        rz_1 = Rotation.from_euler("z", azmuthal_angle_deg[i], degrees=True).as_matrix()
        rx_1 = Rotation.from_euler("x", -polar_angle_deg[i], degrees=True).as_matrix()
        r = rz_1 @ rz_0 @ rx_1 @ rx_0
        collimator.rotation = r
        collimator.material = "Tungsten"

        crystal_box = add_crystal_box(sim, name=f"DetectorCrystal_{i + 1}")
        crystal_box.size = detector_crystal_size_mm
        crystal_box.translation = detector_crystal_translation_mm[i]
        crystal_box.rotation = r
        detector_pixel = sim.add_volume("Box", name=f"pixel_{i + 1}")
        detector_pixel.size = pixel_size_mm
        detector_pixel.mother = crystal_box.name
        pixel_repeater = gate.geometry.volumes.RepeatParametrisedVolume(repeated_volume=detector_pixel)
        pixel_repeater.linear_repeat = n_pixels
        pixel_repeater.translation = pixel_size_mm
        sim.volume_manager.add_volume(pixel_repeater)
        detector_pixel.material = "CsI"

    shielding = gate.geometry.volumes.TesselatedVolume(name="Shielding")
    shielding.mother = "world"
    shielding.file_name = str((stl_dir / "dc_spect_shielding_combined.stl").as_posix())
    shielding.origin_at_cog = False
    sim.add_volume(shielding)
    rz = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    shielding.rotation = rz
    shielding.material = "Lead"


def add_box_source(sim: gate.Simulation, energy_keV: float = 140.0, name: str = "BoxSource", *, args):
    source = gate.sources.generic.GenericSource(name=name)
    source.particle = "gamma"
    source.energy.type = "mono"
    source.activity = args.source_activity_bq * gate.g4_units.Bq
    source.energy.mono = energy_keV * gate.g4_units.keV
    source.position.type = "box"
    source.position.size = [170.1, 170.1, 170.1]
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
        [i * interval_duration, (i + 1) * interval_duration] for i in range(args.num_chunks)
    ]

    expected_events_per_chunk_per_thread = args.source_activity_bq * args.chunk_duration_s
    expected_events_per_chunk = expected_events_per_chunk_per_thread * 1
    expected_total_events = expected_events_per_chunk * args.num_chunks

    print(f"Chunk duration (s): {args.chunk_duration_s}")
    print(f"Number of chunks: {args.num_chunks}")
    print(f"Number of threads: {args.num_threads}")
    print(f"Source activity (Bq): {args.source_activity_bq}")
    print(f"Expected primaries/chunk/thread (mean): {expected_events_per_chunk_per_thread:.3e}")
    print(f"Expected primaries/chunk all threads (mean): {expected_events_per_chunk:.3e}")
    print(f"Expected primaries total all chunks (mean): {expected_total_events:.3e}")

    if expected_events_per_chunk >= args.eventid_warn_threshold:
        print(
            "WARNING: Expected events/chunk is high relative to 32-bit EventID range. "
            "Reduce activity or chunk_duration_s to lower overflow risk."
        )


def process_actors_and_save_srm(readout_actors, output_dir: Path, output_stem: str, args):
    """Convert the in‑memory DigitizerReadoutActor data to sparse SRM .npz files.

    Each actor in *readout_actors* corresponds to one head (index i).
    The actor's ``user_output['hits']`` dictionary provides numpy arrays for the
    required fields (EventPosition_X/Y/Z, PreStepUniqueVolumeID, TotalEnergyDeposit).
    The logic mirrors the previous ROOT‑based implementation but operates entirely
    in memory, avoiding disk I/O and complying with the OpenGate "access output data
    directly from memory" guideline.
    """
    grid_size = args.grid_size
    voxel_size = args.voxel_size
    energy_min = args.energy_min
    energy_max = args.energy_max
    num_voxels = grid_size ** 3
    half_size = grid_size * voxel_size * 0.5
    origin_mm = np.array([-half_size, -half_size, -half_size], dtype=np.float32)

    for i, actor in enumerate(readout_actors):
        # The actor stores hit data in ``user_output['hits']`` – this is a dict of
        # numpy arrays with the same names used in the ROOT trees.
        hits = actor.user_output.get("hits", {})
        if not hits:
            srm = csr_matrix((625, num_voxels), dtype=np.float32)
        else:
            # Apply energy window
            edep = hits.get("TotalEnergyDeposit", np.array([], dtype=np.float32))
            energy_mask = (edep >= energy_min) & (edep <= energy_max)
            if not np.any(energy_mask):
                srm = csr_matrix((625, num_voxels), dtype=np.float32)
            else:
                src_x = hits["EventPosition_X"][energy_mask]
                src_y = hits["EventPosition_Y"][energy_mask]
                src_z = hits["EventPosition_Z"][energy_mask]

                ix = np.floor((src_x - origin_mm[0]) / voxel_size).astype(np.int32)
                iy = np.floor((src_y - origin_mm[1]) / voxel_size).astype(np.int32)
                iz = np.floor((src_z - origin_mm[2]) / voxel_size).astype(np.int32)

                valid_vox = (ix >= 0) & (ix < grid_size) & (iy >= 0) & (iy < grid_size) & (iz >= 0) & (iz < grid_size)
                vol_ids = hits.get("PreStepUniqueVolumeID", np.array([], dtype=object))[energy_mask]
                local_pixel_id = np.array([int(s.rsplit("_", 1)[-1]) for s in vol_ids], dtype=np.int32)
                valid_det = (local_pixel_id >= 0) & (local_pixel_id < 625)
                valid_mask = valid_vox & valid_det

                if not np.any(valid_mask):
                    srm = csr_matrix((625, num_voxels), dtype=np.float32)
                else:
                    final_voxels = ix[valid_mask] * (grid_size * grid_size) + iy[valid_mask] * grid_size + iz[valid_mask]
                    final_pixels = local_pixel_id[valid_mask]
                    ones = np.ones_like(final_voxels, dtype=np.float32)
                    coo = coo_matrix((ones, (final_pixels, final_voxels)), shape=(625, num_voxels))
                    srm = coo.tocsr()
        srm.eliminate_zeros()
        out_name = output_dir / f"srm_{output_stem}_h_{i + 1}.npz"
        save_npz(out_name, srm)
        print(f"Saved SRM for head {i + 1}: {out_name.name} (shape {srm.shape})")


def process_root_and_save_srm(*args, **kwargs):
    raise NotImplementedError("process_root_and_save_srm is obsolete – use process_actors_and_save_srm instead.")


def run_simulation(config: dict, persist_data_dir: Path, args):
    output_dir = Path(args.output_dir).resolve()
    print("Resolved output directory: ", output_dir)
    print(f"Slurm context: job_array_id={args.job_array_id}, job_array_task_id={args.job_array_task_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    job_array_id = args.job_array_id
    job_array_task_id = args.job_array_task_id

    unique_seed = generate_unique_seed(str(job_array_id), str(job_array_task_id))
    print(f"Using random seed: {unique_seed}")

    sim = gate.Simulation(progress_bar=False, output_dir=output_dir)
    sim.random_seed = unique_seed

    add_dc_spect_geometry(sim, config, persist_data_dir)

    if args.debug_geometry:
        print("Geometry debug mode enabled: dumping volume tree and enabling verbose G4 output.")
        print(f"check_volumes_overlap: {sim.check_volumes_overlap}")
        print(sim.volume_manager.dump_volume_tree())
        sim.g4_verbose = True
        sim.g4_verbose_level = 2

    add_box_source(sim, energy_keV=140.0, args=args)
    sim.number_of_threads = int(args.num_threads)
    configure_chunked_run_timing(sim, args)
    print(f"Number of threads: {sim.number_of_threads}")

    output_stem = f"a_{job_array_id}_j_{job_array_task_id}"
    # add_actors now returns a list of the readout actors for in‑memory processing
    readout_actors = add_actors(sim, output_dir, output_stem)
    add_stats_actor(sim, output_dir, output_stem)
    sim.run()

    # Convert the in‑memory actor data to per‑head sparse SRM files
    process_actors_and_save_srm(readout_actors, output_dir, output_stem, args)


def add_stats_actor(sim: gate.Simulation, output_dir: Path, output_stem: str):
    stats_actor = sim.add_actor("SimulationStatisticsActor", "Stats")
    sim.run()
    simulation_stats = stats_actor.user_output["stats"].get_processed_output()
    stats_path = output_dir / f"{output_stem}_sim_stats.txt"
    with open(stats_path, "w") as f:
        for key, value in simulation_stats.items():
            f.write(f"{key}: {value['value']}\n")


def add_actors(sim: gate.Simulation, output_dir: Path, output_stem: str):
    """Create hit and readout actors for all 80 heads.
    Returns a list of the readout actors so that their in‑memory data can be
    accessed after the simulation finishes (see OpenGate docs).
    """
    pixel_array_name = [f"pixel_{i + 1}" for i in range(80)]
    readout_actors = []

    for i in range(80):
        pixel_hits_actor: gate.actors.digitizers.DigitizerHitsCollectionActor = (
            sim.add_actor("DigitizerHitsCollectionActor", f"PixelHits_{i + 1}")
        )
        pixel_hits_actor.attached_to = pixel_array_name[i]
        pixel_hits_actor.output_filename = ""  # no ROOT file written
        pixel_hits_actor.attributes = [
            "RunID",
            "EventID",
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
        pixel_readout_actor.discretize_volume = pixel_array_name[i]
        pixel_readout_actor.policy = "EnergyWeightedCentroidPosition"
        # No file output – keep data in memory
        pixel_readout_actor.output_filename = ""
        readout_actors.append(pixel_readout_actor)

    return readout_actors


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

    sim.visu_commands_vrml = ["/vis/open VRML2FILE", "/vis/drawVolume"]
    sim.visu_commands_vrml.append("/vis/geometry/set/visibility world 0 false")
    for i in range(80):
        sim.visu_commands_vrml.append(f"/vis/geometry/set/visibility DetectorCrystal_{i + 1} 0 false")
    sim.visu_commands_vrml.append("/vis/viewer/flush")

    print("Storing geometry into wrl file only without running the simulation...")
    sim.user_info.visu_filename = str((persist_data_dir.parent / "dev" / "dc_spect_geometry.wrl").resolve())
    print(f"Geometry stored in {sim.user_info.visu_filename}")
    sim.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DC-SPECT simulation and output sparse SRM matrices")
    parser.add_argument("--source-activity-bq", type=float, default=1e6, help="Source activity in Bq.")
    parser.add_argument("--chunk-duration-s", type=float, default=1.0, help="Duration of each run chunk in seconds.")
    parser.add_argument("--num-chunks", type=int, default=1, help="Number of run timing intervals (chunks).")
    parser.add_argument("--eventid-warn-threshold", type=float, default=1.5e9, help="Warn if expected events per chunk exceed this threshold.")
    parser.add_argument("--debug-geometry", action="store_true", help="Dump geometry tree and enable verbose G4 output.")
    parser.add_argument("-t", "--num-threads", type=int, default=1, help="Number of threads requested.")
    parser.add_argument("-x", "--xlsx-path", type=str, default=None, help="Path to geometry configuration xlsx file")
    parser.add_argument("-g", "--geometry-only", action="store_true", help="Store geometry to WRL only.")
    parser.add_argument("-o", "--output-dir", type=str, default=".", help="Directory to store simulation output files")
    parser.add_argument("--job-array-id", type=str, default=None, help="SLURM_ARRAY_JOB_ID for naming output files.")
    parser.add_argument("--job-array-task-id", type=str, default=None, help="SLURM_ARRAY_TASK_ID for naming output files.")

    # SRM parsing options
    parser.add_argument("--grid-size", type=int, default=64, help="Grid dimension (N) for NxNxN voxels.")
    parser.add_argument("--voxel-size", type=float, default=2.0, help="Voxel size in mm.")
    parser.add_argument("--energy-min", type=float, default=0.126, help="Minimum energy window value in MeV.")
    parser.add_argument("--energy-max", type=float, default=0.154, help="Maximum energy window value in MeV.")

    args = parser.parse_args()

    if args.job_array_id is None:
        args.job_array_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"
    if args.job_array_task_id is None:
        args.job_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID") or os.environ.get("SLURM_PROCID") or "0"

    base_dir = Path(__file__).resolve().parents[3]
    persistent_data_dir = base_dir / "persistent_data"

    if args.xlsx_path is not None:
        xlsx_path = Path(args.xlsx_path)
    else:
        xlsx_path = persistent_data_dir / "spreadsheet" / "MDSL.excel80M10RFR.cut-plate.010.150roi.2.30pin.105ellipse.xlsx"

    if not xlsx_path.exists():
        raise FileNotFoundError(f"Geometry configuration xlsx file not found at {xlsx_path}")

    config = get_dc_spect_geometry_config(xlsx_path)
    for key, value in config.items():
        if isinstance(value, np.ndarray):
            assert value.shape[0] == 80
        elif isinstance(value, list):
            assert len(value) == 3

    if args.geometry_only:
        save_simulation_geometry_to_wrl(config, persistent_data_dir, args)
        exit(0)
    else:
        run_simulation(config, persistent_data_dir, args)
