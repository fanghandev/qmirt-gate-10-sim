import pandas as pd
from pathlib import Path
import numpy as np
import opengate as gate
from opengate.geometry.volumes import unite_volumes, subtract_volumes, intersect_volumes
import opengate_core as g4
from scipy.spatial.transform import Rotation


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
    detector_crystal_center_dist_mm_np = hole_fov_center_distance_mm_np + collimator_body_length_mm_np + detector_crystal_size_mm[2]*0.5
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
    collimator_guide_distance_mm_np = hole_fov_center_distance_mm_np - collimator_guide_length_mm_np
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

    return{
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
        "polar_angle_deg": polar_angle_deg
    }

def add_dc_spect_geometry(sim: gate.Simulation, config: dict):

    collimator_body_length_mm_np = config["collimator_body_length_mm_np"]
    collimator_hole_coords_mm_np = config["collimator_hole_coords_mm_np"]
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
        collimator_body_outer.dz = collimator_body_length_mm_np[i]*0.5
        collimator_body_inner.dx1 = collimator_body_inner_top_mm_np[i] * 0.5
        collimator_body_inner.dy1 = collimator_body_inner_top_mm_np[i] * 0.5
        collimator_body_inner.dx2 = collimator_body_inner_bottom_mm_np[i] * 0.5
        collimator_body_inner.dy2 = collimator_body_inner_bottom_mm_np[i] * 0.5
        collimator_body_inner.dz = collimator_body_length_mm_np[i]*0.5+0.1
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
        collimator_guide_outer.dz = collimator_guide_length_mm_np[i]*0.5
        collimator_guide_inner.dx1 = collimator_guide_inner_top_mm_np[i] * 0.5
        collimator_guide_inner.dy1 = collimator_guide_inner_top_mm_np[i] * 0.5
        collimator_guide_inner.dx2 = collimator_guide_inner_bottom_mm_np[i] * 0.5
        collimator_guide_inner.dy2 = collimator_guide_inner_bottom_mm_np[i] * 0.5
        collimator_guide_inner.dz = collimator_guide_length_mm_np[i]*0.5+0.1  # add a small extra length to ensure the subtraction works correctly
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
                collimator_body_length_mm_np[i]*0.5 + collimator_guide_length_mm_np[i]*0.5,
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
        r = rz_1@rz_0@rx_1@rx_0
        collimator.rotation = r

        # Add in detector crystal volume
        detector_crystal = gate.geometry.volumes.BoxVolume(  # type: ignore
            name=f"DetectorCrystal_{i+1}"
        )
        detector_crystal.size = detector_crystal_size_mm
        detector_crystal.mother = "world"
        detector_crystal.translation = detector_crystal_translation_mm[i]
    
        sim.add_volume(detector_crystal, name=f"DetectorCrystal_{i+1}")
        detector_crystal.rotation = r

    front_shielding = gate.geometry.volumes.TesselatedVolume(name="FrontShielding")  # type: ignore
    front_shielding.mother = "world"
    front_shielding.file_name = str(
        (persistent_data_dir / "stl" / "front_shielding.stl").as_posix()
    )
    front_shielding.origin_at_cog = False
    sim.add_volume(front_shielding)
    rz = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    front_shielding.rotation = rz
    # print(sim.volume_manager.dump_volume_tree())



if __name__ == "__main__":
    base_dir = Path(__file__).parent.parents[2]
    persistent_data_dir = base_dir / "persistent_data"
    xlsx_path = (
        persistent_data_dir
        / "spreadsheet"
        / "MDSL.excel80M10RFR.cut-plate.010.150roi.2.30pin.105ellipse.xlsx"
    )
    config = get_dc_spect_geometry_config(xlsx_path)

    # CREATE_MC_GEOM = False
    CREATE_MC_GEOM = True

    if not CREATE_MC_GEOM:
        exit(0)

    sim = gate.Simulation()
    add_dc_spect_geometry(sim, config)


    sim.visu = True
    sim.visu_type = "vrml_file_only"
    sim.visu_filename = "../collimator_geometry.wrl"  # type: ignore
    sim.run()
