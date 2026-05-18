from pathlib import Path

import numpy as np
import opengate as gate
import pandas as pd


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


# Generate a grid of 3D Boxes with center at the origin
def generate_grid_points(grid: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    x_grid, y_grid, z_grid = [
        np.arange(n) * s - (n - 1) * s * 0.5 for n, s in zip(grid, spacing)
    ]
    return np.stack(np.meshgrid(x_grid, y_grid, z_grid, indexing="xy"), axis=-1)


def test_sim():
    sim = gate.Simulation()

    dc_spect_config = get_dc_spect_geometry_config(
        Path("/home/fanghan/Work/RPIL/QMIRT/gate10mc/persistent_data/")
        / "spreadsheet/MDSL.excel80M10RFR.cut-plate.010.150roi.2.30pin.105ellipse.xlsx"
    )
    monolithic_crystal_translation_raw = np.array(
        dc_spect_config["detector_crystal_translation_mm"]
    )
    monolithic_crystal_translation = monolithic_crystal_translation_raw[::2]

    # Create a dummy translation array
    # 20 heads, each is a fixed distance from the origin, but with different angles
    n_heads = 40
    n_rows = 4
    # radius_mm = 200.0  # distance from the origin to each head
    # angles_deg = np.linspace(0, 360, n_heads // n_rows, endpoint=False)
    # # Repeat the angles 4 times to get 80 heads
    # # angles_deg = np.tile(angles_deg, 2)
    # angles_deg = np.tile(angles_deg.reshape(-1, 1), (n_rows, 1)).flatten()

    # row_height_mm = 30.0
    # monolithic_crystal_translation_z = (
    #     np.tile(np.arange(n_rows).reshape(-1, 1), (1, n_heads // n_rows))
    #     * row_height_mm
    #     - (n_rows - 1) * row_height_mm * 0.5
    # ).flatten()

    # monolithic_crystal_translation = np.column_stack(
    #     (
    #         radius_mm * np.cos(np.radians(angles_deg)),
    #         radius_mm * np.sin(np.radians(angles_deg)),
    #         monolithic_crystal_translation_z,
    #     )
    # )

    n_pixels = np.array([2, 2, 1])
    monolithic_crystal_size = np.array([50.0, 50.0, 10.0])
    pixel_box_size = monolithic_crystal_size / n_pixels
    # local_array_translations = gate.geometry.utility.get_grid_repetition(
    #     size=n_pixels, spacing=pixel_box_size
    # )
    local_array_translations = list(
        generate_grid_points(n_pixels, pixel_box_size).reshape(-1, 3)
    )
    global_pixel_translations = []
    for i in range(n_heads):
        global_pixel_translations.extend(
            [
                local_translation + monolithic_crystal_translation[i]
                for local_translation in local_array_translations
            ]
        )
    print(
        f"Global pixel translations shape: {np.array(global_pixel_translations).shape}"
    )
    print(f"Pixel box size: {pixel_box_size} mm")
    detector_pixel = sim.add_volume("Box", "pixel")
    detector_pixel.size = pixel_box_size
    detector_pixel.translation = global_pixel_translations
    sim.user_info.visu = True
    sim.user_info.visu_type = "vrml_file_only"
    print("Storing geometry into wrl file only without running the simulation...")
    sim.user_info.visu_filename = "test_repeat_box_geometry.wrl"
    print(f"Geometry stored in {sim.user_info.visu_filename}")
    sim.run()


if __name__ == "__main__":
    test_sim()
