from pathlib import Path

import numpy as np
import pandas as pd


def get_dc_spect_geometry_config(xlsx_path: Path) -> pd.DataFrame:
    n_heads = 80
    hole_size_mm = 2.3  # unit is mm
    wall_thickness_mm = 2.0  # unit is mm
    guide_length_mm = 3.0
    crystal_size_mm = np.array([50.0, 10.0, 50.0])  # unit is mm

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

    body_l_mm_np = df_coords["length of collimator"].to_numpy(dtype=float)
    hole_center_mm_np = df_coords[
        [
            "x coordinate value at center of hole",
            "y coordinate value at center of hole",
            "z coordinate value at center of hole",
        ]
    ].to_numpy(dtype=float)

    body_l_corrected_mm_np = body_l_mm_np - wall_thickness_mm

    body_inner_bottom_mm_np = np.full((n_heads,), hole_size_mm)
    body_outer_bottom_mm_np = body_inner_bottom_mm_np + wall_thickness_mm * 2
    body_inner_top_mm_np = np.full((n_heads,), crystal_size_mm[0])
    body_inner_top_corrected_mm_np = (
        body_inner_top_mm_np - body_inner_bottom_mm_np
    ) * body_l_corrected_mm_np / body_l_mm_np + body_inner_bottom_mm_np
    body_outer_top_mm_np = body_inner_top_corrected_mm_np + wall_thickness_mm * 2
    hole_r_mm_np = np.linalg.norm(hole_center_mm_np, axis=1)
    z_axis_angle_deg = np.degrees(
        np.arctan2(hole_center_mm_np[:, 1], hole_center_mm_np[:, 0])
    )
    x_axis_angle_deg = np.degrees(np.arctan2(hole_center_mm_np[:, 2], hole_r_mm_np))

    guide_l_mm_np = np.full((n_heads,), guide_length_mm)
    guide_exit_angle_rad = np.arctan2(
        (body_inner_top_mm_np + body_inner_bottom_mm_np) * 0.5, body_l_mm_np
    )
    guide_inner_top_mm_np = np.full((n_heads,), hole_size_mm)
    guide_outer_top_mm_np = guide_inner_top_mm_np + wall_thickness_mm * 2
    guide_inner_bottom_mm_np = (
        guide_inner_top_mm_np + np.tan(guide_exit_angle_rad) * guide_l_mm_np * 2
    )
    guide_outer_bottom_mm_np = guide_inner_bottom_mm_np + wall_thickness_mm * 2

    box_inner_size_mm = np.stack(
        [
            body_inner_top_mm_np + 1.2,
            body_inner_top_mm_np + 1.2,
            np.full((n_heads,), crystal_size_mm[1]),
        ],
        axis=1,
    )
    box_outer_size_mm = np.stack(
        [
            box_inner_size_mm[:, 0] + wall_thickness_mm * 2,
            box_inner_size_mm[:, 1] + wall_thickness_mm * 2,
            box_inner_size_mm[:, 2] + wall_thickness_mm + 4,
        ],
        axis=1,
    )

    # Create a pandas DataFrame to hold the geometry configuration
    geometry_config_df = pd.DataFrame(
        {
            "hole_center_x (mm)": hole_center_mm_np[:, 0],
            "hole_center_y (mm)": hole_center_mm_np[:, 1],
            "hole_center_z (mm)": hole_center_mm_np[:, 2],
            "wall_thickness (mm)": np.full((n_heads,), wall_thickness_mm),
            "hole_r (mm)": hole_r_mm_np,
            "body_l (mm)": body_l_mm_np,
            "body_l_corrected (mm)": body_l_corrected_mm_np,
            "body_inner_top (mm)": body_inner_top_mm_np,
            "body_inner_top_corrected (mm)": body_inner_top_corrected_mm_np,
            "body_outer_top (mm)": body_outer_top_mm_np,
            "body_inner_bottom (mm)": body_inner_bottom_mm_np,
            "body_outer_bottom (mm)": body_outer_bottom_mm_np,
            "guide_l (mm)": guide_l_mm_np,
            "guide_inner_top (mm)": guide_inner_top_mm_np,
            "guide_outer_top (mm)": guide_outer_top_mm_np,
            "guide_inner_bottom (mm)": guide_inner_bottom_mm_np,
            "guide_outer_bottom (mm)": guide_outer_bottom_mm_np,
            "box_inner_size_x (mm)": box_inner_size_mm[:, 0],
            "box_inner_size_y (mm)": box_inner_size_mm[:, 1],
            "box_inner_size_z (mm)": box_inner_size_mm[:, 2],
            "box_outer_size_x (mm)": box_outer_size_mm[:, 0],
            "box_outer_size_y (mm)": box_outer_size_mm[:, 1],
            "box_outer_size_z (mm)": box_outer_size_mm[:, 2],
            "z_axis_angle (deg)": z_axis_angle_deg,
            "x_axis_angle (deg)": x_axis_angle_deg,
        }
    )
    return geometry_config_df


if __name__ == "__main__":
    from qmirt.utils.filesystem import find_project_root

    project_root_dir = find_project_root(start_path=Path(__file__), marker=".git")
    xlsx_path = (
        project_root_dir
        / "persistent_data/cardiac_spect/spreadsheet"
        / "MDSL.excel80M10RFR.cut-plate.010.150roi.2.30pin.105ellipse.xlsx"
    )
    config_df = get_dc_spect_geometry_config(xlsx_path)
    # Save the DataFrame to a CSV file for inspection
    outdir = project_root_dir / "persistent_data/cardiac_spect/spreadsheet"
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv_filename = (
        "MDSL.excel80M10RFR.cut-plate.010.150roi.2.30pin.105ellipse_geometry_config.csv"
    )
    out_html_filename = "MDSL.excel80M10RFR.cut-plate.010.150roi.2.30pin.105ellipse_geometry_config.html"
    config_df.to_csv(
        outdir / out_csv_filename,
        index=False,
    )
    # Save to html for inspection
    config_df.to_html(
        outdir / out_html_filename,
        index=False,
    )
    print(
        "Geometry configuration DataFrame saved:\n"
        + f"Output directory: {outdir}\n"
        + f"csv file:  {out_csv_filename}\n"
        + f"html file: {out_html_filename}"
    )
