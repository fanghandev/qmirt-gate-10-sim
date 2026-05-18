import pandas as pd
from pathlib import Path
import numpy as np
import opengate as gate
from opengate.geometry.volumes import (
    unite_volumes,
    subtract_volumes,
    intersect_volumes,
)
import opengate_core as g4
from scipy.spatial.transform import Rotation

if __name__ == "__main__":

    base_dir = Path(__file__).parent.parents[2]
    persistent_data_dir = base_dir / "persistent_data"
    assert (
        persistent_data_dir.exists()
    ), f"Persistent data directory does not exist: {persistent_data_dir}"
    xlsx_path = (
        persistent_data_dir
        / "spreadsheet"
        / "MDSL.excel80M10RFR.cut-plate.010.150roi.2.30pin.105ellipse.xlsx"
    )

    df_coord = pd.read_excel(
        xlsx_path, sheet_name="Coordinates"
    )  # read the "Values" sheet
    df_coord.columns = df_coord.iloc[0]
    df_coord = df_coord[1:]  # remove the first row which is now the header
    df_coord = df_coord.reset_index(
        drop=True
    )  # reset the index after removing the first row
    df_coord = df_coord.apply(
        pd.to_numeric, errors="coerce"
    )  # convert all columns to numeric, coercing errors to NaN
    df_coord.columns.name = "Coordinate Sheet"

    hole_c_mm = np.array(
        [   
            df_coord["x coordinate value at center of hole"].values,
            df_coord["y coordinate value at center of hole"].values,
            df_coord["z coordinate value at center of hole"].values,
        ]
    ).T

    # Print the shape of the hole center coordinates array
    print(f"Hole center coordinates shape: {hole_c_mm.shape}")
    for i in range(10):
        print(f"Hole {i} center coordinates (mm): {hole_c_mm[i]}")

    sim = gate.Simulation()
    alignment_box_list = []
    front_shielding = gate.geometry.volumes.TesselatedVolume(name="FrontShielding")
    front_shielding.mother = "world"
    front_shielding.origin_at_cog = False
    front_shielding.file_name = str(
        (persistent_data_dir / "stl" / "front_shielding.stl").as_posix()
    )
    front_shielding.rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    print("front shielding volume: ", front_shielding.solid_info.cubic_volume)
    sim.add_volume(front_shielding)

    for i in range(20):
        alignment_box = gate.geometry.volumes.BoxVolume(name=f"AlignmentBox_{i}")
        alignment_box.mother = "world"
        alignment_box.size = [10, 10, 10]
        alignment_box.translation = hole_c_mm[i]
        sim.add_volume(alignment_box)
        alignment_box_list.append(alignment_box)

    sim.volume_manager.dump_volume_tree()

    sim.visu = True
    sim.visu_type = "vrml_file_only"
    sim.visu_filename = "../validation_geometry.wrl"
    sim.run()
