from pathlib import Path

import numpy as np
import opengate as gate
import polars as pl
from opengate.geometry.volumes import subtract_volumes, unite_volumes
from qmirt.utils.simulation import resolve_simulation_runtime_context
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
    crystal_center_r = 179.61

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
    collimator.rotation = get_head_rotation_matrix(pl_df, id)

    collimator.translation = get_collimator_center(config, pl_df, id).tolist()

    collimator.name = f"Collimator_{id + 1}"
    collimator.material = "Tungsten"


def get_collimator_center(config: dict, pl_df: pl.DataFrame, id: int) -> np.ndarray:
    """World position of the collimator volume's origin (Frustum_A's centre).

    The pinhole sits (h_body - h_nozzle) / 2 above that origin along the local z
    axis, so shift the pinhole back by the rotated offset.
    """
    pinhole = np.array(
        [pl_df.item(id, f"Pinhole_{axis}") for axis in ("x", "y", "z")]
    )
    z_offset = (
        config["collimator definition"]["h_nozzle"]
        - config["collimator definition"]["h_body"]
    ) * 0.5
    local_offset = np.array([0.0, 0.0, z_offset])
    return pinhole + get_head_rotation_matrix(pl_df, id) @ local_offset


def get_collimator_outer_primitives(config: dict) -> list[dict]:
    """Outer envelope of construct_collimator_geometry as Geant4 primitives in the
    collimator's local frame (bore subtractions omitted): Frustum_A plus Box_A.
    Trd values are G4Trd half-lengths (dx1/dy1 at -dz, dx2/dy2 at +dz)."""
    c = config["collimator definition"]
    return [
        {
            "type": "trd",
            "dx1": c["l_bottom_outer"] * 0.5,
            "dy1": c["l_bottom_outer"] * 0.5,
            "dx2": c["l_top"] * 0.5,
            "dy2": c["l_top"] * 0.5,
            "dz": (c["h_nozzle"] + c["h_body"]) * 0.5,
            "offset_mm": [0.0, 0.0, 0.0],
        },
        {
            "type": "box",
            "size_mm": [c["l_bottom_outer"], c["l_bottom_outer"], c["h_box"]],
            "offset_mm": [0.0, 0.0, -0.5 * (c["h_nozzle"] + c["h_body"] + c["h_box"])],
        },
    ]


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
    pixel_repeater = gate.geometry.volumes.RepeatParametrisedVolume(
        repeated_volume=detector_pixel
    )
    pixel_repeater.linear_repeat = n_pixels
    pixel_repeater.translation = pixel_size_mm
    sim.volume_manager.add_volume(pixel_repeater)
    detector_pixel.material = "CsI"


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
    shielding.rotation = get_shielding_rotation_matrix()
    shielding.material = "Lead"


def add_shield_pieces_to_gate_sim(sim: gate.Simulation, pieces_dir: str | Path):
    """Shield as closed STL pieces cut from the original by split_brain_shield_stl.py.

    Same solid, but each piece has a small bounding box, so Geant4 only consults
    the pieces near a photon. Pieces are stored in world coordinates.
    """
    import json

    pieces_dir = Path(pieces_dir)
    manifest = json.loads((pieces_dir / "manifest.json").read_text())
    for record in manifest["pieces"]:
        piece_path = pieces_dir / record["file"]
        if not piece_path.exists():
            raise FileNotFoundError(f"Shield piece STL not found at: {piece_path}")
        piece = gate.geometry.volumes.TesselatedVolume(
            name=f"Shielding_{Path(record['file']).stem}"
        )
        piece.mother = "world"
        piece.file_name = piece_path.as_posix()
        piece.origin_at_cog = False
        sim.add_volume(piece)
        piece.material = "Lead"


def get_shielding_rotation_matrix() -> np.ndarray:
    rx = Rotation.from_euler("x", -90, degrees=True).as_matrix()
    rz = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    return rx @ rz


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


def check_fov_clears_collimators(pl_df: pl.DataFrame, shape: str, size_mm: float):
    """Refuse an FOV volume that would overlap a collimator nozzle tip.

    The nozzle tips are the hardware closest to the centre (the lead shield's inner
    surface is at r = 145 mm), so the FOV's farthest point must stay inside them.
    """
    pinhole_r = np.sqrt(
        pl_df["Pinhole_x"] ** 2 + pl_df["Pinhole_y"] ** 2 + pl_df["Pinhole_z"] ** 2
    ).to_numpy()
    # Longest nozzle of any variant, so this holds for every mapping mode.
    h_nozzle = max(
        get_geometry_base_definition(v)["collimator definition"]["h_nozzle"]
        for v in range(4)
    )
    tip_r = float(np.min(pinhole_r) - h_nozzle)
    shape_name = str(shape).lower()
    fov_r = 0.5 * float(size_mm) * (np.sqrt(3.0) if shape_name == "box" else 1.0)
    if fov_r >= tip_r:
        raise ValueError(
            f"FOV {shape_name} of size {size_mm} mm reaches r = {fov_r:.2f} mm, which "
            f"overlaps the collimator nozzle tips at r = {tip_r:.2f} mm "
            f"(max sphere diameter {2 * tip_r:.2f} mm)."
        )


def add_geometry_to_gate_sim(sim: gate.Simulation, pl_df: pl.DataFrame, args):
    n_crystals = pl_df.shape[0]
    check_fov_clears_collimators(pl_df, args.fov_shape, args.fov_size_mm)
    for id in range(n_crystals):
        mapped_id = map_crystal_id(id, n_crystals, args.mapping_mode)
        config = get_geometry_base_definition(mapped_id)
        if args.mode == "geometry-only":
            config["crystal definition"]["n_pixels"] = [1, 1, 1]
        add_collimator_to_gate_sim(sim, config, pl_df, id)
        add_pixelated_detector_to_gate_sim(sim, config, pl_df, id)
    if args.mode != "phantom":
        add_fov_volume_to_gate_sim(sim, shape=args.fov_shape, size_mm=args.fov_size_mm)
    if args.with_shielding:
        model = resolve_shield_model(args)
        if model == "csg":
            from brain_spect_shield_csg import add_csg_shield_to_gate_sim

            add_csg_shield_to_gate_sim(sim, pl_df, layout="tiles")
        elif model == "pieces":
            add_shield_pieces_to_gate_sim(sim, args.shield_pieces_dir)
        else:
            add_shielding_to_gate_sim(sim, config)
    if args.mode == "phantom":
        # after all hardware (incl. the shield), so the whole scanner can move to the
        # parallel world; the phantom replaces the FOV volume (its fittings may reach past it)
        from phantom_gate import add_phantom

        if hardware_in_parallel_world(args):
            move_hardware_to_parallel_world(sim)
        add_phantom(sim, load_phantom_spec(args), files_dir=Path(args.output_dir) / "phantom_image",
                    image_geometry=args.image_geometry)


def resolve_shield_model(args) -> str:
    """'stl' (the 186,772-facet STL), 'pieces' (STL pieces from --shield-pieces-dir,
    e.g. the 0.05 mm simplified shield) or 'csg' (analytic G4Sphere tiles minus the
    apertures, brain_spect_shield_csg.py; ~3x faster than the simplified STL and
    within 0.1% of the STL's crystal counts, see dev/python/brain_shield_csg.md)."""
    model = getattr(args, "shield_model", "stl") or "stl"
    pieces_dir = getattr(args, "shield_pieces_dir", None)
    if model == "stl" and pieces_dir:
        model = "pieces"  # backward compatible: --shield-pieces-dir alone
    if model == "pieces" and not pieces_dir:
        raise ValueError("--shield-model pieces needs --shield-pieces-dir")
    if model == "csg" and pieces_dir:
        raise ValueError("--shield-pieces-dir cannot be combined with --shield-model csg")
    return model


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

    sim.user_info.visu = True
    sim.user_info.visu_type = "vrml_file_only"
    sim.visu_commands_vrml = ["/vis/open VRML2FILE", "/vis/drawVolume"]
    sim.visu_commands_vrml.append("/vis/geometry/set/visibility world 0 false")
    sim.visu_commands_vrml.append("/vis/viewer/flush")
    print("Storing geometry into wrl file only without running the simulation...")
    sim.user_info.visu_filename = str(output_dir / "brain_spect_geometry.wrl")
    sim.run(start_new_process=True)
    print(f"Geometry stored in:\n  {sim.user_info.visu_filename}")


def generate_unique_seed(job_array_id: str, job_array_task_id: str) -> int:
    from hashlib import md5
    from os import times

    seed_string = f"gate_sim_{job_array_id}_{job_array_task_id}"
    # Also add timestamp to ensure uniqueness across different runs, if needed
    seed_string += f"_{times()}"
    return int(md5(seed_string.encode()).hexdigest()[:8], 16)


def add_fov_volume_to_gate_sim(
    sim: gate.Simulation, shape: str = "box", size_mm: float = 150.0
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


def add_fov_box_to_gate_sim(sim: gate.Simulation, size_mm: float = 150.0):
    return add_fov_volume_to_gate_sim(sim, shape="box", size_mm=size_mm)


def add_fov_sphere_to_gate_sim(sim: gate.Simulation, size_mm: float = 150.0):
    return add_fov_volume_to_gate_sim(sim, shape="sphere", size_mm=size_mm)


PHANTOM_SCATTER_ATTRIBUTES = ["PhantomCompton", "PhantomRayleigh"]
PHANTOM_LAST_INTERACTION = "PhantomLastInteraction"
HARDWARE_WORLD = "hardware"


def hardware_in_parallel_world(args) -> bool:
    """Image phantoms are one box (ImageVolume) that reaches through the shield and into
    collimators; overlapping volumes make Geant4 navigation undefined. The standard GATE 10
    remedy: the voxel phantom stays in the mass world and the hardware goes to a parallel
    world with layered mass geometry, whose materials take precedence where they overlap
    (the hardware only overlaps the image's air voxels). Analytic phantoms clear the hardware
    and stay in one world."""
    if args.hardware_world != "auto":
        return args.hardware_world == "parallel"
    return phantom_scatter_volume(args) == "phantom_image"


def phantom_scatter_volume(args) -> str:
    """The phantom's outermost volume (it contains every other phantom volume)."""
    from phantom_models import is_image

    image = is_image(load_phantom_spec(args).phantom)
    return "phantom_image" if image and args.image_geometry == "voxel" else "phantom_body"


def move_hardware_to_parallel_world(sim: gate.Simulation):
    """Re-parent every top-level hardware volume (collimators, crystals, shield) to a
    layered-mass parallel world. Call after the hardware and before the phantom."""
    sim.add_parallel_world(HARDWARE_WORLD)
    moved = 0
    for volume in list(sim.volume_manager.volumes.values()):
        if volume.name != HARDWARE_WORLD and volume.mother == "world":
            volume.mother = HARDWARE_WORLD
            moved += 1
    print(f"Hardware in parallel world '{HARDWARE_WORLD}': {moved} top-level volumes")


def load_phantom_spec(args):
    import json

    from phantom_models import Spec

    return Spec.from_dict(json.loads(Path(args.phantom_spec).read_text()))


def add_phantom_source(sim: gate.Simulation, args):
    """Phantom activity: --phantom-activity-bq decays/s at time 0 in --phantom-activity-region
    ("all": the whole phantom; "brain": an image phantom's brain, the rest scaled by the
    map). opengate gives every thread its own source, so each thread gets 1/threads of it.
    Returns the per-thread source activity in decays/s (for the EventID checks; an upper
    bound on primaries)."""
    from phantom_gate import add_activity_source
    from phantom_models import activity_fraction

    spec = load_phantom_spec(args)
    total_bq = args.phantom_activity_bq / activity_fraction(spec, args.phantom_activity_region)
    radionuclide = None if args.radionuclide == "none" else args.radionuclide
    source = add_activity_source(sim, spec, total_bq / args.num_threads, radionuclide=radionuclide,
                                 half_life=not args.no_decay)
    args.phantom_total_activity_bq = total_bq
    print(f"Phantom {spec.phantom}: {total_bq:.4e} Bq in total ({args.phantom_activity_bq:.4e} Bq in "
          f"{args.phantom_activity_region}), {radionuclide or 'mono 140 keV'}, decay {not args.no_decay}")
    return source.activity / gate.g4_units.Bq


def activate_phantom_scatter_attributes(sim: gate.Simulation, args) -> list[str]:
    """Per-track counts of Compton and Rayleigh steps in the phantom (its outermost volume;
    the count includes every daughter), inherited by secondaries, so each single says
    whether its photon scattered in the phantom (object scatter) before reaching the
    detector. Returns the digitizer attribute names. opengate >= 10.1.1 has auxiliary
    attributes; 10.1.0 (the cluster container) the actor-based
    ProcessDefinedStepInVolumeAttribute, named ProcessDefinedStep__<process>__<volume>;
    both count the same (checked 2026-10-09 on the same toy geometry)."""
    volume = phantom_scatter_volume(args)
    names = []
    if hasattr(sim, "activate_auxiliary_attribute"):
        for name, process in zip(PHANTOM_SCATTER_ATTRIBUTES, ("compt", "Rayl")):
            attribute = sim.activate_auxiliary_attribute("ProcessDefinedStepInVolumeAttribute", name)
            attribute.process_name = process
            attribute.volume_name = volume
            attribute.propagate_from_parent_track = True
            names.append(name)
    else:
        from opengate.actors.digitizers import ProcessDefinedStepInVolumeAttribute

        names = [ProcessDefinedStepInVolumeAttribute(sim, process, volume).name for process in ("compt", "Rayl")]
    if hardware_in_parallel_world(args):
        # with layered mass geometry the counts see only the mass world: Compton steps in
        # hardware inside the image box count as "phantom". The last interaction position
        # in the box tells them apart (reduce_phantom_singles.py: tissue voxel or not).
        if not hasattr(sim, "activate_auxiliary_attribute"):
            raise RuntimeError("--hardware-world parallel needs opengate >= 10.1.1 (LastInteractionPositionInVolumeAttribute)")
        attribute = sim.activate_auxiliary_attribute("LastInteractionPositionInVolumeAttribute", PHANTOM_LAST_INTERACTION)
        attribute.volume_name = volume
        attribute.propagate_from_parent_track = True
        names.append(PHANTOM_LAST_INTERACTION)
    return names


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
    source.activity = args.source_activity_bq * gate.g4_units.Bq
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
        # fov_size_mm is a diameter, matching add_fov_volume_to_gate_sim.
        source.position.radius = fov_size * 0.5 * gate.g4_units.mm
        source_obj = sim.add_source(source, name=name)
        source_obj.attached_to = "FOVSphere"
        return source_obj
    raise ValueError(f"Unsupported FOV shape: {fov_shape!r}. Use 'box' or 'sphere'.")


def add_stats_actor(sim: gate.Simulation, output_dir: Path, output_stem: str):
    stats_actor = sim.add_actor("SimulationStatisticsActor", "Stats")  # type: ignore
    stats_path = output_dir / f"{output_stem}_sim_stats.txt"
    # GATE will automatically write to this file after sim.run() finishes
    stats_actor.output_filename = str(stats_path)


def get_repo_git_commit() -> str:
    import subprocess

    try:
        repo_dir = Path(__file__).resolve().parent
        commit = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return f"{commit}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_run_manifest(
    output_dir: Path, output_stem: str, args, unique_seed: int
) -> None:
    """Persist the exact resolved parameters used for this task, for reproducibility."""
    import json
    import time

    manifest = {
        "script": Path(__file__).name,
        "repo_git_commit": get_repo_git_commit(),
        "generated_at": time.time(),
        "random_seed": unique_seed,
        "parameters": vars(args),
    }
    (output_dir / f"{output_stem}_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )


HIT_ATTRIBUTES = [
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
MERGED_SINGLES_TREE = "Singles"


def add_digitizer_chain(
    sim: gate.Simulation,
    pixel_volumes: str | list[str],
    names: tuple[str, str, str],
    singles_root_path: Path,
    blur_fwhm_kev: float,
    extra_attributes: list[str] | None = None,
):
    """Hits -> readout -> Gaussian energy blur; the blur actor's name is the tree.
    With blur_fwhm_kev = 0 the readout itself writes the tree (true deposited energy;
    blur in post-processing)."""
    hits_name, readout_name, singles_name = names
    if blur_fwhm_kev <= 0:
        readout_name = singles_name
    # Keep hits in-memory only as input to the singles chain.
    hits_actor: gate.actors.digitizers.DigitizerHitsCollectionActor = sim.add_actor(
        "DigitizerHitsCollectionActor", hits_name
    )
    hits_actor.attached_to = pixel_volumes
    hits_actor.output_filename = ""
    hits_actor.attributes = HIT_ATTRIBUTES + list(extra_attributes or [])
    readout_actor = sim.add_actor("DigitizerReadoutActor", readout_name)
    readout_actor.input_digi_collection = hits_actor.name
    # Discretization works by tree depth, which is the same for every head, so
    # any pixel volume serves; hits are grouped per pixel (unique volume ID).
    first_volume = pixel_volumes if isinstance(pixel_volumes, str) else pixel_volumes[0]
    readout_actor.discretize_volume = first_volume
    readout_actor.policy = "EnergyWeightedCentroidPosition"
    readout_actor.output_filename = ""
    if blur_fwhm_kev <= 0:
        readout_actor.output_filename = singles_root_path
        return
    blur_actor: gate.actors.digitizers.DigitizerBlurringActor = sim.add_actor(
        "DigitizerBlurringActor", singles_name
    )
    blur_actor.attached_to = pixel_volumes
    blur_actor.input_digi_collection = readout_actor.name
    blur_actor.blur_attribute = "TotalEnergyDeposit"
    blur_actor.blur_method = "Gaussian"
    blur_actor.blur_fwhm = blur_fwhm_kev * gate.g4_units.keV
    blur_actor.output_filename = singles_root_path


def add_actors(
    sim: gate.Simulation,
    n_crystals: int,
    output_dir: Path,
    output_stem: str,
    energy_resolution: float = 0.10,
    energy_resolution_reference_kev: float = 140.0,
    layout: str = "merged",
    extra_attributes: list[str] | None = None,
):
    """Singles digitizer for all heads.

    layout="merged": one chain over all pixel volumes writing one "Singles" tree;
    the head is in PreStepUniqueVolumeID ("pixel_<head>_param-..."). Every actor
    has a fixed per-event cost, so this is much cheaper than 73 chains.
    layout="per-head": the original 73 chains writing "Pixel_<head>_Singles" trees.
    """
    pixel_array_name = [f"pixel_{i + 1}" for i in range(n_crystals)]
    if energy_resolution < 0:
        raise ValueError("energy_resolution must be >= 0")
    blur_fwhm_kev = energy_resolution * energy_resolution_reference_kev
    singles_root_path = output_dir / f"pixel_singles_{output_stem}.root"

    if layout == "merged":
        add_digitizer_chain(
            sim,
            pixel_array_name,
            ("PixelHits", "PixelReadout", MERGED_SINGLES_TREE),
            singles_root_path,
            blur_fwhm_kev,
            extra_attributes,
        )
    elif layout == "per-head":
        for i in range(n_crystals):
            add_digitizer_chain(
                sim,
                pixel_array_name[i],
                (f"PixelHits_{i + 1}", f"Pixel_{i + 1}_Readout", f"Pixel_{i + 1}_Singles"),
                singles_root_path,
                blur_fwhm_kev,
                extra_attributes,
            )
    else:
        raise ValueError(f"Unsupported actor layout: {layout!r}")


def configure_chunked_run_timing(sim: gate.Simulation, args):
    if args.chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be > 0")
    if args.num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")
    if args.source_activity_bq <= 0:
        raise ValueError("source_activity_bq must be > 0")

    sec = gate.g4_units.s
    interval_duration = args.chunk_duration_s * sec
    # absolute times (a time-sliced job starts at --time-start-s; source decay uses them)
    start = getattr(args, "time_start_s", 0.0) * sec
    sim.run_timing_intervals = [
        [start + i * interval_duration, start + (i + 1) * interval_duration]
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

    if expected_events_per_chunk >= args.eventid_hard_limit:
        raise ValueError(
            "Expected events per timing chunk exceed the 32-bit EventID limit: "
            f"{expected_events_per_chunk:.3e} >= {args.eventid_hard_limit:.3e}. "
            "Reduce source activity, chunk_duration_s, or num_threads."
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
    print(
        "Slurm context: "
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
    # Geant4's overlap check runs by default and repeats in every loop; check once
    # per geometry change instead (--check-overlaps)
    sim.check_volumes_overlap = bool(getattr(args, "check_overlaps", False))
    sim.physics_manager.physics_list_name = getattr(args, "physics_list", "QGSP_BERT_EMV")
    print(f"Physics list: {sim.physics_manager.physics_list_name}")
    sim.volume_manager.add_material_database(persist_data_dir / "GateMaterials.db")
    print(f"Using GateMaterials.db from {persist_data_dir}")
    # Add Geometry to the simulation
    add_geometry_to_gate_sim(sim, geometry_transformation_dataframe, args)

    # Add Source to the simulation
    extra_attributes = None
    if args.mode == "srm-sim":
        add_volume_source(
            sim,
            energy_keV=140.0,
            name="VolumeSource",
            args=args,
            fov_shape=args.fov_shape,
            fov_size_mm=args.fov_size_mm,
        )
    elif args.mode == "phantom":
        # per-thread photons/s; configure_chunked_run_timing checks EventID ranges with it
        args.source_activity_bq = add_phantom_source(sim, args)
        extra_attributes = activate_phantom_scatter_attributes(sim, args)
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
        energy_resolution=args.energy_resolution,
        energy_resolution_reference_kev=args.energy_resolution_reference_kev,
        layout=args.actor_layout,
        extra_attributes=extra_attributes,
    )
    add_stats_actor(sim, output_dir, output_stem)
    write_run_manifest(output_dir, output_stem, args, unique_seed)
    sim.run()


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a GATE simulation with Brain SPECT geometry."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store simulation outputs.",
    )
    parser.add_argument(
        "-j",
        "--job-array-id",
        type=str,
        default=None,
        help="SLURM job array ID used for naming output files.",
    )
    parser.add_argument(
        "-k",
        "--job-array-task-id",
        type=str,
        default=None,
        help="SLURM job array task ID used for naming output files.",
    )
    parser.add_argument(
        "-d",
        "--chunk-duration-s",
        type=float,
        default=1.0,
        help="Duration of each chunk in seconds.",
    )
    parser.add_argument(
        "-c",
        "--num-chunks",
        type=int,
        default=10,
        help="Number of chunks to simulate.",
    )
    parser.add_argument(
        "-t",
        "-n",
        "--num-threads",
        type=int,
        default=None,
        help="Number of threads to use. Defaults to SLURM_CPUS_PER_TASK when running on SLURM.",
    )
    parser.add_argument(
        "-s",
        "--source-activity-bq",
        type=float,
        default=1e6,
        help="Activity of the source in Becquerels.",
    )
    parser.add_argument(
        "--execution-environment",
        type=str,
        default="auto",
        choices=["auto", "ospool", "slurm", "local"],
        help="Select the runtime environment so cluster defaults resolve correctly.",
    )
    parser.add_argument(
        "--eventid-warn-threshold",
        type=int,
        default=1.5e9,
        help="Threshold for expected events per chunk to warn about EventID overflow.",
    )
    parser.add_argument(
        "--eventid-hard-limit",
        type=int,
        default=2_147_483_647,
        help="Maximum expected events per timing chunk for the signed 32-bit EventID range.",
    )
    parser.add_argument(
        "--mapping-mode",
        type=str,
        choices=["sequential", "reverse", "random"],
        default="sequential",
        help="Mapping mode for crystal IDs to collimator configurations.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="srm-sim",
        choices=["srm-sim", "geometry-only", "phantom"],
        help="srm-sim: uniform FOV source (system matrix); phantom: a phantom from "
        "--phantom-spec with its activity; geometry-only: geometry export.",
    )
    parser.add_argument(
        "--phantom-spec",
        type=str,
        default=None,
        help="Phantom spec .json (phantom_models.Spec; a phantom bundle's spec.json) for --mode phantom.",
    )
    parser.add_argument(
        "--phantom-activity-bq",
        type=float,
        default=None,
        help="Phantom activity at time 0 (decays/s) in --phantom-activity-region.",
    )
    parser.add_argument(
        "--phantom-activity-region",
        type=str,
        choices=["all", "brain"],
        default="all",
        help="Where --phantom-activity-bq is: the whole phantom, or an image phantom's brain "
        "(the rest of the map scaled with it).",
    )
    parser.add_argument(
        "--radionuclide",
        type=str,
        choices=["Tc99m", "none"],
        default="Tc99m",
        help="Phantom source: the radionuclide's gamma lines (ICRP-107) and half-life, or "
        "'none' for mono 140 keV gammas without decay.",
    )
    parser.add_argument(
        "--no-decay",
        action="store_true",
        help="Keep the phantom activity constant (no physical decay).",
    )
    parser.add_argument(
        "--image-geometry",
        type=str,
        choices=["mesh", "voxel"],
        default="mesh",
        help="Image phantoms: the model's published nested surface meshes (e.g. Auer's "
        "mesh50_XCAT head; they clear the hardware; default) or one ImageVolume box (overlaps "
        "the hardware, so --hardware-world auto moves the hardware to a parallel world, ~3x slower).",
    )
    parser.add_argument(
        "--hardware-world",
        type=str,
        choices=["auto", "mass", "parallel"],
        default="auto",
        help="Where the scanner hardware goes in phantom mode: 'auto' puts it in a "
        "layered-mass parallel world for image phantoms (their box overlaps it), in the "
        "mass world for analytic phantoms.",
    )
    parser.add_argument(
        "--time-start-s",
        type=float,
        default=0.0,
        help="Start of this job's time window (s after the acquisition start); "
        "chunks follow from there. Time-sliced jobs use it so decay is exact.",
    )
    parser.add_argument(
        "--no-shielding",
        action="store_false",
        dest="with_shielding",
        default=True,
        help="Disable shielding in the simulation (shielding is enabled by default).",
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
        help="FOV size in mm. For box, this is the side length; for sphere, it is the diameter.",
    )
    parser.add_argument(
        "--energy-resolution",
        type=float,
        default=0.10,
        help="Gaussian energy blurring FWHM as a fraction of the reference energy. "
        "Use 0 for no blurring.",
    )
    parser.add_argument(
        "--energy-resolution-reference-kev",
        type=float,
        default=140.0,
        help="Reference energy in keV at which --energy-resolution is specified.",
    )

    parser.add_argument(
        "--shield-model",
        type=str,
        choices=["stl", "pieces", "csg"],
        default="stl",
        help="Shield geometry: the STL, STL pieces (--shield-pieces-dir) or the "
        "analytic CSG tiles (fastest).",
    )
    parser.add_argument(
        "--check-overlaps",
        action="store_true",
        help="Run Geant4's volume overlap check (slow; for validation runs).",
    )
    parser.add_argument(
        "--shield-pieces-dir",
        type=str,
        default=None,
        help="Directory of shield pieces from dev/python/split_brain_shield_stl.py "
        "(manifest.json + STLs) to use instead of the single shield STL.",
    )
    parser.add_argument(
        "--actor-layout",
        type=str,
        choices=["merged", "per-head"],
        default="merged",
        help="Singles digitizer: one chain for all heads writing a 'Singles' tree "
        "(default, much faster) or one chain per head writing 'Pixel_<N>_Singles'.",
    )
    parser.add_argument(
        "--physics-list",
        type=str,
        default="QGSP_BERT_EMV",
        help="Geant4 physics list. QGSP_BERT_EMV (opengate's default, used by the "
        "210 mm campaigns) has no Rayleigh scattering, Klein-Nishina Compton without "
        "binding and no fluorescence; G4EmStandardPhysics_option4 has all three "
        "(~1.4x slower; used for the 288 mm campaign).",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    (
        args.job_array_id,
        args.job_array_task_id,
        args.num_threads,
        args.execution_environment,
    ) = resolve_simulation_runtime_context(
        args.job_array_id,
        args.job_array_task_id,
        args.num_threads,
        args.execution_environment,
    )
    if args.mode == "geometry-only":
        run_simulation_with_geometry_only(args)
    elif args.mode in ("srm-sim", "phantom"):
        if args.mode == "phantom" and not (args.phantom_spec and args.phantom_activity_bq):
            raise ValueError("--mode phantom needs --phantom-spec and --phantom-activity-bq")
        run_simulation(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
