import argparse
from pathlib import Path

import numpy as np
import opengate as gate
from scipy.spatial.transform import Rotation

import qmirt

# Helper function:


def construct_collimator_no_boolean_gate_geometry(
    sim: gate.Simulation, config: dict, id: int
):
    body_translation_mm = config["collimator_body_translation_mm"][id].tolist()
    guide_translation_mm = [
        body_translation_mm[0],
        body_translation_mm[1],
        body_translation_mm[2]
        + config["collimator_body_length_mm_np"][id] * 0.5
        + config["collimator_guide_length_mm_np"][id] * 0.5,
    ]

    body_top_mm = config["collimator_body_outer_top_mm_np"][id]
    body_bottom_mm = config["collimator_body_outer_bottom_mm_np"][id]
    body_length_mm = config["collimator_body_length_mm_np"][id]

    collimator_body_shell = qmirt.utils.simulation.make_gate_shell_trd(
        sim,
        name=f"CollimatorBody_{id + 1}",
        mother="world",
        top_mm=body_top_mm,
        bottom_mm=body_bottom_mm,
        length_mm=body_length_mm,
        translation_mm=body_translation_mm,
        material="Tungsten",
    )

    box_outer_size_mm = np.full(
        (3,),
        config["collimator_body_outer_top_mm_np"][id]
        + 2.0
        + config["collimator_wall_thickness_mm"] * 2.0,
    )
    box_outer_size_mm[2] = config["detector_crystal_size_mm"][2] + 4.0
    box_inner_size_mm = np.full(
        (3,), config["collimator_body_outer_top_mm_np"][id] + 2.0
    )
    box_inner_size_mm[2] = config["detector_crystal_size_mm"][2] + 4.0
    box_translation_mm = [
        body_translation_mm[0],
        body_translation_mm[1],
        body_translation_mm[2] - body_length_mm * 0.5 - box_outer_size_mm[2] * 0.5,
    ]

    collimator_box_shell = qmirt.utils.simulation.make_gate_shell_box(
        sim,
        name=f"CollimatorBox_{id + 1}",
        mother="world",
        outer_size_mm=box_outer_size_mm,
        inner_size_mm=box_inner_size_mm,
        translation_mm=box_translation_mm,
        material="Tungsten",
    )

    guide_top_mm = config["collimator_guide_outer_top_mm_np"][id]
    guide_bottom_mm = config["collimator_guide_outer_bottom_mm_np"][id]
    guide_length_mm = config["collimator_guide_length_mm_np"][id]

    collimator_guide_shell = qmirt.utils.simulation.make_gate_shell_trd(
        sim,
        name=f"CollimatorGuide_{id + 1}",
        mother="world",
        top_mm=guide_top_mm,
        bottom_mm=guide_bottom_mm,
        length_mm=guide_length_mm,
        translation_mm=guide_translation_mm,
        material="Tungsten",
    )

    return collimator_body_shell


def add_collimator_to_gate_sim(sim: gate.Simulation, config: dict, id: int):
    collimator = construct_collimator_no_boolean_gate_geometry(sim, config, id)
    r = get_head_rotation_matrix(config, id)
    collimator.rotation = r


def add_shielding_to_gate_sim(sim: gate.Simulation, config: dict):

    shielding = gate.geometry.volumes.TesselatedVolume(name="Shielding")
    # Make sure the shielding file path is valid before proceeding
    shielding_file_path = Path(config["shielding_file_path"])
    if not shielding_file_path.exists():
        raise FileNotFoundError(
            f"Shielding STL file not found at: {shielding_file_path}"
        )

    shielding.mother = "world"

    shielding.file_name = Path(config["shielding_file_path"]).as_posix()
    shielding.origin_at_cog = False
    sim.add_volume(shielding)
    rz = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    shielding.rotation = rz
    shielding.material = "Lead"


def _configure_wrl_export(
    sim: gate.Simulation, *, force_phantom_wireframe: bool = False
):
    sim.user_info.visu = True
    sim.user_info.visu_type = "vrml_file_only"

    sim.visu_commands_vrml = ["/vis/open VRML2FILE", "/vis/drawVolume"]
    sim.visu_commands_vrml.append("/vis/geometry/set/visibility world 0 false")

    if force_phantom_wireframe:
        sim.visu_commands_vrml.append(
            "/vis/geometry/set/forceWireframe Jaszczak_Phantom 0 true"
        )


def _apply_debug_geometry_settings(sim: gate.Simulation, args):
    if not args.debug_geometry:
        return

    print(
        "Geometry debug mode enabled: dumping volume tree and enabling verbose G4 output."
    )
    print(f"check_volumes_overlap: {sim.check_volumes_overlap}")
    print(sim.volume_manager.dump_volume_tree())
    sim.g4_verbose = True
    sim.g4_verbose_level = 2


def _finalize_wrl_export(sim: gate.Simulation, visu_filename: Path | str):
    sim.visu_commands_vrml.append("/vis/viewer/flush")
    sim.user_info.visu_filename = str(Path(visu_filename).resolve())
    print(f"Geometry stored in {sim.user_info.visu_filename}")
    sim.run()


def _add_scanner_geometry(
    sim: gate.Simulation, config: dict, *, include_shielding: bool
):
    for i in range(80):
        print(f"Adding geometry for head {i + 1}...")
        add_collimator_to_gate_sim(sim, config, id=i)
        add_pixelated_detector_to_gate_sim(sim, config, id=i)

    if include_shielding:
        add_shielding_to_gate_sim(sim, config)


def _add_phantom_geometry(sim: gate.Simulation, config: dict):
    print("Adding Jaszczak phantom geometry...")
    add_Jaszczak_phantom(sim)


def save_geometry_to_wrl(
    config: dict,
    persist_data_dir: Path,
    args,
    export_target: str = "scanner",
):
    """Export scanner, phantom, or combined geometry to a VRML file."""
    sim = gate.Simulation()
    sim.volume_manager.add_material_database(persist_data_dir / "GateMaterials.db")
    wrl_output_dir = Path(args.output_dir).resolve()
    wrl_output_dir.mkdir(parents=True, exist_ok=True)

    export_target = export_target.lower()
    if export_target not in {"scanner", "phantom", "both"}:
        raise ValueError("export_target must be one of: scanner, phantom, both")

    include_shielding = bool(getattr(args, "with_shielding", False))

    if export_target == "scanner":
        _add_scanner_geometry(sim, config, include_shielding=include_shielding)
        _configure_wrl_export(sim)
        _apply_debug_geometry_settings(sim, args)
        print("Storing scanner geometry to WRL without running the simulation...")
        _finalize_wrl_export(
            sim,
            wrl_output_dir / "dc_spect_no_boolean_geometry.wrl",
        )
        return

    if export_target == "phantom":
        _add_phantom_geometry(sim, config)
    else:
        _add_scanner_geometry(sim, config=config, include_shielding=include_shielding)
        _add_phantom_geometry(sim, config)
    _configure_wrl_export(sim, force_phantom_wireframe=True)

    _apply_debug_geometry_settings(sim, args)

    if export_target == "phantom":
        filename = "jaszczak_phantom_only.wrl"
    else:
        filename = "scanner_and_phantom_geometry.wrl"

    print("Storing geometry into wrl file only without running the simulation...")
    _finalize_wrl_export(sim, wrl_output_dir / filename)


def save_simulation_geometry_to_wrl(config: dict, persist_data_dir: Path, args):
    save_geometry_to_wrl(
        config,
        persist_data_dir,
        args,
        export_target=(
            args.geometry_only if isinstance(args.geometry_only, str) else "scanner"
        ),
    )


def render_wrl_to_html(wrl_path: Path, html_output_path: Path):
    import pyvista as pv

    print(f"Rendering WRL geometry from {wrl_path} to HTML for visualization...")
    wrl_path = Path(wrl_path)
    if not wrl_path.exists():
        raise FileNotFoundError(f"WRL file not found at: {wrl_path}")

    original_solid_names = extract_solid_names_from_wrl(wrl_path)

    detector_mesh = load_wrl_as_mesh(wrl_path)

    print(f"\nTotal SOLID objects in WRL: {len(original_solid_names)}")

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(detector_mesh, color="lightblue", show_edges=True, opacity=0.7)

    plotter.export_html(html_output_path)


def extract_solid_names_from_wrl(wrl_path):
    """Extract SOLID names from WRL file comments"""
    solid_names = []
    try:
        with open(wrl_path, "r") as f:
            for line in f:
                if "#---------- SOLID:" in line:
                    # Extract the name after "SOLID: "
                    name = line.split("#---------- SOLID:")[1].strip()
                    solid_names.append(name)
    except Exception as e:
        print(f"Error reading WRL file: {e}")
    return solid_names


def load_wrl_as_mesh(wrl_path):
    import logging

    import pyvista as pv
    import vtk

    vtk.vtkObject.GlobalWarningDisplayOff()
    logging.disable(logging.CRITICAL)

    importer = vtk.vtkVRMLImporter()
    importer.SetFileName(wrl_path)
    importer.Update()

    append_filter = vtk.vtkAppendPolyData()

    renderer = importer.GetRenderer()
    actors = renderer.GetActors()
    actors.InitTraversal()

    for i in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        if actor and actor.GetMapper():
            poly_data = actor.GetMapper().GetInput()
            if poly_data:
                append_filter.AddInputData(poly_data)

    append_filter.Update()

    mesh = pv.wrap(append_filter.GetOutput())

    return mesh


def add_dc_spect_geometry(sim: gate.Simulation, config: dict):

    # Add a box at the center, and attach the source to it
    # The box is used to define the source position,
    # but is not part of the simulation geometry
    # The size of the box is 220 mm x 220 mm x 220 mm
    source_box = sim.add_volume("Box", name="SourceBox")
    source_box.size = [220, 220, 220]  # unit is mm
    source_box.material = "Air"

    for i in range(80):
        # Add the i_th collimator
        add_collimator_to_gate_sim(sim, config, id=i)
        # Add the i_th pixelated detector crystal
        add_pixelated_detector_to_gate_sim(sim, config, id=i)
    # add_shielding_to_gate_sim(sim, config)  # Add the shielding as an example


def run_simulation(config: dict, persist_data_dir: Path, args):
    output_dir = Path(args.output_dir).resolve()
    print("Resolved output directory: ", output_dir)
    print(f"Execution environment: {args.execution_environment}")
    print(
        "Runtime context: "
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

    simulation_mode = getattr(args, "mode", "box")
    print(f"Simulation mode: {simulation_mode}")

    # Add Geometry to the simulation.
    # The phantom mode keeps the center source box out of the world volume to avoid overlaps.
    if simulation_mode == "box":
        _add_scanner_geometry(
            sim, config, include_shielding=bool(getattr(args, "with_shielding", False))
        )
    elif simulation_mode == "jaszczak":
        _add_scanner_geometry(
            sim, config, include_shielding=bool(getattr(args, "with_shielding", False))
        )
        _add_phantom_geometry(sim, config)
    else:
        raise ValueError("simulation_mode must be one of: 'box', 'jaszczak'")

    if args.with_shielding:
        print("Simulate with lead shielding = True")
    else:
        print("Simulate with lead shielding = False")

    # Add Source to the simulation
    if simulation_mode == "box":
        add_box_source(sim, energy_keV=140.0, args=args)
    else:
        add_background_source(sim, args, phantom_name="Jaszczak_Phantom")

    sim.number_of_threads = int(args.num_threads)
    configure_chunked_run_timing(sim, args)
    # In activity mode, expected event count is stochastic and controlled by
    # activity * run_timing_intervals.
    print(f"Number of threads: {sim.number_of_threads}")

    output_stem = f"a_{job_array_id}_j_{job_array_task_id}"
    add_actors(sim, output_dir, output_stem)
    add_stats_actor(sim, output_dir, output_stem)
    sim.run()


def add_stats_actor(sim: gate.Simulation, output_dir: Path, output_stem: str):
    stats_actor = sim.add_actor("SimulationStatisticsActor", "Stats")  # type: ignore
    stats_path = output_dir / f"{output_stem}_sim_stats.txt"
    # GATE will automatically write to this file after sim.run() finishes
    stats_actor.output_filename = str(stats_path)


def add_actors(sim: gate.Simulation, output_dir: Path, output_stem: str):
    pixel_array_name = [f"pixel_{i + 1}" for i in range(80)]

    # Keep hits in-memory only as input to the singles chain.
    for i in range(80):
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


def _resolve_xlsx_path(persistent_data_dir: Path, xlsx_path: str | None) -> Path:
    if xlsx_path is not None:
        candidate = Path(xlsx_path)
    else:
        candidate = (
            persistent_data_dir
            / "cardiac_spect"
            / "spreadsheet"
            / "MDSL.excel80M10RFR.cut-plate.010.150roi.2.30pin.105ellipse.xlsx"
        )
    if not candidate.exists():
        raise FileNotFoundError(
            f"Geometry configuration xlsx file not found at {candidate}"
        )
    return candidate


def parse_args(args=None):
    """
    Parses command line arguments.
    Accepts an optional list of arguments for easy unit testing.
    """

    parser = argparse.ArgumentParser(description="Simulation Runner")

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
        "-d",
        "--chunk-duration-s",
        type=float,
        default=1.0,
        help="Duration of each run chunk in seconds.",
    )
    parser.add_argument(
        "-c",
        "--num-chunks",
        type=int,
        default=1,
        help="Number of run timing intervals (chunks).",
    )
    parser.add_argument(
        "--eventid-warn-threshold",
        type=float,
        default=1.5e9,
        help="Warn if expected events per chunk exceed this threshold.",
    )
    parser.add_argument(
        "--debug-geometry",
        action="store_true",
        help="Dump the geometry tree and enable verbose Geant4 output before running.",
    )
    parser.add_argument(
        "-n",
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
        "--xlsx-path",
        type=str,
        default=None,
        help="Path to the geometry configuration xlsx file",
    )
    parser.add_argument(
        "-g",
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
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store the simulation output files",
    )
    parser.add_argument(
        "-j",
        "--job-array-id",
        type=str,
        default=None,
        help="SLURM_ARRAY_JOB_ID used for naming output files.",
    )
    parser.add_argument(
        "-k",
        "--job-array-task-id",
        type=str,
        default=None,
        help="SLURM_ARRAY_TASK_ID used for naming output files.",
    )
    parser.add_argument(
        "--no-shielding",
        action="store_false",
        dest="with_shielding",
        default=True,
        help="Disable shielding in the simulation (shielding is enabled by default).",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="box",
        choices=["box", "jaszczak"],
        help=(
            "Select the simulation setup: box for SRM generation or "
            "jaszczak for phantom acquisition."
        ),
    )
    parser.add_argument(
        "-t",
        "--source-type",
        type=str,
        default="Tc-99m",
        choices=["Gamma-140", "Tc-99m", "Co-57"],
        help="Radioisotope used for the Jaszczak phantom background source.",
    )
    parsed_args = parser.parse_args(args)
    return parsed_args


def main():
    args = parse_args()

    persistent_data_dir = qmirt.utils.filesystem.search_dir_up(
        "persistent_data", __file__
    )
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

    # Update args directly to pass downstream
    args.job_array_id = resolved_job_array_id
    args.job_array_task_id = resolved_job_array_task_id
    args.num_threads = resolved_num_threads
    args.execution_environment = resolved_execution_environment

    if args.geometry_only:
        config = get_dc_spect_geometry_config(
            _resolve_xlsx_path(persistent_data_dir, args.xlsx_path),
            stl_dir=persistent_data_dir / "cardiac_spect" / "stl",
            n_pixels=(1, 1, 1),
        )
        save_geometry_to_wrl(
            config,
            persistent_data_dir,
            args,
            export_target=args.geometry_only,
        )
    else:
        config = get_dc_spect_geometry_config(
            _resolve_xlsx_path(persistent_data_dir, args.xlsx_path),
            stl_dir=persistent_data_dir / "cardiac_spect" / "stl",
        )
        run_simulation(config, persistent_data_dir, args)


if __name__ == "__main__":
    main()
