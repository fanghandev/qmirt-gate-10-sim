#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import opengate as gate
from opengate.utility import g4_units


def generate_tree(dir_path: Path, prefix: str = ""):
    """
    A recursive generator that yields a visual tree structure of a directory.
    """
    # Visual characters for the tree structure
    space = "    "
    branch = "│   "
    tee = "├── "
    last = "└── "

    # Get directory contents, ignoring hidden files (optional, but standard)
    contents = [path for path in dir_path.iterdir() if not path.name.startswith(".")]

    # Optional: Sort so directories appear first, then files alphabetically
    contents.sort(key=lambda x: (x.is_file(), x.name))

    # Determine the prefix for each item
    pointers = [tee] * (len(contents) - 1) + [last] if contents else []

    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name

        if path.is_dir():
            # If it's a directory, extend the prefix and recurse
            extension = branch if pointer == tee else space
            yield from generate_tree(path, prefix=prefix + extension)


def find_project_root(current_path: Path, marker: str = ".git") -> Path:
    """
    Traverse upward through the directory tree until a directory
    containing the specified marker file/folder is found.
    """
    # .parents generates an iterator over all parent directories
    for parent_dir in [current_path] + list(current_path.parents):
        if (parent_dir / marker).exists():
            return parent_dir

    # If the system root is reached without finding it,
    # you can raise an exception or return a default value
    raise FileNotFoundError(f"Could not find the project root containing {marker}!")


def print_list_aligned(
    items: list,
    *,
    vline=True,
    hline=True,
    box=True,
    fixed_width=None,
    min_width=None,
):
    """
    Print a list of items in a vertically aligned format.
    Each item is printed on a new line, and the width of the printed items
    can be adjusted to be fixed or based on the longest item.

    vline: if True, draw vertical bars at left/right of each row.
    hline: if True, draw a horizontal line (top and bottom) around the block.
    """
    # Expected a list of lists, where each inner list represents a row of items to be printed
    assert all(isinstance(item, list) for item in items), "Expected a list of lists!"
    # Determine the maximum width of items in each column
    num_columns = max(len(row) for row in items)
    column_widths = [0] * num_columns
    for row in items:
        for i, item in enumerate(row):
            column_widths[i] = max(column_widths[i], len(str(item)))
    # If fixed_width is provided, override the calculated column widths
    if fixed_width is not None:
        column_widths = [fixed_width] * num_columns
    # If min_width is provided, ensure each column width is at least min_width
    if min_width is not None:
        column_widths = [max(width, min_width) for width in column_widths]

    # Precompute lengths used for horizontal lines
    # Each column prints as: <content ljust(width)> + "  "  (two trailing spaces)
    len_line = sum((w + 2) for w in column_widths)
    # When vline is True, printed row is "│ {line}│" so there's one extra leading space inside the bars
    inner_width = len_line + (1 if vline else 0)

    def _print_top():
        if vline:
            print("┌" + "─" * inner_width + "┐")
        else:
            print("─" * inner_width)

    def _print_bottom():
        if vline:
            print("└" + "─" * inner_width + "┘")
        else:
            print("─" * inner_width)

    # Print top horizontal line if requested
    if box:
        _print_top()

    # Print each row with aligned columns
    for row in items:
        line = ""
        if not vline:
            for i, item in enumerate(row):
                line += (
                    str(item).ljust(column_widths[i]) + "  "
                )  # Add spacing between columns
        else:
            for i, item in enumerate(row[:-1]):
                line += (
                    str(item).ljust(column_widths[i]) + " │ "
                )  # Add vertical bar after each column (except the last one)
            # Add the last column without a trailing vertical bar
            if row:
                line += str(row[-1]).ljust(column_widths[len(row) - 1]) + " "
        if box:
            line = "│ " + line + "│"  # Add vertical bars and trim trailing spaces
        print(line)

    # Print bottom horizontal line if requested
    if hline:
        _print_bottom()


def run_simulation(
    gate_materials_db_path: Path, optical_properties_file_path: Path | str
):

    # 1. Initialize the Simulation
    sim = gate.Simulation(progress_bar=True, number_of_threads=16)
    sim.output_dir = Path(__file__).parent / "output"
    sim.g4_verbose = True
    sim.visu = False
    sim.random_seed = 123456

    # 2. Enable Optical Physics and Load Material Properties
    # This line explicitly enables optical physics processes in GATE 10
    sim.volume_manager.add_material_database(str(gate_materials_db_path))
    sim.physics_manager.optical_properties_file = str(optical_properties_file_path)

    add_sim_geometry(sim)

    sim.volume_manager.get_volume("world").material = "Air"
    sim.volume_manager.get_volume("crystal").material = "BGO"
    sim.volume_manager.get_volume("optical_photon_detector").material = "Air"
    add_sim_source_and_phyiscs(sim)
    opt_surf_world_crystal = sim.physics_manager.add_optical_surface(
        volume_from="world",
        volume_to="crystal",
        g4_surface_name="polished_teflon_wrapped",
    )
    opt_surf_crystal_world = sim.physics_manager.add_optical_surface(
        volume_from="crystal", volume_to="world", g4_surface_name="rough_teflon_wrapped"
    )
    opt_surf_crystal_to_opt_det = sim.physics_manager.add_optical_surface(
        volume_from="crystal",
        volume_to="optical_photon_detector",
        g4_surface_name="smooth",
    )
    opt_surf_opt_det_to_crystal = sim.physics_manager.add_optical_surface(
        volume_from="optical_photon_detector",
        volume_to="crystal",
        g4_surface_name="smooth",
    )

    print(sim.physics_manager.dump_optical_surfaces())
    add_sim_actors(sim)

    sim.run()


def add_sim_geometry(
    sim: gate.Simulation,
):
    # Define units
    cm = g4_units.cm
    mm = g4_units.mm

    # You MUST have an external 'Materials.xml' file in your directory
    # This file stores properties like scintillation yield, absorption length, and refractive index

    sim.world.size = [6 * cm, 6 * cm, 10 * cm]
    # crystal_wrap.material = "Vacuum"
    # add a simple crystal volume
    crystal = sim.add_volume("Box", "crystal")
    crystal.size = [10 * mm, 3 * mm, 3 * mm]
    crystal.translation = [5 * mm, 0 * mm, 0 * mm]

    # Add a optical photon detector (sensitive detector) to the crystal volume
    optical_photon_detector = sim.add_volume("Box", "optical_detector")
    optical_photon_detector.size = [0.1 * mm, 3 * mm, 3 * mm]
    optical_photon_detector.translation = [5.05 * mm, 0 * mm, 0 * mm]


def add_sim_source_and_phyiscs(sim: gate.Simulation):

    # Define units
    cm = g4_units.cm
    mm = g4_units.mm
    eV = gate.g4_units.eV
    keV = gate.g4_units.keV
    MeV = gate.g4_units.MeV
    # Change source
    source = sim.add_source("GenericSource", "gamma1")
    source.particle = "gamma"
    source.energy.mono = 511 * keV
    # source.activity = 10 * Bq
    source.direction.type = "momentum"
    source.direction.momentum = [1, 0, 0]
    source.n = 1
    source.position.translation = [-2 * cm, 0 * cm, 0 * cm]

    sim.physics_manager.special_physics_constructors.G4OpticalPhysics = True
    # sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    sim.physics_manager.set_production_cut("crystal", "electron", 0.1 * mm)
    sim.physics_manager.energy_range_min = 0.1 * eV
    sim.physics_manager.energy_range_max = 1 * MeV
    sim.physics_manager.special_physics_constructors.G4OpticalPhysics = True


def add_sim_actors(sim: gate.Simulation):
    # add phase actor
    crystal_phsa = sim.add_actor("PhaseSpaceActor", "crystal_phsa")
    crystal_phsa.attached_to = sim.volume_manager.get_volume("crystal")
    crystal_phsa.attributes = [
        "EventID",
        "ParentID",
        "TrackID",
        "CurrentStepNumber",
        "ProcessDefinedStep",
        "PostPosition",
        "PrePosition",
        "ParticleName",
        "TrackCreatorProcess",
        "EventKineticEnergy",
        "KineticEnergy",
        "PDGCode",
    ]
    crystal_phsa.steps_to_store = "entering exiting"
    crystal_phsa.output_filename = "phsa_output.root"

    # add phase
    opt_phsa = sim.add_actor("PhaseSpaceActor", "opt_phsa")
    opt_phsa.attached_to = sim.volume_manager.get_volume("optical_photon_detector")
    opt_phsa.attributes = [
        "EventID",
        "ParentID",
        "TrackID",
        "CurrentStepNumber",
        "ProcessDefinedStep",
        "PostPosition",
        "PrePosition",
        "ParticleName",
        "TrackCreatorProcess",
        "EventKineticEnergy",
        "KineticEnergy",
        "PDGCode",
    ]
    opt_phsa.steps_to_store = "entering exiting"
    opt_phsa.output_filename = "phsa_output.root"


def save_simulation_geometry_to_wrl(ouput_path: Path = Path("sim_geometry.wrl")):
    sim = gate.Simulation()
    add_sim_geometry(sim)
    sim.user_info.visu = True
    sim.user_info.visu_type = "vrml_file_only"
    print("Storing geometry into wrl file only without running the simulation...")
    sim.user_info.visu_filename = str(ouput_path)
    print(f"Geometry stored in {sim.user_info.visu_filename}")
    sim.run()


if __name__ == "__main__":
    project_root = find_project_root(Path(__file__))
    persistent_data_dir = project_root / "persistent_data"
    assert persistent_data_dir.is_dir(), (
        f"Persistent data directory not found: {persistent_data_dir}"
    )
    print_list_aligned(
        [
            ["Project Root:", project_root.as_posix()],
            ["Persistent Data Directory:", persistent_data_dir.as_posix()],
        ],
    )
    print(f"📁 {persistent_data_dir.name}/")
    for line in generate_tree(persistent_data_dir):
        print(line)
    gate_materials_db_path = persistent_data_dir / "GateMaterials.db"
    assert gate_materials_db_path.is_file(), (
        f"GateMaterials.db not found in persistent data directory: {gate_materials_db_path}"
    )
    print("\nGateMaterials.db:\n" + str(gate_materials_db_path))

    # save_simulation_geometry_to_wrl()
    run_simulation(gate_materials_db_path, optical_properties_file_path="Materials.xml")
