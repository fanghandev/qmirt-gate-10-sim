#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import opengate as gate
from opengate.actors.filters import GateFilterBuilder
from opengate.utility import g4_units
from qmirt_utility.utils import find_project_root, generate_tree, print_list_aligned


def run_simulation(
    gate_materials_db_path: Path, optical_properties_file_path: Path | str
):

    # 1. Initialize the Simulation
    sim = gate.Simulation(progress_bar=True, number_of_threads=16)
    sim.output_dir = Path(__file__).parent / "output"
    sim.g4_verbose = True
    sim.visu = False
    # sim.random_seed = 123456

    # 2. Enable Optical Physics and Load Material Properties
    # This line explicitly enables optical physics processes in GATE 10
    sim.volume_manager.add_material_database(str(gate_materials_db_path))
    sim.physics_manager.optical_properties_file = str(optical_properties_file_path)
    sim.physics_manager.surface_properties_file = "SurfaceProperties.xml"

    add_sim_geometry(sim)

    sim.volume_manager.get_volume("world").material = "Air"
    sim.volume_manager.get_volume("crystal").material = "LYSO"
    sim.volume_manager.get_volume("crystal_barrier").material = "LYSO-Barrier"
    sim.volume_manager.get_volume("optical_epoxy").material = "Epoxy"
    sim.volume_manager.get_volume("sipm").material = "Silicon"

    add_sim_actors(sim)
    add_sim_source_and_phyiscs(sim)

    opt_surf_crystal_world = sim.physics_manager.add_optical_surface(
        volume_from="crystal", volume_to="world", g4_surface_name="rough_teflon_wrapped"
    )
    opt_surf_world_crystal = sim.physics_manager.add_optical_surface(
        volume_from="world", volume_to="crystal", g4_surface_name="rough_teflon_wrapped"
    )
    opt_surf_crystal_epoxy = sim.physics_manager.add_optical_surface(
        volume_from="crystal", volume_to="optical_epoxy", g4_surface_name="smooth"
    )
    opt_surf_epoxy_crystal = sim.physics_manager.add_optical_surface(
        volume_from="optical_epoxy", volume_to="crystal", g4_surface_name="smooth"
    )
    opt_surf_epoxy_world = sim.physics_manager.add_optical_surface(
        volume_from="optical_epoxy",
        volume_to="world",
        g4_surface_name="rough_teflon_wrapped",
    )
    opt_surf_world_epoxy = sim.physics_manager.add_optical_surface(
        volume_from="world",
        volume_to="optical_epoxy",
        g4_surface_name="rough_teflon_wrapped",
    )
    opt_surf_epoxy_sipm = sim.physics_manager.add_optical_surface(
        volume_from="optical_epoxy", volume_to="sipm", g4_surface_name="perfect_apd"
    )
    # opt_surf_sipm_epoxy = sim.physics_manager.add_optical_surface(
    #     volume_from="sipm", volume_to="optical_epoxy", g4_surface_name="smooth"
    # )
    opt_surf_lyso_barrier = sim.physics_manager.add_optical_surface(
        volume_from="crystal",
        volume_to="crystal_barrier",
        g4_surface_name="laser_physical_boundary",
    )
    opt_surf_barrier_lyso = sim.physics_manager.add_optical_surface(
        volume_from="crystal_barrier",
        volume_to="crystal",
        g4_surface_name="laser_physical_boundary",
    )

    print(sim.physics_manager.dump_optical_surfaces())

    sim.run()


def add_sim_geometry(
    sim: gate.Simulation,
):
    # Define units
    cm = g4_units.cm
    mm = g4_units.mm
    um = g4_units.um

    # You MUST have an external 'Materials.xml' file in your directory
    # This file stores properties like scintillation yield, absorption length, and refractive index

    ring_half_z_thickness = 1.0 * mm
    barrier_half_z_thickness = 20 * um
    lyso_radial_thickness = 16 * mm
    ring_inner_radius = 160 * mm
    epoxy_layer_thickness = 0.1 * mm
    sipm_layer_thickness = 1.0 * mm

    n_stacks = 10

    ring_outer_radius = ring_inner_radius + lyso_radial_thickness

    crystal = sim.volume_manager.add_volume("Tubs", "crystal")
    crystal.rmin = ring_inner_radius * mm
    crystal.rmax = ring_outer_radius * mm
    crystal.dz = ring_half_z_thickness - barrier_half_z_thickness * mm
    crystal.sphi = 0 * g4_units.deg
    crystal.dphi = 360 * g4_units.deg

    array_translations = gate.geometry.utility.get_grid_repetition(
        size=[1, 1, int(n_stacks)], spacing=[0, 0, ring_half_z_thickness * 2 * mm]
    )
    crystal.translation = array_translations

    crystal_barrier = sim.volume_manager.add_volume("Tubs", "crystal_barrier")
    crystal_barrier.rmin = ring_inner_radius * mm
    crystal_barrier.rmax = ring_outer_radius * mm
    crystal_barrier.dz = barrier_half_z_thickness * mm
    crystal_barrier.sphi = 0 * g4_units.deg
    crystal_barrier.dphi = 360 * g4_units.deg
    barrier_array_translations = gate.geometry.utility.get_grid_repetition(
        size=[1, 1, int(n_stacks) - 1],
        spacing=[0, 0, ring_half_z_thickness * 2 * mm],
        start=[0, 0, array_translations[0][2] + ring_half_z_thickness * mm],
    )
    # print("Barrier array translations:\n")
    # for idx, translation in enumerate(barrier_array_translations):
    #     print(f"Barrier {idx}: {translation}")
    #     print(f"crystal.     : {crystal.translation[idx]}")
    crystal_barrier.translation = barrier_array_translations
    crystal_barrier.mother = sim.volume_manager.get_volume("world")

    # Add optical epoxy to handle the light output coupling between the crystal and the SiPM
    optical_epoxy = sim.volume_manager.add_volume("Tubs", "optical_epoxy")
    optical_epoxy.rmin = ring_outer_radius * mm
    optical_epoxy.rmax = (ring_outer_radius + epoxy_layer_thickness * 2.0) * mm
    optical_epoxy.dz = ring_half_z_thickness * mm
    optical_epoxy.sphi = 0 * g4_units.deg
    optical_epoxy.dphi = 360 * g4_units.deg
    optical_epoxy.translation = array_translations
    optical_epoxy.mother = sim.volume_manager.get_volume("world")

    # Add a box SiPM volume
    sipm = sim.volume_manager.add_volume("Tubs", "sipm")
    sipm.rmin = (ring_outer_radius + epoxy_layer_thickness * 2.0) * mm
    sipm.rmax = (
        ring_outer_radius + epoxy_layer_thickness * 2.0 + sipm_layer_thickness * 2.0
    ) * mm
    sipm.dz = ring_half_z_thickness * mm
    sipm.sphi = 0 * g4_units.deg
    sipm.dphi = 360 * g4_units.deg
    sipm.translation = array_translations
    sipm.mother = sim.volume_manager.get_volume("world")


def add_sim_actors(sim: gate.Simulation):

    # add Hits Collection Actor
    crystal_hits_actor = sim.add_actor("DigitizerHitsCollectionActor", "crystal_hits")
    crystal_hits_actor.attached_to = sim.volume_manager.get_volume("crystal").name
    crystal_hits_actor.attributes = [
        "EventPosition",
        "TrackID",
        "ParentID",
        "EventID",
        "TotalEnergyDeposit",
        "KineticEnergy",
        "PostPosition",
        "PrePosition",
        "PreStepUniqueVolumeID",
        "ParticleName",
        "GlobalTime",
        # "VolumeName",
    ]
    F = GateFilterBuilder()
    optical_photon_filter = F.ParticleName == "opticalphoton"
    crystal_hits_actor.filter = ~optical_photon_filter
    crystal_hits_actor.output_filename = None

    # # add adder actor
    crystal_adder_actor = sim.add_actor("DigitizerAdderActor", "crystal_singles")
    crystal_adder_actor.input_digi_collection = "crystal_hits"
    crystal_adder_actor.policy = "EnergyWeightedCentroidPosition"
    crystal_adder_actor.group_volume = sim.volume_manager.get_volume("crystal").name
    crystal_adder_actor.attached_to = sim.volume_manager.get_volume("crystal").name
    crystal_adder_actor.output_filename = "ring_array_lyso_sipm_output.root"

    # crystal_adder_actor.attributes = [
    #     "EventID",
    #     "ParentID",
    #     "TrackID",
    #     "Position",
    #     "EnergyDeposit",
    #     "KineticEnergy",
    # ]

    opt_epoxy_phsa = sim.add_actor("PhaseSpaceActor", "opt_epoxy_phsa")
    opt_epoxy_phsa.attached_to = sim.volume_manager.get_volume("optical_epoxy").name
    opt_epoxy_phsa.attributes = [
        "EventID",
        "ParentID",
        "TrackID",
        # "CurrentStepNumber",
        # "ProcessDefinedStep",
        "Position",
        # "ParticleName",
        # "TrackCreatorProcess",
        # "EventKineticEnergy",
        "KineticEnergy",
        # "PDGCode",
    ]
    opt_epoxy_phsa.filter = optical_photon_filter
    opt_epoxy_phsa.steps_to_store = "entering"
    opt_epoxy_phsa.output_filename = "ring_array_lyso_sipm_output.root"


def add_sim_source_and_phyiscs(sim: gate.Simulation):

    # Define units
    cm = g4_units.cm
    mm = g4_units.mm
    eV = gate.g4_units.eV
    keV = gate.g4_units.keV
    MeV = gate.g4_units.MeV
    # Change source
    source = sim.add_source("GenericSource", "gamma_source")
    source.particle = "gamma"
    source.energy.mono = 511.0 * keV
    # # source.activity = 10 * Bq
    # source.direction.type = "momentum"
    # source.direction.momentum = [1, 0, 0]
    # source.n = 1
    # source.position.translation = [-5 * cm, 0 * cm, 0 * cm]

    source.position.type = "cylinder"
    source.position.radius = 0.05 * mm
    source.position.dz = 100 * mm
    source.direction.type = "iso"
    source.n = 1000

    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    # sim.physics_manager.set_production_cut("crystal", "electron", 0.1 * mm)
    # sim.physics_manager.set_production_cut("optical_epoxy", "electron", 0.1 * mm)
    sim.physics_manager.special_physics_constructors.G4OpticalPhysics = True
    sim.physics_manager.energy_range_min = 2 * eV
    sim.physics_manager.energy_range_max = 1 * MeV


def save_simulation_geometry_to_wrl(ouput_path: Path = Path("sim_geometry.wrl")):
    sim = gate.Simulation()
    add_sim_geometry(sim)
    sim.user_info.visu = True
    sim.user_info.visu_type = "vrml_file_only"
    print("Storing geometry into wrl file only without running the simulation...")
    sim.user_info.visu_filename = str(ouput_path)
    print(f"Geometry stored in {sim.user_info.visu_filename}")
    sim.run()


def main(args):

    import argparse

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

    parser = argparse.ArgumentParser(
        description="Run a simple GATE simulation of a LYSO crystal coupled to optical epoxy."
    )
    parser.add_argument(
        "--save-wrl-to",
        dest="wrl_output_path",
        help="Save the simulation geometry to a WRL file and exit.",
    )
    args = parser.parse_args()

    if args.wrl_output_path:
        assert args.wrl_output_path.endswith(".wrl"), (
            "Output path must end with .wrl extension!"
        )
        print(f"Saving simulation geometry to WRL file: {args.wrl_output_path}")
        save_simulation_geometry_to_wrl(ouput_path=Path(args.wrl_output_path))
    else:
        # save_simulation_geometry_to_wrl()
        run_simulation(
            gate_materials_db_path, optical_properties_file_path="Materials.xml"
        )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
