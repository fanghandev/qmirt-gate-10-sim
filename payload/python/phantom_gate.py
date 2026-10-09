"""Build phantom_models phantoms in a GATE 10 simulation.

add_phantom() creates one Geant4 volume per part, in the same hierarchy as the voxeliser's
paint order: the inserts (rods, spheres, stems, plates) are daughters of the water, the
water a daughter of the PMMA body, and the lid fittings sit in the mother volume next to
the body. The pose (flip, rotation, shift) is applied once, to the volumes placed in the
mother; daughters are placed in their parent's frame. add_activity_source() fills the
water with a uniform source; confining it to the water volume leaves out the cold inserts.

Check against the voxel phantoms with dev/python/check_gate_phantom.py (voxelises the
Geant4 geometry with opengate and compares it voxel by voxel).
"""

from __future__ import annotations

import numpy as np

import phantom_models as pm

COLORS = {"body": [0.85, 0.95, 1.0, 0.25], "water": [0.1, 0.4, 0.85, 0.25], "plate": [0.85, 0.95, 1.0, 0.6],
          "rod": [1.0, 0.6, 0.0, 0.8], "sphere": [0.6, 0.2, 0.7, 0.8], "stem": [0.6, 0.2, 0.7, 0.8],
          "fitting": [0.9, 0.88, 0.8, 1.0], "fitting_pmma": [0.85, 0.95, 1.0, 0.6]}


axis_rotation = pm.axis_rotation


_IMAGE_FILES = {}  # prefix -> {"files": write_image_files(), "volume": image volume or None, "placement": image box}


def write_image_files(spec: pm.Spec, out_dir) -> dict:
    """Material-label and activity images of an image phantom, in its own image frame
    (centred), for GATE: {labels, activity, materials}."""
    from pathlib import Path

    img = pm.image_data(spec)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preset = spec.options.material_preset or pm.IMAGE_MODELS[spec.phantom].default_preset
    offset = -(img["shape"] - 1) / 2 * img["spacing"]
    labels = out_dir / f"{spec.phantom}_{preset}_labels.mhd"
    activity = out_dir / f"{spec.phantom}_activity.mhd"
    pm.write_mhd(labels, img["labels"], img["spacing"], offset, "MET_UCHAR")
    pm.write_mhd(activity, img["activity"], img["spacing"], offset, "MET_FLOAT")
    return {"labels": labels, "activity": activity, "materials": img["materials"]}


def add_phantom(sim, spec: pm.Spec, *, prefix: str = "phantom", mother: str = "world", files_dir=None,
                image_geometry: str = "voxel") -> dict:
    """Add the phantom's volumes (primitive Geant4 solids: G4Tubs, G4Sphere, placed as
    daughters; no tessellation or voxelisation); returns {part name: volume}. Volume names
    are f"{prefix}_{part name}" (the water is f"{prefix}_water"). Placements come from
    phantom_models.placements(), shared with the descriptive and gammart exports.

    Image phantoms: image_geometry "voxel" is one ImageVolume (a box; it overlaps hardware
    close to the head), "mesh" the model's published nested surface meshes (ImageModel.meshes,
    e.g. Auer's mesh50_XCAT: body f"{prefix}_body" with skeleton, brain and air cavity as
    daughters). The activity is a voxel source either way."""
    if pm.is_image(spec.phantom):
        from pathlib import Path

        files = write_image_files(spec, files_dir or Path(sim.output_dir if hasattr(sim, "output_dir") else ".") / "phantom_image")
        x = pm.placements(spec, prefix, mother)[0]
        _IMAGE_FILES[prefix] = {"files": files, "volume": None, "placement": x}
        if image_geometry == "mesh":
            return _add_image_mesh(sim, spec, prefix, mother)
        # voxel phantom: one ImageVolume (material labels), placed with the same pose
        v = sim.add_volume("ImageVolume", x["name"])
        v.image = str(files["labels"])
        v.material = "G4_AIR"
        v.voxel_materials = [[k - 0.5, k + 0.5, name] for k, name in enumerate(files["materials"])]
        v.mother = x["mother"]
        v.translation = list(x["translation"])
        v.rotation = np.asarray(x["rotation"], float)
        _IMAGE_FILES[prefix]["volume"] = v.name  # for add_activity_source
        return {"image": v}
    volumes = {}
    for x in pm.placements(spec, prefix, mother):
        p, prm = x["part"], x["solid"]["params"]
        v = sim.add_volume("SphereVolume" if p.shape == "sphere" else "TubsVolume", x["name"])
        v.material = p.material
        if p.shape == "sphere":
            v.rmin, v.rmax = 0.0, prm["rmax"]
        else:
            v.rmin, v.rmax, v.dz = prm["rmin"], prm["rmax"], prm["dz"]
        v.mother = x["mother"]
        v.translation = list(x["translation"])
        v.rotation = np.asarray(x["rotation"], float)
        v.color = COLORS.get(p.role, [0.7, 0.7, 0.7, 0.5])
        volumes[p.name] = v
    return volumes


# Radionuclides for add_activity_source: physical half-life (s). Gamma lines and their yield
# (photons per decay) come from opengate's ICRP-107 data.
RADIONUCLIDES = {"Tc99m": {"half_life_s": 6.0067 * 3600.0}}


def _set_energy(source, energy_kev, radionuclide):
    """Mono-energetic gammas, or the radionuclide's gamma lines. With a discrete spectrum
    opengate emits activity x sum(weights) photons per second (weights = photons per
    decay, 0.891 for Tc-99m; checked 2026-10-09), so the activity stays in decays/s."""
    import opengate as gate
    from opengate.sources.utility import set_source_energy_spectrum

    if radionuclide is None:
        source.energy.type = "mono"
        source.energy.mono = energy_kev * gate.g4_units.keV
        return
    set_source_energy_spectrum(source, radionuclide)


def _add_image_mesh(sim, spec: pm.Spec, prefix: str, mother: str) -> dict:
    """The model's nested meshes (image frame): the outermost carries the placement
    (pose x image rotation), the others are its daughters in the same frame."""
    import opengate as gate

    m = pm.IMAGE_MODELS[spec.phantom]
    if not m.meshes:
        raise ValueError(f"{spec.phantom} has no meshes; use image_geometry='voxel'")
    img = pm.image_data(spec)
    volumes = {}
    for r in m.meshes:
        name = f"{prefix}_body" if r["mother"] is None else f"{prefix}_{r['name']}"
        v = sim.add_volume("Tesselated", name)
        v.file_name = str(m.directory() / r["file"])
        v.size_unit = gate.g4_units.mm
        v.origin_at_cog = False
        v.material = r["material"]
        if r["mother"] is None:
            v.mother = mother
            v.translation = list(spec.pose.to_world(pm.image_to_local(img, np.zeros(3))))
            v.rotation = spec.pose.rotation() @ img["M"]
        else:
            v.mother = f"{prefix}_body" if r["mother"] == m.meshes[0]["name"] else f"{prefix}_{r['mother']}"
        v.color = [0.9, 0.75, 0.65, 0.3] if r["mother"] is None else [0.95, 0.95, 0.9, 0.6]
        volumes[r["name"]] = v
    return volumes


def add_activity_source(sim, spec: pm.Spec, activity_bq: float, *, energy_kev: float = 140.0,
                        radionuclide: str | None = None, half_life: bool = True,
                        prefix: str = "phantom", name: str = "phantom_source"):
    """Analytic phantoms: uniform gamma source in the water (cold inserts excluded by
    confinement). Image phantoms: a voxel source from the activity image, attached to the
    image volume (follows its pose); add_phantom must have been called first.

    `activity_bq` is decays per second (at time 0). With `radionuclide` (e.g. "Tc99m") the
    source emits its gamma lines, activity x yield photons per second, and, with
    `half_life`, decays with the physical half-life (GATE's absolute time, so time-sliced
    jobs see the right activity)."""
    import opengate as gate

    if pm.is_image(spec.phantom):
        info = _IMAGE_FILES[prefix]
        source = sim.add_source("VoxelSource", name)
        source.image = str(info["files"]["activity"])
        if info["volume"]:  # follows the image volume
            source.attached_to = info["volume"]
            source.position.translation = [0.0, 0.0, 0.0]
        else:  # mesh geometry: the image box's own placement
            x = info["placement"]
            source.attached_to = x["mother"]
            source.position.translation = list(x["translation"])
            source.position.rotation = np.asarray(x["rotation"], float)
    else:
        parts, _ = pm.build(spec.phantom, spec.options)
        water = next(p for p in parts if p.role == "water")
        source = sim.add_source("GenericSource", name)
        source.position.type = "cylinder"
        source.position.radius = water.radius
        # sampling cylinder at least as tall as the water whether dz is read as half or full height;
        # confinement discards everything outside the water volume
        source.position.dz = water.length
        source.position.translation = list(spec.pose.to_world(np.asarray(water.center, float)))
        source.position.rotation = spec.pose.rotation()
        source.position.confine = f"{prefix}_water"
    source.particle = "gamma"
    _set_energy(source, energy_kev, radionuclide)
    source.activity = activity_bq * gate.g4_units.Bq
    if radionuclide is not None and half_life:
        source.half_life = RADIONUCLIDES[radionuclide]["half_life_s"] * gate.g4_units.s
    return source
