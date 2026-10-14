"""Check that the GATE phantom (payload/python/phantom_gate.py) matches the voxel phantom.

For each spec, builds the phantom alone in a GATE 10 simulation, voxelises the Geant4
geometry with opengate (sim.voxelize_geometry: the volume at each voxel centre), and
compares it voxel by voxel with phantom_models.voxelize() on the same centred grid.
Every voxel is compared by role (air, water, body, plate, rod, sphere, stem, fitting).
Image phantoms: opengate's voxeliser crashes on an ImageVolume, so they are checked by where
the GATE voxel source emits (image_source_check).

The default grid (145 voxels of 2 mm) puts voxel centres on even millimetres, so no part
boundary of the default specs falls exactly on a voxel centre: a mismatch is a geometry
difference, not a tie-breaking one.

Run from dev/python (opengate env):
    python check_gate_phantom.py [--spec spec.json ...] [--grid 145] [--voxel-mm 2]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "payload" / "python"))
import phantom_models as pm  # noqa: E402

DEFAULT_SPECS = [
    pm.Spec("xcat_brain_perfusion", pm.Pose(-20.0, 5.0, True, 30.0)),
    pm.Spec("acr_deluxe"),
    pm.Spec("acr_deluxe", pm.Pose(-20.0, 7.0, True, 33.0)),
    pm.Spec("small"),
    pm.Spec("small", pm.Pose(-26.0, 0.0, True, -150.0)),
]


def _patch_opengate_extent():
    """opengate 10.1.1 accepts an explicit extent only as tuples, then subtracts them
    (needs arrays); convert inside its image helper. The voxeliser runs in a forked
    process, which inherits the patch."""
    import opengate.voxelize as ov

    if getattr(ov, "_extent_patched", False):
        return
    original = ov.create_image_with_extent

    def patched(extent, spacing=(1, 1, 1), margin=0):
        return original([np.asarray(e, float) for e in extent], spacing, margin)

    ov.create_image_with_extent = patched
    ov._extent_patched = True


def gate_labels(spec: pm.Spec, n: int, voxel_mm: float) -> np.ndarray:
    """Role labels [x, y, z] of the Geant4 geometry on the centred grid.

    opengate's label image is 8-bit, so a slab may contain at most 255 distinct volumes:
    the grid is voxelised in z-slabs, halved until each holds fewer than 250."""
    import itk
    import opengate as gate
    from phantom_gate import add_phantom

    _patch_opengate_extent()
    sim = gate.Simulation()
    sim.world.size = [800.0, 800.0, 800.0]
    sim.world.material = "G4_AIR"
    sim.verbose_level = gate.logger.NONE
    sim.progress_bar = False
    add_phantom(sim, spec)
    parts, _ = pm.build(spec.phantom, spec.options)
    role_of = {f"phantom_{p.name}": pm.ROLE_LABEL[p.role] for p in parts}
    half = n * voxel_mm / 2
    out = np.zeros((n, n, n), np.uint8)

    def run(k0, k1):
        lo, hi = (-half, -half, -half + k0 * voxel_mm), (half, half, -half + k1 * voxel_mm)
        labels, image = sim.voxelize_geometry(extent=(lo, hi), spacing=(voxel_mm,) * 3)
        if len(labels) >= 250 and k1 - k0 > 1:
            mid = (k0 + k1) // 2
            run(k0, mid)
            run(mid, k1)
            return
        if len(labels) >= 250:
            raise RuntimeError(f"more than 250 volumes in one voxel slice (z index {k0})")
        arr = itk.array_from_image(image).transpose(2, 1, 0)  # itk (z, y, x) -> [x, y, z]
        origin = np.array(image.GetOrigin())
        if arr.shape != (n, n, k1 - k0) or not np.allclose(origin, np.array(lo) + voxel_mm / 2):
            raise RuntimeError(f"GATE slab {arr.shape} origin {origin}, expected {(n, n, k1 - k0)} at {lo}")
        lut = np.zeros(256, np.uint8)
        for name, v in labels.items():
            lut[v["label"]] = role_of.get(name, 0)  # world -> air
        out[:, :, k0:k1] = lut[arr]

    run(0, n)
    return out


def surface_distance(spec: pm.Spec, points: np.ndarray) -> np.ndarray:
    """Distance (mm) from each world point to the nearest surface of any part."""
    parts, _ = pm.build(spec.phantom, spec.options)
    q = spec.pose.to_local(points)
    best = np.full(len(q), np.inf)
    for p in parts:
        d = q - np.asarray(p.center, float)
        if p.shape == "sphere":
            dist = np.abs(np.linalg.norm(d, axis=1) - p.radius)
        else:
            a = np.asarray(p.axis, float)
            h = d @ a
            rad = np.linalg.norm(d - np.outer(h, a), axis=1)
            dr, dz = rad - p.radius, np.abs(h) - p.length / 2
            # distance to the closed cylinder's surface (inside: nearest of side or caps)
            dist = np.where((dr <= 0) & (dz <= 0), np.minimum(-dr, -dz),
                            np.hypot(np.maximum(dr, 0), np.maximum(dz, 0)))
        best = np.minimum(best, dist)
    return best


def image_source_check(spec: pm.Spec, activity_bq: float = 2e5, tmp_dir=None,
                       image_geometry: str = "voxel") -> dict:
    """Image phantoms (opengate's geometry voxeliser crashes on an ImageVolume): run the
    GATE phantom (image_geometry "voxel" or "mesh") with every material set to vacuum and
    record where the voxel source emits; every emission must fall in a voxel with activity
    (phantom_models.sample_image), and each activity level must get its share of the
    emissions. The mesh case catches a source that does not follow the phantom's rotation
    (opengate's VoxelSource ignores position.rotation)."""
    import tempfile

    import opengate as gate
    import uproot
    from phantom_gate import add_activity_source, add_phantom

    tmp = Path(tmp_dir or tempfile.mkdtemp())
    sim = gate.Simulation()
    sim.world.size = [900.0] * 3
    sim.world.material = "G4_Galactic"
    sim.progress_bar, sim.number_of_threads, sim.random_seed, sim.output_dir = False, 1, 1, str(tmp)
    volumes = add_phantom(sim, spec, files_dir=tmp, image_geometry=image_geometry)
    for v in volumes.values():
        if image_geometry == "voxel":
            v.voxel_materials = [[lo, hi, "G4_Galactic"] for lo, hi, _ in v.voxel_materials]
        v.material = "G4_Galactic"
    add_activity_source(sim, spec, activity_bq)
    shell = sim.add_volume("SphereVolume", "shell")
    shell.rmin, shell.rmax, shell.material = 400.0, 410.0, "G4_Galactic"
    ps = sim.add_actor("PhaseSpaceActor", "ps")
    ps.attached_to, ps.attributes, ps.output_filename = "shell", ["TrackVertexPosition", "TrackID"], "ps.root"
    sim.run_timing_intervals = [[0, 1.0 * gate.g4_units.s]]
    sim.run(start_new_process=True)
    t = uproot.open(ps.get_output_path())["ps"].arrays(library="np")
    p = np.stack([t[f"TrackVertexPosition_{a}"] for a in "XYZ"], 1)[t["TrackID"] == 1]
    _, act = pm.sample_image(spec, p)
    img = pm.image_data(spec)
    levels = np.unique(img["activity"])
    n = np.array([(img["activity"] == L).sum() for L in levels])
    expected = levels * n / (levels * n).sum()
    got = np.array([(act == L).mean() for L in levels])
    return {"spec": spec.to_dict(), "image_geometry": image_geometry, "emissions": int(len(p)), "outside_activity": float((act == 0).mean()),
            "level_share_max_abs_diff": float(np.abs(expected - got).max()),
            "real_differences": int((act == 0).sum()), "ties_on_surface": 0}


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--spec", type=Path, nargs="*", help="spec .json files (default: built-in set)")
    parser.add_argument("--grid", type=int, default=145)
    parser.add_argument("--voxel-mm", type=float, default=2.0)
    parser.add_argument("--report", type=Path, help="write the results as JSON")
    args = parser.parse_args()
    specs = [pm.Spec.from_dict(json.loads(p.read_text())) for p in args.spec] if args.spec else DEFAULT_SPECS

    results, ok = [], True
    for spec in specs:
        if pm.is_image(spec.phantom):
            geometries = ["voxel"] + (["mesh"] if pm.IMAGE_MODELS[spec.phantom].meshes else [])
            for geometry in geometries:
                res = image_source_check(spec, image_geometry=geometry)
                ok &= res["outside_activity"] < 1e-3 and res["level_share_max_abs_diff"] < 0.01
                results.append(res)
                print(f"{spec.phantom:10s} {geometry:5s} {json.dumps(spec.to_dict()['pose'])}: {res['emissions']} emissions, "
                      f"{res['outside_activity']:.4%} outside the activity, level shares within {res['level_share_max_abs_diff']:.4f}")
            continue
        ref = pm.voxelize(spec, args.grid, args.voxel_mm)
        got = gate_labels(spec, args.grid, args.voxel_mm)
        bad = ref != got
        # voxel centres lying on a part surface (within 1 um) are ties, not geometry differences
        c = pm.grid_centers(args.grid, args.voxel_mm)
        ix = np.nonzero(bad)
        ties = surface_distance(spec, np.stack([c[ix[0]], c[ix[1]], c[ix[2]]], 1)) < 1e-3
        real = np.zeros_like(bad)
        real[tuple(i[~ties] for i in ix)] = True
        pairs = {}
        for a, b in zip(ref[bad], got[bad]):
            key = f"{pm.LABELS[int(a)][0]}->{pm.LABELS[int(b)][0]}"
            pairs[key] = pairs.get(key, 0) + 1
        ok &= real.sum() == 0
        res = {"spec": spec.to_dict(), "grid": {"size": args.grid, "voxel_mm": args.voxel_mm}, "voxels": int(ref.size),
               "mismatched": int(bad.sum()), "ties_on_surface": int(ties.sum()), "real_differences": int(real.sum()),
               "mismatch_by_role": pairs}
        results.append(res)
        print(f"{spec.phantom:10s} {json.dumps(spec.to_dict()['pose'])}: {bad.sum()} of {ref.size:,} voxels differ, "
              f"{ties.sum()} of them centred on a surface (ties), {real.sum()} real" + (f" {pairs}" if pairs else ""))
    if args.report:
        args.report.write_text(json.dumps(results, indent=2) + "\n")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
