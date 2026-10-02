"""Simplify (and optionally cut) the brain SPECT lead shield STL for faster Geant4 runs.

The shield is one 186,772-facet G4TessellatedSolid. Most facets are tiny (median
0.008 mm^2) on hole walls and edges, and at 64 threads the full mesh made the
simulation ~13x slower per primary than a simplified one (benchmark 2026-10-01).

Simplification: manifold3d's simplify(tol) keeps a subset of the original vertices
and moves no surface by more than `tol` mm. At 0.05 mm the shield has 43,258 facets,
its volume changes by 5e-5, and its inner surface moves inward by at most 11 um
(min radius 144.7406 vs 144.7518 mm). With a fixed seed, 99.99% of singles were
identical to the original geometry and Geant4 reported no exceptions.

Cutting (--n-az/--n-z > 1) is exact in manifold3d, but the written pieces made
Geant4 report stuck tracks (GeomNav1002) and overlaps at the cut faces, which killed
photons and lost ~0.45% of singles. Do not use cut pieces for production.

Pieces are written in world coordinates (the Gate shielding rotation already
applied), so they are placed with an identity rotation; load them with
gate_sim_brain_spect_boolean.py --shield-pieces-dir <dir>.

Usage:  ma opengate && python dev/python/split_brain_shield_stl.py --simplify-tol-mm 0.05
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import manifold3d as mf
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "payload" / "python"))
import gate_sim_brain_spect_boolean as brain  # noqa: E402


def load_world_manifold():
    stl_path = Path(brain.get_geometry_base_definition(0)["shielding file path"])
    mesh = trimesh.load(stl_path)
    verts = np.asarray(mesh.vertices, float) @ brain.get_shielding_rotation_matrix().T
    manifold = mf.Manifold(
        mf.Mesh64(vert_properties=verts, tri_verts=np.asarray(mesh.faces, np.uint64))
    )
    if manifold.status() != mf.Error.NoError:
        raise RuntimeError(f"shield STL is not a valid manifold: {manifold.status()}")
    return stl_path, manifold


def cut_pieces(manifold, n_az: int, n_z: int):
    """Exact azimuth-sector x z-slab pieces, each split into connected parts."""
    lo, hi = np.asarray(manifold.bounding_box()[:3]), np.asarray(manifold.bounding_box()[3:])
    z_edges = np.linspace(lo[2], hi[2], n_z + 1)
    az_edges = np.linspace(0.0, 2 * np.pi, n_az + 1)
    pieces = []
    for iz in range(n_z):
        slab = manifold
        if iz > 0:  # keep z >= z_edges[iz]
            slab = slab.trim_by_plane((0.0, 0.0, 1.0), float(z_edges[iz]))
        if iz < n_z - 1:  # keep z <= z_edges[iz + 1]
            slab = slab.trim_by_plane((0.0, 0.0, -1.0), -float(z_edges[iz + 1]))
        for ia in range(n_az):
            a0, a1 = az_edges[ia], az_edges[ia + 1]
            sector = slab
            if n_az > 1:
                # keep the side counter-clockwise of a0 and clockwise of a1
                sector = sector.trim_by_plane((-np.sin(a0), np.cos(a0), 0.0), 0.0)
                sector = sector.trim_by_plane((np.sin(a1), -np.cos(a1), 0.0), 0.0)
            for part in sector.decompose():
                if part.num_tri() > 0:
                    pieces.append(((iz, ia), part))
    return pieces


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-az", type=int, default=1,
                    help="azimuth sectors (each < 180 deg); >1 breaks navigation, see above")
    ap.add_argument("--n-z", type=int, default=1,
                    help="z slabs; >1 breaks navigation, see above")
    ap.add_argument("--simplify-tol-mm", type=float, default=0.05,
                    help="max surface displacement for simplification (0 = none)")
    ap.add_argument("--output", type=Path, default=None,
                    help="output directory (default persistent_data/brain_spect/stl/<name>)")
    args = ap.parse_args(argv)
    if args.n_az < 3 and args.n_az != 1:
        ap.error("--n-az must be 1 or >= 3 so each sector is under 180 degrees")

    stl_path, manifold = load_world_manifold()
    source = manifold
    if args.simplify_tol_mm > 0:
        source = manifold.simplify(args.simplify_tol_mm)
    name = f"{stl_path.stem}.simplified_{args.simplify_tol_mm:g}mm" + (
        f"_az{args.n_az}_z{args.n_z}" if args.n_az * args.n_z > 1 else "")
    out = args.output or stl_path.parent / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("piece_*.stl"):
        old.unlink()

    pieces = cut_pieces(source, args.n_az, args.n_z)
    records, vol_sum, tri_sum = [], 0.0, 0
    for k, ((iz, ia), part) in enumerate(pieces):
        m = part.to_mesh64()
        tm = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts),
                             process=False)
        if not tm.is_watertight:
            raise RuntimeError(f"piece {k} (z {iz}, az {ia}) is not closed")
        fname = f"piece_{k:03d}_z{iz}_az{ia:02d}.stl"
        tm.export(out / fname)
        vol_sum += part.volume()
        tri_sum += part.num_tri()
        records.append({"file": fname, "z_slab": iz, "az_sector": ia,
                        "triangles": int(part.num_tri()), "volume_mm3": part.volume(),
                        "bbox_mm": [float(x) for x in part.bounding_box()]})

    rel = vol_sum / manifold.volume() - 1
    manifest = {
        "source_stl": stl_path.name,
        "source_sha256": hashlib.sha256(stl_path.read_bytes()).hexdigest(),
        "frame": "world (Gate shielding rotation applied); place with identity rotation",
        "n_az": args.n_az, "n_z": args.n_z, "simplify_tol_mm": args.simplify_tol_mm,
        "source_volume_mm3": manifold.volume(), "pieces_volume_mm3": vol_sum,
        "volume_relative_difference": rel, "source_triangles": manifold.num_tri(),
        "pieces_triangles": tri_sum, "pieces": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(records)} pieces, {tri_sum} triangles (source {manifold.num_tri()}), "
          f"volume difference {rel:+.2e}, written to {out}")


if __name__ == "__main__":
    main()
