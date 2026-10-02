"""Geometric field-of-view analysis for the brain SPECT system.

For every point on a grid, count how many modules can detect a photon emitted there
(straight line through the pinhole centre that lands on the crystal without crossing
lead shielding or another module's tungsten), and sum a geometric pinhole
sensitivity. Points inside solid material are flagged as inaccessible.

Model (from gate_sim_brain_spect_boolean.py):
- Pinhole at r = 149.61 mm. The back bore (Frustum_C) opens from the pinhole to
  50 mm square over h_body = 25 mm, which is exactly the crystal front face, so each
  module accepts a square pyramid with tan(half-angle) = 25 / 25 = 1. The front
  nozzle (Frustum_B) is slightly wider (tan ~1.01-1.04), so it does not limit it.
- Blockers: the lead shield STL (voxelized exactly at 1 mm) and the outer envelopes
  (Frustum_A + Box_A) of the other 72 collimators. A ray is not marched through
  the last `NOZZLE_SKIP_MM` before its own pinhole (inside its own nozzle bore).
- Septal penetration through the knife edges is ignored.

Usage:  ma opengate && python dev/python/brain_spect_max_fov.py [--voxel-mm 2]
"""

import argparse
import json
import sys
from pathlib import Path

import numba as nb
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "payload" / "python"))
import gate_sim_brain_spect_boolean as brain  # noqa: E402

SOLID_SHIELD = 100
NOZZLE_SKIP_MM = 6.0
ACCEPT_TAN = 1.0


def module_table():
    df = brain.get_geometry_definitions()
    n = df.shape[0]
    pinholes = np.array(
        [[df.item(i, f"Pinhole_{a}") for a in "xyz"] for i in range(n)], dtype=float
    )
    rotations = np.array([brain.get_head_rotation_matrix(df, i) for i in range(n)])
    variants = [brain.map_crystal_id(i, n, "sequential") for i in range(n)]
    configs = [brain.get_geometry_base_definition(v) for v in variants]
    centers = np.array(
        [brain.get_collimator_center(c, df, i) for i, c in enumerate(configs)]
    )
    w_pinhole = np.array([c["collimator definition"]["w_pinhole"] for c in configs])
    # sanity: local +z points from the pinhole toward the FOV centre
    assert np.all(np.einsum("nij,nj->ni", rotations.transpose(0, 2, 1), -pinholes)[:, 2] > 0)
    return pinholes, rotations, centers, configs, w_pinhole


# ---------------------------------------------------------------- voxelization
@nb.njit(cache=True)
def _scanline_hits(tri, x0, y0, step, nx, ny):
    """z of every triangle crossing for each (ix, iy) column centre (parity fill)."""
    cap = tri.shape[0] * 8 + 1024
    col = np.empty(cap, np.int64)
    zs = np.empty(cap, np.float64)
    k = 0
    for t in range(tri.shape[0]):
        ax, ay, az = tri[t, 0]
        bx, by, bz = tri[t, 1]
        cx, cy, cz = tri[t, 2]
        ix0 = max(int(np.ceil((min(ax, bx, cx) - x0) / step)), 0)
        ix1 = min(int(np.floor((max(ax, bx, cx) - x0) / step)), nx - 1)
        iy0 = max(int(np.ceil((min(ay, by, cy) - y0) / step)), 0)
        iy1 = min(int(np.floor((max(ay, by, cy) - y0) / step)), ny - 1)
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if det == 0.0:
            continue
        for ix in range(ix0, ix1 + 1):
            px = x0 + ix * step
            for iy in range(iy0, iy1 + 1):
                py = y0 + iy * step
                l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
                l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
                l3 = 1.0 - l1 - l2
                # exact edge hits are rare with float vertices; any double count
                # shows up as an odd column, which voxelize_mesh reports and skips
                if l1 < 0 or l2 < 0 or l3 < 0:
                    continue
                if k >= cap:
                    return col[:k], zs[:k], -1
                col[k] = ix * ny + iy
                zs[k] = l1 * az + l2 * bz + l3 * cz
                k += 1
    return col[:k], zs[:k], 0


def voxelize_mesh(vertices, faces, origin, step, shape):
    nx, ny, nz = shape
    tri = vertices[faces].astype(np.float64)
    col, zs, err = _scanline_hits(tri, origin[0], origin[1], step, nx, ny)
    assert err == 0, "scanline buffer overflow"
    order = np.lexsort((zs, col))
    col, zs = col[order], zs[order]
    occ = np.zeros(shape, dtype=bool)
    starts = np.flatnonzero(np.r_[True, col[1:] != col[:-1]])
    ends = np.r_[starts[1:], col.size]
    z_centres = origin[2] + step * np.arange(nz)
    odd = 0
    for s, e in zip(starts, ends):
        if (e - s) % 2:
            odd += 1
            continue
        ix, iy = divmod(int(col[s]), ny)
        for a, b in zip(zs[s:e:2], zs[s + 1 : e : 2]):
            occ[ix, iy, (z_centres >= a) & (z_centres < b)] = True
    return occ, odd


def collimator_envelope_labels(labels, origin, step, centers, rotations, configs):
    """Write module id+1 into voxels inside each collimator's outer envelope."""
    shape = np.array(labels.shape)
    for i, (c, r, cfg) in enumerate(zip(centers, rotations, configs)):
        prims = brain.get_collimator_outer_primitives(cfg)
        lo = np.floor((c - 60 - origin) / step).astype(int).clip(0, shape - 1)
        hi = np.ceil((c + 60 - origin) / step).astype(int).clip(0, shape - 1)
        ax = [origin[k] + step * np.arange(lo[k], hi[k] + 1) for k in range(3)]
        g = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
        loc = (g - c) @ r  # world -> local (r is local->world)
        inside = np.zeros(g.shape[:3], bool)
        for p in prims:
            q = loc - np.asarray(p["offset_mm"])
            if p["type"] == "box":
                h = np.asarray(p["size_mm"]) / 2
                inside |= np.all(np.abs(q) <= h, axis=-1)
            else:
                t = (q[..., 2] + p["dz"]) / (2 * p["dz"])
                hx = p["dx1"] + (p["dx2"] - p["dx1"]) * t
                hy = p["dy1"] + (p["dy2"] - p["dy1"]) * t
                inside |= (
                    (np.abs(q[..., 2]) <= p["dz"])
                    & (np.abs(q[..., 0]) <= hx)
                    & (np.abs(q[..., 1]) <= hy)
                )
        sub = labels[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1]
        sub[inside & (sub == 0)] = i + 1


# ---------------------------------------------------------------- ray casting
@nb.njit(parallel=True, cache=True)
def visibility(points, pinholes, rotations, w_pinhole, labels, origin, step,
               accept_tan, skip_mm, march_mm):
    n = points.shape[0]
    m = pinholes.shape[0]
    nx, ny, nz = labels.shape
    count = np.zeros(n, np.int16)
    sens = np.zeros(n, np.float64)
    solid = np.zeros(n, np.bool_)
    for p in nb.prange(n):
        px, py, pz = points[p, 0], points[p, 1], points[p, 2]
        ix = int(np.floor((px - origin[0]) / step + 0.5))
        iy = int(np.floor((py - origin[1]) / step + 0.5))
        iz = int(np.floor((pz - origin[2]) / step + 0.5))
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz and labels[ix, iy, iz] != 0:
            solid[p] = True
            continue
        for k in range(m):
            dx = px - pinholes[k, 0]
            dy = py - pinholes[k, 1]
            dz = pz - pinholes[k, 2]
            # world -> local: R^T d
            lx = rotations[k, 0, 0] * dx + rotations[k, 1, 0] * dy + rotations[k, 2, 0] * dz
            ly = rotations[k, 0, 1] * dx + rotations[k, 1, 1] * dy + rotations[k, 2, 1] * dz
            lz = rotations[k, 0, 2] * dx + rotations[k, 1, 2] * dy + rotations[k, 2, 2] * dz
            if lz <= 0 or abs(lx) > accept_tan * lz or abs(ly) > accept_tan * lz:
                continue
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            nsteps = int((dist - skip_mm) / march_mm)
            blocked = False
            for s in range(1, nsteps + 1):
                f = 1.0 - s * march_mm / dist
                qx = pinholes[k, 0] + f * dx
                qy = pinholes[k, 1] + f * dy
                qz = pinholes[k, 2] + f * dz
                jx = int(np.floor((qx - origin[0]) / step + 0.5))
                jy = int(np.floor((qy - origin[1]) / step + 0.5))
                jz = int(np.floor((qz - origin[2]) / step + 0.5))
                if jx < 0 or jy < 0 or jz < 0 or jx >= nx or jy >= ny or jz >= nz:
                    continue
                lab = labels[jx, jy, jz]
                if lab != 0 and lab != k + 1:
                    blocked = True
                    break
            if blocked:
                continue
            count[p] += 1
            cos_a = lz / dist
            # square aperture area seen at angle a, as a fraction of 4 pi
            sens[p] += w_pinhole[k] ** 2 * cos_a / (4.0 * np.pi * dist * dist)
    return count, sens, solid


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--voxel-mm", type=float, default=2.0)
    ap.add_argument("--label-mm", type=float, default=1.0)
    ap.add_argument("--extent-mm", type=float, nargs=6,
                    default=[-200, 200, -200, 200, -300, 170],
                    help="xmin xmax ymin ymax zmin zmax of the analysis grid")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent / "brain_fov_analysis")
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    pinholes, rotations, centers, configs, w_pinhole = module_table()
    print(f"{len(pinholes)} modules; pinhole |r| = {np.linalg.norm(pinholes, axis=1).mean():.2f} mm")

    ext = np.array(args.extent_mm, float).reshape(3, 2)
    # label grid (blockers), padded so rays to any pinhole stay inside it
    lab_lo = np.minimum(ext[:, 0], -200.0)
    lab_hi = np.maximum(ext[:, 1], 200.0)
    lab_shape = tuple(np.round((lab_hi - lab_lo) / args.label_mm).astype(int) + 1)
    print("label grid", lab_shape)

    mesh = trimesh.load(configs[0]["shielding file path"])
    verts = mesh.vertices @ brain.get_shielding_rotation_matrix().T
    occ, odd = voxelize_mesh(verts, mesh.faces, lab_lo, args.label_mm, lab_shape)
    print(f"shield voxels: {occ.sum()} ({occ.sum() * args.label_mm**3 / 1e3:.0f} cm^3 "
          f"vs mesh volume {mesh.volume / 1e3:.0f} cm^3), odd columns: {odd}")
    labels = np.zeros(lab_shape, np.int8)
    labels[occ] = SOLID_SHIELD
    collimator_envelope_labels(labels, lab_lo, args.label_mm, centers, rotations, configs)

    axes = [np.arange(lo, hi + 1e-9, args.voxel_mm) for lo, hi in ext]
    grid_shape = tuple(len(a) for a in axes)
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    print("analysis grid", grid_shape, f"{len(pts):.3e} points")
    count, sens, solid = visibility(
        pts, pinholes, rotations, w_pinhole, labels, lab_lo, args.label_mm,
        ACCEPT_TAN, NOZZLE_SKIP_MM, 0.5,
    )
    np.savez_compressed(
        args.output / "brain_fov_maps.npz",
        x=axes[0], y=axes[1], z=axes[2],
        count=count.reshape(grid_shape), sens=sens.reshape(grid_shape),
        solid=solid.reshape(grid_shape),
        pinholes=pinholes, w_pinhole=w_pinhole,
    )
    tip_radius = float(min(
        np.linalg.norm(pinholes, axis=1).min() - c["collimator definition"]["h_nozzle"]
        for c in configs
    ))
    summary = summarize(axes, count.reshape(grid_shape), solid.reshape(grid_shape),
                        sens.reshape(grid_shape), args.voxel_mm, tip_radius)
    (args.output / "brain_fov_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cavity_radial_profile"}, indent=2))


def summarize(axes, count, solid, sens, dv, tip_radius_mm):
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    seen = count >= 1
    cavity = (R < tip_radius_mm) & ~solid
    out = {"voxel_mm": dv, "nozzle_tip_min_radius_mm": tip_radius_mm,
           "max_sphere_diameter_without_overlap_mm": 2 * tip_radius_mm}
    out["solid_min_radius_on_grid_mm"] = float(R[solid].min())
    out["cavity_min_count"] = int(count[cavity].min())
    # largest origin-centred sphere seen by >= k modules everywhere
    for k in (1, 8, 24, 48, 73):
        bad = (count < k) | solid
        out[f"centred_sphere_diameter_mm_min{k}_modules"] = float(2 * R[bad].min())
    edges = np.arange(0, tip_radius_mm + dv, 5.0)
    prof = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = cavity & (R >= a) & (R < b)
        if m.any():
            prof.append({"r_mm": float(0.5 * (a + b)), "min_count": int(count[m].min()),
                         "mean_count": float(count[m].mean()),
                         "mean_sens": float(sens[m].mean())})
    out["cavity_radial_profile"] = prof
    # detectable space outside the nozzle-tip sphere (via the neck / face openings)
    outside = seen & ~solid & (R >= tip_radius_mm)
    out["outside_cavity_detectable_cm3"] = float(outside.sum() * dv**3 / 1e3)
    for name, m in {"below_z_-113 (neck opening)": outside & (Z < -113),
                    "z_>=_-113 (face window / gaps)": outside & (Z >= -113)}.items():
        out[name] = {"volume_cm3": float(m.sum() * dv**3 / 1e3),
                     "sens_fraction_of_cavity_total": float(sens[m].sum() / sens[cavity].sum())}
    return out


if __name__ == "__main__":
    main()
