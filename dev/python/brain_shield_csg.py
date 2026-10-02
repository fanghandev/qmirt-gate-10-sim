"""Constructive solid geometry (CSG) model of the brain SPECT lead shield.

The shield STL (BrainFrame.008.Lead_Shield.STL, 186,772 facets) is reproduced by:

- a spherical shell, r = 145-155 mm, keeping elevation >= -47.0 deg (bottom cut is
  a cone through the centre, i.e. a G4Sphere theta limit);
- minus a two-step face window bounded by constant azimuth and elevation
  (G4Sphere segments): azimuth 90 +/- 33.75 deg for elevation -26.5..-5.75 deg and
  90 +/- 39.476 deg for elevation -47.0..-26.5 deg;
- minus 73 square apertures, one per module, in the module frame (origin at the
  pinhole, +z toward the centre): a tapered square (G4Trd) of width
  27.528 - 1.5861 * z mm up to z = 3.8 mm, then a 21.65 mm square lip through the
  inner surface. The top module's aperture is rotated 2.5 deg about its axis.

Optional rounded edges (build_csg(rounded=True)), measured on the STL on 2026-10-02:
- every inner edge of the window and bottom cut (where a cut meets the r = 145 mm
  sphere) is rounded with r = 4.73 mm. The cuts contain the centre, so each round
  is an exact torus segment (axis z for the cones, the plane normal for the window
  sides);
- the window outline corners are rounded with r = 6.0 mm and the two window-bottom
  junctions with r = 14.0 mm, as straight cylinders along the corner's radial
  direction (same radius at r = 152 and 154 mm in the STL).
Still not modelled: where the inner-edge round follows a rounded corner (vertex
blends), and the small hole-corner fillets and lip chamfer inside each aperture.
All parameters were fitted to the STL on 2026-10-02 (see the plotting script).

This module builds the CSG with manifold3d for checking against the STL; it is not
yet a Geant4 geometry.
"""

import sys
from functools import lru_cache
from pathlib import Path

import manifold3d as mf
import numpy as np
import trimesh
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "payload" / "python"))
import gate_sim_brain_spect_boolean as brain  # noqa: E402

SHIELD = {
    "r_inner_mm": 145.0,
    "r_outer_mm": 155.0,
    "bottom_elevation_deg": -47.0,
    # (azimuth min, azimuth max), (elevation min, elevation max)
    "window": [
        ((56.25, 123.75), (-26.5, -5.75)),
        ((50.524, 129.476), (-47.0, -26.5)),
    ],
    "rounds": {
        "inner_edge_radius_mm": 4.73,
        "window_corner_radius_mm": 6.0,
        "bottom_junction_radius_mm": 14.0,
        # corner centre offsets in the tangent plane (along azimuth, along elevation)
        "bottom_junction_centre_mm": (14.0, 13.4),
    },
    "aperture": {
        "width_at_pinhole_mm": 27.528,
        "width_slope": -1.5861,  # d(width)/d(z), z toward the centre
        "lip_start_z_mm": 3.8,
        "lip_width_mm": 21.65,
        "top_module_rotation_deg": 2.5,
    },
}
_BIG = 400.0


def module_frames():
    """Pinholes (N, 3) and module rotations (N, 3, 3), local -> world."""
    df = brain.get_geometry_definitions()
    n = df.shape[0]
    pinholes = np.array([[df.item(i, f"Pinhole_{a}") for a in "xyz"] for i in range(n)])
    rotations = np.array([brain.get_head_rotation_matrix(df, i) for i in range(n)])
    return pinholes, rotations


def aperture_rotation(k: int, rotations: np.ndarray) -> np.ndarray:
    r = rotations[k]
    if k == len(rotations) - 1:
        a = np.radians(SHIELD["aperture"]["top_module_rotation_deg"])
        r = r @ np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return r


def _below_elevation(el_deg: float, segments: int):
    """All directions with elevation < el_deg (< 0): a downward cone, apex at 0."""
    radius = _BIG / np.tan(np.radians(-el_deg))
    return mf.Manifold.cylinder(_BIG, radius, 0.0, segments).translate((0, 0, -_BIG))


def _azimuth_wedge(az0: float, az1: float):
    box = mf.Manifold.cube((2 * _BIG,) * 3, True)
    a0, a1 = np.radians(az0), np.radians(az1)
    box = box.trim_by_plane((-np.sin(a0), np.cos(a0), 0.0), 0.0)
    return box.trim_by_plane((np.sin(a1), -np.cos(a1), 0.0), 0.0)


def aperture_local():
    p = SHIELD["aperture"]

    def square(width, z):
        h = width / 2
        return [[-h, -h, z], [h, -h, z], [h, h, z], [-h, h, z]]

    z0, z1 = -9.0, p["lip_start_z_mm"]
    width = lambda z: p["width_at_pinhole_mm"] + p["width_slope"] * z  # noqa: E731
    taper = mf.Manifold.hull_points(square(width(z0), z0) + square(width(z1), z1))
    lip = mf.Manifold.hull_points(square(p["lip_width_mm"], z1 - 0.01)
                                  + square(p["lip_width_mm"], 9.0))
    return taper + lip


def _direction(az_deg, el_deg):
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _ccw(poly):
    poly = np.asarray(poly, float)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    return poly if area > 0 else poly[::-1]


def _edge_round_profile(f, r_inner, material_below=False, eps=0.05, n=32):
    """2D sliver removed by rounding (radius f) the convex edge between the circle
    r = r_inner (lead outside it) and the line y = 0 through the centre (lead at
    y > 0). It reaches eps into the void so the subtraction leaves no skin."""
    cx, cy = np.sqrt((r_inner + f) ** 2 - f ** 2), f
    t_c = r_inner * np.array([cx, cy]) / (r_inner + f)  # tangent point on the circle
    phi_c = np.arctan2(t_c[1], t_c[0])
    a_c = np.arctan2(-cy, -cx)
    poly = [(cx, -eps), (r_inner - eps, -eps)]
    poly += [((r_inner - eps) * np.cos(t), (r_inner - eps) * np.sin(t)) for t in np.linspace(0, phi_c, n)]
    poly += [tuple(t_c)]
    poly += [(cx + f * np.cos(t), cy + f * np.sin(t)) for t in np.linspace(a_c, -np.pi / 2, n)[1:]]
    poly = np.array(poly)
    if material_below:
        poly[:, 1] *= -1
    return poly


def _revolved(poly, axis, start_dir, span_deg, segments):
    """Sweep a 2D profile (x = distance from the axis, y = along the axis) about
    `axis`, starting at `start_dir` and turning right-handedly about it."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    x = np.asarray(start_dir, float)
    x = x - axis * (x @ axis)
    x /= np.linalg.norm(x)
    y = np.cross(axis, x)
    m = mf.Manifold.revolve(mf.CrossSection([_ccw(poly)]), segments, float(span_deg))
    return m.transform(np.c_[x, y, axis, np.zeros(3)])


def _inner_edge_rounds(segments):
    s = SHIELD
    f, r_in = s["rounds"]["inner_edge_radius_mm"], s["r_inner_mm"]
    (up_az, up_el), (lo_az, lo_el) = s["window"]
    pieces = []
    # cones of constant elevation (lead above the cut): torus segments about z
    cones = [
        (s["bottom_elevation_deg"], lo_az[1], 360.0 + lo_az[0]),  # outside the window
        (up_el[1], up_az[0], up_az[1]),  # window top edge
        (lo_el[1], lo_az[0], up_az[0]),  # step, left
        (lo_el[1], up_az[1], lo_az[1]),  # step, right
    ]
    for el, az0, az1 in cones:
        e = np.radians(el)
        rot = np.array([[np.cos(e), -np.sin(e)], [np.sin(e), np.cos(e)]])
        prof = _edge_round_profile(f, r_in) @ rot.T  # line y = 0 turned to the cone
        pieces.append(_revolved(prof, (0, 0, 1), _direction(az0, 0), az1 - az0, segments))
    # window side planes (constant azimuth): torus segments about the plane normal;
    # turning about +n lowers the elevation, so start at the upper end
    sides = [(up_az[0], up_el, True), (up_az[1], up_el, False),
             (lo_az[0], lo_el, True), (lo_az[1], lo_el, False)]
    for az, (el0, el1), lead_on_low_azimuth in sides:
        a = np.radians(az)
        n = np.array([-np.sin(a), np.cos(a), 0.0])  # toward increasing azimuth
        prof = _edge_round_profile(f, r_in, material_below=lead_on_low_azimuth)
        pieces.append(_revolved(prof, n, _direction(az, el1), el1 - el0, segments))
    return mf.Manifold.batch_boolean(pieces, mf.OpType.Add)


def _corner_slivers(segments):
    """Window outline corners: (sliver solid, True if lead is added there)."""
    s = SHIELD
    rr, rb = s["rounds"]["window_corner_radius_mm"], s["rounds"]["bottom_junction_radius_mm"]
    bx, by = s["rounds"]["bottom_junction_centre_mm"]
    (up_az, up_el), (lo_az, lo_el) = s["window"]
    bottom = s["bottom_elevation_deg"]
    # (az, el, centre along azimuth, centre along elevation, radius, add lead?)
    corners = [
        (up_az[0], up_el[1], +rr, -rr, rr, True), (up_az[1], up_el[1], -rr, -rr, rr, True),
        (up_az[0], lo_el[1], -rr, +rr, rr, False), (up_az[1], lo_el[1], +rr, +rr, rr, False),
        (lo_az[0], lo_el[1], +rr, -rr, rr, True), (lo_az[1], lo_el[1], -rr, -rr, rr, True),
        (lo_az[0], bottom, -bx, +by, rb, False), (lo_az[1], bottom, +bx, +by, rb, False),
    ]
    eps, r0, depth = 0.05, s["r_inner_mm"] - 2.0, s["r_outer_mm"] - s["r_inner_mm"] + 4.0
    out = []
    for az, el, cx, cy, radius, add in corners:
        d = _direction(az, el)
        e1 = np.cross([0.0, 0.0, 1.0], d)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        # rectangle from just past the sharp corner (eps) to the round's centre
        xs = sorted((-np.sign(cx) * eps, cx))
        ys = sorted((-np.sign(cy) * eps, cy))
        rect = mf.CrossSection.square((xs[1] - xs[0], ys[1] - ys[0])).translate((xs[0], ys[0]))
        disc = mf.CrossSection.circle(radius, segments).translate((cx, cy))
        sliver = mf.Manifold.extrude(rect - disc, depth)
        out.append((sliver.transform(np.c_[e1, e2, d, d * r0]), add))
    return out


@lru_cache(maxsize=4)
def build_csg(segments: int = 720, rounded: bool = False):
    s = SHIELD
    shell = (mf.Manifold.sphere(s["r_outer_mm"], segments)
             - mf.Manifold.sphere(s["r_inner_mm"], segments))
    band = shell
    shell = shell - _below_elevation(s["bottom_elevation_deg"], segments)
    for (az0, az1), (el0, el1) in s["window"]:
        region = _azimuth_wedge(az0, az1) ^ _below_elevation(el1, segments)
        # a window boundary on the bottom cut would leave a zero-thickness sliver
        if el0 > s["bottom_elevation_deg"]:
            region = region - _below_elevation(el0, segments)
        shell = shell - region
    if rounded:
        shell = shell - _inner_edge_rounds(segments)
        for sliver, add in _corner_slivers(segments):
            shell = (shell + (sliver ^ band)) if add else (shell - sliver)
    pinholes, rotations = module_frames()
    hole = aperture_local()
    holes = [hole.transform(np.c_[aperture_rotation(k, rotations), pinholes[k]])
             for k in range(len(pinholes))]
    return shell - mf.Manifold.batch_boolean(holes, mf.OpType.Add)


@lru_cache(maxsize=1)
def stl_world_mesh():
    """The shield STL in world coordinates (Gate rotation applied), as (V, F)."""
    m = trimesh.load(brain.get_geometry_base_definition(0)["shielding file path"])
    v = np.asarray(m.vertices, float) @ brain.get_shielding_rotation_matrix().T
    return v, np.asarray(m.faces, np.int64)


def stl_manifold():
    v, f = stl_world_mesh()
    return mf.Manifold(mf.Mesh64(vert_properties=v, tri_verts=f.astype(np.uint64)))


def manifold_mesh(manifold):
    m = manifold.to_mesh64()
    return np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts, np.int64)


def closest_points_on_triangles(p, a, b, c):
    """Closest point on triangle (a, b, c) to p; all (N, 3). Ericson's region test."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        out = a + ab * (vb / denom)[:, None] + ac * (vc / denom)[:, None]
        t_ab, t_ac = d1 / (d1 - d3), d2 / (d2 - d6)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    for mask, val in (
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * t_ab[:, None]),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * t_ac[:, None]),
        ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b + (c - b) * t_bc[:, None]),
        ((d1 <= 0) & (d2 <= 0), a),
        ((d3 >= 0) & (d4 <= d3), b),
        ((d6 >= 0) & (d5 <= d6), c),
    ):
        out[mask] = val[mask]
    return out


def _surface_samples(tri, spacing):
    """Points on every triangle, no more than `spacing` apart, with their triangle index."""
    edge = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
    n_div = np.maximum(1, np.ceil(edge / spacing).astype(int))
    pts, owner = [], []
    for n in np.unique(n_div):
        sel = np.flatnonzero(n_div == n)
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (i + j) <= n
        bary = np.c_[i[keep], j[keep], n - i[keep] - j[keep]] / n  # (m, 3)
        p = np.einsum("mk,tkd->tmd", bary, tri[sel]).reshape(-1, 3)
        pts.append(p)
        owner.append(np.repeat(sel, len(bary)))
    return np.concatenate(pts), np.concatenate(owner)


def signed_distance(points, verts, faces, spacing=0.5, chunk=20_000):
    """Exact signed distance from points to the surface (verts, faces); + outside.

    Every triangle is sampled at <= `spacing`; the true closest triangle always has a
    sample within (nearest-sample distance + spacing), so all triangles owning a
    sample inside that ball are tested exactly.
    """
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1)[:, None] + 1e-300
    samples, owner = _surface_samples(tri, spacing)
    tree = cKDTree(samples)
    out = np.empty(len(points))
    for s in range(0, len(points), chunk):
        p = points[s:s + chunk]
        d_nn, _ = tree.query(p)
        balls = tree.query_ball_point(p, d_nn + spacing * 1.01)
        rows = np.repeat(np.arange(len(p)), [len(b) for b in balls])
        cand = owner[np.concatenate(balls).astype(int)]
        key = np.unique(rows * len(tri) + cand)
        rows, cand = key // len(tri), key % len(tri)
        t = tri[cand]
        q = closest_points_on_triangles(p[rows], t[:, 0], t[:, 1], t[:, 2])
        dist = np.linalg.norm(p[rows] - q, axis=1)
        order = np.lexsort((dist, rows))
        first = order[np.r_[True, rows[order][1:] != rows[order][:-1]]]
        best_q, best_t, best_r = q[first], cand[first], rows[first]
        sign = np.sign(np.einsum("ij,ij->i", p[best_r] - best_q, nrm[best_t]))
        sign[sign == 0] = 1
        out[s + best_r] = sign * dist[first]
    return out


def _thickness_rows(args):
    manifold_key, directions, r0, r1 = args
    manifold = _WORKER_MANIFOLDS[manifold_key]
    out = np.zeros(len(directions))
    for i, u in enumerate(directions):
        hits = manifold.ray_cast(tuple(u * r0), tuple(u * r1))
        inside, total, last = False, 0.0, 0.0
        for h in hits:
            entering = float(np.dot(h.normal, u)) < 0
            if entering and not inside:
                inside, last = True, h.distance
            elif not entering and inside:
                inside, total = False, total + (h.distance - last)
        out[i] = total * (r1 - r0)
    return out


_WORKER_MANIFOLDS = {}


def radial_thickness_map(manifolds: dict, az_deg, el_deg, r0=140.0, r1=160.0, workers=32):
    """Lead path length (mm) along radial rays from r0 to r1, per manifold."""
    import multiprocessing as mp

    A, E = np.meshgrid(np.radians(az_deg), np.radians(el_deg))
    u = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
    _WORKER_MANIFOLDS.clear()
    _WORKER_MANIFOLDS.update(manifolds)
    chunks = np.array_split(np.arange(len(u)), workers * 8)
    result = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for key in manifolds:
            parts = pool.map(_thickness_rows, [(key, u[c], r0, r1) for c in chunks])
            result[key] = np.concatenate(parts).reshape(A.shape)
    return result
