"""2D and 3D comparison plots of the CSG shield model against the shield STL.

Outputs (default dev/python/brain_shield_csg_comparison/):
  whole_thickness_maps.png   radial lead thickness (STL, CSG, difference) vs az/el
  whole_sections.png         meridional sections, STL vs CSG outlines, with zooms
  aperture_summary.png       per-aperture small residual volumes and max deviation
  apertures.pdf              one page per aperture: vertical cut at its azimuth and
                             12 slices perpendicular to its axis (STL vs CSG)
  aperture_NN.png            the aperture page for a few representative modules
  whole_3d.html              both surfaces coloured by signed distance to the other
  regions_3d.html            per-aperture (and window/bottom edge) 3D close-ups

Usage:  ma opengate && python dev/python/plot_brain_shield_csg_vs_stl.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import trimesh  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brain_shield_csg as S  # noqa: E402

STL_STYLE = {"color": "black", "lw": 1.1, "label": "STL"}
CSG_STYLE = {"color": "#d1495b", "lw": 1.1, "ls": "--", "label": "CSG"}
DEV_CLIP = 0.5  # mm, colour range for signed deviations
CROP = 32.0  # mm, half-size of the aperture neighbourhood in the module frame


# ----------------------------------------------------------------------------- helpers
def to_local(v, pinhole, rot):
    return (v - pinhole) @ rot


def crop_mesh(v, f, keep_vertex, whole_faces_only=False):
    """Faces touching the kept vertices (or, for display, faces fully inside)."""
    keep = keep_vertex[f].all(1) if whole_faces_only else keep_vertex[f].any(1)
    ff = f[keep]
    used, inv = np.unique(ff, return_inverse=True)
    return v[used], inv.reshape(-1, 3), used


def section_lines(v, f, origin, normal, axes):
    """Polylines of the mesh cut by a plane, projected on two axis vectors."""
    if len(f) == 0:
        return []
    mesh = trimesh.Trimesh(v, f, process=False)
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None:
        return []
    return [np.c_[np.asarray(seg) @ axes[0], np.asarray(seg) @ axes[1]] for seg in sec.discrete]


def draw(ax, lines, style, label=True):
    for i, ln in enumerate(lines):
        kw = dict(style)
        if not label or i:
            kw.pop("label", None)
        ax.plot(ln[:, 0], ln[:, 1], **kw)


# ----------------------------------------------------------------------------- data
def prepare(out: Path, rounded: bool = False):
    cache = out / "cache.npz"
    stl_v, stl_f = S.stl_world_mesh()
    pinholes, rotations = S.module_frames()
    if cache.exists():
        d = dict(np.load(cache, allow_pickle=True))
        print("loaded cache", cache)
        return d | {"stl_v": stl_v, "stl_f": stl_f, "pinholes": pinholes, "rotations": rotations}
    print("building CSG (720 segments for analysis, 360 for display) ...")
    csg = S.build_csg(720, rounded)
    csg_v, csg_f = S.manifold_mesh(csg)
    disp_v, disp_f = S.manifold_mesh(S.build_csg(360, rounded))
    stl = S.stl_manifold()
    print("signed distances ...")
    stl_dev = S.signed_distance(stl_v, csg_v, csg_f)       # + : STL surface outside CSG
    disp_dev = S.signed_distance(disp_v, stl_v, stl_f)     # + : CSG surface outside STL
    print("residual pieces ...")
    res = {}
    for name, diff in (("stl_minus_csg", stl - csg), ("csg_minus_stl", csg - stl)):
        parts = sorted(diff.decompose(), key=lambda p: -p.volume())
        # the largest piece is the thin film from the STL's faceted spheres
        res[name + "_film"] = parts[0].volume()
        cen = np.array([np.mean(np.asarray(p.bounding_box()).reshape(2, 3), 0) for p in parts[1:]])
        res[name + "_vol"] = np.array([p.volume() for p in parts[1:]])
        res[name + "_cen"] = cen
    print("thickness maps ...")
    az = np.arange(0, 360, 0.25) + 0.125
    el = np.arange(-50, 90, 0.25) + 0.125
    th = S.radial_thickness_map({"stl": stl, "csg": csg}, az, el)
    print("local thickness maps per aperture ...")
    g = np.arange(-CROP, CROP + 1e-9, 0.4)
    gx, gy = np.meshgrid(g, g)
    dirs = []
    for k in range(len(pinholes)):
        p = pinholes[k] + (np.stack([gx, gy, np.zeros_like(gx)], -1).reshape(-1, 3) @ rotations[k].T)
        dirs.append(p / np.linalg.norm(p, axis=1)[:, None])
    loc = _local_thickness({"stl": stl, "csg": csg}, np.concatenate(dirs))
    n_g = len(g)
    d = {
        "csg_v": csg_v, "csg_f": csg_f, "disp_v": disp_v, "disp_f": disp_f,
        "stl_dev": stl_dev, "disp_dev": disp_dev, "az": az, "el": el,
        "th_stl": th["stl"], "th_csg": th["csg"], "loc_g": g,
        "loc_stl": loc["stl"].reshape(len(pinholes), n_g, n_g),
        "loc_csg": loc["csg"].reshape(len(pinholes), n_g, n_g),
        "vol_stl": stl.volume(), "vol_csg": csg.volume(),
        **{k: v for k, v in res.items()},
    }
    np.savez_compressed(cache, **d)
    return d | {"stl_v": stl_v, "stl_f": stl_f, "pinholes": pinholes, "rotations": rotations}


def _local_thickness(manifolds, directions, r0=140.0, r1=160.0, workers=32):
    import multiprocessing as mp

    S._WORKER_MANIFOLDS.clear()
    S._WORKER_MANIFOLDS.update(manifolds)
    chunks = np.array_split(np.arange(len(directions)), workers * 8)
    out = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for key in manifolds:
            parts = pool.map(S._thickness_rows, [(key, directions[c], r0, r1) for c in chunks])
            out[key] = np.concatenate(parts)
    return out


# ----------------------------------------------------------------------------- 2D
def plot_thickness_maps(d, out: Path):
    az, el = d["az"], d["el"]
    ext = [az[0] - 0.125, az[-1] + 0.125, el[0] - 0.125, el[-1] + 0.125]
    diff = d["th_stl"] - d["th_csg"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), constrained_layout=True)
    for ax, img, title in ((axes[0], d["th_stl"], "STL"), (axes[1], d["th_csg"], "CSG")):
        im = ax.imshow(img, origin="lower", extent=ext, aspect="auto", cmap="Greys", vmin=0, vmax=12)
        ax.set_title(f"{title}: lead thickness along radial rays (r = 140-160 mm)")
        fig.colorbar(im, ax=ax, label="mm")
    im = axes[2].imshow(diff, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r",
                        vmin=-1, vmax=1)
    axes[2].set_title("STL − CSG thickness (red: STL has more lead)  ·  "
                      f"|diff| > 0.5 mm in {np.mean(np.abs(diff) > 0.5):.3%} of directions, "
                      f"> 0.1 mm in {np.mean(np.abs(diff) > 0.1):.2%}")
    fig.colorbar(im, ax=axes[2], label="mm")
    ph = d["pinholes"]
    pel = np.degrees(np.arcsin(ph[:, 2] / np.linalg.norm(ph, axis=1)))
    paz = np.degrees(np.arctan2(ph[:, 1], ph[:, 0])) % 360
    for ax in axes:
        ax.scatter(paz, pel, s=6, c="#2c6fbb", label="pinhole")
        ax.set_xlabel("azimuth (deg)")
        ax.set_ylabel("elevation (deg)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.savefig(out / "whole_thickness_maps.png", dpi=110)
    plt.close(fig)


def plot_sections(d, out: Path):
    stl_v, stl_f, csg_v, csg_f = d["stl_v"], d["stl_f"], d["csg_v"], d["csg_f"]
    # 53 deg cuts only the lower window step; 131 deg is just outside the window
    cuts = [0.0, 90.0, 180.0, 270.0, 53.0, 131.0]
    fig, axes = plt.subplots(3, 4, figsize=(18, 13.5), constrained_layout=True)
    for ax, az in zip(axes[:2].ravel(), cuts + [None, None]):
        if az is None:
            ax.axis("off")
            continue
        phi = np.radians(az)
        e_rho, e_z = np.array([np.cos(phi), np.sin(phi), 0]), np.array([0, 0, 1.0])
        n = np.array([-np.sin(phi), np.cos(phi), 0])
        for v, f, st in ((stl_v, stl_f, STL_STYLE), (csg_v, csg_f, CSG_STYLE)):
            lines = section_lines(v, f, [0, 0, 0], n, (e_rho, e_z))
            draw(ax, [ln[ln[:, 0] > -1] for ln in lines if (ln[:, 0] > -1).any()], st)
        ax.set_aspect("equal")
        ax.set_xlim(-5, 165)
        ax.set_ylim(-125, 165)
        ax.set_title(f"meridional section, azimuth {az:g}°", fontsize=10)
        ax.set_xlabel("rho (mm)")
        ax.set_ylabel("z (mm)")
        ax.legend(fontsize=7, loc="upper left")
    # zooms: window corners and bottom edge
    zooms = [
        ("bottom edge, az 0°", 0.0, (95, 125), (-120, -95)),
        ("window top edge, az 90°", 90.0, (135, 160), (-25, 0)),
        ("window step, az 53°", 53.0, (125, 150), (-80, -55)),
        ("window side, z = −40 cut", None, None, None),
    ]
    for ax, (title, az, xl, yl) in zip(axes[2], zooms):
        if az is None:
            # horizontal cut through the window side walls
            n = np.array([0, 0, 1.0])
            for v, f, st in ((stl_v, stl_f, STL_STYLE), (csg_v, csg_f, CSG_STYLE)):
                draw(ax, section_lines(v, f, [0, 0, -40], n, (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))), st)
            ax.set_xlim(-110, 110)
            ax.set_ylim(60, 160)
        else:
            phi = np.radians(az)
            e_rho = np.array([np.cos(phi), np.sin(phi), 0])
            n = np.array([-np.sin(phi), np.cos(phi), 0])
            for v, f, st in ((stl_v, stl_f, STL_STYLE), (csg_v, csg_f, CSG_STYLE)):
                draw(ax, section_lines(v, f, [0, 0, 0], n, (e_rho, np.array([0, 0, 1.0]))), st)
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.set_title("zoom: " + title, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("Shield: STL (black) vs CSG (red dashed) cross-sections", fontsize=13)
    fig.savefig(out / "whole_sections.png", dpi=110)
    plt.close(fig)


def aperture_stats(d):
    ph, rot = d["pinholes"], d["rotations"]
    n = len(ph)
    stats = {"stl_minus_csg": np.zeros(n), "csg_minus_stl": np.zeros(n), "max_abs_dev": np.zeros(n),
             "p99_abs_dev": np.zeros(n)}
    other = {"stl_minus_csg": 0.0, "csg_minus_stl": 0.0}
    for name in ("stl_minus_csg", "csg_minus_stl"):
        for vol, c in zip(d[name + "_vol"], d[name + "_cen"]):
            loc = np.einsum("nij,ni->nj", rot, c[None] - ph)  # (c - ph) @ R for every module
            inside = (np.abs(loc[:, 0]) < CROP) & (np.abs(loc[:, 1]) < CROP) & (np.abs(loc[:, 2]) < 12)
            if inside.any():
                stats[name][np.argmin(np.linalg.norm(loc, axis=1) + 1e9 * ~inside)] += vol
            else:
                other[name] += vol
    for k in range(n):
        L = to_local(d["stl_v"], ph[k], rot[k])
        sel = (np.abs(L[:, 0]) < CROP) & (np.abs(L[:, 1]) < CROP) & (np.abs(L[:, 2]) < 12)
        dev = np.abs(d["stl_dev"][sel])
        stats["max_abs_dev"][k] = dev.max()
        stats["p99_abs_dev"][k] = np.quantile(dev, 0.99)
    return stats, other


def plot_aperture_summary(d, stats, other, out: Path):
    n = len(stats["max_abs_dev"])
    x = np.arange(1, n + 1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 7.5), constrained_layout=True, sharex=True)
    a1.bar(x - 0.2, stats["stl_minus_csg"], 0.4, color="#4a4a4a", label="in STL, not in CSG")
    a1.bar(x + 0.2, stats["csg_minus_stl"], 0.4, color="#d1495b", label="in CSG, not in STL")
    a1.set_ylabel("volume (mm³)")
    a1.set_title("Residual volume near each aperture, excluding the sphere-facet films "
                 f"(films: STL−CSG {d['stl_minus_csg_film'] / 1e3:.1f} cm³, CSG−STL "
                 f"{d['csg_minus_stl_film'] / 1e3:.1f} cm³; away from apertures: "
                 f"{other['stl_minus_csg']:.0f} / {other['csg_minus_stl']:.0f} mm³)", fontsize=10)
    a1.legend()
    a2.bar(x - 0.2, stats["max_abs_dev"], 0.4, color="#2c6fbb", label="max")
    a2.bar(x + 0.2, stats["p99_abs_dev"], 0.4, color="#9ecae1", label="99th percentile")
    a2.set_ylabel("|STL vertex − CSG surface| (mm)")
    a2.set_xlabel("module / aperture")
    a2.set_title("Distance of STL vertices near each aperture to the CSG surface", fontsize=10)
    a2.legend()
    for a in (a1, a2):
        a.grid(alpha=0.3, axis="y")
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "aperture_summary.png", dpi=110)
    plt.close(fig)


LEAD_FILL = "#cfcfcf"
SLICE_R = [154.5, 153.5, 152.5, 151.5, 150.5, None, 148.5, 147.5, 146.5, 145.5, 144.5, 143.5]
SLICE_HALF = 40.0  # mm, half-width of the axial slice panels
ZOOM_HALF = 32.0  # mm, half-width of the meridional zoom


def plane_loops(v, f, origin, normal, e1, e2):
    """Outlines where a plane cuts the mesh, in plane coordinates about `origin`.

    Only triangles that cross the plane are passed to the section, which gives the
    same outlines as the full mesh (closed loops) at a fraction of the cost.
    """
    # nudge the plane off mesh vertices that lie exactly on it (symmetry planes),
    # which otherwise break the section into fragments; 1e-4 mm is invisible here
    origin = np.asarray(origin, float) + 1e-4 * np.asarray(normal, float)
    sv = (v - origin) @ normal
    sf = sv[f]
    cross = (sf.min(1) <= 0) & (sf.max(1) >= 0)
    if not cross.any():
        return []
    ff = f[cross]
    used, inv = np.unique(ff, return_inverse=True)
    sec = trimesh.Trimesh(v[used], inv.reshape(-1, 3), process=False).section(
        plane_origin=origin, plane_normal=normal)
    if sec is None:
        return []
    return [np.c_[(np.asarray(p) - origin) @ e1, (np.asarray(p) - origin) @ e2] for p in sec.discrete]


def _closed(loops):
    return [ln for ln in loops if len(ln) > 3 and np.allclose(ln[0], ln[-1], atol=1e-6)]


def fill_lead(ax, loops, **kw):
    """Fill the region inside an odd number of closed loops (lead), holes left white."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    closed = _closed(loops)
    if not closed:
        return
    paths = [MPath(ln) for ln in closed]
    verts, codes = [], []
    for i, ln in enumerate(closed):
        depth = sum(paths[j].contains_point(ln[0]) for j in range(len(closed)) if j != i)
        area = 0.5 * np.sum(ln[:-1, 0] * ln[1:, 1] - ln[1:, 0] * ln[:-1, 1])
        if (depth % 2 == 0) != (area > 0):  # outer loops CCW, holes CW (nonzero = even-odd)
            ln = ln[::-1]
        verts.append(ln)
        codes.append([MPath.MOVETO] + [MPath.LINETO] * (len(ln) - 2) + [MPath.CLOSEPOLY])
    ax.add_patch(PathPatch(MPath(np.concatenate(verts), np.concatenate(codes)), **kw))


def in_lead(loops, pt):
    from matplotlib.path import Path as MPath

    return sum(MPath(ln).contains_point(pt) for ln in _closed(loops)) % 2 == 1


def draw_both(ax, stl_loops, csg_loops):
    fill_lead(ax, stl_loops, facecolor=LEAD_FILL, edgecolor="none", zorder=0)
    for ln in stl_loops:
        ax.plot(ln[:, 0], ln[:, 1], color="black", lw=1.1, zorder=2)
    for ln in csg_loops:
        ax.plot(ln[:, 0], ln[:, 1], color="#d1495b", lw=1.1, ls="--", zorder=3)


COL_FILL, COL_EDGE = "#f6d5a8", "#c0782b"
_COLLIMATORS = None


def collimator_world_meshes():
    """Every module's collimator as Gate builds it (construct_collimator_geometry):
    Frustum_A minus the nozzle bore (Frustum_B) and body bore (Frustum_C), united with
    the hollow box, placed with the module rotation and collimator centre."""
    global _COLLIMATORS
    if _COLLIMATORS is not None:
        return _COLLIMATORS
    import manifold3d as mf

    def trd(dx1, dy1, dx2, dy2, dz, z0=0.0):
        pts = [[sx * dx1, sy * dy1, z0 - dz] for sx in (-1, 1) for sy in (-1, 1)]
        pts += [[sx * dx2, sy * dy2, z0 + dz] for sx in (-1, 1) for sy in (-1, 1)]
        return mf.Manifold.hull_points(pts)

    df = S.brain.get_geometry_definitions()
    n = df.shape[0]
    out = []
    for k in range(n):
        cfg = S.brain.get_geometry_base_definition(S.brain.map_crystal_id(k, n, "sequential"))
        c = cfg["collimator definition"]
        w, lt, hn, hb = c["w_pinhole"], c["l_top"], c["h_nozzle"], c["h_body"]
        lbo, lbi, hbox, ww = c["l_bottom_outer"], c["l_bottom_inner"], c["h_box"], c["w_wall"]
        dz = 0.1  # same over-length as the Gate code, for clean subtraction
        db = (lt - w) / hn * dz * 0.5
        dc = (w - lbi) / hb * dz * 0.5
        frustum_a = trd(lbo / 2, lbo / 2, lt / 2, lt / 2, (hn + hb) / 2)
        frustum_b = trd(w / 2 - db, w / 2 - db, lt / 2 + db, lt / 2 + db, hn / 2 + dz, z0=hb / 2)
        frustum_c = trd(lbi / 2 - dc, lbi / 2 - dc, w / 2 + dc, w / 2 + dc, hb / 2 + dz, z0=-hn / 2)
        box = (mf.Manifold.cube((lbo, lbo, hbox), True)
               - mf.Manifold.cube((lbo - 2 * ww, lbo - 2 * ww, hbox + 2.0), True))
        col = (frustum_a - frustum_b - frustum_c) + box.translate((0, 0, -0.5 * (hn + hb + hbox)))
        rot = S.brain.get_head_rotation_matrix(df, k)
        centre = S.brain.get_collimator_center(cfg, df, k)
        out.append(S.manifold_mesh(col.transform(np.c_[rot, centre])))
    _COLLIMATORS = out
    return out


def collimator_bore_width(c, z):
    """Square bore width of the collimator at depth z from the pinhole (+ toward the
    centre), from the Gate parameters; None beyond the nozzle tip."""
    if z > c["h_nozzle"]:
        return None
    if z >= 0:
        return c["w_pinhole"] + (c["l_top"] - c["w_pinhole"]) * z / c["h_nozzle"]
    if z >= -c["h_body"]:
        return c["w_pinhole"] + (c["l_bottom_inner"] - c["w_pinhole"]) * (-z) / c["h_body"]
    return c["l_bottom_outer"] - 2 * c["w_wall"]  # inside the hollow box


def aperture_basis(u):
    """Unit vectors along increasing azimuth and elevation, perpendicular to u."""
    e_az = np.cross([0.0, 0.0, 1.0], u)
    if np.linalg.norm(e_az) < 1e-9:  # top module: azimuth undefined, use +x / +y
        return np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    e_az /= np.linalg.norm(e_az)
    return e_az, np.cross(u, e_az)


def aperture_page(d, k):
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ph = d["pinholes"]
    u_all = ph / np.linalg.norm(ph, axis=1)[:, None]
    p_k = ph[k]
    r_k = np.linalg.norm(p_k)
    u = u_all[k]
    el = np.degrees(np.arcsin(u[2]))
    az = np.degrees(np.arctan2(u[1], u[0])) % 360 if abs(el) < 89.9 else 0.0
    meshes = ((d["stl_v"], d["stl_f"]), (d["csg_v"], d["csg_f"]))

    fig = plt.figure(figsize=(19, 35))
    gs = GridSpec(5, 4, figure=fig, height_ratios=[2.0, 2.0, 1, 1, 1], hspace=0.2, wspace=0.2,
                  top=0.935, bottom=0.025, left=0.05, right=0.98)
    all_el = np.degrees(np.arcsin(u_all[:, 2]))
    all_az = np.degrees(np.arctan2(u_all[:, 1], u_all[:, 0])) % 360
    zoom_box = dict(fill=False, ec="#2c6fbb", lw=0.8)

    def style(ax, cx, cy, half, title, xlabel, ylabel):
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)

    # --- horizontal cut at the pinhole height, fixed world x-y (shows the azimuth)
    z_k = p_k[2]
    stl_h, csg_h = (plane_loops(v, f, np.array([0, 0, z_k]), np.array([0, 0, 1.0]),
                                np.array([1.0, 0, 0]), np.array([0, 1.0, 0])) for v, f in meshes)
    same_ring = [j for j in range(len(ph)) if abs(ph[j][2] - z_k) < 1e-3]
    for row, (cx, cy, half) in enumerate(((0.0, 0.0, 168.0), (p_k[0], p_k[1], ZOOM_HALF))):
        ax = fig.add_subplot(gs[row, :2])
        draw_both(ax, stl_h, csg_h)
        ax.plot([0, p_k[0] * 1.08], [0, p_k[1] * 1.08], color="#2c6fbb", lw=0.9, ls=":", zorder=1)
        for j in same_ring:
            if abs(ph[j][0] - cx) < half and abs(ph[j][1] - cy) < half:
                ax.annotate(f"A{j + 1}", (ph[j][0], ph[j][1]), ha="center", va="center",
                            fontsize=10 if j == k else 8, fontweight="bold" if j == k else "normal",
                            color="#d1495b" if j == k else "#2c6fbb", zorder=4)
        style(ax, cx, cy, half,
              (f"horizontal cut at the pinhole height z = {z_k:.1f} mm (fixed scanner x-y)"
               if row == 0 else f"horizontal cut, zoom on aperture {k + 1}"),
              "x (mm)", "y (mm)")
        if row == 0:
            ax.add_patch(plt.Rectangle((p_k[0] - ZOOM_HALF, p_k[1] - ZOOM_HALF), 2 * ZOOM_HALF,
                                       2 * ZOOM_HALF, **zoom_box))
            ax.text(0.01, 0.01, "azimuth 0° = +x, 90° = +y (face window side)", fontsize=8,
                    transform=ax.transAxes, color="#6b6f73")

    # --- vertical cut through the z axis at the aperture azimuth (shows the elevation)
    phi = np.radians(az)
    e_rho, e_z = np.array([np.cos(phi), np.sin(phi), 0.0]), np.array([0, 0, 1.0])
    normal = np.array([-np.sin(phi), np.cos(phi), 0.0])
    stl_m, csg_m = (plane_loops(v, f, np.zeros(3), normal, e_rho, e_z) for v, f in meshes)
    rho_k = p_k @ e_rho
    same_meridian = [j for j in range(len(ph)) if abs(u_all[j] @ normal) < 1e-3]
    for row, (cx, cz, half) in enumerate(((0.0, 20.0, 168.0), (rho_k, z_k, ZOOM_HALF))):
        ax = fig.add_subplot(gs[row, 2:])
        draw_both(ax, stl_m, csg_m)
        ax.plot([0, rho_k * 1.08], [0, z_k * 1.08], color="#2c6fbb", lw=0.9, ls=":", zorder=1)
        for j in same_meridian:
            rj, zj = ph[j] @ e_rho, ph[j][2]
            if abs(rj - cx) < half and abs(zj - cz) < half:
                ax.annotate(f"A{j + 1}", (rj, zj), ha="center", va="center",
                            fontsize=10 if j == k else 8, fontweight="bold" if j == k else "normal",
                            color="#d1495b" if j == k else "#2c6fbb", zorder=4)
        style(ax, cx, cz, half,
              (f"vertical cut containing the z axis at azimuth {az:.2f}°\n"
               "(this plane turns with the aperture: it shows the elevation, not the azimuth)"
               if row == 0 else f"vertical cut, zoom on aperture {k + 1}"),
              f"horizontal distance from the z axis toward azimuth {az:.1f}° (mm)", "z (mm)")
        if row == 0:
            ax.add_patch(plt.Rectangle((rho_k - ZOOM_HALF, z_k - ZOOM_HALF), 2 * ZOOM_HALF,
                                       2 * ZOOM_HALF, **zoom_box))
            # locator: where this aperture sits among all 73
            loc = ax.inset_axes([0.22, 0.30, 0.56, 0.36])
            loc.scatter(all_az, all_el, s=14, c="#9aa0a6")
            loc.scatter([all_az[k]], [all_el[k]], s=60, c="#d1495b", zorder=3)
            for j in range(len(ph)):
                loc.annotate(str(j + 1), (all_az[j], all_el[j]), fontsize=5.5, ha="center",
                             va="bottom", xytext=(0, 3), textcoords="offset points",
                             color="#d1495b" if j == k else "#6b6f73")
            loc.set_xlim(0, 360)
            loc.set_ylim(-48, 98)
            loc.set_xticks([0, 90, 180, 270, 360])
            loc.set_xlabel("azimuth (deg)", fontsize=7)
            loc.set_ylabel("elevation (deg)", fontsize=7)
            loc.tick_params(labelsize=6)
            loc.set_title("aperture positions (red: this one)", fontsize=7)

    # --- slices perpendicular to the aperture axis
    e1, e2 = aperture_basis(u)
    col_cfg = S.brain.get_geometry_base_definition(
        S.brain.map_crystal_id(k, len(ph), "sequential"))["collimator definition"]
    for i, r_axis in enumerate(SLICE_R):
        ax = fig.add_subplot(gs[2 + i // 4, i % 4])
        r_axis = r_k if r_axis is None else r_axis
        origin = u * r_axis
        stl_s, csg_s = (plane_loops(v, f, origin, u, e1, e2) for v, f in meshes)
        draw_both(ax, stl_s, csg_s)
        # this aperture's own collimator only, with its bore
        col_s = plane_loops(*collimator_world_meshes()[k], origin, u, e1, e2)
        fill_lead(ax, col_s, facecolor=COL_FILL, edgecolor="none", zorder=1)
        for ln in col_s:
            ax.plot(ln[:, 0], ln[:, 1], color=COL_EDGE, lw=0.9, zorder=2)
        bore = collimator_bore_width(col_cfg, r_k - r_axis)
        ax.text(0.02, 0.02, "no collimator (past the nozzle tip)" if bore is None
                else f"collimator bore {bore:.2f} mm", transform=ax.transAxes, fontsize=8,
                color=COL_EDGE, va="bottom")

        def crossing(direction):
            """Where the ray along `direction` meets this plane, if that point lies
            inside the shell thickness (only then is the opening really in the slice)."""
            cos_w = direction @ u
            if cos_w <= 0.3:
                return None
            x = direction * (r_axis / cos_w)
            if not 145.0 <= np.linalg.norm(x) <= 155.0:
                return None
            pt = ((x - origin) @ e1, (x - origin) @ e2)
            if max(abs(pt[0]), abs(pt[1])) > SLICE_HALF - 4 or in_lead(stl_s, pt):
                return None
            return pt

        for j in range(len(ph)):
            pt = None if j == k else crossing(u_all[j])  # own label would cover the bore
            if pt is not None:
                ax.annotate(f"A{j + 1}", pt, ha="center", va="center", fontsize=9, color="#2c6fbb",
                            zorder=4)
        win = S.SHIELD["window"]
        openings = (
            ("face window", np.clip(az, win[0][0][0] + 1, win[0][0][1] - 1), win[0][1][1] - 4.0),
            ("bottom opening", az, S.SHIELD["bottom_elevation_deg"] - 4.0),
        )
        for name, a_deg, e_deg in openings:
            a_r, e_r = np.radians(a_deg), np.radians(e_deg)
            pt = crossing(np.array([np.cos(e_r) * np.cos(a_r), np.cos(e_r) * np.sin(a_r), np.sin(e_r)]))
            if pt is not None:
                ax.annotate(name, pt, ha="center", va="center", fontsize=8, color="#7a5195",
                            style="italic", zorder=4)
        ax.set_xlim(-SLICE_HALF, SLICE_HALF)
        ax.set_ylim(-SLICE_HALF, SLICE_HALF)
        ax.set_aspect("equal")
        where = "pinhole plane" if abs(r_axis - r_k) < 1e-6 else f"{r_k - r_axis:+.1f} mm from pinhole"
        ax.set_title(f"axial slice at r = {r_axis:.2f} mm ({where})", fontsize=10)
        ax.set_xlabel("along azimuth (mm)" if abs(el) < 89.9 else "x (mm)", fontsize=9)
        ax.set_ylabel("along elevation (mm)" if abs(el) < 89.9 else "y (mm)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    handles = [
        Patch(facecolor=LEAD_FILL, label="lead (STL)"),
        Line2D([], [], color="black", lw=1.1, label="STL outline"),
        Line2D([], [], color="#d1495b", lw=1.1, ls="--", label="CSG outline"),
        Patch(facecolor=COL_FILL, edgecolor=COL_EDGE,
              label="this aperture's collimator, tungsten (axial slices; bore white)"),
        Line2D([], [], color="#2c6fbb", lw=0.9, ls=":", label=f"axis of aperture {k + 1} (pinhole to centre)"),
        Line2D([], [], ls="none", marker="$A$", color="#2c6fbb",
               label="An: shield aperture n (at its axis); italic: face window / bottom opening"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.972))
    ring = np.flatnonzero(np.isclose(np.round(all_el, 2), round(el, 2)))
    fig.suptitle(f"Shield aperture {k + 1}: ring at elevation {el:.1f}°, number {list(ring).index(k) + 1} "
                 f"of {len(ring)} in the ring, azimuth {az:.1f}°; collimator pinhole "
                 f"{col_cfg['w_pinhole']:.3f} mm.  STL vs CSG.\n"
                 "Axial slices are perpendicular to the aperture axis, viewed from outside "
                 "looking toward the centre; r is the distance from the centre along the axis.",
                 fontsize=12, y=0.988)
    return fig


def plot_apertures(d, stats, out: Path, overview=(0, 40, 65, 72)):
    n = len(d["pinholes"])
    with PdfPages(out / "apertures.pdf") as pdf:
        for k in range(n):
            fig = aperture_page(d, k)
            pdf.savefig(fig)
            if k in overview:
                fig.savefig(out / f"aperture_{k + 1:02d}.png", dpi=90)
            plt.close(fig)
    print("wrote apertures.pdf")


# ----------------------------------------------------------------------------- 3D
def _mesh_trace(v, f, dev, name, scene, visible=True, show_scale=False):
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
        intensity=np.clip(dev, -DEV_CLIP, DEV_CLIP), cmin=-DEV_CLIP, cmax=DEV_CLIP,
        colorscale="RdBu_r", showscale=show_scale, name=name, scene=scene, visible=visible,
        flatshading=True, text=[f"{x:+.3f} mm" for x in dev] if len(dev) < 40000 else None,
        hovertemplate=("%{text}<extra>" + name + "</extra>") if len(dev) < 40000 else
        "%{intensity:+.3f} mm<extra>" + name + "</extra>",
        colorbar={"title": "signed distance<br>to the other<br>model (mm)", "x": 1.0},
    )


def plot_whole_3d(d, out: Path):
    fig = go.Figure()
    fig.add_trace(_mesh_trace(d["stl_v"], d["stl_f"], d["stl_dev"], "STL", "scene", True, True))
    fig.add_trace(_mesh_trace(d["disp_v"], d["disp_f"], d["disp_dev"], "CSG", "scene", False, True))
    fig.update_layout(
        title=("Shield, STL vs CSG: surface coloured by signed distance to the other model "
               "(red = this surface lies outside the other solid, blue = inside; clipped at "
               f"±{DEV_CLIP} mm)"),
        height=850, template="plotly_white",
        scene={"aspectmode": "data", "camera": {"eye": {"x": 1.2, "y": 1.6, "z": 0.4}}},
        updatemenus=[{"type": "buttons", "direction": "right", "x": 0, "y": 1.05, "xanchor": "left",
                      "buttons": [
                          {"label": "STL", "method": "restyle", "args": [{"visible": [True, False]}]},
                          {"label": "CSG", "method": "restyle", "args": [{"visible": [False, True]}]},
                          {"label": "Both", "method": "restyle", "args": [{"visible": [True, True]}]},
                      ]}],
    )
    fig.write_html(out / "whole_3d.html", include_plotlyjs=True)


def plot_regions_3d(d, out: Path):
    ph, rot = d["pinholes"], d["rotations"]
    regions = [(f"Aperture {k + 1}", ph[k], rot[k]) for k in range(len(ph))]

    def frame_at(az, el):
        a, e = np.radians(az), np.radians(el)
        u = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        z = -u  # toward the centre, like the module frames
        x = np.cross([0, 0, 1.0], z)
        x /= np.linalg.norm(x)
        return 150 * u, np.c_[x, np.cross(z, x), z]

    for title, az, el in (("Window top corner (az 56°, el −6°)", 56.25, -5.75),
                          ("Window step corner (az 51°, el −27°)", 50.524, -26.5),
                          ("Bottom edge (az 0°, el −47°)", 0.0, -47.0)):
        regions.append((title, *frame_at(az, el)))
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("STL (coloured by distance to CSG)",
                                        "CSG (coloured by distance to STL)"),
                        horizontal_spacing=0.02)
    for r, (title, origin, R) in enumerate(regions):
        for col, (v, f, dev, name) in enumerate(((d["stl_v"], d["stl_f"], d["stl_dev"], "STL"),
                                                 (d["disp_v"], d["disp_f"], d["disp_dev"], "CSG"))):
            L = to_local(v, origin, R)
            keep = (np.abs(L[:, 0]) < CROP) & (np.abs(L[:, 1]) < CROP) & (np.abs(L[:, 2]) < 9)
            lv, lf, used = crop_mesh(L, f, keep, whole_faces_only=True)
            fig.add_trace(_mesh_trace(lv, lf, dev[used], name, "scene" if col == 0 else "scene2",
                                      visible=(r == 0), show_scale=(col == 1)), row=1, col=col + 1)
    n_tr = len(fig.data)
    buttons = []
    for r, (title, *_rest) in enumerate(regions):
        vis = [False] * n_tr
        vis[2 * r] = vis[2 * r + 1] = True
        buttons.append({"label": title, "method": "update", "args": [{"visible": vis},
                                                                     {"title.text": title}]})
    scene = {"aspectmode": "data", "xaxis_title": "local x", "yaxis_title": "local y",
             "zaxis_title": "local z (toward centre)", "camera": {"eye": {"x": 0.9, "y": -1.2, "z": 1.1}}}
    fig.update_layout(
        title={"text": regions[0][0]}, height=780, template="plotly_white", scene=scene, scene2=scene,
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0, "y": 1.12, "xanchor": "left",
                      "showactive": True}],
        annotations=list(fig.layout.annotations) + [go.layout.Annotation(
            text=(f"red = surface outside the other solid, blue = inside; clipped at ±{DEV_CLIP} mm. "
                  "Apertures in the module frame (origin at the pinhole, z toward the centre). "
                  "Blotches on the CSG spheres are the STL's flat facets (up to 0.2 mm off the true sphere)."),
            x=0.5, y=-0.04, xref="paper", yref="paper", showarrow=False, font={"size": 11})],
    )
    fig.write_html(out / "regions_3d.html", include_plotlyjs=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounded", action="store_true",
                    help="use the CSG with rounded window/bottom edges and corners")
    ap.add_argument("--output", type=Path, default=None,
                    help="default: brain_shield_csg_comparison[_rounded] next to this script")
    args = ap.parse_args(argv)
    if args.output is None:
        args.output = Path(__file__).resolve().parent / (
            "brain_shield_csg_comparison" + ("_rounded" if args.rounded else ""))
    args.output.mkdir(parents=True, exist_ok=True)
    d = prepare(args.output, args.rounded)
    stats, other = aperture_stats(d)
    summary = {
        "stl_volume_cm3": float(d["vol_stl"]) / 1e3, "csg_volume_cm3": float(d["vol_csg"]) / 1e3,
        "film_stl_minus_csg_cm3": float(d["stl_minus_csg_film"]) / 1e3,
        "film_csg_minus_stl_cm3": float(d["csg_minus_stl_film"]) / 1e3,
        "other_stl_minus_csg_mm3": float(d["stl_minus_csg_vol"].sum()),
        "other_csg_minus_stl_mm3": float(d["csg_minus_stl_vol"].sum()),
        "stl_vertex_to_csg_abs_mm": {q: float(np.quantile(np.abs(d["stl_dev"]), p))
                                     for q, p in (("p50", .5), ("p99", .99), ("p999", .999), ("max", 1))},
        "csg_vertex_to_stl_abs_mm": {q: float(np.quantile(np.abs(d["disp_dev"]), p))
                                     for q, p in (("p50", .5), ("p99", .99), ("p999", .999), ("max", 1))},
        "rounded": args.rounded,
        "shield_parameters": S.SHIELD,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "shield_parameters"}, indent=2))
    plot_thickness_maps(d, args.output)
    plot_sections(d, args.output)
    plot_aperture_summary(d, stats, other, args.output)
    plot_apertures(d, stats, args.output)
    plot_whole_3d(d, args.output)
    plot_regions_3d(d, args.output)
    print("wrote plots to", args.output)


if __name__ == "__main__":
    main()
