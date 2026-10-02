"""Interactive 3D view of the brain SPECT hardware and an FOV sphere, with clearances.

Draws the lead shield STL, the outer envelope of every collimator (Frustum_A + Box_A,
bores omitted, as in get_collimator_outer_primitives) and the crystals, all placed
exactly as gate_sim_brain_spect_boolean.py places them, plus the FOV sphere. The
clearance of each part is the exact distance from the centre to its surface (closest
point on every triangle) minus the sphere radius.

Usage:  ma opengate && python dev/python/plot_brain_spect_fov_3d.py [--fov-size-mm 288]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh
from plotly.subplots import make_subplots

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "payload" / "python"))
import gate_sim_brain_spect_boolean as brain  # noqa: E402

# Unit cube corners in G4 order: bottom face (z = -1) then top face (z = +1).
_CORNERS = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float
)
_HEX_FACES = np.array(
    [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
     [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
)


def primitive_vertices(p: dict) -> np.ndarray:
    """8 local corners of a box / G4Trd primitive, including its offset."""
    if p["type"] == "box":
        v = _CORNERS * (np.asarray(p["size_mm"]) / 2)
    else:
        half = np.where(
            _CORNERS[:, 2:3] < 0, [p["dx1"], p["dy1"], p["dz"]], [p["dx2"], p["dy2"], p["dz"]]
        )
        v = _CORNERS * half
    return v + np.asarray(p["offset_mm"])


def closest_point_distance(tri: np.ndarray) -> np.ndarray:
    """Distance from the origin to each triangle (N, 3, 3), Ericson's region test."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    p = np.zeros_like(a)
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    with np.errstate(divide="ignore", invalid="ignore"):
        v = vb / denom
        w = vc / denom
        out = a + ab * v[:, None] + ac * w[:, None]  # interior
        # edges
        t_ab = d1 / (d1 - d3)
        t_ac = d2 / (d2 - d6)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    out[m] = (a + ab * t_ab[:, None])[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    out[m] = (a + ac * t_ac[:, None])[m]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    out[m] = (b + (c - b) * t_bc[:, None])[m]
    # vertices
    out[(d1 <= 0) & (d2 <= 0)] = a[(d1 <= 0) & (d2 <= 0)]
    out[(d3 >= 0) & (d4 <= d3)] = b[(d3 >= 0) & (d4 <= d3)]
    out[(d6 >= 0) & (d5 <= d6)] = c[(d6 >= 0) & (d5 <= d6)]
    return np.linalg.norm(out, axis=1)


def build_parts():
    df = brain.get_geometry_definitions()
    n = df.shape[0]
    modules = []
    for i in range(n):
        cfg = brain.get_geometry_base_definition(brain.map_crystal_id(i, n, "sequential"))
        rot = brain.get_head_rotation_matrix(df, i)
        center = brain.get_collimator_center(cfg, df, i)
        coll = [center + primitive_vertices(p) @ rot.T
                for p in brain.get_collimator_outer_primitives(cfg)]
        crystal_center = np.array([df.item(i, f"Crystal_{a}") for a in "xyz"])
        size = np.asarray(cfg["crystal definition"]["size_mm"])
        crystal = crystal_center + (_CORNERS * size / 2) @ rot.T
        pinhole = np.array([df.item(i, f"Pinhole_{a}") for a in "xyz"])
        modules.append({"collimator": coll, "crystal": crystal, "pinhole": pinhole,
                        "variant": brain.map_crystal_id(i, n, "sequential")})
    mesh = trimesh.load(cfg["shielding file path"])
    shield_v = mesh.vertices @ brain.get_shielding_rotation_matrix().T
    return modules, shield_v, np.asarray(mesh.faces)


def hexes_to_mesh(hexes: list[np.ndarray]):
    verts = np.concatenate(hexes)
    faces = np.concatenate([_HEX_FACES + 8 * k for k in range(len(hexes))])
    return verts, faces


def sphere_mesh(radius: float, n: int = 72):
    th, ph = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, 2 * n))
    x = radius * np.sin(th) * np.cos(ph)
    y = radius * np.sin(th) * np.sin(ph)
    z = radius * np.cos(th)
    return x, y, z


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fov-size-mm", type=float, default=288.0)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent
                    / "brain_fov_analysis" / "brain_fov_3d_overlap_check.html")
    args = ap.parse_args(argv)
    r_fov = args.fov_size_mm / 2

    modules, shield_v, shield_f = build_parts()
    n = len(modules)

    # exact clearances
    coll_clear, crys_clear = [], []
    for m in modules:
        v, f = hexes_to_mesh(m["collimator"])
        coll_clear.append(closest_point_distance(v[f]).min() - r_fov)
        v, f = hexes_to_mesh([m["crystal"]])
        crys_clear.append(closest_point_distance(v[f]).min() - r_fov)
    coll_clear, crys_clear = np.array(coll_clear), np.array(crys_clear)
    shield_d = closest_point_distance(shield_v[shield_f])
    shield_clear = shield_d.min() - r_fov
    worst = int(np.argmin(coll_clear))
    print(f"FOV sphere D = {args.fov_size_mm} mm (r = {r_fov} mm)")
    print(f"collimator clearance: min {coll_clear.min():.3f} mm (module {worst + 1}), "
          f"max {coll_clear.max():.3f} mm")
    print(f"crystal clearance:    min {crys_clear.min():.3f} mm")
    print(f"shield clearance:     min {shield_clear:.3f} mm")
    ok = min(coll_clear.min(), crys_clear.min(), shield_clear) > 0

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.68, 0.32],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("Hardware and FOV sphere (drag to rotate, scroll to zoom)",
                        "Gap between sphere and each collimator"),
        horizontal_spacing=0.06,
    )

    # shield: full and cut-away (faces with centroid y > 0 removed)
    cent_y = shield_v[shield_f].mean(axis=1)[:, 1]
    # opaque on purpose: WebGL mis-sorts a large translucent mesh among opaque ones
    for name, faces, visible in (("Lead shield", shield_f, False),
                                 ("Lead shield (cut-away)", shield_f[cent_y <= 0], True)):
        fig.add_trace(go.Mesh3d(
            x=shield_v[:, 0], y=shield_v[:, 1], z=shield_v[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="#9aa0a6", opacity=1.0, name="Lead shield", showlegend=True,
            legendgroup="shield", visible=visible, hoverinfo="skip", flatshading=True,
        ), row=1, col=1)

    # collimators, one trace per half so the cut-away view can hide the +y half
    front = np.array([m["pinhole"][1] > 0 for m in modules])
    for half, sel in (("+y half", front), ("-y half", ~front)):
        idx = np.flatnonzero(sel)
        hexes = [h for k in idx for h in modules[k]["collimator"]]
        v, f = hexes_to_mesh(hexes)
        text = np.repeat([f"Collimator_{k + 1} (variant {modules[k]['variant']})<br>"
                          f"gap to sphere: {coll_clear[k]:.3f} mm" for k in idx], 16)
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color="#c0782b", opacity=1.0, flatshading=True, name="Collimators",
            text=text, hovertemplate="%{text}<extra></extra>", meta=half,
            legendgroup="coll", showlegend=(half == "-y half"),
            visible=(half == "-y half"),
        ), row=1, col=1)
        v, f = hexes_to_mesh([modules[k]["crystal"] for k in idx])
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color="#3b7dd8", opacity=1.0, flatshading=True, name="Crystals",
            hoverinfo="skip", legendgroup="crys", showlegend=(half == "-y half"),
            meta=half, visible=(half == "-y half"),
        ), row=1, col=1)

    for d, color, opacity, visible, label in (
        (args.fov_size_mm, "#2ca02c", 0.55, True, f"FOV sphere D = {args.fov_size_mm:g} mm"),
        (210.0, "#444444", 0.12, "legendonly", "Previous FOV D = 210 mm"),
    ):
        x, y, z = sphere_mesh(d / 2)
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, opacity=opacity, showscale=False, name=label,
            colorscale=[[0, color], [1, color]], showlegend=True, visible=visible,
            hoverinfo="skip",
        ), row=1, col=1)

    # nearest-approach markers on the worst collimator nozzle tip
    tip = modules[worst]["pinhole"] * (1 - 5.04 / np.linalg.norm(modules[worst]["pinhole"]))
    fig.add_trace(go.Scatter3d(
        x=[tip[0], tip[0] * r_fov / np.linalg.norm(tip)],
        y=[tip[1], tip[1] * r_fov / np.linalg.norm(tip)],
        z=[tip[2], tip[2] * r_fov / np.linalg.norm(tip)],
        mode="markers", marker={"size": 4, "color": ["#c0782b", "#2ca02c"]},
        name=f"Closest approach (module {worst + 1})",
        hovertemplate="%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>",
    ), row=1, col=1)

    # clearance per module (2D)
    fig.add_trace(go.Bar(
        x=np.arange(1, n + 1), y=coll_clear, marker_color="#c0782b",
        name="Collimator gap", showlegend=False,
        hovertemplate="module %{x}<br>gap %{y:.3f} mm<extra></extra>",
    ), row=1, col=2)
    # add_hline does not work in a figure that also holds a 3D scene
    fig.add_trace(go.Scatter(
        x=[0.5, n + 0.5], y=[shield_clear] * 2, mode="lines+text",
        line={"dash": "dash", "color": "#6b6f73"}, showlegend=False,
        text=[f"shield gap {shield_clear:.2f} mm", ""], textposition="top right",
        hoverinfo="skip",
    ), row=1, col=2)
    fig.update_yaxes(title_text="gap to FOV sphere (mm)", range=[0, 1.0], row=1, col=2)
    fig.update_xaxes(title_text="module (Collimator_N)", row=1, col=2)

    # default view is the cut-away (set above); "Full" shows every part
    cut_vis = [t.visible if t.visible is not None else True for t in fig.data]
    full_vis = list(cut_vis)
    for k, t in enumerate(fig.data):
        if t.name == "Lead shield":
            full_vis[k] = not cut_vis[k]
        elif t.meta == "+y half":
            full_vis[k] = True

    verdict = ("NO OVERLAP" if ok else "OVERLAP")
    fig.update_layout(
        title={"text": (f"Brain SPECT: {args.fov_size_mm:g} mm FOV sphere vs hardware — "
                        f"<b>{verdict}</b><br><sup>closest collimator gap "
                        f"{coll_clear.min():.3f} mm (module {worst + 1}); shield gap "
                        f"{shield_clear:.2f} mm; crystal gap {crys_clear.min():.1f} mm. "
                        "Geant4 CheckOverlaps on the Gate geometry: no overlaps at 288 mm; "
                        "a 292 mm sphere overlaps all 73 collimators and the shield.</sup>"),
               "x": 0.01},
        height=820, template="plotly_white", margin={"t": 140},
        legend={"x": 0.0, "y": 0.0, "bgcolor": "rgba(255,255,255,0.7)"},
        scene={"aspectmode": "data", "xaxis_title": "x (mm)", "yaxis_title": "y (mm)",
               "zaxis_title": "z (mm)", "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 0.9}}},
        updatemenus=[{
            "type": "buttons", "direction": "right", "x": 0.0, "y": 1.06,
            "xanchor": "left", "showactive": True,
            "buttons": [
                {"label": "Cut away +y half", "method": "restyle",
                 "args": [{"visible": cut_vis}]},
                {"label": "Full", "method": "restyle", "args": [{"visible": full_vis}]},
            ],
        }],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
