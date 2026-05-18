import numpy as np
import plotly.graph_objects as pgo
import uproot
import re
from pathlib import Path

def parse_vrml_indexed_face_sets(wrl_path):
    """Parse VRML V2.0 IndexedFaceSet blocks into named meshes."""
    meshes = []
    current_name = "unnamed"
    with open(wrl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("#---------- SOLID:"):
            current_name = line.split(":", 1)[1].strip()

        if "point [" in line:
            vertices = []
            i += 1
            while i < len(lines):
                s = lines[i].strip()
                if s.startswith("]"):
                    break
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
                if len(nums) >= 3:
                    vertices.append([float(nums[0]), float(nums[1]), float(nums[2])])
                i += 1

            while i < len(lines) and "coordIndex [" not in lines[i]:
                i += 1
            if i >= len(lines):
                break

            polygons = []
            poly = []
            i += 1
            while i < len(lines):
                s = lines[i].strip()
                if s.startswith("]"):
                    if len(poly) >= 3:
                        polygons.append(poly)
                    break
                nums = re.findall(r"-?\d+", s)
                for n in nums:
                    idx = int(n)
                    if idx == -1:
                        if len(poly) >= 3:
                            polygons.append(poly)
                        poly = []
                    else:
                        poly.append(idx)
                i += 1

            if vertices and polygons:
                meshes.append({
                    "name": current_name,
                    "vertices": np.array(vertices, dtype=float),
                    "polygons": polygons,
                })
        i += 1

    return meshes

def triangulate_polygons(polygons):
    tri_i, tri_j, tri_k = [], [], []
    for poly in polygons:
        if len(poly) < 3:
            continue
        root = poly[0]
        for t in range(1, len(poly) - 1):
            tri_i.append(root)
            tri_j.append(poly[t])
            tri_k.append(poly[t + 1])
    return np.array(tri_i), np.array(tri_j), np.array(tri_k)

def extract_sharp_edges(verts, tri_i, tri_j, tri_k, angle_deg=35.0, include_boundary=True):
    triangles = np.column_stack((tri_i, tri_j, tri_k)).astype(int)
    if triangles.shape[0] == 0:
        return []

    p0 = verts[triangles[:, 0]]
    p1 = verts[triangles[:, 1]]
    p2 = verts[triangles[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    norm_len = np.linalg.norm(normals, axis=1)
    valid = norm_len > 0
    normals[valid] = normals[valid] / norm_len[valid, None]

    edge_faces = {}
    for face_id, tri in enumerate(triangles):
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for a, b in edges:
            key = (a, b) if a < b else (b, a)
            edge_faces.setdefault(key, []).append(face_id)

    cos_threshold = np.cos(np.deg2rad(angle_deg))
    sharp_edges = []
    for edge, faces in edge_faces.items():
        if len(faces) == 1:
            if include_boundary:
                sharp_edges.append(edge)
            continue
        if len(faces) >= 2:
            f1, f2 = faces[0], faces[1]
            c = float(np.clip(np.dot(normals[f1], normals[f2]), -1.0, 1.0))
            if c < cos_threshold:
                sharp_edges.append(edge)

    return sharp_edges

def edges_to_lines(verts, edges):
    x, y, z = [], [], []
    for a, b in edges:
        va = verts[a]
        vb = verts[b]
        x.extend([va[0], vb[0], None])
        y.extend([va[1], vb[1], None])
        z.extend([va[2], vb[2], None])
    return x, y, z

def load_postion_data_from_root(simulation_dir, dc_spect_mc_root_filename):

    output= {}
    with uproot.open(simulation_dir / dc_spect_mc_root_filename) as file:
        detectorActorTree = file["DetectorHitsActor"]
        # Read EventPosition_X, EventPosition_Y, EventPosition_Z into numpy arrays
        event_pos_data = detectorActorTree.arrays(["EventPosition_X", "EventPosition_Y", "EventPosition_Z"], library="np")
        # Combine into (N, 3) array
        output["event_pos_xyz"] = np.column_stack((event_pos_data["EventPosition_X"], 
                                event_pos_data["EventPosition_Y"], 
                                event_pos_data["EventPosition_Z"]))

        output["event_id"] = detectorActorTree["EventID"].array(library="np")
        # Read PostPosition_X, PostPosition_Y, PostPosition_Z into numpy arrays
        hit_post_pos_data = detectorActorTree.arrays(["PostPosition_X ", "PostPosition_Y", "PostPosition_Z"], library="np")
        # Read PrePosition_X, PrePosition_Y, PrePosition_Z into numpy arrays
        hit_pre_pos_data = detectorActorTree.arrays(["PrePosition_X ", "PrePosition_Y", "PrePosition_Z"], library="np")
        # Combine into (N, 3) array
        output["hit_post_pos_xyz"] = np.column_stack((hit_post_pos_data["PostPosition_X "], 
                                hit_post_pos_data["PostPosition_Y"], 
                                hit_post_pos_data["PostPosition_Z"]))
        output["hit_pre_pos_xyz"] = np.column_stack((hit_pre_pos_data["PrePosition_X "], 
                                hit_pre_pos_data["PrePosition_Y"], 
                                hit_pre_pos_data["PrePosition_Z"]))
    return output                                


def plot_hit_positions_and_geometry(position_data_dict, base_dir):
    event_pos_xyz = position_data_dict["event_pos_xyz"]
    hit_post_pos_xyz = position_data_dict["hit_post_pos_xyz"]
    hit_pre_pos_xyz = position_data_dict["hit_pre_pos_xyz"]

    fig = pgo.Figure()

    wrl_path = base_dir / "opengate" / "dev" / "collimator_geometry.wrl"
    print(f"Parsing geometry from: {wrl_path}")
    wrl_meshes = parse_vrml_indexed_face_sets(wrl_path)

    geometry_styles = {
        "collimator": dict(color="gray", opacity=0.12, edge_color="black", legend_name="Collimator"),
        "detector": dict(color="orange", opacity=0.18, edge_color="darkorange", legend_name="Detector"),
        "shielding": dict(color="cyan", opacity=0.20, edge_color="indigo", legend_name="Shielding"),
    }

    # Aggregate all WRL solids per component so each component has one surface entry and one edge entry in legend.
    grouped = {
        key: dict(x=[], y=[], z=[], i=[], j=[], k=[], edge_x=[], edge_y=[], edge_z=[], offset=0)
        for key in geometry_styles.keys()
    }

    for mesh in wrl_meshes:
        mesh_name_lower = mesh["name"].lower()
        if "collimator" in mesh_name_lower:
            geom_key = "collimator"
        elif "detector" in mesh_name_lower:
            geom_key = "detector"
        elif "shield" in mesh_name_lower:
            geom_key = "shielding"
        else:
            continue

        verts = mesh["vertices"]
        i_idx, j_idx, k_idx = triangulate_polygons(mesh["polygons"])
        if len(i_idx) == 0:
            continue

        g = grouped[geom_key]
        off = g["offset"]

        g["x"].extend(verts[:, 0].tolist())
        g["y"].extend(verts[:, 1].tolist())
        g["z"].extend(verts[:, 2].tolist())
        g["i"].extend((i_idx + off).tolist())
        g["j"].extend((j_idx + off).tolist())
        g["k"].extend((k_idx + off).tolist())
        g["offset"] += len(verts)

        sharp_edges = extract_sharp_edges(verts, i_idx, j_idx, k_idx, angle_deg=35.0, include_boundary=True)
        if len(sharp_edges) > 0:
            edge_x, edge_y, edge_z = edges_to_lines(verts, sharp_edges)
            g["edge_x"].extend(edge_x)
            g["edge_y"].extend(edge_y)
            g["edge_z"].extend(edge_z)

    for geom_key, style in geometry_styles.items():
        g = grouped[geom_key]
        if len(g["i"]) == 0:
            continue

        legend_group = f"Geometry {style['legend_name']}"
        print(f"Added geometry: {style['legend_name']}")
        fig.add_trace(
            pgo.Mesh3d(
                x=g["x"],
                y=g["y"],
                z=g["z"],
                i=g["i"],
                j=g["j"],
                k=g["k"],
                name="Surface",
                legendgroup=legend_group,
                legendgrouptitle_text=style["legend_name"],
                opacity=style["opacity"],
                color=style["color"],
                flatshading=True,
                showlegend=True,
            )
        )

        if len(g["edge_x"]) > 0:
            print(f"Added geometry edges: {style['legend_name']}")
            fig.add_trace(
                pgo.Scatter3d(
                    x=g["edge_x"],
                    y=g["edge_y"],
                    z=g["edge_z"],
                    mode="lines",
                    name="Edges",
                    legendgroup=legend_group,
                    showlegend=True,
                    line=dict(color=style["edge_color"], width=2),
                    hoverinfo="skip",
                )
            )

    # Add hit/source traces
    post_pos_3DScatter = pgo.Scatter3d(
        x=hit_post_pos_xyz[:, 0],
        y=hit_post_pos_xyz[:, 1],
        z=hit_post_pos_xyz[:, 2],
        legendgroup="Hit Post Position",
        name="Post Position",
        mode="markers",
        showlegend=True,
        marker=dict(size=1, color="blue"),
    )
    pre_pos_3DScatter = pgo.Scatter3d(
        x=hit_pre_pos_xyz[:, 0],
        y=hit_pre_pos_xyz[:, 1],
        z=hit_pre_pos_xyz[:, 2],
        legendgroup="Hit Pre Position",
        name="Pre Position",
        mode="markers",
        showlegend=True,
        marker=dict(size=1, color="red"),
    )
    source_pos_3DScatter = pgo.Scatter3d(
        x=event_pos_xyz[:, 0],
        y=event_pos_xyz[:, 1],
        z=event_pos_xyz[:, 2],
        legendgroup="Source Position",
        name="Source Position",
        mode="markers",
        showlegend=True,
        marker=dict(size=1, color="green"),
    )

    fig.add_trace(post_pos_3DScatter)
    fig.add_trace(pre_pos_3DScatter)
    fig.add_trace(source_pos_3DScatter)

    fig.update_layout(scene_camera=dict(projection=dict(type="orthographic")))

    fig.update_layout(
        width=2000,
        height=1400,
        margin=dict(l=10, r=10, t=10, b=10),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="black", gridwidth=0.5),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="black", gridwidth=0.5),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="black", gridwidth=0.5),
            aspectmode="data",
        ),
        legend=dict(title="Geometry and Hit Traces", x=0.01, y=0.5),
    )

    # fig.show()
    # Save interactive Plotly figure as HTML
    output_html = Path(base_dir) / "opengate" / "dev" / "jupyter" / "output" / "mc_geometry_hits_plot.html"
    # if folder doesn't exist, create it
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_html,
        include_plotlyjs="cdn",  # smaller file; use True for fully self-contained
        full_html=True
    )

    print(f"Saved: {output_html}")


def plot_geometry_with_points(
    point_xyz,
    wrl_path,
    point_values=None,
    point_name="Event Positions",
    point_color="green",
    point_size=1,
    point_colorbar_title="Values",
    title="Detector geometry with event positions",
    width=None,
    height=900,
    margin=None,
    show=True,
    output_html=None,
):
    fig = pgo.Figure()
    print(f"Parsing geometry from: {wrl_path}")
    wrl_meshes = parse_vrml_indexed_face_sets(wrl_path)

    geometry_styles = {
        "collimator": dict(color="gray", opacity=0.12, edge_color="black", legend_name="Collimator"),
        "detector": dict(color="orange", opacity=0.18, edge_color="darkorange", legend_name="Detector"),
        "shielding": dict(color="cyan", opacity=0.20, edge_color="indigo", legend_name="Shielding"),
    }

    grouped = {
        key: dict(x=[], y=[], z=[], i=[], j=[], k=[], edge_x=[], edge_y=[], edge_z=[], offset=0)
        for key in geometry_styles.keys()
    }

    for mesh in wrl_meshes:
        mesh_name_lower = mesh["name"].lower()
        if "collimator" in mesh_name_lower:
            geom_key = "collimator"
        elif "detector" in mesh_name_lower:
            geom_key = "detector"
        elif "shield" in mesh_name_lower:
            geom_key = "shielding"
        else:
            continue

        verts = mesh["vertices"]
        i_idx, j_idx, k_idx = triangulate_polygons(mesh["polygons"])
        if len(i_idx) == 0:
            continue

        g = grouped[geom_key]
        off = g["offset"]

        g["x"].extend(verts[:, 0].tolist())
        g["y"].extend(verts[:, 1].tolist())
        g["z"].extend(verts[:, 2].tolist())
        g["i"].extend((i_idx + off).tolist())
        g["j"].extend((j_idx + off).tolist())
        g["k"].extend((k_idx + off).tolist())
        g["offset"] += len(verts)

        sharp_edges = extract_sharp_edges(verts, i_idx, j_idx, k_idx, angle_deg=35.0, include_boundary=True)
        if len(sharp_edges) > 0:
            edge_x, edge_y, edge_z = edges_to_lines(verts, sharp_edges)
            g["edge_x"].extend(edge_x)
            g["edge_y"].extend(edge_y)
            g["edge_z"].extend(edge_z)

    for geom_key, style in geometry_styles.items():
        g = grouped[geom_key]
        if len(g["i"]) == 0:
            continue

        legend_group = f"Geometry {style['legend_name']}"
        print(f"Added geometry: {style['legend_name']}")
        fig.add_trace(
            pgo.Mesh3d(
                x=g["x"],
                y=g["y"],
                z=g["z"],
                i=g["i"],
                j=g["j"],
                k=g["k"],
                name="Surface",
                legendgroup=legend_group,
                legendgrouptitle_text=style["legend_name"],
                opacity=style["opacity"],
                color=style["color"],
                flatshading=True,
                showlegend=True,
            )
        )

        if len(g["edge_x"]) > 0:
            print(f"Added geometry edges: {style['legend_name']}")
            fig.add_trace(
                pgo.Scatter3d(
                    x=g["edge_x"],
                    y=g["edge_y"],
                    z=g["edge_z"],
                    mode="lines",
                    name="Edges",
                    legendgroup=legend_group,
                    showlegend=True,
                    line=dict(color=style["edge_color"], width=2),
                    hoverinfo="skip",
                )
            )

    point_marker = dict(
        size=point_size,
        opacity=0.85,
    )
    if point_values is None:
        point_marker["color"] = point_color
    else:
        point_marker["color"] = point_values
        point_marker["colorscale"] = "Viridis"
        point_marker["colorbar"] = dict(title=point_colorbar_title)

    fig.add_trace(
        pgo.Scatter3d(
            x=point_xyz[:, 0],
            y=point_xyz[:, 1],
            z=point_xyz[:, 2],
            legendgroup=point_name,
            name=point_name,
            mode="markers",
            showlegend=True,
            marker=point_marker,
        )
    )

    if margin is None:
        margin = dict(l=0, r=0, t=40, b=0)

    fig.update_layout(scene_camera=dict(projection=dict(type="orthographic")))
    fig.update_layout(
        title=title,
        autosize=True if width is None else False,
        width=width,
        height=height,
        margin=margin,
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="black", gridwidth=0.5),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="black", gridwidth=0.5),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="black", gridwidth=0.5),
            aspectmode="data",
        ),
        legend=dict(
            title="Geometry and Event Traces",
            orientation="h",
            x=0.01,
            y=1.02,
            xanchor="left",
            yanchor="bottom",
        ),
    )

    if output_html is not None:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(
            output_html,
            include_plotlyjs="cdn",
            full_html=True,
        )
        print(f"Saved: {output_html}")

    if show:
        fig.show(config={"responsive": True})

    return fig

if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parents[2]
    print(f"Base directory: {base_dir}")
    simulation_data_dir = Path("/data/fanghan/opengate_sim/data/")
    dc_spect_mc_root_filename = "run_0_detector_hits.root"

    position_data_dict = load_postion_data_from_root(simulation_data_dir, dc_spect_mc_root_filename)
    plot_hit_positions_and_geometry(position_data_dict, base_dir)
    

    