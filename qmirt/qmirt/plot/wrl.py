"""Geometry plotting helpers for qmirt_utility.

This module parses VRML geometry and builds Plotly figures for the
simulation meshes. It is intentionally self-contained so the package does
not depend on the legacy gate10_tools module.
"""

from __future__ import annotations

import json
import re
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go

Mesh = dict[str, Any]
StyleMap = dict[str, dict[str, Any]]
StyleSheetInput = dict[str, Any] | str | Path
PatternList = list[str] | tuple[str, ...]
ExcludeMode = Literal["merge", "replace", "style_only"]


def parse_vrml_indexed_face_sets(wrl_path: str | Path) -> list[Mesh]:
    """Parse VRML V2.0 IndexedFaceSet blocks into named meshes."""
    meshes: list[Mesh] = []
    current_name = "unnamed"
    lines = Path(wrl_path).read_text(encoding="utf-8").splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("#---------- SOLID:"):
            current_name = line.split(":", 1)[1].strip()

        if "point [" in line:
            vertices: list[list[float]] = []
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

            polygons: list[list[int]] = []
            poly: list[int] = []
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
                meshes.append(
                    {
                        "name": current_name,
                        "vertices": np.array(vertices, dtype=float),
                        "polygons": polygons,
                    }
                )

        i += 1

    return meshes


def triangulate_polygons(
    polygons: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def extract_sharp_edges(
    verts: np.ndarray,
    tri_i: np.ndarray,
    tri_j: np.ndarray,
    tri_k: np.ndarray,
    angle_deg: float = 35.0,
    include_boundary: bool = True,
) -> list[tuple[int, int]]:
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

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, tri in enumerate(triangles):
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for a, b in edges:
            key = (a, b) if a < b else (b, a)
            edge_faces.setdefault(key, []).append(face_id)

    cos_threshold = np.cos(np.deg2rad(angle_deg))
    sharp_edges: list[tuple[int, int]] = []
    for edge, faces in edge_faces.items():
        if len(faces) == 1:
            if include_boundary:
                sharp_edges.append(edge)
            continue
        f1, f2 = faces[0], faces[1]
        c = float(np.clip(np.dot(normals[f1], normals[f2]), -1.0, 1.0))
        if c < cos_threshold:
            sharp_edges.append(edge)

    return sharp_edges


def edges_to_lines(
    verts: np.ndarray, edges: list[tuple[int, int]]
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for a, b in edges:
        va = verts[a]
        vb = verts[b]
        x.extend([float(va[0]), float(vb[0]), None])
        y.extend([float(va[1]), float(vb[1]), None])
        z.extend([float(va[2]), float(vb[2]), None])
    return x, y, z


def _default_geometry_styles(wrl_meshes: list[Mesh]) -> StyleMap:
    palette = [
        ("#7f7f7f", "#222222"),
        ("#ff8c00", "#8b4513"),
        ("#17becf", "#1f3a93"),
        ("#2ca02c", "#145a32"),
        ("#d62728", "#6e2c00"),
        ("#1f77b4", "#0b3c5d"),
    ]
    styles: StyleMap = {}
    for idx, mesh in enumerate(wrl_meshes):
        name = str(mesh["name"])
        color, edge_color = palette[idx % len(palette)]
        styles[name] = {
            "color": color,
            "opacity": 0.2,
            "edge_color": edge_color,
            "legend_name": name,
        }
    return styles


def _load_style_sheet(style_sheet: StyleSheetInput) -> dict[str, Any]:
    if isinstance(style_sheet, dict):
        return style_sheet

    path = Path(style_sheet)
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(raw)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "YAML stylesheet requires pyyaml. Install it or use JSON stylesheet."
            ) from exc
        loaded = yaml.safe_load(raw)
        return loaded or {}

    raise ValueError(
        f"Unsupported stylesheet extension '{suffix}'. Use .json, .yaml, or .yml"
    )


def _merge_style(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


def _styles_from_style_sheet(
    wrl_meshes: list[Mesh], style_sheet: dict[str, Any]
) -> StyleMap:
    default_style = style_sheet.get(
        "default_style",
        {
            "color": "#7f7f7f",
            "opacity": 0.2,
            "edge_color": "#222222",
        },
    )
    default_enabled = bool(style_sheet.get("default_enabled", True))
    rules = style_sheet.get("rules", [])

    styles: StyleMap = {}
    for mesh in wrl_meshes:
        name = str(mesh["name"])
        enabled = default_enabled
        style = _merge_style(default_style, {"legend_name": name})

        for rule in rules:
            pattern = str(rule.get("pattern", "*"))
            if not fnmatchcase(name, pattern):
                continue

            if "enabled" in rule:
                enabled = bool(rule["enabled"])

            if isinstance(rule.get("style"), dict):
                style = _merge_style(style, rule["style"])

        if enabled:
            style.setdefault("legend_name", name)
            styles[name] = style

    return styles


def _matches_any_pattern(name: str, patterns: PatternList) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def _filter_excluded_meshes(
    wrl_meshes: list[Mesh], exclude_patterns: PatternList | None
) -> list[Mesh]:
    if not exclude_patterns:
        return wrl_meshes
    return [
        mesh
        for mesh in wrl_meshes
        if not _matches_any_pattern(str(mesh["name"]), exclude_patterns)
    ]


def _combine_exclude_patterns(
    style_sheet_excludes: PatternList | None,
    call_excludes: PatternList | None,
    exclude_mode: ExcludeMode,
) -> PatternList | None:
    if exclude_mode == "replace":
        return call_excludes
    if exclude_mode == "style_only":
        return style_sheet_excludes

    combined: list[str] = []
    if style_sheet_excludes:
        combined.extend(style_sheet_excludes)
    if call_excludes:
        combined.extend(call_excludes)
    return combined


def plot_meshes(
    wrl_meshes: list[Mesh],
    geometry_styles: StyleMap | None = None,
    style_sheet: StyleSheetInput | None = None,
    exclude_patterns: PatternList | None = None,
    exclude_mode: ExcludeMode = "merge",
) -> go.Figure:
    """Create a Plotly figure from parsed WRL meshes.

    If style_sheet is provided, it is used to select geometries and style traces.
    """
    fig = go.Figure()
    if style_sheet is not None:
        loaded_style_sheet = _load_style_sheet(style_sheet)
        style_sheet_excludes = loaded_style_sheet.get("exclude_patterns", [])
        combined_excludes = _combine_exclude_patterns(
            style_sheet_excludes if isinstance(style_sheet_excludes, list) else None,
            exclude_patterns,
            exclude_mode,
        )
        filtered_meshes = _filter_excluded_meshes(wrl_meshes, combined_excludes)
        styles = _styles_from_style_sheet(filtered_meshes, loaded_style_sheet)
    else:
        filtered_meshes = _filter_excluded_meshes(wrl_meshes, exclude_patterns)
        styles = geometry_styles or _default_geometry_styles(filtered_meshes)
        if exclude_patterns:
            styles = {
                name: style
                for name, style in styles.items()
                if not _matches_any_pattern(str(name), exclude_patterns)
            }

    grouped = {
        key: dict(
            x=[],
            y=[],
            z=[],
            i=[],
            j=[],
            k=[],
            edge_x=[],
            edge_y=[],
            edge_z=[],
            offset=0,
        )
        for key in styles.keys()
    }

    selected_meshes = [mesh for mesh in filtered_meshes if str(mesh["name"]) in styles]
    for mesh in selected_meshes:
        geom_key = str(mesh["name"])
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

        sharp_edges = extract_sharp_edges(
            verts, i_idx, j_idx, k_idx, angle_deg=35.0, include_boundary=True
        )
        if sharp_edges:
            edge_x, edge_y, edge_z = edges_to_lines(verts, sharp_edges)
            g["edge_x"].extend(edge_x)
            g["edge_y"].extend(edge_y)
            g["edge_z"].extend(edge_z)

    for geom_key, style in styles.items():
        g = grouped[geom_key]
        if not g["i"]:
            continue

        legend_group = f"Geometry {style['legend_name']}"
        fig.add_trace(
            go.Mesh3d(
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
                hovertemplate=f"<b>Mesh: {geom_key}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>",
            )
        )

        if g["edge_x"]:
            fig.add_trace(
                go.Scatter3d(
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

    return fig


def plot_wrl_file(
    wrl_path: str | Path,
    *,
    geometry_styles: StyleMap | None = None,
    style_sheet: StyleSheetInput | None = None,
    exclude_patterns: PatternList | None = None,
    exclude_mode: ExcludeMode = "merge",
    title: str = "Simulation Geometry from VRML",
) -> go.Figure:
    """Parse a WRL file and return a styled Plotly 3D figure.

    style_sheet can be a dict or a path to a JSON/YAML stylesheet.
    exclude_patterns uses shell-style wildcards on mesh names, e.g. ["world:*"]
    """
    meshes = parse_vrml_indexed_face_sets(wrl_path)
    fig = plot_meshes(
        meshes,
        geometry_styles=geometry_styles,
        style_sheet=style_sheet,
        exclude_patterns=exclude_patterns,
        exclude_mode=exclude_mode,
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig
