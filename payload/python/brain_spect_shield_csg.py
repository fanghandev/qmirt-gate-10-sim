"""Brain SPECT lead shield as Geant4 constructive solid geometry (CSG) for Gate 10.

Replaces the tessellated BrainFrame.008.Lead_Shield.STL with analytic solids:

- a spherical shell, r = 145-155 mm, elevation >= -47 deg (a G4Sphere theta cut);
- minus a two-step face window bounded by constant azimuth and elevation;
- minus 73 square apertures, one per module, in the module frame (origin at the
  pinhole, +z toward the centre): a tapered square (G4Trd) of width
  27.528 - 1.5861 * z mm up to z = 3.8 mm, then a 21.65 mm square lip (G4Box)
  through the inner surface. The top module's aperture is rotated 2.5 deg in-plane.

The STL's rounded edges (4.73 mm on the inner edges of the window and bottom cut,
6.0 / 14.0 mm at the window corners) are not built here: they need G4Torus /
G4GenericPolycone / G4ExtrudedSolid, which opengate_core does not expose. See
dev/python/brain_shield_csg.md.

Two layouts give the same solid:
- "tiles" (default): 73 world volumes, each a G4Sphere segment around one aperture
  minus that aperture. Ring boundaries are the mid-elevations between rings, which
  are also the window edges, so the window is simply where no tile is placed.
- "single": one G4Sphere minus the window segments minus all 73 apertures (a long
  boolean chain; kept for comparison).
"""

import numpy as np
import opengate as gate
from opengate.geometry.volumes import subtract_volumes, unite_volumes
from scipy.spatial.transform import Rotation

SHIELD_CSG = {
    "r_inner_mm": 145.0,
    "r_outer_mm": 155.0,
    "bottom_elevation_deg": -47.0,
    # (azimuth min, azimuth max), (elevation min, elevation max); azimuth 90 = +y
    "window": [
        ((56.25, 123.75), (-26.5, -5.75)),
        ((50.524, 129.476), (-47.0, -26.5)),
    ],
    "aperture": {
        "width_at_pinhole_mm": 27.528,
        "width_slope": -1.5861,  # d(width)/dz, z from the pinhole toward the centre
        "lip_start_z_mm": 3.8,
        "lip_width_mm": 21.65,
        "top_module_rotation_deg": 2.5,
    },
    # measured on the STL, used by the manifold3d model only (not built in Gate)
    "rounds": {
        "inner_edge_radius_mm": 4.73,
        "window_corner_radius_mm": 6.0,
        "bottom_junction_radius_mm": 14.0,
        "bottom_junction_centre_mm": (14.0, 13.4),
    },
}
# the aperture solid is cut long enough to pass through both spheres
_APERTURE_Z_RANGE = (-9.0, 9.0)


def module_frames(pl_df):
    """Pinholes (N, 3) and module rotations (N, 3, 3), local -> world."""
    import gate_sim_brain_spect_boolean as brain

    n = pl_df.shape[0]
    pinholes = np.array([[pl_df.item(i, f"Pinhole_{a}") for a in "xyz"] for i in range(n)])
    rotations = np.array([brain.get_head_rotation_matrix(pl_df, i) for i in range(n)])
    return pinholes, rotations


def aperture_rotation(k, rotations):
    r = rotations[k]
    if k == len(rotations) - 1:
        a = SHIELD_CSG["aperture"]["top_module_rotation_deg"]
        r = r @ Rotation.from_euler("z", a, degrees=True).as_matrix()
    return r


def _aperture_volume(name):
    """Aperture solid in its own frame; returns (volume, z of its origin in the
    module frame). The trd's origin is its centre."""
    p = SHIELD_CSG["aperture"]
    z0, z_lip = _APERTURE_Z_RANGE[0], p["lip_start_z_mm"]
    width = lambda z: p["width_at_pinhole_mm"] + p["width_slope"] * z  # noqa: E731
    trd = gate.geometry.volumes.TrdVolume(name=f"{name}_taper")
    trd.dz = (z_lip - z0) / 2
    trd.dx1 = trd.dy1 = width(z0) / 2
    trd.dx2 = trd.dy2 = width(z_lip) / 2
    trd_centre = (z_lip + z0) / 2
    lip = gate.geometry.volumes.BoxVolume(name=f"{name}_lip")
    lip_z0, lip_z1 = z_lip - 0.01, _APERTURE_Z_RANGE[1]
    lip.size = [p["lip_width_mm"], p["lip_width_mm"], lip_z1 - lip_z0]
    lip_centre = (lip_z0 + lip_z1) / 2
    hole = unite_volumes(trd, lip, translation=[0, 0, lip_centre - trd_centre], new_name=name)
    return hole, trd_centre


def _subtract_aperture(solid, k, pinholes, rotations, name):
    hole, z_c = _aperture_volume(f"{name}_aperture")
    r = aperture_rotation(k, rotations)
    centre = pinholes[k] + r @ np.array([0.0, 0.0, z_c])
    # opengate passes the rotation to G4 booleans as a frame rotation (passive), so
    # the object rotation r (local -> world) is given as its transpose
    return subtract_volumes(solid, hole, translation=centre.tolist(), rotation=r.T, new_name=name)


def _sphere_segment(name, az0, az1, el0, el1, r_in=None, r_out=None):
    deg = gate.g4_units.deg
    seg = gate.geometry.volumes.SphereVolume(name=name)
    seg.rmin = SHIELD_CSG["r_inner_mm"] if r_in is None else r_in
    seg.rmax = SHIELD_CSG["r_outer_mm"] if r_out is None else r_out
    seg.sphi, seg.dphi = az0 * deg, (az1 - az0) * deg
    seg.stheta, seg.dtheta = (90.0 - el1) * deg, (el1 - el0) * deg  # theta from +z
    return seg


def _ring_bands(pinholes):
    """Rings by pinhole elevation, with band edges midway between rings."""
    el = np.degrees(np.arcsin(pinholes[:, 2] / np.linalg.norm(pinholes, axis=1)))
    rings = np.unique(np.round(el, 3))
    edges = [SHIELD_CSG["bottom_elevation_deg"]]
    edges += [(a + b) / 2 for a, b in zip(rings[:-1], rings[1:])]
    edges += [90.0]
    members = [np.flatnonzero(np.isclose(np.round(el, 3), e)) for e in rings]
    return rings, edges, members


def _band_azimuth_limits(el0, el1):
    """Azimuth range kept in a band (the window removes [az0, az1] over its rows)."""
    for (az0, az1), (w0, w1) in SHIELD_CSG["window"]:
        if np.isclose(el0, w0) and np.isclose(el1, w1):
            return az1, az0 + 360.0
    return None


def add_csg_shield_to_gate_sim(sim, pl_df, layout="tiles", material="Lead", prefix="ShieldCSG"):
    """Add the CSG shield to the world; returns the list of volume names."""
    pinholes, rotations = module_frames(pl_df)
    names = []
    if layout == "single":
        shell = _sphere_segment(f"{prefix}_shell", 0.0, 360.0,
                                SHIELD_CSG["bottom_elevation_deg"], 90.0)
        solid = shell
        for i, ((az0, az1), (el0, el1)) in enumerate(SHIELD_CSG["window"]):
            win = _sphere_segment(f"{prefix}_window_{i}", az0, az1, el0, el1,
                                  r_in=SHIELD_CSG["r_inner_mm"] - 5, r_out=SHIELD_CSG["r_outer_mm"] + 5)
            solid = subtract_volumes(solid, win, new_name=f"{prefix}_w{i}")
        for k in range(len(pinholes)):
            solid = _subtract_aperture(solid, k, pinholes, rotations, f"{prefix}_a{k + 1}")
        solid.name = prefix
        solid.material = material
        sim.volume_manager.add_volume(solid)
        return [solid.name]
    if layout != "tiles":
        raise ValueError(f"unknown CSG shield layout {layout!r}")
    azimuth = np.degrees(np.arctan2(pinholes[:, 1], pinholes[:, 0])) % 360
    rings, edges, members = _ring_bands(pinholes)
    for b, (ring_el, idx) in enumerate(zip(rings, members)):
        el0, el1 = edges[b], edges[b + 1]
        limits = _band_azimuth_limits(el0, el1)
        if len(idx) == 1 and np.isclose(ring_el, 90.0):  # top cap
            sectors = [(idx[0], 0.0, 360.0)]
        else:
            az = azimuth[idx]
            if limits is not None:
                az = np.where(az < limits[0], az + 360.0, az)
            order = np.argsort(az)
            idx, az = idx[order], az[order]
            mids = (az + np.roll(az, -1) + np.where(np.arange(len(az)) == len(az) - 1, 360.0, 0.0)) / 2
            starts = np.r_[mids[-1] - 360.0, mids[:-1]]
            ends = mids.copy()
            if limits is not None:  # the window takes the gap between the end sectors
                starts[0], ends[-1] = limits[0], limits[1]
            sectors = list(zip(idx, starts, ends))
        for k, a0, a1 in sectors:
            tile = _sphere_segment(f"{prefix}_tile{k + 1}_seg", a0, a1, el0, el1)
            if np.isclose(ring_el, 90.0):
                tile.stheta, tile.dtheta = 0.0, (90.0 - el0) * gate.g4_units.deg
            solid = _subtract_aperture(tile, k, pinholes, rotations, f"{prefix}_tile{k + 1}")
            solid.material = material
            sim.volume_manager.add_volume(solid)
            names.append(solid.name)
    return names
