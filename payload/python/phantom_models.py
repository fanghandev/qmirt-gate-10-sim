"""Shared phantom definitions: one description per phantom, every output built from it.

A phantom is a list of parts (cylinders, tubes, spheres) in its own frame: origin at the
centre of the water volume, z towards the lid, mm. Parts are listed parents first; the
voxeliser paints them in that order, so inserts overwrite the water they sit in and the
water overwrites the PMMA body around it. A Pose places the phantom in the scanner frame
(z = helmet axis, +y = face window): stand it on its lid (flip: 180 deg about x), turn it
about z, then shift it.

Outputs built from the same parts and pose:
    voxelize()            label volume on a centred grid (voxel centres at
                          -N v / 2 + (i + 1/2) v, inclusion tested at voxel centres) and
                          the hot (activity) mask, written as the sparse .npz the
                          forward projectors read (save_sparse_mask)
    page_data()           the geometry the interactive phantom page draws
    clearance_solids()    swept solids for the helmet clearance check
    GATE                  phantom_gate.py builds Geant4 volumes from scene()

The two models are the Data Spectrum phantoms as worked out in
dev/python/phantom_3d/photo_fit/README.md and dev/python/phantom_fit/README.md; values
are tagged there as datasheet, measured (vendor photos) or estimated.

numpy only, so GATE jobs can import it without torch.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

INCH = 25.4

# Materials (Geant4 NIST names) and the voxel labels painted for each role
MATERIALS = {"water": "G4_WATER", "pmma": "G4_PLEXIGLASS", "nylon": "G4_NYLON-6-6"}
# Compositions for the descriptive export (density g/cm3, element mass fractions):
# Geant4 NIST values; SpineBone and Brain from the GATE materials database (XCAT bone and
# brain, as persistent_data/GateMaterials.db and Auer's mesh50_XCAT).
MATERIAL_DATA = {
    "G4_AIR": (0.00120479, {"C": 0.000124, "N": 0.755268, "O": 0.231781, "Ar": 0.012827}),
    "G4_WATER": (1.0, {"H": 0.111894, "O": 0.888106}),
    "G4_PLEXIGLASS": (1.19, {"H": 0.080538, "C": 0.599848, "O": 0.319614}),
    "G4_NYLON-6-6": (1.14, {"H": 0.097976, "C": 0.636856, "N": 0.123779, "O": 0.141389}),
    "SpineBone": (1.42, {"H": 0.063, "C": 0.261, "N": 0.039, "O": 0.436, "Na": 0.001, "Mg": 0.001,
                         "P": 0.061, "S": 0.003, "Cl": 0.001, "K": 0.001, "Ca": 0.133}),
    "Brain": (1.04, {"H": 0.107, "C": 0.145, "N": 0.022, "O": 0.712, "Na": 0.002, "P": 0.004, "S": 0.002,
                     "Cl": 0.003, "K": 0.003}),
}
ELEMENTS = {"H": (1, 1.008), "C": (6, 12.011), "N": (7, 14.007), "O": (8, 15.999), "Na": (11, 22.990),
            "Mg": (12, 24.305), "P": (15, 30.974), "S": (16, 32.06), "Cl": (17, 35.45), "Ar": (18, 39.948),
            "K": (19, 39.098), "Ca": (20, 40.078)}
LABELS = {  # label: (role, material key)
    0: ("air", None),
    1: ("water", "water"),
    2: ("body", "pmma"),
    3: ("plate", "pmma"),
    4: ("rod", "pmma"),
    5: ("sphere", "pmma"),
    6: ("stem", "pmma"),
    7: ("fitting", "nylon"),
    8: ("fitting_pmma", "pmma"),
}
ROLE_LABEL = {role: k for k, (role, _) in LABELS.items()}
HOT_ROLES = ("water",)


@dataclass
class Part:
    name: str
    role: str  # key of ROLE_LABEL
    shape: str  # "cylinder" | "tube" | "sphere"
    center: tuple  # phantom frame, mm
    radius: float
    length: float = 0.0  # cylinder / tube, along axis
    inner_radius: float = 0.0  # tube
    axis: tuple = (0.0, 0.0, 1.0)
    parent: str | None = None  # containing part (GATE mother); None = outside the body

    @property
    def material(self) -> str:
        return MATERIALS[LABELS[ROLE_LABEL[self.role]][1]]


@dataclass
class Pose:
    z_center_mm: float = 0.0  # water centre on the helmet axis
    y_shift_mm: float = 0.0
    flip: bool = False  # stood on its lid: 180 deg about x
    rotate_deg: float = 0.0  # counter-clockwise about z, after the flip

    def rotation(self) -> np.ndarray:
        a = np.deg2rad(self.rotate_deg)
        rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        rx = np.diag([1.0, -1.0, -1.0]) if self.flip else np.eye(3)
        return rz @ rx

    def translation(self) -> np.ndarray:
        return np.array([0.0, self.y_shift_mm, self.z_center_mm])

    def to_world(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p, float) @ self.rotation().T + self.translation()

    def to_local(self, p: np.ndarray) -> np.ndarray:
        return (np.asarray(p, float) - self.translation()) @ self.rotation()


@dataclass
class Options:
    stems: bool = True
    plates: bool = True  # rod-insert plates (thin; thickness estimated)
    fittings: bool = True  # caps, boss, screws, knobs (outside the water)
    base_mm: float | None = None  # end-plate overrides (None: the model's values)
    lid_mm: float | None = None
    material_preset: str | None = None  # image phantoms: attenuation-level -> material table


@dataclass
class Spec:
    phantom: str  # key of MODELS
    pose: Pose = field(default_factory=Pose)
    options: Options = field(default_factory=Options)

    def to_dict(self) -> dict:
        return {"phantom": self.phantom, "pose": asdict(self.pose), "options": asdict(self.options),
                "model_version": model_version(self.phantom, self.options)}

    @staticmethod
    def from_dict(d: dict) -> "Spec":
        return Spec(d["phantom"], Pose(**d.get("pose", {})), Options(**d.get("options", {})))


# ======================================================================================
# Layouts (the simulations' rod and sphere layouts)
# ======================================================================================
def acr_rod_centers(diameters, *, cylinder_radius_mm=104.5, spacing_multiplier=2.0, sector_gap_mm=2.0,
                    wall_margin_mm=2.0, inner_apex_radius_mm=None):
    """Rule-based sector lattices of the ACR Deluxe phantom: one 60 deg sector per diameter,
    bisectors at 30 + 60 k deg, rods 2 d apart (as ACR_Jaszczak_digital_phantom.get_rod_centers)."""
    sectors = []
    for k, d in enumerate(diameters):
        pitch = d * spacing_multiplier
        row_h = pitch * np.sqrt(3) / 2.0
        coords, row = [], 0
        while True:
            x = inner_apex_radius_mm[k] + row * row_h
            if x > cylinder_radius_mm:
                break
            added = False
            for j in range(row + 1):
                y = (j - row / 2.0) * pitch
                if np.hypot(x, y) > cylinder_radius_mm - d / 2 - wall_margin_mm:
                    continue
                if x * np.sin(np.pi / 6) - abs(y) * np.cos(np.pi / 6) < d / 2 + sector_gap_mm:
                    continue
                coords.append((x, y))
                added = True
            if not added and row > 0:
                break
            row += 1
        a = k * (np.pi / 3) + np.pi / 6
        c, s = np.cos(a), np.sin(a)
        sectors.append([(x * c - y * s, x * s + y * c, d) for x, y in coords])
    return sectors


def small_rod_centers():
    """Sector lattices of the small phantom as drawn on its datasheet
    (small_Jaszczak_digital_phantom.ROD_SECTORS)."""
    sectors_def = [
        (6.4, 0.0, 19.2, [1, 2, 3, 4, (1, 2, 3)]),
        (4.8, 60.0, 17.6, [1, 2, 3, 4, 5, 6]),
        (12.7, 120.0, 28.7, [1, 2]),
        (11.1, 180.0, 27.3, [1, 2]),
        (9.5, 240.0, 22.5, [1, 2, 3]),
        (7.9, 300.0, 20.8, [1, 2, 3, (1, 2)]),
    ]
    out = []
    for d, bisector, apex, rows in sectors_def:
        pitch = 2.0 * d
        row_h = pitch * np.sqrt(3) / 2
        local = []
        for r, keep in enumerate(rows):
            sites = range(r + 1) if isinstance(keep, int) else keep
            for k in sites:
                local.append((apex + r * row_h, (k - r / 2) * pitch))
        a = np.deg2rad(bisector)
        c, s = np.cos(a), np.sin(a)
        out.append([(x * c - y * s, x * s + y * c, d) for x, y in local])
    return out


def ring(diameters, orbit_mm, first_deg):
    """Spheres every 60 deg counter-clockwise from first_deg: (x, y, d)."""
    return [(orbit_mm * np.cos(np.deg2rad(first_deg + 60 * i)), orbit_mm * np.sin(np.deg2rad(first_deg + 60 * i)), d)
            for i, d in enumerate(diameters)]


# ======================================================================================
# Models
# ======================================================================================
def _vcyl(name, role, x, y, z0, z1, r, parent=None):
    return Part(name, role, "cylinder", (x, y, (z0 + z1) / 2), r, length=z1 - z0, parent=parent)


def _vcyl_split(name, role, x, y, z0, z1, r, cuts, parent="water"):
    """Vertical insert from z0 to z1, cut where it passes through plates (z ranges in
    `cuts`), so no two daughters of the water overlap (Geant4 requirement). The plate's
    PMMA fills the cut, so the material is unchanged."""
    pieces, lo = [], z0
    for a, b in sorted(cuts):
        if b <= lo or a >= z1:
            continue
        if a > lo:
            pieces.append((lo, a))
        lo = max(lo, b)
    if lo < z1:
        pieces.append((lo, z1))
    if len(pieces) == 1:
        return [_vcyl(name, role, x, y, pieces[0][0], pieces[0][1], r, parent)]
    return [_vcyl(f"{name}_seg{k}", role, x, y, a, b, r, parent) for k, (a, b) in enumerate(pieces)]


def _body_and_water(R, r, h, base, lid):
    zf = -h / 2
    return [_vcyl("body", "body", 0, 0, zf - base, h / 2 + lid, R),
            _vcyl("water", "water", 0, 0, zf, h / 2, r, parent="body")]


def _rod_insert(rods, zf, rod_len, plate_t, hole_mm, plate_r, plates):
    """Cold-rod insert: two identical solid PMMA plates whose blind holes (depth `hole_mm`,
    hole floor f = plate_t - hole_mm) hold rods of the datasheet length `rod_len`
    (dev/python/phantom_fit/jaszczak_gate_geometry_spec.md).

    A rod end inside its blind hole is PMMA in PMMA, so each rod is modelled only over the
    clear gap between the plates (rod_len - 2 hole_mm), its end faces on the plate faces; the
    portions in the holes are part of the plates. Insert height H = rod_len + 2 f. Without
    plates, the rods keep their physical extent f .. f + rod_len above the floor.
    Returns (parts, cuts for stems, rod z range above the floor, insert height)."""
    f = plate_t - hole_mm
    H = rod_len + 2 * f
    parts, cuts = [], []
    if plates:
        cuts = [(zf, zf + plate_t), (zf + H - plate_t, zf + H)]
        parts += [_vcyl("plate_bottom", "plate", 0, 0, *cuts[0], plate_r, "water"),
                  _vcyl("plate_top", "plate", 0, 0, *cuts[1], plate_r, "water")]
        z0, z1 = plate_t, H - plate_t
    else:
        z0, z1 = f, f + rod_len
    for i, (x, y, d) in enumerate(rods):
        parts.append(_vcyl(f"rod_{i}", "rod", x, y, zf + z0, zf + z1, d / 2, "water"))
    return parts, cuts, (z0, z1), H


def acr_deluxe(opt: Options) -> tuple[list[Part], dict]:
    """Data Spectrum Flangeless Deluxe Jaszczak (ECT/FL-DLX/P)."""
    R, r, h = 104.5 + 6.4, 104.5, 186.0  # datasheet
    base, lid = INCH / 2, INCH * 3 / 8  # standard sheets matching the photo's base + lid = 22.5
    base = base if opt.base_mm is None else opt.base_mm
    lid = lid if opt.lid_mm is None else opt.lid_mm
    zf = -h / 2
    parts = _body_and_water(R, r, h, base, lid)
    rod_len, plate_t, hole = 88.0, INCH * 3 / 16, 2.0  # datasheet rods; plates est. from the 5.5 L volume; holes (user)
    rods = [rod for sector in acr_rod_centers([6.4, 4.8, 12.7, 11.1, 9.5, 7.9],
                                              inner_apex_radius_mm=[18.0, 16.0, 24.0, 22.0, 22.0, 20.0])
            for rod in sector]
    ins, cuts, rod_z, H = _rod_insert(rods, zf, rod_len, plate_t, hole, 104.0, opt.plates)  # plate dia 208 (datasheet)
    parts += ins
    plate_top = H
    z_sph = 22.0  # simulations' reading of "12.7 cm from base plate" (127 above the outer bottom)
    spheres = ring([9.5, 31.8, 25.4, 19.1, 15.9, 12.7], 57.5, 60.0)
    stem_d = INCH * 3 / 16
    for i, (x, y, d) in enumerate(spheres):
        if opt.stems:  # from the top plate up to the sphere's bottom point (photo, TRELLIS.2 mesh)
            parts.append(_vcyl(f"stem_{i}", "stem", x, y, zf + plate_top, z_sph - d / 2, stem_d / 2, "water"))
        parts.append(Part(f"sphere_{i}", "sphere", "sphere", (x, y, z_sph), d / 2, parent="water"))
    top = h / 2 + lid
    if opt.fittings:  # photo measurements; outside the water
        def polar(rr, az):
            return rr * np.cos(np.deg2rad(az)), rr * np.sin(np.deg2rad(az))
        for name, rr, az, d, z0, z1, role in [
            ("cap_centre", 0.0, 0.0, 24.0, 0.0, 21.0, "fitting"),
            ("cap_back", 75.2, 113.5, 24.0, 0.0, 21.0, "fitting"),
            ("boss", 91.3, 302.3, 37.0, 0.0, 18.0, "fitting_pmma"),
            ("cap_boss", 91.3, 302.3, 28.0, 18.0, 39.0, "fitting"),
        ] + [(f"screw_{k}", 105.7, 30.7 + 60 * k, 10.0, 0.0, 6.0, "fitting") for k in range(6)]:
            x, y = polar(rr, az)
            parts.append(_vcyl(name, role, x, y, top + z0, top + z1, d / 2))
    meta = {"label": "ACR Deluxe flangeless Jaszczak", "outer_radius_mm": R, "inner_radius_mm": r,
            "inner_height_mm": h, "base_mm": base, "lid_mm": lid, "rod_length_mm": rod_len, "rod_z_mm": list(rod_z),
            "insert_height_mm": H, "plate_mm": plate_t, "blind_hole_mm": hole,
            "sphere_above_floor_mm": z_sph - zf, "plates_estimated": True}
    return parts, meta


def small_jaszczak(opt: Options) -> tuple[list[Part], dict]:
    """Data Spectrum Small Jaszczak SPECT phantom (ECT/SM/P)."""
    R, r, h = 153.0 / 2, 139.0 / 2, 150.0  # datasheet
    base = INCH * 3 / 8  # body 184.4 mm (photo); base 3/8 in, lid 1 in (inferred)
    lid = 184.4 - h - base
    base = base if opt.base_mm is None else opt.base_mm
    lid = lid if opt.lid_mm is None else opt.lid_mm
    zf = -h / 2
    parts = _body_and_water(R, r, h, base, lid)
    rod_len, plate_t, hole = 40.0, INCH / 8, 2.0  # datasheet rods; plates measured (photo); holes (user)
    rods = [rod for sector in small_rod_centers() for rod in sector]
    ins, cuts, rod_z, H = _rod_insert(rods, zf, rod_len, plate_t, hole, r, opt.plates)  # plate dia 139 (datasheet)
    parts += ins
    z_sph = zf + 78.0  # on stems standing on the floor, 78 mm up (datasheet + photos)
    spheres = ring([9.5, 6.4, 25.4, 19.1, 15.9, 12.7], 97.0 / 2, 30.0)
    stem_d, necks = INCH / 4, {6.4: (3.0, 6.0), 9.5: (3.0, 6.0)}
    for i, (x, y, d) in enumerate(spheres):
        if opt.stems:  # stands on the floor, through the insert plates, up to the sphere's bottom point
            top = z_sph - d / 2
            nd, nl = necks.get(d, (None, 0.0))
            parts += _vcyl_split(f"stem_{i}", "stem", x, y, zf, top - nl, stem_d / 2, cuts)
            if nl:
                parts.append(_vcyl(f"stem_{i}_neck", "stem", x, y, top - nl, top, nd / 2, "water"))
        parts.append(Part(f"sphere_{i}", "sphere", "sphere", (x, y, z_sph), d / 2, parent="water"))
    top = h / 2 + lid
    if opt.fittings:
        off = -25.7  # photo azimuths -> layout frame
        for name, rr, az, d, z0, z1 in [("cap_left", 61.0, -136.2, 24.5, 0.0, 24.8),
                                        ("cap_centre", 17.7, -94.2, 24.5, 0.0, 24.4),
                                        ("boss_and_cap", 46.6, 22.1, 24.8, 0.0, 47.8)]:
            a = np.deg2rad(az + off)
            parts.append(_vcyl(name, "fitting", rr * np.cos(a), rr * np.sin(a), top + z0, top + z1, d / 2))
        a, kd, ko = np.deg2rad(-22.6 + off), 13.0, 11.0  # twist-lock knob, centred 166.3 above the outer bottom
        mid = R + ko / 2  # from the body surface outwards
        parts.append(Part("knob", "fitting", "cylinder", (mid * np.cos(a), mid * np.sin(a), zf - base + 166.3), kd / 2,
                          length=ko, axis=(np.cos(a), np.sin(a), 0.0)))
    meta = {"label": "Small Jaszczak SPECT", "outer_radius_mm": R, "inner_radius_mm": r, "inner_height_mm": h,
            "base_mm": base, "lid_mm": lid, "rod_length_mm": rod_len, "rod_z_mm": list(rod_z), "insert_height_mm": H,
            "plate_mm": plate_t, "blind_hole_mm": hole, "sphere_above_floor_mm": 78.0, "plates_estimated": True}
    return parts, meta


MODELS = {"acr_deluxe": acr_deluxe, "small": small_jaszczak}


def build(phantom: str, opt: Options | None = None) -> tuple[list[Part], dict]:
    return MODELS[phantom](opt or Options())


def model_version(phantom: str, opt: Options | None = None) -> str:
    if is_image(phantom):
        m = IMAGE_MODELS[phantom]
        blob = json.dumps([asdict(m), image_material_table(phantom, opt)], sort_keys=True, default=str)
        blob += "".join(f"{f.stat().st_size}:{f.stat().st_mtime_ns}" for f in sorted(m.directory().glob("*.raw")))
        blob += "".join(f"{f.stat().st_size}" for f in sorted(m.directory().glob("*.stl")))
        return hashlib.sha1(blob.encode()).hexdigest()[:12]
    parts, meta = build(phantom, opt)
    blob = json.dumps([asdict(p) for p in parts] + [meta], sort_keys=True, default=float)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


# ======================================================================================
# Image (voxel) phantoms
# ======================================================================================
PHANTOM_DATA = Path(os.environ.get("QMIRT_PHANTOM_DATA",
                                   Path(__file__).resolve().parents[2] / "dev" / "python" / "phantom_sources"))
_MHD_TYPES = {"MET_FLOAT": "<f4", "MET_DOUBLE": "<f8", "MET_UCHAR": "u1", "MET_CHAR": "i1", "MET_SHORT": "<i2",
              "MET_USHORT": "<u2", "MET_INT": "<i4", "MET_UINT": "<u4"}


def read_mhd(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(array indexed [x, y, z], spacing, offset = centre of the first voxel) of a MetaImage."""
    path = Path(path)
    hdr = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            hdr[k.strip()] = v.strip()
    dims = [int(v) for v in hdr["DimSize"].split()]
    raw = path.parent / hdr["ElementDataFile"]
    a = np.fromfile(raw, _MHD_TYPES[hdr["ElementType"]]).reshape(dims[::-1]).transpose(2, 1, 0)
    return a, np.array(hdr["ElementSpacing"].split(), float), np.array(hdr.get("Offset", "0 0 0").split(), float)


def write_mhd(path, array: np.ndarray, spacing, offset, element_type="MET_FLOAT"):
    """Write an [x, y, z] array as MetaImage (.mhd + .raw next to it)."""
    path = Path(path)
    raw = path.with_suffix(".raw")
    np.ascontiguousarray(array.transpose(2, 1, 0)).astype(_MHD_TYPES[element_type]).tofile(raw)
    path.write_text("\n".join([
        "ObjectType = Image", "NDims = 3", "BinaryData = True", "BinaryDataByteOrderMSB = False",
        "CompressedData = False", "TransformMatrix = 1 0 0 0 1 0 0 0 1",
        "Offset = " + " ".join(f"{v:g}" for v in offset), "CenterOfRotation = 0 0 0",
        "ElementSpacing = " + " ".join(f"{v:g}" for v in spacing),
        "DimSize = " + " ".join(str(v) for v in array.shape), f"ElementType = {element_type}",
        f"ElementDataFile = {raw.name}"]) + "\n")


@dataclass
class ImageModel:
    label: str
    directory_name: str  # under PHANTOM_DATA
    activity: str  # .mhd: relative activity concentration
    attenuation: str  # .mhd: values mapped to materials by a preset
    presets: dict  # name -> [(lo, hi, material)], value v in [lo, hi) -> material; others: outside (air)
    default_preset: str
    local_from_image: list  # 3x3, rows: phantom-frame axes in image coordinates (proper rotation)
    origin_activity_min: float  # phantom origin: centroid of voxels with activity >= this (the brain)
    source: str
    # GATE geometry as nested surface meshes in the image frame (phantom_gate, image_geometry
    # "mesh"): [{"name", "file", "material", "mother": None (outermost) or a name}]
    meshes: list | None = None

    def directory(self) -> Path:
        return PHANTOM_DATA / self.directory_name


IMAGE_MODELS = {
    "xcat_brain_perfusion": ImageModel(
        label="XCAT brain perfusion (Yuemeng)", directory_name="xcat_brain_perfusion",
        activity="activity.mhd", attenuation="attenuation.mhd",
        presets={
            # by tissue, from where each level lies (phantom_sources/xcat_brain_perfusion/README.md)
            "tissue": [(30.0, 100.0, "G4_AIR"), (100.0, 230.0, "G4_TISSUE_SOFT_ICRP"),
                       (230.0, 256.0, "G4_BONE_CORTICAL_ICRP")],
            # her GATE range table (puts the brain, level 209.3, in SpineBone)
            "yuemeng": [(10.0, 201.0, "G4_WATER"), (201.0, 256.0, "SpineBone")],
        },
        default_preset="tissue",
        # image +x superior -> phantom z, image +z anterior -> phantom y, image y -> phantom x
        local_from_image=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        origin_activity_min=8.0,  # brain (white and grey matter)
        source="Yuemeng Feng's BrainPET GATE example (XCAT perfusion), 141 x 134 x 186 x 1.5 mm"),
    "mesh50_xcat_head_perfusion": ImageModel(
        label="mesh50_XCAT head, brain perfusion (Auer)", directory_name="mesh50_xcat_head",
        activity="activity.mhd", attenuation="labels.mhd",
        presets={  # the published labels and materials (their GateMaterials.db; body = water)
            "mesh50": [(0.5, 1.5, "G4_WATER"), (1.5, 2.5, "G4_AIR"), (2.5, 3.5, "Brain"), (6.5, 7.5, "SpineBone")],
        },
        default_preset="mesh50",
        # image -y anterior, +z superior: 180 deg about z puts the face at phantom +y
        local_from_image=[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        origin_activity_min=200.0,  # white (230) and grey matter (920), striatum (~1012); background 100
        source="Auer et al. mesh50_XCAT (MIT; github.com/BenAuer2021/Mesh-based-Human-Phantom-for-Simulation): "
               "head meshes + 1 mm voxel labels, brain perfusion activity (ratio x 100) cut from the "
               "head-torso-abdomen map (dev/python/prepare_mesh50_xcat_head.py)",
        # the published meshes welded; brain and air cavity cut by the skeleton (they overlap it by
        # 7.3 and 1.6 mm3), dev/python/prepare_mesh50_xcat_head.py
        meshes=[{"name": "body", "file": "gate_body.stl", "material": "G4_WATER", "mother": None},
                {"name": "skeleton", "file": "gate_skeleton.stl", "material": "SpineBone", "mother": "body"},
                {"name": "brain", "file": "gate_brain.stl", "material": "Brain", "mother": "body"},
                {"name": "air_cavity", "file": "gate_air_cavity.stl", "material": "G4_AIR", "mother": "body"}]),
}


def is_image(phantom: str) -> bool:
    return phantom in IMAGE_MODELS


def image_material_table(phantom: str, opt: Options | None = None) -> list:
    m = IMAGE_MODELS[phantom]
    return m.presets[(opt.material_preset if opt and opt.material_preset else m.default_preset)]


@lru_cache(maxsize=4)
def _load_image(phantom: str, preset_key: str) -> dict:
    m = IMAGE_MODELS[phantom]
    act, sp, off = read_mhd(m.directory() / m.activity)
    atn, sp2, off2 = read_mhd(m.directory() / m.attenuation)
    if act.shape != atn.shape or not np.allclose(sp, sp2):
        raise ValueError(f"{phantom}: activity and attenuation grids differ")
    table = m.presets[preset_key]
    names = list(dict.fromkeys(["G4_AIR"] + [mat for _, _, mat in table]))  # label 0: air (outside and cavities)
    lab = np.zeros(atn.shape, np.uint8)
    for lo, hi, mat in table:
        lab[(atn >= lo) & (atn < hi)] = names.index(mat)
    M = np.asarray(m.local_from_image, float)
    idx = np.argwhere(act >= m.origin_activity_min)
    c = off + idx.mean(0) * sp  # image-frame origin of the phantom frame
    return {"activity": act.astype(np.float32), "labels": lab, "materials": names, "spacing": sp, "offset": off,
            "M": M, "origin": c, "shape": np.array(act.shape)}


def image_data(spec: "Spec") -> dict:
    m = IMAGE_MODELS[spec.phantom]
    return _load_image(spec.phantom, spec.options.material_preset or m.default_preset)


def image_to_local(img: dict, p: np.ndarray) -> np.ndarray:
    return (np.asarray(p, float) - img["origin"]) @ img["M"].T


def local_to_image(img: dict, q: np.ndarray) -> np.ndarray:
    return np.asarray(q, float) @ img["M"] + img["origin"]


def sample_image(spec: "Spec", world_points: np.ndarray):
    """Nearest-voxel (material label, activity) at world points; outside the image: (0, 0)."""
    img = image_data(spec)
    p = local_to_image(img, spec.pose.to_local(world_points))
    k = np.rint((p - img["offset"]) / img["spacing"]).astype(np.int64)
    ok = np.all((k >= 0) & (k < img["shape"]), axis=1)
    lab = np.zeros(len(p), np.uint8)
    act = np.zeros(len(p), np.float32)
    kk = k[ok]
    lab[ok] = img["labels"][kk[:, 0], kk[:, 1], kk[:, 2]]
    act[ok] = img["activity"][kk[:, 0], kk[:, 1], kk[:, 2]]
    return lab, act


# ======================================================================================
# Scene and voxeliser
# ======================================================================================
def scene(spec: Spec) -> list[dict]:
    """Parts with their world placement (centre, axis, rotation matrix of the part's frame)."""
    parts, _ = build(spec.phantom, spec.options)
    R = spec.pose.rotation()
    out = []
    for p in parts:
        d = asdict(p)
        d.update(world_center=spec.pose.to_world(np.array(p.center)).tolist(),
                 world_axis=(R @ np.asarray(p.axis, float)).tolist(), material=p.material,
                 label=ROLE_LABEL[p.role])
        out.append(d)
    return out


def grid_centers(n: int, voxel_mm: float) -> np.ndarray:
    return -n * voxel_mm / 2 + (np.arange(n) + 0.5) * voxel_mm


def _inside(part: Part, q: np.ndarray) -> np.ndarray:
    """Points q (N, 3) in the phantom frame inside the part (boundary included)."""
    d = q - np.asarray(part.center, float)
    if part.shape == "sphere":
        return np.einsum("ij,ij->i", d, d) <= part.radius ** 2
    a = np.asarray(part.axis, float)
    h = d @ a
    radial2 = np.einsum("ij,ij->i", d, d) - h * h
    ok = (np.abs(h) <= part.length / 2) & (radial2 <= part.radius ** 2)
    if part.shape == "tube":
        ok &= radial2 >= part.inner_radius ** 2
    return ok


def _part_bbox(part: Part) -> np.ndarray:
    """Phantom-frame bounding box corners (8, 3) of a part."""
    c, r = np.asarray(part.center, float), part.radius
    if part.shape == "sphere":
        ext = np.array([r, r, r])
    else:
        a = np.asarray(part.axis, float)
        ext = np.abs(a) * part.length / 2 + r * np.sqrt(np.clip(1 - a * a, 0, 1))
    return np.array([[c[0] + sx * ext[0], c[1] + sy * ext[1], c[2] + sz * ext[2]]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def voxelize(spec: Spec, n: int, voxel_mm: float) -> np.ndarray:
    """Label volume (n, n, n), uint8, indexed [x, y, z] on the centred grid."""
    parts, _ = build(spec.phantom, spec.options)
    labels = np.zeros((n, n, n), np.uint8)
    c = grid_centers(n, voxel_mm)
    for p in parts:
        box = spec.pose.to_world(_part_bbox(p))
        lo = np.searchsorted(c, box.min(0) - 1e-9)
        hi = np.searchsorted(c, box.max(0) + 1e-9, side="right")
        if np.any(hi <= lo):
            continue
        ix, iy, iz = (np.arange(lo[k], hi[k]) for k in range(3))
        gx, gy, gz = np.meshgrid(c[ix], c[iy], c[iz], indexing="ij")
        q = spec.pose.to_local(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1))
        sub = labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].reshape(-1)
        sub[_inside(p, q)] = ROLE_LABEL[p.role]
        labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = sub.reshape(gx.shape)
    return labels


def hot_mask(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, [ROLE_LABEL[r] for r in HOT_ROLES])


def _grid_slabs(n: int, voxel_mm: float, slab: int = 16):
    """World voxel centres of the centred grid, in z-slabs: (k0, k1, points (n*n*(k1-k0), 3))."""
    c = grid_centers(n, voxel_mm)
    for k0 in range(0, n, slab):
        k1 = min(n, k0 + slab)
        gx, gy, gz = np.meshgrid(c, c, c[k0:k1], indexing="ij")
        yield k0, k1, np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)


def material_volume(spec: Spec, n: int, voxel_mm: float) -> tuple[np.ndarray, list]:
    """Material label volume [x, y, z] (uint8) and the material names (label 0: outside, air)."""
    if is_image(spec.phantom):
        names = image_data(spec)["materials"]
        out = np.zeros((n, n, n), np.uint8)
        for k0, k1, pts in _grid_slabs(n, voxel_mm):
            out[:, :, k0:k1] = sample_image(spec, pts)[0].reshape(n, n, k1 - k0)
        return out, names
    roles = voxelize(spec, n, voxel_mm)
    names = ["G4_AIR"] + sorted({MATERIALS[m] for _, m in LABELS.values() if m})
    lut = np.array([0] + [names.index(MATERIALS[LABELS[k][1]]) for k in range(1, len(LABELS))], np.uint8)
    return lut[roles], names


def activity_volume(spec: Spec, n: int, voxel_mm: float) -> np.ndarray:
    """Relative activity concentration [x, y, z] (float32): 1 in the water of analytic
    phantoms; the activity image (nearest voxel) of image phantoms."""
    if is_image(spec.phantom):
        out = np.zeros((n, n, n), np.float32)
        for k0, k1, pts in _grid_slabs(n, voxel_mm):
            out[:, :, k0:k1] = sample_image(spec, pts)[1].reshape(n, n, k1 - k0)
        return out
    return hot_mask(voxelize(spec, n, voxel_mm)).astype(np.float32)


def read_stl(path) -> tuple[np.ndarray, np.ndarray]:
    """Binary STL -> (vertices (n, 3), faces (m, 3)), identical vertices merged."""
    data = Path(path).read_bytes()
    n = int(np.frombuffer(data, "<u4", count=1, offset=80)[0])
    rec = np.frombuffer(data, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
    verts, inv = np.unique(rec["v"].reshape(-1, 3), axis=0, return_inverse=True)
    return verts.astype(float), inv.reshape(-1, 3)


def image_meshes(spec: "Spec") -> list[dict]:
    """The model's GATE meshes (ImageModel.meshes) in the phantom frame (mm):
    [{"name", "material", "mother", "vertices", "faces"}]; [] for models without meshes."""
    m = IMAGE_MODELS[spec.phantom]
    if not m.meshes:
        return []
    img = image_data(spec)
    out = []
    for r in m.meshes:
        v, f = read_stl(m.directory() / r["file"])
        out.append({"name": r["name"], "material": r["material"], "mother": r["mother"],
                    "vertices": image_to_local(img, v), "faces": f})
    return out


def activity_fraction(spec: Spec, region: str = "all") -> float:
    """Fraction of the phantom's activity in `region`: "all" (1), or "brain" for image
    phantoms (voxels at or above the model's brain level, e.g. XCAT grey + white matter)."""
    if region == "all":
        return 1.0
    if region != "brain" or not is_image(spec.phantom):
        raise ValueError(f"region {region!r} is not defined for {spec.phantom}")
    act = image_data(spec)["activity"].astype(float)
    return float(act[act >= IMAGE_MODELS[spec.phantom].origin_activity_min].sum() / act.sum())


def save_sparse_mask(mask: np.ndarray, path, values: np.ndarray | None = None) -> None:
    """The sparse format of the digital phantoms: indices (3, N) int64, values, shape."""
    idx = np.stack(np.nonzero(mask)).astype(np.int64)
    vals = np.ones(idx.shape[1], np.float32) if values is None else values[mask].astype(np.float32)
    np.savez_compressed(path, indices=idx, values=vals, shape=np.array(mask.shape))


# ======================================================================================
# Placements, descriptive export and analytic ray path lengths
# ======================================================================================
def axis_rotation(axis) -> np.ndarray:
    """Rotation matrix taking +z to the unit vector `axis` (cylinders are built along z)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(z, a), float(z @ a)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1 / (1 + c))


def placements(spec: Spec, prefix: str = "phantom", mother: str = "world") -> list[dict]:
    """Geant4-style placement of every part: mother volume, translation in the mother's
    frame and active rotation (x_mother = R x_local + t). Parts outside the body are placed
    in `mother` with the pose applied; daughters in their parent's frame."""
    if is_image(spec.phantom):
        img = image_data(spec)
        size = (img["shape"] * img["spacing"]).tolist()
        centre_img = img["offset"] + (img["shape"] - 1) / 2 * img["spacing"]
        return [{"part": None, "name": f"{prefix}_image", "mother": mother, "image": True,
                 "translation": spec.pose.to_world(image_to_local(img, centre_img)).tolist(),
                 "rotation": (spec.pose.rotation() @ img["M"]).tolist(),
                 "solid": {"type": "box", "params": {"size": size}}}]
    parts, _ = build(spec.phantom, spec.options)
    by_name = {p.name: p for p in parts}
    R = spec.pose.rotation()
    out = []
    for p in parts:
        rp = axis_rotation(p.axis)
        if p.parent is None:
            t = spec.pose.to_world(np.asarray(p.center, float))
            r = R @ rp
            m = mother
        else:
            q = by_name[p.parent]
            rq = axis_rotation(q.axis)
            t = rq.T @ (np.asarray(p.center, float) - np.asarray(q.center, float))
            r = rq.T @ rp
            m = f"{prefix}_{p.parent}"
        if p.shape == "sphere":
            solid = {"type": "sphere", "params": {"rmin": 0.0, "rmax": p.radius}}
        else:
            solid = {"type": "tubs", "params": {"rmin": p.inner_radius, "rmax": p.radius, "dz": p.length / 2,
                                                "sphi": 0.0, "dphi": 2 * np.pi}}
        out.append({"part": p, "name": f"{prefix}_{p.name}", "mother": m, "translation": t.tolist(),
                    "rotation": r.tolist(), "solid": solid})
    return out


def describe(spec: Spec, prefix: str = "phantom") -> dict:
    """Self-contained analytic description: materials (density, element mass fractions),
    the volume tree (solid, material, mother, placement, relative activity concentration).
    The material at a point is that of the deepest volume containing it (Geant4 rule)."""
    pl = placements(spec, prefix)
    if is_image(spec.phantom):
        img, m = image_data(spec), IMAGE_MODELS[spec.phantom]
        v = pl[0]
        return {
            "schema": "qmirt.phantom/1",
            "units": {"length": "mm", "angle": "rad", "density": "g/cm3"},
            "conventions": {"frame": "scanner frame: z = helmet axis, +y = face window",
                            "placement": "x_mother = rotation @ x_image + translation (active); the image is centred "
                                         "on the box, voxel [i, j, k] at (i, j, k) * spacing - size / 2 + spacing / 2",
                            "material_rule": "material of the voxel containing the point; outside: world_material"},
            "spec": spec.to_dict(), "world_material": "G4_AIR",
            "materials": {n: ({"density_g_cm3": MATERIAL_DATA[n][0], "mass_fractions": MATERIAL_DATA[n][1]}
                              if n in MATERIAL_DATA else {"nist": n}) for n in img["materials"]},
            "volumes": [{"name": v["name"], "role": "image", "solid": v["solid"], "material": "G4_AIR",
                         "mother": v["mother"], "translation": v["translation"], "rotation": v["rotation"],
                         "image": {"shape": img["shape"].tolist(), "spacing_mm": img["spacing"].tolist(),
                                   "materials": img["materials"], "material_table": image_material_table(spec.phantom, spec.options),
                                   "labels_file": "image_labels.npy", "activity_file": "image_activity.npy",
                                   "source": m.source}}],
        }
    mats = sorted({x["part"].material for x in pl} | {"G4_AIR"})
    return {
        "schema": "qmirt.phantom/1",
        "units": {"length": "mm", "angle": "rad", "density": "g/cm3"},
        "conventions": {
            "frame": "scanner frame: z = helmet axis, +y = face window",
            "placement": "x_mother = rotation @ x_local + translation (active)",
            "tubs": "Geant4 G4Tubs: radii, half length dz along local z",
            "material_rule": "deepest volume containing the point; outside all volumes: world_material",
        },
        "spec": spec.to_dict(),
        "world_material": "G4_AIR",
        "materials": {m: {"density_g_cm3": MATERIAL_DATA[m][0], "mass_fractions": MATERIAL_DATA[m][1]} for m in mats},
        "volumes": [{"name": x["name"], "role": x["part"].role, "solid": x["solid"], "material": x["part"].material,
                     "mother": x["mother"], "translation": x["translation"], "rotation": x["rotation"],
                     "activity": 1.0 if x["part"].role in HOT_ROLES else 0.0} for x in pl],
    }


def materials_db_text(names) -> str:
    """GATE-format material database for those `names` with a composition in MATERIAL_DATA
    (as read by GATE and gammart; Geant4 NIST names without one are resolved by name)."""
    names = [n for n in names if n in MATERIAL_DATA]
    full = {"H": "Hydrogen", "C": "Carbon", "N": "Nitrogen", "O": "Oxygen", "Na": "Sodium", "Mg": "Magnesium",
            "P": "Phosphor", "S": "Sulfur", "Cl": "Chlorine", "Ar": "Argon", "K": "Potassium", "Ca": "Calcium"}
    used = sorted({e for n in names for e in MATERIAL_DATA[n][1]}, key=lambda e: ELEMENTS[e][0])
    lines = ["[Elements]"] + [f"{full[e]}: S= {e} ; Z= {ELEMENTS[e][0]}. ; A= {ELEMENTS[e][1]} g/mole" for e in used]
    lines += ["", "[Materials]"]
    for n in names:
        d, fr = MATERIAL_DATA[n]
        lines.append(f"{n}: d={d:g} g/cm3 ; n={len(fr)}")
        lines += [f"        +el: name={full[e]} ; f={w}" for e, w in fr.items()]
        lines.append("")
    return "\n".join(lines) + "\n"


def gammart_scene(spec: Spec, material_db: str | None = None, prefix: str = "phantom",
                  cell_material_file: str | None = None) -> dict:
    """The phantom as a gammart scene (schema gammart.scene/1, same placements as GATE)."""
    if is_image(spec.phantom):
        img, v = image_data(spec), placements(spec, prefix)[0]
        half = (img["shape"] * img["spacing"] / 2).tolist()
        grid = {"shape": img["shape"].tolist(), "lower": [-h for h in half], "upper": half,
                "materials": img["materials"], "cell_material": None, "cell_material_file": cell_material_file}
        vols = [{"name": v["name"], "solid": v["solid"], "material": "G4_AIR", "mother": v["mother"],
                 "translation": v["translation"], "rotation": v["rotation"], "sensitive": False, "det_base": None,
                 "grid": grid, "tags": {"role": "image"}}]
        return {"schema": "gammart.scene/1", "world_material": "G4_AIR", "material_db": material_db,
                "metadata": {"phantom_spec": spec.to_dict()}, "volumes": vols, "transmission_bounds": []}
    return {"schema": "gammart.scene/1", "world_material": "G4_AIR", "material_db": material_db,
            "metadata": {"phantom_spec": spec.to_dict()},
            "volumes": [{"name": x["name"], "solid": x["solid"], "material": x["part"].material, "mother": x["mother"],
                         "translation": x["translation"], "rotation": x["rotation"], "sensitive": False,
                         "det_base": None, "grid": None, "tags": {"role": x["part"].role}} for x in placements(spec, prefix)],
            "transmission_bounds": []}


def _crossings(part: Part, o: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Ray parameters where the ray o + t d (phantom frame) crosses the part's surfaces."""
    c = np.asarray(part.center, float)
    oc = o - c
    if part.shape == "sphere":
        b, cc = oc @ d, oc @ oc - part.radius ** 2
        disc = b * b - cc
        return np.array([]) if disc < 0 else np.array([-b - np.sqrt(disc), -b + np.sqrt(disc)])
    a = np.asarray(part.axis, float)
    ho, hd = oc @ a, d @ a
    out = []
    if abs(hd) > 1e-15:  # end planes
        out += [(part.length / 2 - ho) / hd, (-part.length / 2 - ho) / hd]
    po, pd = oc - ho * a, d - hd * a  # components across the axis
    A, B = pd @ pd, po @ pd
    for r in (part.radius, part.inner_radius):
        if r <= 0 or A < 1e-30:
            continue
        disc = B * B - A * (po @ po - r * r)
        if disc >= 0:
            out += [(-B - np.sqrt(disc)) / A, (-B + np.sqrt(disc)) / A]
    return np.array(out)


def path_lengths(spec: Spec, origins, directions, tmax=None) -> dict:
    """Exact path length (mm) in each material along rays from `origins` along unit
    `directions` (world frame, (N, 3)) up to `tmax` (scalar or (N,); default: through the
    whole phantom). Material at a point: last part in build order containing it (= deepest
    volume)."""
    if is_image(spec.phantom):
        return _image_path_lengths(spec, origins, directions, tmax)
    parts, _ = build(spec.phantom, spec.options)
    O = spec.pose.to_local(np.atleast_2d(origins))
    D = np.atleast_2d(directions) @ spec.pose.rotation()  # world -> phantom frame (orthonormal)
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    tm = np.full(len(O), np.inf) if tmax is None else np.broadcast_to(np.asarray(tmax, float), (len(O),))
    centers = np.array([p.center for p in parts], float)
    bound = np.array([np.linalg.norm(np.ptp(_part_bbox(p), axis=0)) / 2 for p in parts])
    names = sorted({p.material for p in parts})
    out = {m: np.zeros(len(O)) for m in names}
    for k in range(len(O)):
        o, d = O[k], D[k]
        rel = centers - o
        along = rel @ d
        miss = np.linalg.norm(rel - np.outer(along, d), axis=1) > bound
        cand = [p for p, m in zip(parts, miss) if not m]
        if not cand:
            continue
        ts = np.concatenate([_crossings(p, o, d) for p in cand] + [np.array([0.0])])
        ts = np.unique(np.clip(ts[np.isfinite(ts)], 0.0, tm[k]))
        if len(ts) < 2:
            continue
        mids = (ts[:-1] + ts[1:]) / 2
        pts = o + np.outer(mids, d)
        owner = np.full(len(mids), -1)
        for j, p in enumerate(cand):
            owner[_inside(p, pts)] = j
        seg = np.diff(ts)
        for j, p in enumerate(cand):
            out[p.material][k] += seg[owner == j].sum()
    return out


def _image_path_lengths(spec: Spec, origins, directions, tmax=None) -> dict:
    """Exact per-material path lengths through the voxel image (all voxel-plane crossings);
    air (label 0) is not reported."""
    img = image_data(spec)
    O = local_to_image(img, spec.pose.to_local(np.atleast_2d(origins)))
    D = (np.atleast_2d(directions) @ spec.pose.rotation()) @ img["M"]
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    tm = np.full(len(O), np.inf) if tmax is None else np.broadcast_to(np.asarray(tmax, float), (len(O),))
    lo = img["offset"] - img["spacing"] / 2
    hi = lo + img["shape"] * img["spacing"]
    out = {m: np.zeros(len(O)) for m in img["materials"][1:]}
    for k in range(len(O)):
        o, d = O[k], D[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_a, t_b = (lo - o) / d, (hi - o) / d
        t0 = np.nanmax(np.where(d != 0, np.minimum(t_a, t_b), -np.inf))
        t1 = np.nanmin(np.where(d != 0, np.maximum(t_a, t_b), np.inf))
        t0, t1 = max(t0, 0.0), min(t1, tm[k])
        if t1 <= t0:
            continue
        ts = [np.array([t0, t1])]
        for ax in range(3):
            if d[ax] == 0:
                continue
            planes = lo[ax] + np.arange(img["shape"][ax] + 1) * img["spacing"][ax]
            t = (planes - o[ax]) / d[ax]
            ts.append(t[(t > t0) & (t < t1)])
        ts = np.unique(np.concatenate(ts))
        mid = o + np.outer((ts[:-1] + ts[1:]) / 2, d)
        idx = np.clip(np.floor((mid - lo) / img["spacing"]).astype(int), 0, img["shape"] - 1)
        lab = img["labels"][idx[:, 0], idx[:, 1], idx[:, 2]]
        seg = np.diff(ts)
        for j, name in enumerate(img["materials"]):
            if j:
                out[name][k] += seg[lab == j].sum()
    return out


# ======================================================================================
# Page and clearance views of the same parts
# ======================================================================================
def page_data(phantom: str, opt: Options | None = None) -> dict:
    """What phantom_fit_brain_scanner.html draws, in the page's conventions (heights above
    the outer bottom for the body, above the inner floor for the inserts)."""
    opt = opt or Options()
    parts, meta = build(phantom, opt)
    zf = -meta["inner_height_mm"] / 2
    zb0 = zf - meta["base_mm"]  # outer bottom
    top = meta["inner_height_mm"] / 2 + meta["lid_mm"]
    rods, spheres, stems, plates, fittings, knobs = {}, [], {}, [], [], []
    for p in parts:
        x, y, z = p.center
        base_name = p.name.split("_seg")[0]  # pieces of one rod or stem (cut by the plates)
        z0, z1 = z - p.length / 2 - zf, z + p.length / 2 - zf
        if p.role == "rod":
            rods[base_name] = [x, y, 2 * p.radius]
        elif p.role == "sphere":
            spheres.append([x, y, 2 * p.radius])
        elif p.role == "stem":
            prev = stems.get(base_name)
            stems[base_name] = [x, y, min(z0, prev[2]) if prev else z0, max(z1, prev[3]) if prev else z1, 2 * p.radius]
        elif p.role == "plate":
            plates.append([p.radius, z - p.length / 2 - zf, z + p.length / 2 - zf])
        elif p.role.startswith("fitting") and abs(p.axis[2]) > 0.99:
            fittings.append([p.name, float(np.hypot(x, y)), float(np.degrees(np.arctan2(y, x))), 2 * p.radius,
                             z - p.length / 2 - top, z + p.length / 2 - top])
        elif p.role.startswith("fitting"):
            az = float(np.degrees(np.arctan2(p.axis[1], p.axis[0])))
            knobs.append([p.name, az, 2 * p.radius, p.length, z - zb0])
    return {**meta, "model": phantom, "rods": list(rods.values()), "spheres": spheres, "stems": list(stems.values()), "plates": plates,
            "fittings": fittings, "side_knobs": knobs, "model_version": model_version(phantom, opt)}


def clearance_solids(phantom: str, opt: Options | None = None) -> list[tuple]:
    """Annular solids (r_in, r_out, z0, z1) in the phantom frame: the body, and each lid
    fitting and side knob swept round the axis (valid for any rotation). Image phantoms:
    one disc per voxel layer along the phantom z, out to the farthest non-air voxel corner."""
    if is_image(phantom):
        img = image_data(Spec(phantom, Pose(), opt or Options()))
        idx = np.argwhere(img["labels"] > 0)
        q = image_to_local(img, img["offset"] + idx * img["spacing"])
        half_diag = float(np.linalg.norm(img["spacing"])) / 2
        layer = float(img["spacing"].min())
        kz = np.floor(q[:, 2] / layer).astype(int)
        r = np.hypot(q[:, 0], q[:, 1])
        out = []
        for k in np.unique(kz):
            out.append((0.0, float(r[kz == k].max() + half_diag), (k - 0.5) * layer, (k + 1.5) * layer))
        return out
    parts, meta = build(phantom, opt or Options())
    out = []
    for p in parts:
        if p.parent is not None:
            continue
        x, y, z = p.center
        if p.role == "body":
            out.append((0.0, p.radius, z - p.length / 2, z + p.length / 2))
        elif abs(p.axis[2]) > 0.99:
            rr = float(np.hypot(x, y))
            out.append((max(rr - p.radius, 0.0), rr + p.radius, z - p.length / 2, z + p.length / 2))
        else:
            rr = float(np.hypot(x, y))
            out.append((rr - p.length / 2, rr + p.length / 2, z - p.radius, z + p.radius))
    return out
