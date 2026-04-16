#!/usr/bin/env python3
"""Convert otaniemi.osm → Mitsuba 2.1.0 XML scene for Sionna RT.

Outputs
-------
ply/ground.ply            — flat ground plane
ply/building_<id>.ply     — one extruded mesh per OSM building
OtaniemiScene.xml         — Sionna-compatible Mitsuba scene

Coordinate system
-----------------
Origin = bounding-box centre of the OSM file.
  +x → East, +y → North, +z → up  (metres)

Usage
-----
    python osm_to_mitsuba.py            # from the Project directory
    python osm_to_mitsuba.py --help     # see options
"""

import argparse
import logging
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ── defaults ──────────────────────────────────────────────────────────────────
OSM_FILE       = "otaniemi.osm"
PLY_DIR        = Path("ply")
SCENE_FILE     = "OtaniemiScene.xml"
DEFAULT_HEIGHT = 8.0    # metres — fallback when no height/levels tag present
LEVEL_HEIGHT   = 3.0    # metres per building:levels storey
GROUND_Z       = 0.0    # elevation of the ground plane

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── projection ────────────────────────────────────────────────────────────────
_EARTH_R = 6_371_000.0  # metres


def latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float):
    """Equirectangular projection centred at (lat0, lon0) → (x_east, y_north) in metres."""
    cos0 = math.cos(math.radians(lat0))
    x = (lon - lon0) * cos0 * math.pi / 180.0 * _EARTH_R
    y = (lat - lat0)         * math.pi / 180.0 * _EARTH_R
    return x, y


# ── PLY I/O ───────────────────────────────────────────────────────────────────

def write_ply(path: Path, verts, faces) -> None:
    """Write a triangle mesh to a binary-little-endian PLY file.

    Parameters
    ----------
    verts : array-like, shape (N, 3), float32
    faces : array-like, shape (M, 3), int32  (vertex indices)
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    face_buf = bytearray()
    for f in faces:
        face_buf += struct.pack("<Biii", 3, int(f[0]), int(f[1]), int(f[2]))

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(verts.tobytes())
        fh.write(bytes(face_buf))


# ── polygon geometry ──────────────────────────────────────────────────────────

def _signed_area(ring):
    """Signed area of a 2-D polygon (positive = CCW)."""
    n = len(ring)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += ring[i][0] * ring[j][1] - ring[j][0] * ring[i][1]
    return a * 0.5


def _ensure_ccw(ring):
    """Return ring in counter-clockwise order (copy if needed)."""
    return ring if _signed_area(ring) > 0 else ring[::-1]


def _point_in_triangle(p, a, b, c):
    """Return True if 2-D point p is strictly inside triangle abc."""
    def _cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    d1, d2, d3 = _cross(p, a, b), _cross(p, b, c), _cross(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def ear_clip(ring):
    """Ear-clipping triangulation of a simple polygon.

    Works for both convex and concave (non-self-intersecting) polygons.

    Parameters
    ----------
    ring : list of (x, y) — CCW polygon vertices

    Returns
    -------
    list of (i, j, k) index triples referencing *ring*
    """
    n = len(ring)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    idx = list(range(n))   # working index list
    tris = []
    guard = n * n          # iteration safety limit

    while len(idx) > 3 and guard > 0:
        guard -= 1
        found = False
        for pos in range(len(idx)):
            prev_pos = (pos - 1) % len(idx)
            next_pos = (pos + 1) % len(idx)
            ia, ib, ic = idx[prev_pos], idx[pos], idx[next_pos]
            a, b, c = ring[ia], ring[ib], ring[ic]

            # Ear must be a convex vertex (cross product > 0 for CCW ring)
            cross = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            if cross <= 0:
                continue

            # No other polygon vertex may lie inside the ear triangle
            ear = True
            for j_pos in range(len(idx)):
                if j_pos in (prev_pos, pos, next_pos):
                    continue
                if _point_in_triangle(ring[idx[j_pos]], a, b, c):
                    ear = False
                    break

            if ear:
                tris.append((ia, ib, ic))
                idx.pop(pos)
                found = True
                break

        if not found:
            # Degenerate / self-intersecting polygon — bail out
            break

    if len(idx) == 3:
        tris.append(tuple(idx))

    return tris


# ── 3-D building mesh ─────────────────────────────────────────────────────────

def extrude_building(ring_xy, height: float):
    """Create a closed building mesh (walls + floor + roof).

    Parameters
    ----------
    ring_xy : list of (x, y) — building footprint vertices (any winding)
    height  : float — building height in metres

    Returns
    -------
    verts : list of (x, y, z)
    faces : list of (i, j, k)
    """
    ring_xy = list(_ensure_ccw(ring_xy))
    n = len(ring_xy)

    # Bottom ring: indices 0 … n-1  (z = 0)
    # Top ring:    indices n … 2n-1 (z = height)
    verts = [(x, y, GROUND_Z)          for x, y in ring_xy]
    verts += [(x, y, GROUND_Z + height) for x, y in ring_xy]

    faces = []

    # Walls — each edge → 2 triangles (outward normal for CCW ring)
    for i in range(n):
        j = (i + 1) % n
        faces.append((i,     j,     j + n))
        faces.append((i,     j + n, i + n))

    # Floor (normal pointing down → reverse winding)
    for ia, ib, ic in ear_clip(ring_xy):
        faces.append((ia, ic, ib))

    # Roof (normal pointing up → keep CCW winding)
    for ia, ib, ic in ear_clip(ring_xy):
        faces.append((ia + n, ib + n, ic + n))

    return verts, faces


def make_ground_plane(min_x, min_y, max_x, max_y):
    """Simple rectangular ground quad (2 triangles, normal pointing up)."""
    verts = [
        (min_x, min_y, GROUND_Z),
        (max_x, min_y, GROUND_Z),
        (max_x, max_y, GROUND_Z),
        (min_x, max_y, GROUND_Z),
    ]
    faces = [(0, 1, 2), (0, 2, 3)]
    return verts, faces


# ── parse height tag ("10", "10 m", "10.5m" → float) ─────────────────────────

def _parse_height(s: str) -> float:
    """Parse an OSM height string to float metres; return 0.0 on failure."""
    if not s:
        return 0.0
    try:
        return float(s.split()[0].rstrip("m").rstrip("M"))
    except (ValueError, IndexError):
        return 0.0


# ── Mitsuba / Sionna XML generation ──────────────────────────────────────────

_MATERIALS = {
    "itu_concrete":          "0.92, 0.88, 0.80",
    "itu_glass":             "0.75, 0.90, 0.95",
    "itu_metal":             "0.35, 0.38, 0.42",
    "itu_wood":              "0.72, 0.50, 0.20",
    "itu_medium_dry_ground": "0.60, 0.55, 0.40",
}


def _build_xml(shape_entries: list) -> str:
    """Render the Mitsuba 2.1.0 scene XML."""
    bsdf_lines = "\n".join(
        f'  <bsdf type="diffuse" id="{mid}">\n'
        f'    <rgb name="reflectance" value="{rgb}"/>\n'
        f'  </bsdf>'
        for mid, rgb in _MATERIALS.items()
    )
    shape_lines = "\n".join(
        f'  <shape type="ply" id="{e["sid"]}" name="{e["sid"]}">\n'
        f'    <string name="filename" value="ply/{e["fname"]}"/>\n'
        f'    <boolean name="face_normals" value="true"/>\n'
        f'    <ref id="{e["mat"]}" name="bsdf"/>\n'
        f'  </shape>'
        for e in shape_entries
    )
    return (
        '<scene version="2.1.0">\n\n'
        "  <!-- ITU radio materials — itu_ prefix → Sionna maps to EM properties -->\n"
        + bsdf_lines + "\n\n"
        + shape_lines + "\n\n"
        "</scene>\n"
    )


# ── OSM parsing ───────────────────────────────────────────────────────────────

def parse_osm(osm_path: str):
    """Parse an OSM file and return building data.

    Returns
    -------
    nodes : dict[str, (lat, lon)]
    buildings : list[dict]  — each has keys: id, refs, height, mat_id
    origin : (lat0, lon0)
    bbox_xy : (min_x, min_y, max_x, max_y) in local metres
    """
    log.info("Parsing %s …", osm_path)
    tree = ET.parse(osm_path)
    root = tree.getroot()

    bounds = root.find("bounds")
    min_lat = float(bounds.get("minlat"))
    max_lat = float(bounds.get("maxlat"))
    min_lon = float(bounds.get("minlon"))
    max_lon = float(bounds.get("maxlon"))
    lat0 = (min_lat + max_lat) / 2
    lon0 = (min_lon + max_lon) / 2
    log.info("  Bounding box : lat [%.4f, %.4f]  lon [%.4f, %.4f]",
             min_lat, max_lat, min_lon, max_lon)
    log.info("  Scene origin : lat0=%.5f  lon0=%.5f", lat0, lon0)

    # ── nodes ──────────────────────────────────────────────────────────────
    nodes: dict = {}
    for node in root.iter("node"):
        nodes[node.get("id")] = (float(node.get("lat")), float(node.get("lon")))
    log.info("  Loaded %d nodes", len(nodes))

    # ── scene extents in local coords ──────────────────────────────────────
    corners = [
        latlon_to_xy(min_lat, min_lon, lat0, lon0),
        latlon_to_xy(min_lat, max_lon, lat0, lon0),
        latlon_to_xy(max_lat, min_lon, lat0, lon0),
        latlon_to_xy(max_lat, max_lon, lat0, lon0),
    ]
    min_x = min(c[0] for c in corners) - 5.0
    max_x = max(c[0] for c in corners) + 5.0
    min_y = min(c[1] for c in corners) - 5.0
    max_y = max(c[1] for c in corners) + 5.0
    scene_w = max_x - min_x
    scene_h = max_y - min_y
    log.info("  Scene extent : %.0f m (E–W) × %.0f m (N–S)", scene_w, scene_h)

    # ── cache all way node-refs (needed for multipolygon relation lookup) ──
    way_refs: dict = {}
    for way in root.iter("way"):
        way_refs[way.get("id")] = [nd.get("ref") for nd in way.iter("nd")]

    # ── building ways ──────────────────────────────────────────────────────
    buildings = []
    for way in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in way.iter("tag")}
        if "building" not in tags:
            continue

        refs = way_refs[way.get("id")][:]
        # OSM closed way: last node repeats first — drop it
        if len(refs) > 1 and refs[0] == refs[-1]:
            refs = refs[:-1]

        # Height
        height = _parse_height(tags.get("height", ""))
        if height <= 0:
            height = _parse_height(tags.get("building:levels", "")) * LEVEL_HEIGHT
        if height <= 0:
            height = DEFAULT_HEIGHT

        # Material
        mat_tag = tags.get("building:material", "").lower()
        if "glass" in mat_tag:
            mat_id = "itu_glass"
        elif "metal" in mat_tag or "steel" in mat_tag:
            mat_id = "itu_metal"
        elif "wood" in mat_tag or "timber" in mat_tag:
            mat_id = "itu_wood"
        else:
            mat_id = "itu_concrete"

        buildings.append({"id": way.get("id"), "refs": refs,
                          "height": height, "mat_id": mat_id})

    log.info("  Found %d building ways", len(buildings))

    # ── multipolygon building relations ────────────────────────────────────
    # Buildings like Aalto Undergraduate Centre (Kandidaattikeskus) carry
    # building=* on the relation only.  Assemble the outer ring from
    # "outer"-role member ways and extrude it.
    def _mat_from_tags(t):
        m = t.get("building:material", "").lower()
        if "glass" in m:
            return "itu_glass"
        if "metal" in m or "steel" in m:
            return "itu_metal"
        if "wood" in m or "timber" in m:
            return "itu_wood"
        return "itu_concrete"

    rel_count = 0
    for rel in root.iter("relation"):
        rel_tags = {t.get("k"): t.get("v") for t in rel.iter("tag")}
        if rel_tags.get("type") != "multipolygon" or "building" not in rel_tags:
            continue

        has_explicit_outer = any(
            m.get("type") == "way" and m.get("role") == "outer"
            for m in rel.iter("member")
        )
        outer_segs = []
        for member in rel.iter("member"):
            if member.get("type") != "way":
                continue
            role = member.get("role") or ""
            if has_explicit_outer and role != "outer":
                continue
            if not has_explicit_outer and role == "inner":
                continue
            wid_m = member.get("ref")
            if wid_m not in way_refs:
                continue
            seg = way_refs[wid_m][:]
            if len(seg) > 1 and seg[0] == seg[-1]:
                seg = seg[:-1]
            if len(seg) >= 2:
                outer_segs.append(seg)

        if not outer_segs:
            continue

        # Chain segments into one closed outer ring
        if len(outer_segs) == 1:
            outer_refs = outer_segs[0]
        else:
            ring = list(outer_segs[0])
            remaining = list(outer_segs[1:])
            changed = True
            while remaining and changed:
                changed = False
                for i, seg in enumerate(remaining):
                    if seg[0] == ring[-1]:
                        ring.extend(seg[1:]); remaining.pop(i); changed = True; break
                    elif seg[-1] == ring[-1]:
                        ring.extend(reversed(seg[:-1])); remaining.pop(i); changed = True; break
                    elif seg[0] == ring[0]:
                        ring[:0] = list(reversed(seg[1:])); remaining.pop(i); changed = True; break
                    elif seg[-1] == ring[0]:
                        ring[:0] = seg[:-1]; remaining.pop(i); changed = True; break
            for seg in remaining:
                ring.extend(seg)
            outer_refs = ring

        if len(outer_refs) < 3:
            continue

        height = _parse_height(rel_tags.get("height", ""))
        if height <= 0:
            height = _parse_height(rel_tags.get("building:levels", "")) * LEVEL_HEIGHT
        if height <= 0:
            height = DEFAULT_HEIGHT

        rel_id = f"rel_{rel.get('id')}"
        buildings.append({"id": rel_id, "refs": outer_refs,
                          "height": height, "mat_id": _mat_from_tags(rel_tags)})
        log.info("  Multipolygon building: %s  (h=%.1f m, %d nodes)",
                 rel_tags.get("name", rel_id), height, len(outer_refs))
        rel_count += 1

    log.info("  Found %d multipolygon building relations", rel_count)
    return nodes, buildings, (lat0, lon0), (min_x, min_y, max_x, max_y)


# ── main ──────────────────────────────────────────────────────────────────────

def convert(osm_file=OSM_FILE, ply_dir=PLY_DIR, scene_file=SCENE_FILE):
    ply_dir = Path(ply_dir)
    ply_dir.mkdir(parents=True, exist_ok=True)

    nodes, buildings, (lat0, lon0), (min_x, min_y, max_x, max_y) = parse_osm(osm_file)

    shape_entries = []

    # Ground plane
    gv, gf = make_ground_plane(min_x, min_y, max_x, max_y)
    write_ply(ply_dir / "ground.ply", gv, gf)
    shape_entries.append({"sid": "ground", "fname": "ground.ply",
                          "mat": "itu_medium_dry_ground"})
    log.info("Wrote ground.ply")

    # Buildings
    written = skipped = 0
    for bld in buildings:
        ring_xy = []
        ok = True
        for ref in bld["refs"]:
            if ref not in nodes:
                ok = False
                break
            lat, lon = nodes[ref]
            ring_xy.append(latlon_to_xy(lat, lon, lat0, lon0))
        if not ok or len(ring_xy) < 3:
            skipped += 1
            continue

        verts, faces = extrude_building(ring_xy, bld["height"])
        if not faces:
            skipped += 1
            continue

        sid   = f"building_{bld['id']}"
        fname = f"building_{bld['id']}.ply"
        write_ply(ply_dir / fname, verts, faces)
        shape_entries.append({"sid": sid, "fname": fname, "mat": bld["mat_id"]})
        written += 1

    log.info("Buildings: %d written, %d skipped", written, skipped)

    # Scene XML
    xml_str = _build_xml(shape_entries)
    with open(scene_file, "w", encoding="utf-8") as fh:
        fh.write(xml_str)
    log.info("Scene XML → %s  (%d shapes)", scene_file, len(shape_entries))
    log.info("Done.")
    return scene_file


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osm",   default=OSM_FILE,   help="Input .osm file")
    ap.add_argument("--ply",   default=str(PLY_DIR), help="Output PLY directory")
    ap.add_argument("--scene", default=SCENE_FILE, help="Output Mitsuba XML file")
    ap.add_argument("--default-height", type=float, default=DEFAULT_HEIGHT,
                    metavar="M", help="Fallback building height in metres")
    args = ap.parse_args()
    DEFAULT_HEIGHT = args.default_height
    convert(args.osm, args.ply, args.scene)
