#!/usr/bin/env python3
"""Convert a GLB scene into Mitsuba XML plus referenced PLY meshes.

Outputs
-------
ply/<name>.ply         -- one mesh per GLB scene node / primitive
scene.xml              -- Mitsuba 2.1.0 XML referencing the exported PLY files

Usage
-----
    python glb_to_mitsuba.py input.glb
    python glb_to_mitsuba.py input.glb --scene MyScene.xml --ply-dir ply_out
"""

import argparse
import logging
import os
import re
import struct
from pathlib import Path

import numpy as np
import trimesh

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_MATERIALS = {
    "itu_concrete": "0.92, 0.88, 0.80",
    "itu_glass": "0.75, 0.90, 0.95",
    "itu_metal": "0.35, 0.38, 0.42",
    "itu_wood": "0.72, 0.50, 0.20",
    "itu_medium_dry_ground": "0.60, 0.55, 0.40",
}

# GLB assets are typically Y-up, while Sionna/Mitsuba scenes here assume Z-up.
_Y_UP_TO_Z_UP = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])


def write_ply(path: Path, verts, faces) -> None:
    """Write a triangle mesh to a binary-little-endian PLY file."""
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
    for face in faces:
        face_buf += struct.pack("<Biii", 3, int(face[0]), int(face[1]), int(face[2]))

    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(verts.tobytes())
        handle.write(bytes(face_buf))


def _slugify(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (text or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _material_id(*labels: str) -> str:
    joined = " ".join(label for label in labels if label).lower()
    if "glass" in joined:
        return "itu_glass"
    if "metal" in joined or "steel" in joined:
        return "itu_metal"
    if "wood" in joined or "timber" in joined:
        return "itu_wood"
    return "itu_concrete"


def _build_xml(shape_entries: list[dict]) -> str:
    bsdf_lines = "\n".join(
        f'  <bsdf type="diffuse" id="{material_id}">\n'
        f'    <rgb name="reflectance" value="{rgb}"/>\n'
        f'  </bsdf>'
        for material_id, rgb in _MATERIALS.items()
    )
    shape_lines = "\n".join(
        f'  <shape type="ply" id="{entry["sid"]}" name="{entry["sid"]}">\n'
        f'    <string name="filename" value="{entry["filename"]}"/>\n'
        f'    <boolean name="face_normals" value="true"/>\n'
        f'    <ref id="{entry["material_id"]}" name="bsdf"/>\n'
        f'  </shape>'
        for entry in shape_entries
    )
    return (
        '<scene version="2.1.0">\n\n'
        "  <!-- ITU radio materials — itu_ prefix → Sionna maps to EM properties -->\n"
        + bsdf_lines + "\n\n"
        + shape_lines + "\n\n"
        + "</scene>\n"
    )


def _iter_scene_meshes(scene: trimesh.Scene):
    counts: dict[str, int] = {}
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        source_geometry = scene.geometry[geometry_name]
        mesh = source_geometry.copy()
        mesh.apply_transform(transform)
        mesh.apply_transform(_Y_UP_TO_Z_UP)

        if not isinstance(mesh, trimesh.Trimesh):
            continue
        if mesh.faces is None or len(mesh.faces) == 0:
            continue

        mesh = mesh.copy()
        mesh.remove_unreferenced_vertices()
        visual_material = getattr(getattr(mesh.visual, "material", None), "name", "")

        base_name = _slugify(node_name or geometry_name, "mesh")
        counts[base_name] = counts.get(base_name, 0) + 1
        suffix = "" if counts[base_name] == 1 else f"_{counts[base_name]:03d}"
        yield {
            "name": f"{base_name}{suffix}",
            "mesh": mesh,
            "material_id": _material_id(node_name, geometry_name, visual_material),
        }


def convert(glb_path: str, scene_file: str, ply_dir: str) -> str:
    input_path = Path(glb_path)
    ply_dir_path = Path(ply_dir)
    scene_path = Path(scene_file)

    if not input_path.exists():
        raise FileNotFoundError(f"GLB file not found: {input_path}")

    ply_dir_path.mkdir(parents=True, exist_ok=True)
    scene_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s", input_path)
    loaded = trimesh.load(input_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        scene = loaded
    else:
        scene = trimesh.Scene()
        scene.add_geometry(loaded, node_name=_slugify(input_path.stem, "mesh"))

    shape_entries = []
    written = 0
    for item in _iter_scene_meshes(scene):
        mesh = item["mesh"]
        file_name = f"{item['name']}.ply"
        output_path = ply_dir_path / file_name
        xml_mesh_path = Path(os.path.relpath(output_path, start=scene_path.parent))

        write_ply(output_path, mesh.vertices, mesh.faces)
        shape_entries.append({
            "sid": item["name"],
            "filename": str(xml_mesh_path),
            "material_id": item["material_id"],
        })
        written += 1

    if not shape_entries:
        raise RuntimeError("No triangle meshes were found in the GLB scene.")

    xml = _build_xml(shape_entries)
    with open(scene_path, "w", encoding="utf-8") as handle:
        handle.write(xml)

    log.info("Wrote %d PLY files into %s", written, ply_dir_path)
    log.info("Wrote scene XML to %s", scene_path)
    return str(scene_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("glb", help="Input GLB file")
    parser.add_argument(
        "--scene",
        default="glb_scene.xml",
        help="Output Mitsuba XML file",
    )
    parser.add_argument(
        "--ply-dir",
        default="ply",
        help="Directory for exported PLY meshes",
    )
    args = parser.parse_args()
    convert(args.glb, args.scene, args.ply_dir)


if __name__ == "__main__":
    main()