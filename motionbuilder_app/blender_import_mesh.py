"""Blender headless helper: import a static mesh file and export vertices/faces NPZ."""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
import numpy as np


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _import(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    elif suffix == ".stl":
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.stl(filepath=str(path))
    elif suffix == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(path))
    else:
        raise RuntimeError(f"unsupported static mesh format: {suffix}")


def _triangulated_mesh_arrays() -> tuple[np.ndarray, np.ndarray]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        try:
            base = len(vertices)
            world = obj.matrix_world
            for v in mesh.vertices:
                p = world @ v.co
                vertices.append([float(p.x), float(p.y), float(p.z)])
            for poly in mesh.polygons:
                idx = list(poly.vertices)
                if len(idx) < 3:
                    continue
                for i in range(1, len(idx) - 1):
                    faces.append([base + idx[0], base + idx[i], base + idx[i + 1]])
        finally:
            eval_obj.to_mesh_clear()
    if not vertices or not faces:
        raise RuntimeError("no triangle mesh found in imported file")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def main() -> int:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("usage: blender --background --python blender_import_mesh.py -- input output")
    args = argv[argv.index("--") + 1 :]
    if len(args) != 2:
        raise SystemExit("usage: blender --background --python blender_import_mesh.py -- input output")
    source = Path(args[0]).resolve()
    output = Path(args[1]).resolve()
    _clear_scene()
    _import(source)
    vertices, faces = _triangulated_mesh_arrays()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=vertices, faces=faces)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
