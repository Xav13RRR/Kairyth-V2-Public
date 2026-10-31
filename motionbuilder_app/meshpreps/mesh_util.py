from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import trimesh

from .config import FACES_TARGET_COUNT


BLENDER_STATIC_FORMATS = {".fbx", ".glb", ".gltf", ".obj", ".ply", ".stl"}


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    source_path: Path


def normalize_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts = vertices.astype(np.float32)
    faces = faces.astype(np.int64)
    center = verts.mean(axis=0)
    verts[:, 0] -= center[0]
    verts[:, 2] -= center[2]
    min_y = verts[:, 1].min()
    verts[:, 1] -= min_y
    height = max(verts[:, 1].max() - verts[:, 1].min(), 1e-6)
    verts *= 1.7 / height
    return verts, faces


def load_mesh(path: Path) -> MeshData:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"无法加载三角网格: {path}")
    vertices, faces = normalize_mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    return MeshData(vertices=vertices, faces=faces, source_path=path)


def _load_with_trimesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = tuple(g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh))
        if not meshes:
            raise ValueError(f"no triangle mesh found in scene: {path}")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"cannot load triangle mesh: {path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"empty triangle mesh: {path}")
    return vertices, faces


def _load_with_blender(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from ..exporter import find_blender

    blender = find_blender()
    script = Path(__file__).resolve().parents[1] / "blender_import_mesh.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    with tempfile.TemporaryDirectory(prefix="motionbuilder_mesh_import_") as temp_dir:
        out_path = Path(temp_dir) / "mesh.npz"
        cmd = [str(blender), "--background", "--python", str(script), "--", str(path.resolve()), str(out_path)]
        kwargs = {
            "cwd": str(Path(__file__).resolve().parents[2]),
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        process = subprocess.run(cmd, **kwargs)
        if process.returncode != 0:
            output = (process.stdout + "\n" + process.stderr)[-4000:]
            raise RuntimeError(output.strip() or "Blender mesh import failed")
        if not out_path.is_file():
            raise RuntimeError("Blender did not write imported mesh data")
        with np.load(out_path, allow_pickle=False) as data:
            vertices = np.asarray(data["vertices"], dtype=np.float32)
            faces = np.asarray(data["faces"], dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"empty triangle mesh: {path}")
    return vertices, faces


def load_mesh(path: Path) -> MeshData:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    errors: list[str] = []
    try:
        raw_vertices, raw_faces = _load_with_trimesh(path)
    except Exception as exc:
        errors.append(f"trimesh: {exc}")
        if path.suffix.lower() not in BLENDER_STATIC_FORMATS:
            raise ValueError(f"cannot load triangle mesh: {path}") from exc
        try:
            raw_vertices, raw_faces = _load_with_blender(path)
        except Exception as blender_exc:
            errors.append(f"blender: {blender_exc}")
            raise ValueError("static model import failed: " + " | ".join(errors)) from blender_exc
    vertices, faces = normalize_mesh(raw_vertices, raw_faces)
    return MeshData(vertices=vertices, faces=faces, source_path=path)


def write_obj(mesh: MeshData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    tm.export(path)


def write_raw_data_npz(mesh: MeshData, npz_path: Path, target_count: int = FACES_TARGET_COUNT) -> None:
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    vertices = np.asarray(tm.vertices, dtype=np.float32)
    faces = np.asarray(tm.faces, dtype=np.int64)
    if faces.shape[0] > target_count:
        try:
            simplified = tm.simplify_quadric_decimation(target_count)
            vertices = np.asarray(simplified.vertices, dtype=np.float32)
            faces = np.asarray(simplified.faces, dtype=np.int64)
            tm = simplified
        except Exception:
            pass
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        vertices=vertices,
        vertex_normals=np.asarray(tm.vertex_normals, dtype=np.float32),
        faces=faces,
        face_normals=np.asarray(tm.face_normals, dtype=np.float32),
        joints=None,
        tails=None,
        skin=None,
        no_skin=None,
        parents=None,
        names=None,
        matrix_local=None,
        path=None,
        cls=None,
    )


def preview_mesh(mesh: MeshData, max_faces: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    if tm.faces.shape[0] > max_faces:
        try:
            tm = tm.simplify_quadric_decimation(max_faces)
        except Exception:
            pass
    return np.asarray(tm.vertices, dtype=np.float32), np.asarray(tm.faces, dtype=np.int64)
