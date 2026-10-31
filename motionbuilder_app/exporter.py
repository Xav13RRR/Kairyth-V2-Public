from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .meshpreps.binding_io import load_binding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLENDER_SCRIPT = Path(__file__).resolve().parent / "blender_export_fbx.py"
EXPORTER_LOGIC = Path(__file__).resolve()
FBX_EXPORT_IDENTITY_VERSION = 1


def _file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def _export_identity(source: Path, bind_pose_only: bool) -> dict:
    return {
        "version": FBX_EXPORT_IDENTITY_VERSION,
        "source": _file_identity(source),
        "script": _file_identity(BLENDER_SCRIPT),
        "packing_logic": _file_identity(EXPORTER_LOGIC),
        "bind_pose_only": bool(bind_pose_only),
    }


def _export_marker_path(output: Path) -> Path:
    return output.with_name(output.name + ".export.json")


def is_fbx_export_current(
    source_npz: str | Path, output_fbx: str | Path, *, bind_pose_only: bool = False
) -> bool:
    """Old/unmarked, changed or damaged exports are stale regardless of mtime."""
    source, output = Path(source_npz), Path(output_fbx)
    try:
        marker_path = _export_marker_path(output)
        # A legitimate marker is tiny; do not parse an arbitrary large sidecar.
        if not output.is_file() or output.stat().st_size < 64 or marker_path.stat().st_size > 16384:
            return False
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        return (isinstance(marker, dict)
                and marker.get("identity") == _export_identity(source, bind_pose_only)
                and marker.get("output") == _file_identity(output))
    except (OSError, ValueError, TypeError):
        return False


def find_blender() -> Path:
    configured = os.environ.get("MOTIONBUILDER_BLENDER", "").strip()
    if configured and Path(configured).is_file():
        return Path(configured)
    runtime = PROJECT_ROOT / "runtime" / "blender"
    direct = runtime / "blender.exe"
    if direct.is_file():
        return direct
    matches = list(runtime.glob("**/blender.exe")) if runtime.is_dir() else []
    if matches:
        return matches[0]
    raise FileNotFoundError("未找到项目内 Blender；请运行 install.ps1 完成便携 Blender 部署")


def _write_blender_bundle(source: Path, destination: Path, *, bind_pose_only: bool = False) -> dict:
    data = np.load(source, allow_pickle=True)
    if "world_motion" in data.files:
        world = np.asarray(data["world_motion"], dtype=np.float32)
        vertices = data["rest_vertices"]
        bind = data["target_bind"]
        parents = np.asarray(data["parents"], dtype=np.int64)
        names = np.asarray(data["names"], dtype=object)
        lbs_indices = np.asarray(data["lbs_indices"], dtype=np.int64)
        if "virtual_root_motion" in data.files:
            virtual_root = np.asarray(data["virtual_root_motion"], dtype=np.float32)
            reconstructed_world = np.einsum("fab,fjbc->fjac", virtual_root, world).astype(np.float32)
            root_bind = virtual_root[0].astype(np.float32)
            world = np.concatenate([virtual_root[:, None], reconstructed_world], axis=1)
            bind = np.concatenate([root_bind[None], np.asarray(bind, dtype=np.float32)], axis=0)
            parents = np.concatenate([np.asarray([-1], dtype=np.int64), np.where(parents >= 0, parents + 1, 0).astype(np.int64)])
            names = np.concatenate([np.asarray(["VirtualRoot"], dtype=object), names.astype(object)])
            lbs_indices = lbs_indices + 1
    else:
        binding = load_binding(source)
        world = binding.bind_matrices[None]
        vertices = binding.vertices
        bind = binding.bind_matrices
        np.savez_compressed(
            destination,
            vertices=vertices,
            faces=binding.faces,
            joints=bind[:, :3, 3],
            parents=binding.joint_parents,
            names=np.asarray(binding.joint_names, dtype=object),
            posed_joints=world[:, :, :3, 3],
            global_rot_mats=world[:, :, :3, :3],
            fps=np.asarray([30.0], dtype=np.float32),
            lbs_indices=binding.lbs_indices,
            lbs_weights=binding.lbs_weights,
            bind_matrices=bind,
        )
        return {"bind_pose_only": True}
    np.savez_compressed(
        destination,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(data["faces"], dtype=np.int64),
        joints=np.asarray(bind[:, :3, 3], dtype=np.float32),
        parents=parents,
        names=names,
        posed_joints=np.asarray(world[:, :, :3, 3], dtype=np.float32),
        global_rot_mats=np.asarray(world[:, :, :3, :3], dtype=np.float32),
        fps=np.asarray(data["fps"], dtype=np.float32),
        lbs_indices=lbs_indices,
        lbs_weights=np.asarray(data["lbs_weights"], dtype=np.float32),
        bind_matrices=np.asarray(bind, dtype=np.float32),
    )
    return {"bind_pose_only": bind_pose_only}


def export_fbx(source_npz: str | Path, output_fbx: str | Path, *, bind_pose_only: bool = False) -> Path:
    source = Path(source_npz).resolve()
    output = Path(output_fbx).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    blender = find_blender()
    identity = _export_identity(source, bind_pose_only)
    # Stage on the destination filesystem so replacement is atomic. A failed
    # Blender run must not truncate the last successful output or its marker.
    with tempfile.TemporaryDirectory(prefix="motionbuilder_fbx_", dir=output.parent) as temp_dir:
        temp = Path(temp_dir)
        bundle = temp / "export_bundle.npz"
        meta = _write_blender_bundle(source, bundle, bind_pose_only=bind_pose_only)
        staged_output = temp / "export.fbx"
        meta["output_fbx"] = str(staged_output)
        meta_path = temp / "export_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        process = subprocess.run(
            [str(blender), "--background", "--python", str(BLENDER_SCRIPT), "--", str(bundle), str(meta_path)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if process.returncode != 0:
            raise RuntimeError((process.stdout + "\n" + process.stderr)[-5000:])
        if not staged_output.is_file() or staged_output.stat().st_size < 64:
            raise RuntimeError("Blender 未生成有效 FBX")
        if _export_identity(source, bind_pose_only) != identity:
            raise RuntimeError("FBX source or export code changed during export; previous export was not replaced")
        marker = {"identity": identity, "output": _file_identity(staged_output)}
        staged_marker = temp / "export.json"
        staged_marker.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
        os.replace(staged_output, output)
        # If this second replacement fails, an old marker cannot validate the
        # new FBX because its output digest differs. The next request rebuilds.
        os.replace(staged_marker, _export_marker_path(output))
    return output


def export_binding_fbx(binding_npz: str | Path, *, bind_matrices_override: np.ndarray | None = None) -> Path:
    source = Path(binding_npz)
    if bind_matrices_override is not None:
        binding = load_binding(source)
        binding.bind_matrices = np.asarray(bind_matrices_override, dtype=np.float32)
        from .meshpreps.binding_io import save_binding
        temp_source = source.with_name("binding_export_override.npz")
        save_binding(temp_source, binding)
        source = temp_source
    return export_fbx(source, source.with_name("binding_rest.fbx"), bind_pose_only=True)
