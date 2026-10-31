# SPDX-FileCopyrightText: Copyright (c) 2026 MKA Project contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .config import BLENDER_EXE, PROJECT_ROOT
from .session import StudioSession
from .skin_frames import custom_binding_preview_dict

_BLENDER_SCRIPT = Path(__file__).resolve().parent / "scripts" / "blender_export_fbx.py"


def _find_blender_exe() -> Path:
    if BLENDER_EXE:
        path = Path(BLENDER_EXE)
        if path.is_file():
            return path
    candidates = [
        Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    found = shutil.which("blender")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "未找到 Blender。请安装 Blender 或设置环境变量 BLENDER_EXE 指向 blender.exe"
    )


def _write_export_bundle(session: StudioSession, bundle_dir: Path) -> Path:
    motion = session.motion
    if motion is None:
        raise RuntimeError("no motion to export")

    binding = session.remapped_binding
    preview = custom_binding_preview_dict(session)
    rest_verts = np.asarray(preview["vertices"], dtype=np.float32)
    lbs_idx = None
    lbs_w = None

    if binding.has_soma30_rig():
        from kimodo.rigging.soma30_retarget import retarget_world_motion

        parents = np.asarray(binding.soma30_joint_parents, dtype=np.int64)
        names = np.asarray([str(n) for n in (binding.soma30_joint_names or [])], dtype=np.str_)
        rest_joints = np.asarray(binding.soma30_joint_positions, dtype=np.float32)
        world = retarget_world_motion(binding, motion.posed_joints, motion.global_rot_mats)
        posed = world[:, :, :3, 3].astype(np.float32)
        rots = world[:, :, :3, :3].astype(np.float32)
        lbs_idx = binding.soma30_lbs_indices
        lbs_w = binding.soma30_lbs_weights
    else:
        from kimodo.skeleton import SOMASkeleton77

        from .skin_frames import _custom_binding_rest_pose_legacy

        rest_verts, rest_joints = _custom_binding_rest_pose_legacy(session)
        skel77 = SOMASkeleton77(load=True)
        parents = skel77.joint_parents.detach().cpu().numpy().astype(np.int64)
        names = np.asarray([str(n) for n, _ in skel77.bone_order_names_with_parents], dtype=np.str_)
        posed = motion.posed_joints.astype(np.float32)
        rots = motion.global_rot_mats.astype(np.float32)

    if posed.ndim == 4:
        posed = posed[0]
    if rots.ndim == 5:
        rots = rots[0]

    npz_path = bundle_dir / "export_bundle.npz"
    payload = {
        "vertices": rest_verts.astype(np.float32),
        "faces": binding.faces.astype(np.int64),
        "joints": rest_joints.astype(np.float32),
        "parents": parents,
        "names": names,
        "posed_joints": posed,
        "global_rot_mats": rots,
        "fps": np.array([motion.fps], dtype=np.float32),
    }
    if lbs_idx is not None and lbs_w is not None:
        payload["lbs_indices"] = lbs_idx
        payload["lbs_weights"] = lbs_w
    np.savez_compressed(npz_path, **payload)

    meta = {
        "fps": float(motion.fps),
        "num_frames": int(posed.shape[0]),
        "num_joints": int(posed.shape[1]),
        "output_fbx": str(bundle_dir / "kimodo_custom_anim.fbx"),
        "use_soma30_weights": binding.has_soma30_rig(),
    }
    (bundle_dir / "export_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return npz_path


def export_custom_mesh_fbx(session: StudioSession) -> Path:
    blender = _find_blender_exe()
    if not _BLENDER_SCRIPT.is_file():
        raise FileNotFoundError(f"Blender export script missing: {_BLENDER_SCRIPT}")

    bundle_dir = Path(tempfile.mkdtemp(prefix="kimodo_fbx_"))
    npz_path = _write_export_bundle(session, bundle_dir)
    out_fbx = bundle_dir / "kimodo_custom_anim.fbx"
    meta_path = bundle_dir / "export_meta.json"

    cmd = [
        str(blender),
        "--background",
        "--python",
        str(_BLENDER_SCRIPT),
        "--",
        str(npz_path),
        str(meta_path),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0 or not out_fbx.is_file():
        tail = (result.stderr or result.stdout or "")[-3000:]
        raise RuntimeError(f"Blender FBX export failed (exit {result.returncode}):\n{tail}")
    return out_fbx


def build_blender_command(session: StudioSession, bundle_dir: Path) -> list[str]:
    """Return Blender CLI argv for tests."""
    blender = _find_blender_exe()
    npz_path = _write_export_bundle(session, bundle_dir)
    meta_path = bundle_dir / "export_meta.json"
    return [
        str(blender),
        "--background",
        "--python",
        str(_BLENDER_SCRIPT),
        "--",
        str(npz_path),
        str(meta_path),
    ]
