# SPDX-FileCopyrightText: Copyright (c) 2026 MKA Project contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77

from kimodo.rigging.soma30_retarget import target_bind_matrices
from kimodo.viz.custom_skin import CustomMeshSkin
from kimodo.viz.soma30_mesh_skin import Soma30MeshSkin
from kimodo.viz.soma_skin import SOMASkin

from .session import MotionResult, StudioSession

_scene_cache_key: tuple[str, float] | None = None
_scene_cache_data: dict | None = None


def clear_scene_cache() -> None:
    global _scene_cache_key, _scene_cache_data
    _scene_cache_key = None
    _scene_cache_data = None


def _preview_dict(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray | None = None,
    parents: np.ndarray | None = None,
    names: list[str] | None = None,
    label: str,
    extra: dict | None = None,
) -> dict:
    out: dict = {
        "label": label,
        "vertices": vertices.astype(np.float32).tolist(),
        "faces": faces.astype(np.int64).tolist(),
        "n_verts": int(vertices.shape[0]),
        "n_faces": int(faces.shape[0]),
    }
    if joints is not None and parents is not None and names is not None:
        out["joints"] = joints.astype(np.float32).tolist()
        out["parents"] = parents.astype(np.int64).tolist()
        out["names"] = names
    if extra:
        out.update(extra)
    return out


def _ground_align_bind_pose(skel77: SOMASkeleton77, skel30: SOMASkeleton30) -> torch.Tensor:
    bind_pos = skel77.neutral_joints.clone()
    min_height = skel30.neutral_joints[:, 1].min().item()
    bind_pos[:, 1] -= min_height
    bind_pos[:, 1] += 0.02
    return bind_pos


def _build_default_skin(session: StudioSession) -> SOMASkin:
    skel30 = SOMASkeleton30(load=True)
    if session.model_bundle is not None:
        skel30 = skel30.to(session.model_bundle.model.device)
    return SOMASkin(skel30)


def _build_custom_skin_legacy(session: StudioSession) -> CustomMeshSkin:
    bundle = session.model_bundle
    if bundle is None:
        raise RuntimeError("model not loaded")
    skeleton = bundle.model.motion_rep.skeleton
    if not isinstance(skeleton, SOMASkeleton30):
        skel30 = SOMASkeleton30(load=True).to(bundle.model.device)
    else:
        skel30 = skeleton
    return CustomMeshSkin(skel30, session.remapped_binding)


def _build_custom_skin(session: StudioSession) -> Soma30MeshSkin | CustomMeshSkin:
    binding = session.remapped_binding
    if binding.has_soma30_rig():
        return Soma30MeshSkin(binding)
    return _build_custom_skin_legacy(session)


def default_soma_preview_dict(session: StudioSession) -> dict:
    skel30 = SOMASkeleton30(load=True)
    skel77 = SOMASkeleton77(load=True)
    skin = SOMASkin(skel30)

    bind_pos = _ground_align_bind_pose(skel77, skel30)
    bind_rot = torch.eye(3, device=bind_pos.device).repeat(bind_pos.shape[0], 1, 1)
    with torch.no_grad():
        display_verts = skin.skin(bind_rot[None], bind_pos[None], rot_is_global=True)[0]

    faces = skin.faces.detach().cpu().numpy().astype(np.int64)
    return _preview_dict(
        vertices=display_verts.detach().cpu().numpy(),
        faces=faces,
        label="default_soma",
        extra={"show_skeleton": False},
    )


def _custom_binding_rest_pose_legacy(session: StudioSession) -> tuple[np.ndarray, np.ndarray]:
    skel30 = SOMASkeleton30(load=True)
    skin = CustomMeshSkin(skel30, session.remapped_binding)
    skel77 = skin.skeleton_skin

    bind_pos = _ground_align_bind_pose(skel77, skel30)
    bind_rot = torch.eye(3, device=bind_pos.device).repeat(bind_pos.shape[0], 1, 1)
    with torch.no_grad():
        display_verts = skin.skin(bind_rot[None], bind_pos[None], rot_is_global=True)[0]
    return (
        display_verts.detach().cpu().numpy().astype(np.float32),
        bind_pos.detach().cpu().numpy().astype(np.float32),
    )


def custom_binding_preview_dict(session: StudioSession) -> dict:
    binding = session.remapped_binding
    if binding.has_soma30_rig():
        display_verts = Soma30MeshSkin(binding).rest_vertices()
        target_bind = target_bind_matrices(binding)
        joint_positions = target_bind[:, :3, 3].astype(np.float32)
        parents = np.asarray(binding.soma30_joint_parents, dtype=np.int64)
        names = list(binding.soma30_joint_names or [])
    else:
        display_verts, joint_positions = _custom_binding_rest_pose_legacy(session)
        skel77 = SOMASkeleton77(load=True)
        parents = skel77.joint_parents.detach().cpu().numpy().astype(np.int64)
        names = [n for n, _ in skel77.bone_order_names_with_parents]

    return _preview_dict(
        vertices=display_verts,
        faces=binding.faces,
        joints=joint_positions,
        parents=parents,
        names=names,
        label="custom_mesh",
        extra={
            "binding_path": str(session.binding_path),
            "show_skeleton": False,
        },
    )


def scene_dict(session: StudioSession) -> dict:
    global _scene_cache_key, _scene_cache_data

    binding_path = session.binding_path.resolve()
    cache_key = (str(binding_path), binding_path.stat().st_mtime)
    if _scene_cache_key == cache_key and _scene_cache_data is not None:
        data = dict(_scene_cache_data)
        data["custom_offset"] = session.custom_mesh_offset.astype(np.float32).tolist()
        return data

    data = {
        "default": default_soma_preview_dict(session),
        "custom": custom_binding_preview_dict(session),
        "custom_offset": session.custom_mesh_offset.astype(np.float32).tolist(),
    }
    _scene_cache_key = cache_key
    _scene_cache_data = data
    return data


def binding_preview_dict(session: StudioSession) -> dict:
    return custom_binding_preview_dict(session)


def _skin_frames_torch(
    skin,
    motion: MotionResult,
    start: int,
    end: int,
) -> list[list[list[float]]]:
    device = skin.bind_vertices.device
    posed = torch.from_numpy(motion.posed_joints).to(device=device, dtype=torch.float)
    rots = torch.from_numpy(motion.global_rot_mats).to(device=device, dtype=torch.float)

    n_frames = posed.shape[0]
    start = max(0, start)
    end = min(n_frames, end)
    if start >= end:
        return []

    out: list[list[list[float]]] = []
    for fi in range(start, end):
        verts = skin.skin(rots[fi : fi + 1], posed[fi : fi + 1], rot_is_global=True)
        arr = verts[0].detach().cpu().numpy().astype(np.float32)
        out.append(arr.tolist())
    return out


def skin_default_frames(
    session: StudioSession,
    motion: MotionResult,
    start: int,
    end: int,
) -> list[list[list[float]]]:
    return _skin_frames_torch(_build_default_skin(session), motion, start, end)


def skin_custom_frames(
    session: StudioSession,
    motion: MotionResult,
    start: int,
    end: int,
) -> list[list[list[float]]]:
    skin = _build_custom_skin(session)
    if isinstance(skin, Soma30MeshSkin):
        frames = skin.skin_frames(motion.posed_joints, motion.global_rot_mats, start, end)
        return [f.tolist() for f in frames]
    return _skin_frames_torch(skin, motion, start, end)


def skin_motion_frames(
    session: StudioSession,
    motion: MotionResult,
    start: int,
    end: int,
    *,
    target: str = "custom",
) -> list[list[list[float]]]:
    if target == "default":
        return skin_default_frames(session, motion, start, end)
    if target == "custom":
        return skin_custom_frames(session, motion, start, end)
    raise ValueError(f"unknown skin target: {target}")


def default_motion_joint_frames(
    motion: MotionResult,
    start: int,
    end: int,
) -> list[list[list[float]]]:
    posed = np.asarray(motion.posed_joints, dtype=np.float32)
    if posed.ndim == 4:
        posed = posed[0]
    n_frames = posed.shape[0]
    start = max(0, start)
    end = min(n_frames, end)
    if start >= end:
        return []
    return [posed[fi].tolist() for fi in range(start, end)]


def custom_motion_joint_frames(
    session: StudioSession,
    motion: MotionResult,
    start: int,
    end: int,
) -> list[list[list[float]]]:
    binding = session.remapped_binding
    if binding.has_soma30_rig():
        skin = Soma30MeshSkin(binding)
        joints = skin.retargeted_joint_positions(motion.posed_joints, motion.global_rot_mats)
        n_frames = joints.shape[0]
        start = max(0, start)
        end = min(n_frames, end)
        return [joints[fi].tolist() for fi in range(start, end)]

    posed = np.asarray(motion.posed_joints, dtype=np.float32)
    if posed.ndim == 4:
        posed = posed[0]
    n_frames = posed.shape[0]
    start = max(0, start)
    end = min(n_frames, end)
    return [posed[fi].tolist() for fi in range(start, end)]


def motion_joint_frames(
    session: StudioSession,
    motion: MotionResult,
    start: int,
    end: int,
    *,
    target: str = "custom",
) -> list[list[list[float]]]:
    if target == "default":
        return default_motion_joint_frames(motion, start, end)
    if target == "custom":
        return custom_motion_joint_frames(session, motion, start, end)
    raise ValueError(f"unknown joint target: {target}")


def custom_retargeted_joints(session: StudioSession, motion: MotionResult) -> np.ndarray | None:
    binding = session.remapped_binding
    if not binding.has_soma30_rig():
        return None
    skin = Soma30MeshSkin(binding)
    return skin.retargeted_joint_positions(motion.posed_joints, motion.global_rot_mats)
