# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77

from .types import RemappedSOMABinding


@dataclass
class Soma30BoneMapping:
    """Per-bone rest alignment entry (source == target name for SOMA-30)."""

    name: str
    source_rest: np.ndarray
    target_rest: np.ndarray
    align_offset: np.ndarray


def _normalize_bind_mats(bind: np.ndarray) -> np.ndarray:
    mats = np.asarray(bind, dtype=np.float64)
    if mats.ndim == 2 and mats.shape == (4, 4):
        mats = mats.reshape(1, 4, 4)
    elif mats.ndim == 2 and mats.shape[1] == 16:
        mats = mats.reshape(-1, 4, 4)
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"bind matrices must be [J, 4, 4], got {mats.shape}")
    return mats


def _rot_pos_to_mat4(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot
    m[:3, 3] = pos
    return m.astype(np.float32)


def _world_mats_from_rot_pos(rots: np.ndarray, pos: np.ndarray) -> np.ndarray:
    n_frames = pos.shape[0]
    n_joints = pos.shape[1]
    out = np.zeros((n_frames, n_joints, 4, 4), dtype=np.float32)
    for fi in range(n_frames):
        for j in range(n_joints):
            out[fi, j] = _rot_pos_to_mat4(rots[fi, j], pos[fi, j])
    return out


def _joint_positions_from_world(world: np.ndarray) -> np.ndarray:
    return np.asarray(world[..., :3, 3], dtype=np.float32)


def _topo_order(parents: np.ndarray) -> list[int]:
    n = len(parents)
    depth = np.zeros(n, dtype=np.int64)
    for j in range(n):
        d = 0
        p = int(parents[j])
        while p >= 0:
            d += 1
            p = int(parents[p])
        depth[j] = d
    return list(np.argsort(depth))


def _local_from_world(world: np.ndarray, parents: np.ndarray) -> np.ndarray:
    n = world.shape[0]
    local = np.zeros_like(world)
    for j in range(n):
        p = int(parents[j])
        if p < 0:
            local[j] = world[j]
        else:
            local[j] = np.linalg.inv(world[p]) @ world[j]
    return local


def _world_from_local(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    n = local.shape[0]
    world = np.zeros_like(local)
    order = _topo_order(parents)
    for j in order:
        p = int(parents[j])
        world[j] = local[j] if p < 0 else world[p] @ local[j]
    return world


def source_bind_matrices(binding: RemappedSOMABinding | None = None) -> np.ndarray:
    """Kimodo SOMA-30 neutral bind pose as full world 4x4 matrices."""
    if binding is not None and binding.soma30_source_rest_bind is not None:
        return _normalize_bind_mats(binding.soma30_source_rest_bind).astype(np.float32)

    skel30 = SOMASkeleton30(load=True)
    device = torch.device("cpu")
    local_rots = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(skel30.nbjoints, 1, 1)
    root_pos = skel30.neutral_joints[skel30.root_idx].clone().to(device)
    with torch.no_grad():
        global_rots, posed, _ = skel30.fk(local_rots.unsqueeze(0), root_pos.unsqueeze(0))
    n = skel30.nbjoints
    world = np.zeros((n, 4, 4), dtype=np.float32)
    gr = global_rots[0].cpu().numpy()
    pj = posed[0].cpu().numpy()
    for j in range(n):
        world[j] = _rot_pos_to_mat4(gr[j], pj[j])
    return world


def target_bind_matrices(binding: RemappedSOMABinding) -> np.ndarray:
    if not binding.has_soma30_rig():
        raise ValueError("binding has no SOMA-30 target rig")
    return _normalize_bind_mats(binding.soma30_bind_matrices).astype(np.float32)


def align_offsets_for_binding(binding: RemappedSOMABinding) -> np.ndarray:
    n = target_bind_matrices(binding).shape[0]
    if binding.soma30_align_offsets is None:
        return np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    offsets = _normalize_bind_mats(binding.soma30_align_offsets)
    if offsets.shape[0] != n:
        raise ValueError(f"align_offsets joint count {offsets.shape[0]} != {n}")
    return offsets.astype(np.float32)


def effective_target_bind(binding: RemappedSOMABinding) -> np.ndarray:
    """Target bind with user rest-pose alignment offsets applied."""
    target = target_bind_matrices(binding).astype(np.float64)
    offsets = align_offsets_for_binding(binding).astype(np.float64)
    return (target @ offsets).astype(np.float32)


def build_soma30_bone_mappings(binding: RemappedSOMABinding) -> list[Soma30BoneMapping]:
    source = source_bind_matrices(binding)
    target = target_bind_matrices(binding)
    offsets = align_offsets_for_binding(binding)
    names = list(binding.soma30_joint_names or [])
    if len(names) != source.shape[0]:
        names = [f"joint_{i}" for i in range(source.shape[0])]
    return [
        Soma30BoneMapping(
            name=names[j],
            source_rest=source[j].copy(),
            target_rest=target[j].copy(),
            align_offset=offsets[j].copy(),
        )
        for j in range(source.shape[0])
    ]


def compute_align_offsets(base_target_bind: np.ndarray, adjusted_target_bind: np.ndarray) -> np.ndarray:
    """Compute per-bone offsets: adjusted = base @ offset."""
    base = _normalize_bind_mats(base_target_bind)
    adjusted = _normalize_bind_mats(adjusted_target_bind)
    if base.shape != adjusted.shape:
        raise ValueError(f"bind shape mismatch {base.shape} vs {adjusted.shape}")
    offsets = np.zeros_like(base)
    for j in range(base.shape[0]):
        offsets[j] = np.linalg.inv(base[j]) @ adjusted[j]
    return offsets.astype(np.float32)


def extract_source_world_mats(
    posed_joints: np.ndarray,
    global_rot_mats: np.ndarray,
) -> np.ndarray:
    """Extract SOMA-30 joint world 4x4 matrices from Kimodo motion."""
    skel30 = SOMASkeleton30(load=True)
    skel77 = SOMASkeleton77(load=True)
    skel_slice = list(skel30.get_skel_slice(skel77))

    posed = np.asarray(posed_joints, dtype=np.float64)
    rots = np.asarray(global_rot_mats, dtype=np.float64)
    if posed.ndim == 4:
        posed = posed[0]
    if rots.ndim == 5:
        rots = rots[0]

    n_frames = posed.shape[0]
    n30 = len(skel_slice)
    if posed.shape[1] == n30:
        return _world_mats_from_rot_pos(rots, posed)

    out_rots = np.zeros((n_frames, n30, 3, 3), dtype=np.float64)
    out_pos = np.zeros((n_frames, n30, 3), dtype=np.float64)
    for j30, j77 in enumerate(skel_slice):
        out_rots[:, j30] = rots[:, j77]
        out_pos[:, j30] = posed[:, j77]
    return _world_mats_from_rot_pos(out_rots, out_pos)


def _compose_local_rot_trans(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot
    m[:3, 3] = trans
    return m.astype(np.float32)


def _local_rot_delta(bind_local: np.ndarray, anim_local: np.ndarray) -> np.ndarray:
    r_bind = bind_local[:3, :3].astype(np.float64)
    r_anim = anim_local[:3, :3].astype(np.float64)
    return (np.linalg.inv(r_bind) @ r_anim).astype(np.float64)


def retarget_world_matrices_proportional(
    source_anim_world: np.ndarray,
    source_bind: np.ndarray,
    target_bind: np.ndarray,
    parents: np.ndarray,
    root_idx: int,
) -> np.ndarray:
    """Retarget with fixed target bone lengths: root translation delta + local rotation delta FK."""
    source_bind = _normalize_bind_mats(source_bind).astype(np.float64)
    target_bind = _normalize_bind_mats(target_bind).astype(np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    n_frames = source_anim_world.shape[0]
    n_joints = source_bind.shape[0]
    src_bind_local = _local_from_world(source_bind, parents).astype(np.float64)
    tgt_bind_local = _local_from_world(target_bind, parents).astype(np.float64)
    src_root_ref = source_bind[root_idx, :3, 3].copy()
    tgt_root_base = target_bind[root_idx, :3, 3].copy()

    out = np.zeros((n_frames, n_joints, 4, 4), dtype=np.float32)
    order = _topo_order(parents)
    for fi in range(n_frames):
        src_anim_local = _local_from_world(
            source_anim_world[fi].astype(np.float64),
            parents,
        )
        tgt_local = np.zeros((n_joints, 4, 4), dtype=np.float64)
        root_delta = source_anim_world[fi, root_idx, :3, 3].astype(np.float64) - src_root_ref
        for j in order:
            r_delta = _local_rot_delta(src_bind_local[j], src_anim_local[j])
            r_tgt = r_delta @ tgt_bind_local[j, :3, :3]
            if j == root_idx:
                pos = tgt_root_base + root_delta
                tgt_local[j] = _compose_local_rot_trans(r_tgt, pos)
            else:
                tgt_local[j] = _compose_local_rot_trans(r_tgt, tgt_bind_local[j, :3, 3])
        out[fi] = _world_from_local(tgt_local, parents)
    return out


def _resolve_root_idx(binding: RemappedSOMABinding) -> int:
    names = list(binding.soma30_joint_names or [])
    if "Hips" in names:
        return names.index("Hips")
    skel30 = SOMASkeleton30(load=True)
    return int(skel30.root_idx)


def retarget_world_matrices(
    source_anim_world: np.ndarray,
    source_bind: np.ndarray,
    target_bind: np.ndarray,
    parents: np.ndarray,
    root_idx: int,
) -> np.ndarray:
    return retarget_world_matrices_proportional(
        source_anim_world,
        source_bind,
        target_bind,
        parents,
        root_idx,
    )


def world_to_skin_matrices(world: np.ndarray, target_bind: np.ndarray) -> np.ndarray:
    target_bind = _normalize_bind_mats(target_bind)
    target_bind_inv = np.linalg.inv(target_bind)
    return (world.astype(np.float64) @ target_bind_inv[None]).astype(np.float32)


def retarget_skin_matrices(
    source_world: np.ndarray,
    source_bind: np.ndarray,
    target_bind: np.ndarray,
    parents: np.ndarray,
    root_idx: int,
) -> np.ndarray:
    world = retarget_world_matrices_proportional(
        source_world,
        source_bind,
        target_bind,
        parents,
        root_idx,
    )
    return world_to_skin_matrices(world, target_bind)


def retarget_motion_to_target_bind(
    binding: RemappedSOMABinding,
    posed_joints: np.ndarray,
    global_rot_mats: np.ndarray,
) -> np.ndarray:
    """Return per-frame SOMA-30 skin matrices [F, J, 4, 4] for target mesh bind."""
    if not binding.has_soma30_rig():
        raise ValueError("binding has no SOMA-30 target rig")
    target_bind = target_bind_matrices(binding)
    source_bind = source_bind_matrices(binding)
    if source_bind.shape[0] != target_bind.shape[0]:
        raise ValueError(
            f"SOMA-30 joint count mismatch: source {source_bind.shape[0]} vs target {target_bind.shape[0]}"
        )
    parents = np.asarray(binding.soma30_joint_parents, dtype=np.int64)
    root_idx = _resolve_root_idx(binding)
    source_world = extract_source_world_mats(posed_joints, global_rot_mats)
    return retarget_skin_matrices(source_world, source_bind, target_bind, parents, root_idx)


def retarget_world_motion(
    binding: RemappedSOMABinding,
    posed_joints: np.ndarray,
    global_rot_mats: np.ndarray,
) -> np.ndarray:
    """Return retargeted world 4x4 matrices [F, J, 4, 4]."""
    target_bind = target_bind_matrices(binding)
    source_bind = source_bind_matrices(binding)
    parents = np.asarray(binding.soma30_joint_parents, dtype=np.int64)
    root_idx = _resolve_root_idx(binding)
    source_world = extract_source_world_mats(posed_joints, global_rot_mats)
    return retarget_world_matrices_proportional(
        source_world,
        source_bind,
        target_bind,
        parents,
        root_idx,
    )


def retargeted_joint_world_positions(
    binding: RemappedSOMABinding,
    posed_joints: np.ndarray,
    global_rot_mats: np.ndarray,
) -> np.ndarray:
    """World joint positions [F, J, 3] after retarget."""
    world = retarget_world_motion(binding, posed_joints, global_rot_mats)
    return _joint_positions_from_world(world)


def retarget_rest_skin_matrices(binding: RemappedSOMABinding) -> np.ndarray:
    """Rest-pose skin matrices: identity at target bind."""
    if not binding.has_soma30_rig():
        raise ValueError("binding has no SOMA-30 target rig")
    n = target_bind_matrices(binding).shape[0]
    return np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))


def source_rest_preview_dict() -> dict:
    """Kimodo SOMA-30 neutral skeleton for meshpreps rest-align source overlay."""
    skel30 = SOMASkeleton30(load=True)
    bind = source_bind_matrices()
    names = [n for n, _ in skel30.bone_order_names_with_parents]
    parents = np.array(
        [-1 if p is None else names.index(p) for _, p in skel30.bone_order_names_with_parents],
        dtype=np.int64,
    )
    joints = bind[:, :3, 3].astype(np.float32)
    return {
        "joints": joints.tolist(),
        "parents": parents.tolist(),
        "names": names,
        "bind_matrices": [m.reshape(-1).tolist() for m in bind],
    }
