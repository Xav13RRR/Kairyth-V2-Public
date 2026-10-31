# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np

from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77

from .humanoid_template import build_bind_matrices
from .soma_retarget_profile import (
    ALIGN_LANDMARKS,
    PRIMARY_SOMA_JOINTS,
    SOMA_LARM_CHAIN,
    SOMA_LLEG_CHAIN,
    SOMA_NECK_CHAIN,
    SOMA_RARM_CHAIN,
    SOMA_RLEG_CHAIN,
    SOMA_SKIN_JOINTS,
    SOMA_SPINE_CHAIN,
    SOMA_WEIGHT_EXCLUDE,
    UNIRIG_TO_SOMA,
)
from .types import RemappedSOMABinding, UniRigBinding

# Re-export for backward compatibility
__all__ = ["UNIRIG_TO_SOMA", "remap_to_soma"]
MAX_INFLUENCES = 5
_GROUND_OFFSET = 0.02
# When reposing the captured mesh into the SOMA rest, drop an influencing bone if
# its reposed vertex position lands further than this (SOMA units, ~metres) from
# the dominant bone's. This prunes the weight that UniRig leaks onto limb bones
# in the captured A-pose, which would otherwise shred the torso in the T-pose.
_REPOSE_REJECT_DIST = 0.12


def _load_soma_bind() -> tuple[np.ndarray, list[str]]:
    skel = SOMASkeleton77(load=True)
    skin_path = Path(skel.folder).parent / "somaskel77" / "skin_standard.npz"
    data = np.load(skin_path)
    bind = np.array(data["bind_rig_transform"], dtype=np.float64)
    names = [str(n) for n in data["rig_joint_names"]]
    return bind, names


def _build_rest_transform(soma_names: list[str]) -> np.ndarray:
    """Match Character custom_skin init: ground-aligned SOMA neutral + identity rotation."""
    skel77 = SOMASkeleton77(load=True)
    skel30 = SOMASkeleton30(load=True)
    neutral = skel77.neutral_joints.detach().cpu().numpy().astype(np.float64)
    min_height = float(skel30.neutral_joints[:, 1].min().item())
    rest_pos = neutral.copy()
    rest_pos[:, 1] -= min_height
    rest_pos[:, 1] += _GROUND_OFFSET

    n_joints = len(soma_names)
    rest = np.tile(np.eye(4, dtype=np.float64)[None], (n_joints, 1, 1))
    rest[:, :3, 3] = rest_pos[:n_joints]
    return rest


def _load_soma_skin_meta() -> tuple[np.ndarray, list[str], np.ndarray]:
    bind, names = _load_soma_bind()
    skin_path = Path(SOMASkeleton77(load=True).folder).parent / "somaskel77" / "skin_standard.npz"
    data = np.load(skin_path, allow_pickle=True)
    parents = _parents_from_connections(len(names), np.asarray(data["rig_joint_connections"]))
    return bind, names, parents


def _parents_from_connections(n_joints: int, connections: np.ndarray) -> np.ndarray:
    parents = np.full(n_joints, -1, dtype=np.int64)
    for row in connections:
        p, c = int(row[0]), int(row[1])
        if 0 <= c < n_joints:
            parents[c] = p
    return parents


def _nearest_skin_ancestor(
    joint_idx: int,
    soma_names: list[str],
    soma_parents: np.ndarray,
) -> int:
    j = int(joint_idx)
    seen: set[int] = set()
    while j >= 0 and j not in seen:
        seen.add(j)
        if soma_names[j] in SOMA_SKIN_JOINTS:
            return j
        j = int(soma_parents[j])
    return 0


def _soma_name_for_unirig_bone(unirig_name: str) -> str | None:
    return UNIRIG_TO_SOMA.get(unirig_name)


def _apply_name_overrides(
    joint_names: list[str],
    lut: list[int],
    bone_map: dict[str, str],
    bone_map_modes: dict[str, str],
    soma_name_to_idx: dict[str, int],
    rest_pos: np.ndarray,
    target_pos: np.ndarray | None,
) -> None:
    for j, name in enumerate(joint_names):
        soma_name = _soma_name_for_unirig_bone(name)
        if soma_name is None or soma_name not in soma_name_to_idx:
            continue
        lut[j] = soma_name_to_idx[soma_name]
        bone_map[name] = soma_name
        bone_map_modes[name] = "name"
        if target_pos is not None:
            target_pos[j] = rest_pos[lut[j]]


def _resolve_chain_fallbacks(
    joint_names: list[str],
    lut: list[int],
    bone_map: dict[str, str],
    bone_map_modes: dict[str, str],
    soma_names: list[str],
    parents: np.ndarray,
    hips_idx: int,
) -> None:
    n = len(joint_names)
    root_candidates = np.where(parents < 0)[0]
    root_idx = int(root_candidates[0]) if root_candidates.size else 0

    for _ in range(n):
        changed = False
        for j in range(n):
            name = joint_names[j]
            if bone_map_modes.get(name) == "name":
                continue
            p = int(parents[j])
            if lut[j] == hips_idx and j != root_idx and p >= 0 and lut[p] != hips_idx:
                lut[j] = lut[p]
                bone_map[name] = soma_names[lut[j]]
                bone_map_modes[name] = "chain-fallback"
                changed = True
            elif not bone_map_modes.get(name):
                bone_map_modes[name] = "chain"
        if not changed:
            break


def _build_bone_map(
    joint_names: list[str],
    soma_name_to_idx: dict[str, int],
) -> tuple[list[int], dict[str, str]]:
    """Per UniRig joint index -> SOMA bind matrix index."""
    lut: list[int] = []
    bone_map: dict[str, str] = {}
    for name in joint_names:
        soma_name = _soma_name_for_unirig_bone(name)
        if soma_name is None:
            soma_name = "Hips"
        bone_map[name] = soma_name
        lut.append(soma_name_to_idx[soma_name])
    return lut, bone_map


def _count_recognized_names(joint_names: list[str]) -> int:
    return sum(1 for n in joint_names if n in UNIRIG_TO_SOMA)


def _build_hybrid_mapping(
    binding: UniRigBinding,
    soma_names: list[str],
    soma_name_to_idx: dict[str, int],
    rest_pos: np.ndarray,
) -> tuple[list[int], dict[str, str], dict[str, str], np.ndarray, np.ndarray, np.ndarray]:
    """Chain topology + name overrides + parent fallback (always used)."""
    aligned_vertices, aligned_bind, aligned_joints = _similarity_align(
        binding, rest_pos, soma_name_to_idx
    )
    target_pos: np.ndarray | None = None
    bone_map_modes: dict[str, str] = {}

    structural = _structural_bone_map(
        binding, soma_names, soma_name_to_idx, rest_pos, aligned_joints
    )
    if structural is not None:
        lut, bone_map, target_pos = structural
        for name in bone_map:
            bone_map_modes[name] = "chain"
    else:
        lut, bone_map, aligned_vertices, aligned_bind = _spatial_correspondence(
            binding, rest_pos, soma_names, soma_name_to_idx
        )
        for name in bone_map:
            bone_map_modes[name] = "spatial"

    _apply_name_overrides(
        binding.joint_names,
        lut,
        bone_map,
        bone_map_modes,
        soma_name_to_idx,
        rest_pos,
        target_pos,
    )
    _resolve_chain_fallbacks(
        binding.joint_names,
        lut,
        bone_map,
        bone_map_modes,
        soma_names,
        np.asarray(binding.joint_parents, dtype=np.int64),
        soma_name_to_idx.get("Hips", 0),
    )

    aligned_vertices, aligned_bind = _align_to_soma_skeleton(binding, rest_pos, soma_name_to_idx)

    if target_pos is None:
        target_pos = rest_pos[np.asarray(lut, dtype=np.int64)].copy()
    else:
        for j, name in enumerate(binding.joint_names):
            if bone_map_modes.get(name) in ("name", "chain-fallback"):
                target_pos[j] = rest_pos[lut[j]]

    return lut, bone_map, bone_map_modes, target_pos, aligned_vertices, aligned_bind


def _similarity_align(
    binding: UniRigBinding,
    rest_pos: np.ndarray,
    soma_name_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hip-align + height-scale UniRig into SOMA rest space (no rotation)."""
    joints = binding.joint_positions.astype(np.float64)
    parents = np.asarray(binding.joint_parents, dtype=np.int64)

    root_candidates = np.where(parents < 0)[0]
    root_idx = int(root_candidates[0]) if root_candidates.size else int(np.argmin(joints[:, 1]))

    u_h = float(joints[:, 1].max() - joints[:, 1].min())
    s_h = float(rest_pos[:, 1].max() - rest_pos[:, 1].min())
    scale = s_h / max(u_h, 1e-6)

    hips_idx = soma_name_to_idx.get("Hips", 0)
    offset = rest_pos[hips_idx] - scale * joints[root_idx]

    aligned_joints = scale * joints + offset
    aligned_vertices = (scale * binding.vertices.astype(np.float64) + offset).astype(np.float32)
    aligned_bind = build_bind_matrices(aligned_joints.astype(np.float32), parents).astype(np.float64)
    return aligned_vertices, aligned_bind, aligned_joints


def _path_to_root(parents: np.ndarray, leaf: int) -> list[int]:
    """Joint indices from root down to `leaf` (inclusive)."""
    path: list[int] = []
    seen: set[int] = set()
    j = int(leaf)
    while j >= 0 and j not in seen:
        seen.add(j)
        path.append(j)
        j = int(parents[j])
    return path[::-1]


def _lca(parents: np.ndarray, a: int, b: int) -> int:
    """Lowest common ancestor of two joints."""
    anc = set(_path_to_root(parents, a))
    j = int(b)
    while j >= 0:
        if j in anc:
            return j
        j = int(parents[j])
    return -1


def _assign_chain(
    uni_path: list[int],
    soma_chain_idx: list[int],
    lut: list[int],
    assigned: list[bool],
) -> None:
    """Proportionally map a UniRig joint chain onto a SOMA joint chain."""
    if not uni_path or not soma_chain_idx:
        return
    length = len(uni_path)
    m = len(soma_chain_idx)
    for i, uj in enumerate(uni_path):
        t = (i / (length - 1)) if length > 1 else 0.0
        sj = soma_chain_idx[min(m - 1, int(round(t * (m - 1))))]
        lut[uj] = sj
        assigned[uj] = True


def _polyline_point(pts: np.ndarray, f: float) -> np.ndarray:
    """Point at normalized arc-length fraction `f` along an open polyline."""
    if len(pts) == 1:
        return pts[0]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-9:
        return pts[0]
    target = float(np.clip(f, 0.0, 1.0)) * total
    acc = 0.0
    for i, length in enumerate(seg):
        if acc + length >= target or i == len(seg) - 1:
            t = (target - acc) / max(float(length), 1e-9)
            return pts[i] + (pts[i + 1] - pts[i]) * float(np.clip(t, 0.0, 1.0))
        acc += float(length)
    return pts[-1]


def _structural_bone_map(
    binding: UniRigBinding,
    soma_names: list[str],
    soma_name_to_idx: dict[str, int],
    rest_pos: np.ndarray,
    aligned_joints: np.ndarray,
) -> tuple[list[int], dict[str, str], np.ndarray] | None:
    """Topology-based UniRig->SOMA mapping (pose-independent).

    Detects spine, head, both arms and both legs from the skeleton tree and maps
    each chain to the corresponding SOMA chain. Returns ``(lut, bone_map,
    target_pos)`` where ``lut`` is the per-bone SOMA index (for weight remapping)
    and ``target_pos`` is each bone's continuous destination along the SOMA limb
    (for geometry), interpolated so bone directions never collapse. Unlike spatial
    nearest-neighbour, an A-pose arm maps to the SOMA arm even though the SOMA arm
    is in a T-pose far from the captured arm position.
    """
    parents = np.asarray(binding.joint_parents, dtype=np.int64)
    joints = binding.joint_positions.astype(np.float64)
    n = len(parents)

    root_candidates = np.where(parents < 0)[0]
    if root_candidates.size == 0:
        return None
    root = int(root_candidates[0])

    children: dict[int, list[int]] = {}
    for j in range(n):
        p = int(parents[j])
        if p >= 0:
            children.setdefault(p, []).append(j)
    leaves = [j for j in range(n) if j not in children]
    if len(leaves) < 4:
        return None

    rx = float(joints[root, 0])
    feet = sorted(leaves, key=lambda j: joints[j, 1])[:2]
    head = max(leaves, key=lambda j: joints[j, 1])
    others = [j for j in leaves if j not in feet and j != head]
    if len(others) < 2:
        return None
    hands = sorted(others, key=lambda j: abs(joints[j, 0] - rx))[-2:]

    def _left_right(a: int, b: int) -> tuple[int, int]:
        # SOMA convention: character-left is +X.
        return (a, b) if (joints[a, 0] - rx) >= (joints[b, 0] - rx) else (b, a)

    l_foot, r_foot = _left_right(feet[0], feet[1])
    l_hand, r_hand = _left_right(hands[0], hands[1])

    chest = _lca(parents, l_hand, r_hand)
    if chest < 0:
        chest = _lca(parents, l_hand, head)
    if chest < 0:
        return None

    def _idx(chain: list[str]) -> list[int]:
        return [soma_name_to_idx[c] for c in chain if c in soma_name_to_idx]

    lut = [soma_name_to_idx.get("Hips", 0)] * n
    assigned = [False] * n

    _assign_chain(_path_to_root(parents, chest), _idx(SOMA_SPINE_CHAIN), lut, assigned)

    def _suffix(path: list[int], anchor: int) -> list[int]:
        return path[path.index(anchor) + 1 :] if anchor in path else path

    _assign_chain(_suffix(_path_to_root(parents, head), chest), _idx(SOMA_NECK_CHAIN), lut, assigned)
    _assign_chain(_suffix(_path_to_root(parents, l_hand), chest), _idx(SOMA_LARM_CHAIN), lut, assigned)
    _assign_chain(_suffix(_path_to_root(parents, r_hand), chest), _idx(SOMA_RARM_CHAIN), lut, assigned)
    _assign_chain(_suffix(_path_to_root(parents, l_foot), root), _idx(SOMA_LLEG_CHAIN), lut, assigned)
    _assign_chain(_suffix(_path_to_root(parents, r_foot), root), _idx(SOMA_RLEG_CHAIN), lut, assigned)

    lut[root] = soma_name_to_idx.get("Hips", 0)
    assigned[root] = True

    for _ in range(n):
        changed = False
        for j in range(n):
            if assigned[j]:
                continue
            p = int(parents[j])
            if p >= 0 and assigned[p]:
                lut[j] = lut[p]
                assigned[j] = True
                changed = True
        if not changed:
            break

    # Continuous per-bone target positions along the SOMA limbs (drives geometry).
    target_pos = np.tile(rest_pos[soma_name_to_idx.get("Hips", 0)], (n, 1)).astype(np.float64)
    target_set = [False] * n

    def _poly(chain: list[str]) -> np.ndarray:
        return np.stack([rest_pos[soma_name_to_idx[c]] for c in chain if c in soma_name_to_idx])

    def _assign_target(uni_path: list[int], chain: list[str]) -> None:
        poly = _poly(chain)
        length = len(uni_path)
        for i, uj in enumerate(uni_path):
            f = (i / (length - 1)) if length > 1 else 0.0
            target_pos[uj] = _polyline_point(poly, f)
            target_set[uj] = True

    def _branch(path: list[int], anchor: int) -> list[int]:
        return path[path.index(anchor) :] if anchor in path else path

    _assign_target(_path_to_root(parents, chest), SOMA_SPINE_CHAIN)
    _assign_target(_branch(_path_to_root(parents, head), chest), ["Chest", *SOMA_NECK_CHAIN])
    _assign_target(_branch(_path_to_root(parents, l_hand), chest), ["Chest", *SOMA_LARM_CHAIN])
    _assign_target(_branch(_path_to_root(parents, r_hand), chest), ["Chest", *SOMA_RARM_CHAIN])
    _assign_target(_branch(_path_to_root(parents, l_foot), root), ["Hips", *SOMA_LLEG_CHAIN])
    _assign_target(_branch(_path_to_root(parents, r_foot), root), ["Hips", *SOMA_RLEG_CHAIN])

    target_pos[root] = rest_pos[soma_name_to_idx.get("Hips", 0)]
    target_set[root] = True
    for _ in range(n):
        changed = False
        for j in range(n):
            if target_set[j]:
                continue
            p = int(parents[j])
            if p >= 0 and target_set[p]:
                target_pos[j] = target_pos[p] + (aligned_joints[j] - aligned_joints[p])
                target_set[j] = True
                changed = True
        if not changed:
            break

    bone_map: dict[str, str] = {}
    for j in range(n):
        name = binding.joint_names[j] if j < len(binding.joint_names) else f"bone_{j}"
        bone_map[name] = soma_names[lut[j]]
    return lut, bone_map, target_pos


def _spatial_correspondence(
    binding: UniRigBinding,
    rest_pos: np.ndarray,
    soma_names: list[str],
    soma_name_to_idx: dict[str, int],
) -> tuple[list[int], dict[str, str], np.ndarray, np.ndarray]:
    """Fallback: hip-align + height-scale UniRig, then nearest SOMA primary joint."""
    aligned_vertices, aligned_bind, aligned_joints = _similarity_align(
        binding, rest_pos, soma_name_to_idx
    )
    primary_idx = [soma_name_to_idx[n] for n in PRIMARY_SOMA_JOINTS if n in soma_name_to_idx]
    primary_pos = rest_pos[primary_idx]

    lut: list[int] = []
    bone_map: dict[str, str] = {}
    for ji, jp in enumerate(aligned_joints):
        d = np.linalg.norm(primary_pos - jp[None, :], axis=1)
        soma_idx = int(primary_idx[int(np.argmin(d))])
        lut.append(soma_idx)
        name = binding.joint_names[ji] if ji < len(binding.joint_names) else f"bone_{ji}"
        bone_map[name] = soma_names[soma_idx]

    return lut, bone_map, aligned_vertices, aligned_bind


def _kabsch_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R, t such that dst approx R @ src + t (no scale)."""
    src_cent = src.mean(axis=0)
    dst_cent = dst.mean(axis=0)
    src_c = src - src_cent
    dst_c = dst - dst_cent
    cov = src_c.T @ dst_c
    u, _, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = dst_cent - (r @ src_cent)
    return r, t


def _align_to_soma_skeleton(
    binding: UniRigBinding,
    rest_pos: np.ndarray,
    soma_name_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly align UniRig mesh/joints to ground-aligned SOMA rest landmarks."""
    name_to_idx = {name: i for i, name in enumerate(binding.joint_names)}
    src_pts: list[np.ndarray] = []
    dst_pts: list[np.ndarray] = []
    for unirig_name, soma_name in ALIGN_LANDMARKS:
        if unirig_name not in name_to_idx or soma_name not in soma_name_to_idx:
            continue
        src_pts.append(binding.joint_positions[name_to_idx[unirig_name]])
        dst_pts.append(rest_pos[soma_name_to_idx[soma_name]])

    if len(src_pts) < 3:
        return binding.vertices.copy(), binding.bind_matrices.copy()

    src = np.stack(src_pts, axis=0).astype(np.float64)
    dst = np.stack(dst_pts, axis=0).astype(np.float64)
    r, t = _kabsch_rigid(src, dst)
    aligned_vertices = (binding.vertices.astype(np.float64) @ r.T) + t
    aligned_joints = (binding.joint_positions.astype(np.float64) @ r.T) + t
    aligned_bind = build_bind_matrices(aligned_joints.astype(np.float32), binding.joint_parents)
    return aligned_vertices.astype(np.float32), aligned_bind.astype(np.float64)


def _axis_frame(direction: np.ndarray) -> np.ndarray:
    """Orthonormal bone frame whose primary (X) axis follows `direction`.

    Roll is resolved with a fixed world-up reference so source and target frames
    are constructed identically and the relative rotation has no spurious twist.
    """
    n = float(np.linalg.norm(direction))
    if n < 1e-8:
        return np.eye(3)
    x = direction / n
    ref = np.array([0.0, 1.0, 0.0]) if abs(x[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
    y = np.cross(ref, x)
    ny = float(np.linalg.norm(y))
    if ny < 1e-8:
        return np.eye(3)
    y /= ny
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def _compute_rest_target(
    vertices: np.ndarray,
    lbs_indices: np.ndarray,
    lbs_weights: np.ndarray,
    unirig_bind: np.ndarray,
    target_pos: np.ndarray,
    joint_parents: np.ndarray,
) -> np.ndarray:
    """Repose the captured UniRig mesh into the SOMA rest (T-pose).

    Per-bone *similarity* retarget. ``target_pos`` holds, per UniRig bone, its
    destination position in SOMA rest space (continuous along each limb, so bone
    directions never collapse). For every bone ``b`` with parent ``p`` we build the
    similarity transform (rotation + uniform scale + translation) that maps the
    captured bone segment ``[joint_p, joint_b]`` onto the SOMA target segment
    ``[target_p, target_b]``. Because the transform maps ``joint_b -> target_b`` and
    a child's transform maps that same ``joint_b -> target_b`` too, neighbouring
    bone transforms agree exactly at every shared joint (no LBS tearing), while
    anchoring directly on SOMA target positions avoids any limb-length drift.
    """
    n_bones = unirig_bind.shape[0]
    # build_bind_matrices stores the forward joint transform ([I | joint]),
    # so the joint position is the translation column directly.
    joints = unirig_bind[:, :3, 3]
    target = np.asarray(target_pos, dtype=np.float64)
    parents = np.asarray(joint_parents, dtype=np.int64)

    # Topological (parent-before-child) order so parents resolve first.
    children: dict[int, list[int]] = {}
    roots: list[int] = []
    for j in range(n_bones):
        p = int(parents[j])
        (children.setdefault(p, []).append(j) if p >= 0 else roots.append(j))
    order: list[int] = []
    stack = list(roots)
    while stack:
        b = stack.pop()
        order.append(b)
        stack.extend(children.get(b, []))

    # Forward-kinematics target skeleton: each bone keeps its *captured* length
    # (so limbs/head are never stretched into spikes) but points along the SOMA
    # target direction. Child anchors derive from the parent so neighbouring bone
    # transforms agree exactly at every shared joint (no LBS tearing). Anchoring
    # on the FK chain rather than raw target positions keeps natural proportions
    # and avoids the funnel that limb-length forcing produces.
    t_fk = np.array(target_pos, dtype=np.float64)
    rot = np.zeros((n_bones, 3, 3), dtype=np.float64)
    for b in order:
        p = int(parents[b])
        if p < 0:
            rot[b] = np.eye(3)
            t_fk[b] = target[b]
            continue
        d_u = joints[b] - joints[p]
        d_t = target[b] - target[p]
        n_u = float(np.linalg.norm(d_u))
        n_t = float(np.linalg.norm(d_t))
        if n_u < 1e-9 or n_t < 1e-9:
            rot[b] = rot[p]
            t_fk[b] = t_fk[p] + (rot[b] @ d_u if n_u > 1e-9 else np.zeros(3))
            continue
        rot[b] = _axis_frame(d_t) @ _axis_frame(d_u).T
        t_fk[b] = t_fk[p] + rot[b] @ d_u

    to_rest = np.zeros((n_bones, 4, 4), dtype=np.float64)
    for b in range(n_bones):
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = rot[b]
        mat[:3, 3] = t_fk[b] - rot[b] @ joints[b]
        to_rest[b] = mat

    n_verts = vertices.shape[0]
    v_h = np.concatenate([vertices.astype(np.float64), np.ones((n_verts, 1))], axis=1)

    # Candidate reposed position from every influencing bone.
    n_k = lbs_weights.shape[1]
    pos_k = np.empty((n_verts, n_k, 3), dtype=np.float64)
    for k in range(n_k):
        pos_k[:, k] = np.einsum("nij,nj->ni", to_rest[lbs_indices[:, k].astype(np.int64)], v_h)[:, :3]

    # UniRig predicts the captured A-pose, where limbs sit close to the torso, so
    # it leaks small weights from torso/shoulder/hip vertices onto limb bones that
    # rotate ~90 degrees away in the SOMA T-pose. Linearly blending those produces
    # the shredding/candy-wrapper. Reject the influences whose reposed position
    # disagrees with the dominant bone, then blend only the agreeing ones.
    w = lbs_weights.astype(np.float64)
    dom_k = np.argmax(w, axis=1)
    p_dom = pos_k[np.arange(n_verts), dom_k]
    dist = np.linalg.norm(pos_k - p_dom[:, None, :], axis=2)
    keep = dist <= _REPOSE_REJECT_DIST
    w_eff = w * keep
    w_sum = w_eff.sum(axis=1, keepdims=True)
    w_sum[w_sum < 1e-12] = 1.0
    w_eff /= w_sum

    return (w_eff[:, :, None] * pos_k).sum(axis=1).astype(np.float32)


def _solve_bind_vertices(
    rest_target: np.ndarray,
    lbs_indices: np.ndarray,
    lbs_weights: np.ndarray,
    soma_bind: np.ndarray,
    rest_transform: np.ndarray,
) -> np.ndarray:
    """Inverse SOMA rest LBS: find bind-space verts for the remapped SOMA weights."""
    soma_bind_inv = np.linalg.inv(soma_bind)
    affine = rest_transform @ soma_bind_inv

    n_verts = rest_target.shape[0]
    bind_vertices = np.zeros((n_verts, 3), dtype=np.float32)

    gathered = affine[lbs_indices]
    rot = np.einsum("nw,nwij->nij", lbs_weights, gathered[:, :, :3, :3])
    trans = np.einsum("nw,nwi->ni", lbs_weights, gathered[:, :, :3, 3])
    rhs = rest_target.astype(np.float64) - trans

    dets = np.linalg.det(rot)
    bad = np.abs(dets) < 1e-8
    good = ~bad
    if np.any(good):
        solved = np.linalg.solve(rot[good], rhs[good][:, :, None])[..., 0]
        bind_vertices[good] = solved.astype(np.float32)

    if np.any(bad):
        for i in np.where(bad)[0]:
            k = int(np.argmax(lbs_weights[i]))
            j = int(lbs_indices[i, k])
            a = affine[j]
            bind_vertices[i] = np.linalg.solve(a[:3, :3], rest_target[i] - a[:3, 3]).astype(np.float32)

    return bind_vertices


def _forward_rest_lbs(
    bind_vertices: np.ndarray,
    lbs_indices: np.ndarray,
    lbs_weights: np.ndarray,
    soma_bind: np.ndarray,
    rest_transform: np.ndarray,
    root_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Apply rest-pose LBS (matches CustomMeshSkin at Character init)."""
    verts = bind_vertices.astype(np.float64)
    if root_offset is not None:
        verts = verts + root_offset.astype(np.float64)

    soma_bind_inv = np.linalg.inv(soma_bind)
    affine = rest_transform @ soma_bind_inv
    gathered = affine[lbs_indices]
    rot = np.einsum("nw,nwij->nij", lbs_weights, gathered[:, :, :3, :3])
    trans = np.einsum("nw,nwi->ni", lbs_weights, gathered[:, :, :3, 3])
    return np.einsum("nij,nj->ni", rot, verts) + trans


def _remap_weights_sanitized(
    binding: UniRigBinding,
    soma_names: list[str],
    unirig_to_soma_idx: list[int],
    soma_parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_verts = binding.vertices.shape[0]
    n_soma = len(soma_names)
    soma_w = np.zeros((n_verts, n_soma), dtype=np.float64)
    w_u = binding.lbs_weights.shape[1]
    soma_idx_lut = np.array(unirig_to_soma_idx, dtype=np.int64)
    for k in range(w_u):
        bone_idx = binding.lbs_indices[:, k].astype(np.int64)
        soma_j = soma_idx_lut[bone_idx]
        np.add.at(soma_w, (np.arange(n_verts), soma_j), binding.lbs_weights[:, k].astype(np.float64))

    exclude_idx = [i for i, name in enumerate(soma_names) if name in SOMA_WEIGHT_EXCLUDE]
    for j in exclude_idx:
        col = soma_w[:, j]
        if not np.any(col > 1e-8):
            continue
        target = _nearest_skin_ancestor(j, soma_names, soma_parents)
        if target != j:
            soma_w[:, target] += col
        soma_w[:, j] = 0.0

    top_idx = np.argsort(soma_w, axis=1)[:, -MAX_INFLUENCES:]
    top_w = np.take_along_axis(soma_w, top_idx, axis=1).astype(np.float32)
    row_sum = top_w.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-8] = 1.0
    top_w = top_w / row_sum
    return top_idx.astype(np.int64), top_w


def _remap_weights(
    binding: UniRigBinding,
    soma_names: list[str],
    unirig_to_soma_idx: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    n_verts = binding.vertices.shape[0]
    soma_w = np.zeros((n_verts, len(soma_names)), dtype=np.float32)
    w_u = binding.lbs_weights.shape[1]
    soma_idx_lut = np.array(unirig_to_soma_idx, dtype=np.int64)
    for k in range(w_u):
        bone_idx = binding.lbs_indices[:, k].astype(np.int64)
        soma_j = soma_idx_lut[bone_idx]
        np.add.at(soma_w, (np.arange(n_verts), soma_j), binding.lbs_weights[:, k])

    top_idx = np.argsort(soma_w, axis=1)[:, -MAX_INFLUENCES:]
    top_w = np.take_along_axis(soma_w, top_idx, axis=1)
    row_sum = top_w.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-8] = 1.0
    top_w = top_w / row_sum
    return top_idx.astype(np.int64), top_w.astype(np.float32)


def remap_to_soma(binding: UniRigBinding) -> RemappedSOMABinding:
    soma_bind, soma_names, soma_parents = _load_soma_skin_meta()
    soma_name_to_idx = {n: i for i, n in enumerate(soma_names)}
    rest_transform = _build_rest_transform(soma_names)

    (
        unirig_to_soma_idx,
        bone_map,
        bone_map_modes,
        target_pos,
        aligned_vertices,
        aligned_bind,
    ) = _build_hybrid_mapping(binding, soma_names, soma_name_to_idx, rest_transform[:, :3, 3])

    top_idx, top_w = _remap_weights_sanitized(
        binding, soma_names, unirig_to_soma_idx, soma_parents
    )
    rest_target = _compute_rest_target(
        aligned_vertices,
        binding.lbs_indices,
        binding.lbs_weights,
        aligned_bind,
        target_pos,
        binding.joint_parents,
    )
    bind_vertices = _solve_bind_vertices(rest_target, top_idx, top_w, soma_bind, rest_transform)

    return RemappedSOMABinding(
        vertices=bind_vertices,
        faces=binding.faces.copy(),
        lbs_indices=top_idx,
        lbs_weights=top_w,
        bind_rig_transform=soma_bind.astype(np.float32),
        bone_map=bone_map,
        bone_map_modes=bone_map_modes,
        soma_scale=1.0,
        soma_root_offset=np.zeros(3, dtype=np.float32),
    )
