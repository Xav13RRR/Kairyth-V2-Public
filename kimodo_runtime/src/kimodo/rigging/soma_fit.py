# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit SOMA-30 joint positions to UniRig skeleton (rigid snap + local offsets)."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import numpy as np

from kimodo.skeleton import SOMASkeleton30

from .soma_retarget_profile import (
    SOMA_DIRECT_SNAP,
    SOMA_HEAD_ACCESSORY_ANCHOR,
    SOMA_LARM_CHAIN,
    SOMA_LLEG_CHAIN,
    SOMA_MIDDLE_END_JOINTS,
    SOMA_NECK_CHAIN,
    SOMA_RARM_CHAIN,
    SOMA_RLEG_CHAIN,
    SOMA_SPINE_SNAP_ORDER,
    SOMA_THUMB_END_INFER,
    SOMA_TOE_BASE_JOINTS,
    UNIRIG_ANONYMOUS_40,
    UNIRIG_ANONYMOUS_52,
    UNIRIG_TO_SOMA,
)

# SOMA landmark bones used to estimate pelvis-anchored similarity vs UniRig.
_SOMA_LANDMARKS = (
    "Hips",
    "Chest",
    "Head",
    "LeftHand",
    "RightHand",
    "LeftFoot",
    "RightFoot",
)


@dataclass
class SomaFitResult:
    names: list[str]
    joints: np.ndarray  # [J, 3] float32
    parents: np.ndarray  # [J] int64


@dataclass
class UniRigRoles:
    hips: int
    chest: int
    head: int
    spine_chain: list[int]
    neck_chain: list[int]
    left_arm_chain: list[int]
    right_arm_chain: list[int]
    left_leg_chain: list[int]
    right_leg_chain: list[int]
    left_hand: int
    right_hand: int
    left_foot: int
    right_foot: int
    left_finger_tip: int
    right_finger_tip: int
    left_toe: int
    right_toe: int
    left_leg_four: list[int]
    right_leg_four: list[int]
    name_to_idx: dict[str, int] = field(default_factory=dict)


def _load_soma_rest() -> tuple[list[str], np.ndarray, np.ndarray]:
    skel = SOMASkeleton30(load=True)
    hierarchy = skel.bone_order_names_with_parents
    names = [name for name, _ in hierarchy]
    rest = np.asarray(skel.neutral_joints.detach().cpu().numpy(), dtype=np.float64)
    if rest.shape[0] != len(names):
        rest = rest[: len(names)]
    name_to_idx = {name: i for i, name in enumerate(names)}
    parents = np.array(
        [name_to_idx[parent] if parent is not None else -1 for _, parent in hierarchy],
        dtype=np.int64,
    )
    return names, rest, parents


def _path_to_root(parents: np.ndarray, leaf: int) -> list[int]:
    path: list[int] = []
    seen: set[int] = set()
    j = int(leaf)
    while j >= 0 and j not in seen:
        seen.add(j)
        path.append(j)
        j = int(parents[j])
    return path[::-1]


def _lca(parents: np.ndarray, a: int, b: int) -> int:
    anc = set(_path_to_root(parents, a))
    j = int(b)
    while j >= 0:
        if j in anc:
            return j
        j = int(parents[j])
    return -1


def _polyline_point(pts: np.ndarray, f: float) -> np.ndarray:
    if len(pts) == 1:
        return pts[0].copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-9:
        return pts[0].copy()
    target = float(np.clip(f, 0.0, 1.0)) * total
    acc = 0.0
    for i, length in enumerate(seg):
        if acc + length >= target or i == len(seg) - 1:
            t = (target - acc) / max(float(length), 1e-9)
            return pts[i] + (pts[i + 1] - pts[i]) * float(np.clip(t, 0.0, 1.0))
        acc += float(length)
    return pts[-1].copy()


def _branch(path: list[int], anchor: int) -> list[int]:
    return path[path.index(anchor) :] if anchor in path else path


def _limb_branch(parents: np.ndarray, end_idx: int, anchor_idx: int) -> list[int]:
    """Path from anchor to end, excluding the anchor joint itself."""
    path = _branch(_path_to_root(parents, end_idx), anchor_idx)
    if path and path[0] == anchor_idx and len(path) > 1:
        return path[1:]
    return path


def _split_left_right(
    joints: np.ndarray,
    root_x: float,
    a: int,
    b: int,
) -> tuple[int, int]:
    """Return (left_idx, right_idx) where left = lower X, right = higher X."""
    if (joints[a, 0] - root_x) <= (joints[b, 0] - root_x):
        return a, b
    return b, a


def _ensure_left_right(
    positions: np.ndarray,
    root_x: float,
    left_idx: int,
    right_idx: int,
) -> tuple[int, int]:
    """Swap indices if spatial X does not match left/right convention."""
    if left_idx == right_idx:
        return left_idx, right_idx
    left_x = float(positions[left_idx, 0] - root_x)
    right_x = float(positions[right_idx, 0] - root_x)
    if left_x > right_x:
        return right_idx, left_idx
    return left_idx, right_idx


def _resolve_lateral_pair(
    canonical_to_idx: dict[str, int],
    positions: np.ndarray,
    root_x: float,
    left_key: str,
    right_key: str,
    topo_left: int,
    topo_right: int,
) -> tuple[int, int]:
    """Prefer explicit Left/Right bone names; fall back to topology + spatial check."""
    if left_key in canonical_to_idx and right_key in canonical_to_idx:
        left_idx = canonical_to_idx[left_key]
        right_idx = canonical_to_idx[right_key]
        if left_idx != right_idx:
            return left_idx, right_idx
    return _ensure_left_right(positions, root_x, topo_left, topo_right)


def _leg_chain_four(
    parents: np.ndarray,
    positions: np.ndarray,
    hips_idx: int,
    foot_hint: int,
) -> tuple[list[int], int, int]:
    """Return last 4 joints on hip->toe branch plus explicit (foot, toe) indices."""
    toe_leaf = _deepest_descendant(parents, positions, foot_hint)
    branch = _limb_branch(parents, toe_leaf, hips_idx)
    if not branch:
        branch = [foot_hint]
    if len(branch) >= 4:
        chain = branch[-4:]
        foot_bone = chain[-2]
        toe_bone = chain[-1]
    elif branch:
        chain = branch
        foot_bone = chain[-1]
        toe_bone = chain[-1]
    else:
        chain = [foot_hint]
        foot_bone = toe_bone = foot_hint
    return chain, foot_bone, toe_bone


def _finger_tip_under_wrist(
    parents: np.ndarray,
    positions: np.ndarray,
    wrist_idx: int,
    root_x: float,
) -> int:
    """Deepest finger tip under wrist, kept on the same lateral side as the wrist."""
    tip = _deepest_descendant(parents, positions, wrist_idx)
    if tip == wrist_idx:
        return tip
    wrist_x = float(positions[wrist_idx, 0] - root_x)
    tip_x = float(positions[tip, 0] - root_x)
    if abs(wrist_x) < 1e-5 or abs(tip_x) < 1e-5:
        return tip
    if (wrist_x > 0.0) == (tip_x > 0.0):
        return tip
    return wrist_idx


def _canonical_unirig_names(raw_names: list[str]) -> list[str]:
    """Map anonymous UniRig bone_N names to canonical humanoid labels when possible."""
    if not raw_names:
        return list(raw_names)
    bone_pattern = re.compile(r"^bone_(\d+)$", re.IGNORECASE)
    indices: list[int | None] = []
    for name in raw_names:
        match = bone_pattern.match(str(name).strip())
        indices.append(int(match.group(1)) if match else None)
    if any(idx is None for idx in indices):
        return list(raw_names)

    n = len(raw_names)
    role_map = UNIRIG_ANONYMOUS_52 if n >= 52 else UNIRIG_ANONYMOUS_40 if n >= 40 else None
    if role_map is None:
        return list(raw_names)

    canonical: list[str] = []
    for idx in indices:
        assert idx is not None
        canonical.append(role_map.get(idx, raw_names[idx]))
    return canonical


def _children_map(parents: np.ndarray) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for j in range(len(parents)):
        p = int(parents[j])
        if p >= 0:
            children.setdefault(p, []).append(j)
    return children


def _detect_topology(
    joints: np.ndarray,
    parents: np.ndarray,
) -> tuple[int, int, int, int, int, int, int] | None:
    """Return root, chest, head, l_hand, r_hand, l_foot, r_foot UniRig indices."""
    n = len(parents)
    root_candidates = np.where(parents < 0)[0]
    if root_candidates.size == 0:
        return None
    root = int(root_candidates[0])

    children = _children_map(parents)
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

    l_foot, r_foot = _split_left_right(joints, rx, feet[0], feet[1])
    l_hand, r_hand = _split_left_right(joints, rx, hands[0], hands[1])
    chest = _lca(parents, l_hand, r_hand)
    if chest < 0:
        chest = _lca(parents, l_hand, head)
    if chest < 0:
        return None
    return root, chest, head, l_hand, r_hand, l_foot, r_foot


def _resolve_pelvis_index(
    unirig_names: list[str],
    u_pos: np.ndarray,
    u_parents: np.ndarray,
) -> int:
    """UniRig pelvis / hips joint index."""
    name_to_idx = {name: i for i, name in enumerate(unirig_names)}
    for key in ("Hips", "hip", "hips", "pelvis", "root"):
        if key in name_to_idx:
            return name_to_idx[key]
    roots = np.where(u_parents < 0)[0]
    if roots.size:
        return int(roots[0])
    return 0


def _deepest_descendant(
    parents: np.ndarray,
    positions: np.ndarray,
    root_idx: int,
) -> int:
    """Deepest leaf in ``root_idx`` subtree; fallback to ``root_idx`` if no children."""
    children = _children_map(parents)
    best_idx = root_idx
    best_depth = 0
    best_dist = 0.0
    origin = positions[root_idx]

    def _walk(j: int, depth: int) -> None:
        nonlocal best_idx, best_depth, best_dist
        kids = children.get(j, [])
        if not kids:
            dist = float(np.linalg.norm(positions[j] - origin))
            if depth > best_depth or (depth == best_depth and dist > best_dist):
                best_idx = j
                best_depth = depth
                best_dist = dist
            return
        for child in kids:
            _walk(child, depth + 1)

    _walk(root_idx, 0)
    return best_idx


def _chain_from_names(names: list[str], chain_names: list[str]) -> list[int]:
    name_to_idx = {name: i for i, name in enumerate(names)}
    out: list[int] = []
    for key in chain_names:
        if key in name_to_idx:
            out.append(name_to_idx[key])
    return out


def _resolve_unirig_roles(
    unirig_names: list[str],
    u_pos: np.ndarray,
    u_parents: np.ndarray,
) -> UniRigRoles:
    canonical_names = _canonical_unirig_names(unirig_names)
    name_to_idx = {name: i for i, name in enumerate(unirig_names)}
    canonical_to_idx: dict[str, int] = {}
    for i, canon in enumerate(canonical_names):
        canonical_to_idx.setdefault(canon, i)

    hips = _resolve_pelvis_index(canonical_names, u_pos, u_parents)
    root_x = float(u_pos[hips, 0])

    topo = _detect_topology(u_pos, u_parents)
    if topo is not None:
        root, chest_topo, head_topo, l_hand, r_hand, l_foot, r_foot = topo
    else:
        root = hips
        chest_topo = hips
        head_topo = hips
        l_hand = hips
        r_hand = hips
        l_foot = hips
        r_foot = hips

    def _idx(*keys: str, fallback: int) -> int:
        for key in keys:
            if key in canonical_to_idx:
                return canonical_to_idx[key]
            if key in name_to_idx:
                return name_to_idx[key]
        return fallback

    chest = _idx("Spine2", "Chest", fallback=chest_topo)
    head = _idx("Head", fallback=head_topo)
    l_hand_idx, r_hand_idx = _resolve_lateral_pair(
        canonical_to_idx, u_pos, root_x, "LeftHand", "RightHand", l_hand, r_hand
    )
    l_foot_hint, r_foot_hint = _resolve_lateral_pair(
        canonical_to_idx, u_pos, root_x, "LeftFoot", "RightFoot", l_foot, r_foot
    )

    left_leg_four, left_foot_bone, left_toe_bone = _leg_chain_four(
        u_parents, u_pos, hips, l_foot_hint
    )
    right_leg_four, right_foot_bone, right_toe_bone = _leg_chain_four(
        u_parents, u_pos, hips, r_foot_hint
    )

    spine_path = _path_to_root(u_parents, chest)
    if hips in spine_path:
        spine_chain = spine_path[spine_path.index(hips) :]
    else:
        spine_chain = _path_to_root(u_parents, chest)

    head_path = _path_to_root(u_parents, head)
    if chest in head_path:
        neck_chain = head_path[head_path.index(chest) + 1 :]
    else:
        neck_chain = [head]

    left_arm_chain = _chain_from_names(
        canonical_names, ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"]
    )
    if not left_arm_chain:
        left_arm_chain = _limb_branch(u_parents, l_hand_idx, chest)

    right_arm_chain = _chain_from_names(
        canonical_names, ["RightShoulder", "RightArm", "RightForeArm", "RightHand"]
    )
    if not right_arm_chain:
        right_arm_chain = _limb_branch(u_parents, r_hand_idx, chest)

    left_finger_tip = _finger_tip_under_wrist(u_parents, u_pos, l_hand_idx, root_x)
    right_finger_tip = _finger_tip_under_wrist(u_parents, u_pos, r_hand_idx, root_x)

    return UniRigRoles(
        hips=hips,
        chest=chest,
        head=head,
        spine_chain=spine_chain,
        neck_chain=neck_chain,
        left_arm_chain=left_arm_chain,
        right_arm_chain=right_arm_chain,
        left_leg_chain=left_leg_four,
        right_leg_chain=right_leg_four,
        left_hand=l_hand_idx,
        right_hand=r_hand_idx,
        left_foot=left_foot_bone,
        right_foot=right_foot_bone,
        left_finger_tip=left_finger_tip,
        right_finger_tip=right_finger_tip,
        left_toe=left_toe_bone,
        right_toe=right_toe_bone,
        left_leg_four=left_leg_four,
        right_leg_four=right_leg_four,
        name_to_idx=canonical_to_idx,
    )


def _snap_chain(
    pos: np.ndarray,
    placed: np.ndarray,
    soma_idx: dict[str, int],
    soma_chain: list[str],
    uni_chain: list[int],
    u_pos: np.ndarray,
) -> None:
    if not uni_chain:
        return
    if len(soma_chain) == len(uni_chain):
        for soma_name, uni_idx in zip(soma_chain, uni_chain):
            if soma_name not in soma_idx:
                continue
            j = soma_idx[soma_name]
            pos[j] = u_pos[uni_idx]
            placed[j] = True
        return

    uni_pts = u_pos[np.asarray(uni_chain, dtype=np.int64)]
    m = len(soma_chain)
    for i, soma_name in enumerate(soma_chain):
        if soma_name not in soma_idx:
            continue
        j = soma_idx[soma_name]
        f = (i / (m - 1)) if m > 1 else 0.0
        pos[j] = _polyline_point(uni_pts, f)
        placed[j] = True


def _snap_spine_chain(
    pos: np.ndarray,
    placed: np.ndarray,
    soma_idx: dict[str, int],
    roles: UniRigRoles,
    u_pos: np.ndarray,
) -> None:
    for soma_name, unirig_name in SOMA_DIRECT_SNAP.items():
        if soma_name not in SOMA_SPINE_SNAP_ORDER:
            continue
        if soma_name not in soma_idx or unirig_name not in roles.name_to_idx:
            continue
        j = soma_idx[soma_name]
        pos[j] = u_pos[roles.name_to_idx[unirig_name]]
        placed[j] = True

    missing = [name for name in SOMA_SPINE_SNAP_ORDER if name in soma_idx and not placed[soma_idx[name]]]
    if missing and roles.spine_chain:
        _snap_chain(pos, placed, soma_idx, missing, roles.spine_chain, u_pos)


def _snap_leg_side(
    pos: np.ndarray,
    placed: np.ndarray,
    soma_idx: dict[str, int],
    soma_chain: list[str],
    uni_chain: list[int],
    roles: UniRigRoles,
    u_pos: np.ndarray,
    toe_idx: int,
) -> None:
    if len(uni_chain) == len(soma_chain):
        _snap_chain(pos, placed, soma_idx, soma_chain, uni_chain, u_pos)
        return

    named_ok = True
    for soma_name in soma_chain[:3]:
        unirig_name = SOMA_DIRECT_SNAP.get(soma_name)
        if (
            soma_name not in soma_idx
            or not unirig_name
            or unirig_name not in roles.name_to_idx
        ):
            named_ok = False
            break

    if named_ok:
        for soma_name in soma_chain[:3]:
            unirig_name = SOMA_DIRECT_SNAP[soma_name]
            j = soma_idx[soma_name]
            pos[j] = u_pos[roles.name_to_idx[unirig_name]]
            placed[j] = True
    elif len(uni_chain) >= 3:
        _snap_chain(pos, placed, soma_idx, soma_chain[:3], uni_chain, u_pos)
    elif uni_chain:
        _snap_chain(pos, placed, soma_idx, soma_chain, uni_chain, u_pos)

    toe_name = soma_chain[3]
    if toe_name in soma_idx:
        j = soma_idx[toe_name]
        pos[j] = u_pos[toe_idx]
        placed[j] = True


def _snap_direct(
    pos: np.ndarray,
    placed: np.ndarray,
    soma_names: list[str],
    roles: UniRigRoles,
    u_pos: np.ndarray,
) -> None:
    soma_idx = {name: i for i, name in enumerate(soma_names)}

    _snap_spine_chain(pos, placed, soma_idx, roles, u_pos)

    _snap_chain(pos, placed, soma_idx, SOMA_LARM_CHAIN, roles.left_arm_chain, u_pos)
    _snap_chain(pos, placed, soma_idx, SOMA_RARM_CHAIN, roles.right_arm_chain, u_pos)
    _snap_leg_side(
        pos, placed, soma_idx, SOMA_LLEG_CHAIN, roles.left_leg_four, roles, u_pos, roles.left_toe
    )
    _snap_leg_side(
        pos, placed, soma_idx, SOMA_RLEG_CHAIN, roles.right_leg_four, roles, u_pos, roles.right_toe
    )

    if "Head" in soma_idx:
        pos[soma_idx["Head"]] = u_pos[roles.head]
        placed[soma_idx["Head"]] = True


def _place_neck_interp(
    pos: np.ndarray,
    placed: np.ndarray,
    soma_idx: dict[str, int],
    roles: UniRigRoles,
    u_pos: np.ndarray,
    u_parents: np.ndarray,
) -> None:
    neck_names = [name for name in ("Neck1", "Neck2") if name in soma_idx]
    if not neck_names:
        return

    head_path = _path_to_root(u_parents, roles.head)
    if roles.chest in head_path:
        neck_path = head_path[head_path.index(roles.chest) :]
    else:
        neck_path = [roles.chest, roles.head]

    neck_pts = u_pos[np.asarray(neck_path, dtype=np.int64)]
    m = len(neck_names) + 1
    for i, name in enumerate(neck_names, start=1):
        j = soma_idx[name]
        pos[j] = _polyline_point(neck_pts, i / m)
        placed[j] = True


def _apply_rest_offset(
    pos: np.ndarray,
    rest: np.ndarray,
    soma_idx: dict[str, int],
    anchor_name: str,
    joint_name: str,
) -> None:
    if anchor_name not in soma_idx or joint_name not in soma_idx:
        return
    anchor_i = soma_idx[anchor_name]
    joint_i = soma_idx[joint_name]
    pos[joint_i] = pos[anchor_i] + (rest[joint_i] - rest[anchor_i])


def _rotation_from_to(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Minimal rotation mapping src direction onto dst direction."""
    src_n = float(np.linalg.norm(src))
    dst_n = float(np.linalg.norm(dst))
    if src_n < 1e-8 or dst_n < 1e-8:
        return np.eye(3, dtype=np.float64)
    a = src / src_n
    b = dst / dst_n
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-8:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        perp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(a, perp)
        axis /= float(np.linalg.norm(axis))
        return np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _infer_thumb_end(
    pos: np.ndarray,
    rest: np.ndarray,
    soma_idx: dict[str, int],
    thumb_name: str,
    hand_name: str,
    middle_name: str,
) -> None:
    """Place thumb tip from wrist + middle-finger tip using rest hand-local geometry."""
    if thumb_name not in soma_idx or hand_name not in soma_idx or middle_name not in soma_idx:
        return
    hand_i = soma_idx[hand_name]
    mid_i = soma_idx[middle_name]
    thumb_i = soma_idx[thumb_name]

    rest_mid = rest[mid_i] - rest[hand_i]
    rest_thumb = rest[thumb_i] - rest[hand_i]
    fit_mid = pos[mid_i] - pos[hand_i]

    rest_mid_len = float(np.linalg.norm(rest_mid))
    fit_mid_len = float(np.linalg.norm(fit_mid))
    if rest_mid_len < 1e-8 or fit_mid_len < 1e-8:
        pos[thumb_i] = pos[hand_i] + rest_thumb
        return

    scale = fit_mid_len / rest_mid_len
    rot = _rotation_from_to(rest_mid, fit_mid)
    pos[thumb_i] = pos[hand_i] + rot @ (rest_thumb * scale)


def _snap_end_effectors(
    pos: np.ndarray,
    placed: np.ndarray,
    soma_idx: dict[str, int],
    roles: UniRigRoles,
    u_pos: np.ndarray,
) -> None:
    """Snap finger tips; toes are mapped via the 4-bone leg chain unless missing."""
    middle_targets = {
        SOMA_MIDDLE_END_JOINTS[0]: roles.left_finger_tip,
        SOMA_MIDDLE_END_JOINTS[1]: roles.right_finger_tip,
    }
    for soma_name, uni_idx in middle_targets.items():
        if soma_name not in soma_idx:
            continue
        j = soma_idx[soma_name]
        pos[j] = u_pos[uni_idx]
        placed[j] = True

    if len(roles.left_leg_four) < 4:
        soma_name = SOMA_TOE_BASE_JOINTS[0]
        if soma_name in soma_idx:
            pos[soma_idx[soma_name]] = u_pos[roles.left_toe]
            placed[soma_idx[soma_name]] = True
    if len(roles.right_leg_four) < 4:
        soma_name = SOMA_TOE_BASE_JOINTS[1]
        if soma_name in soma_idx:
            pos[soma_idx[soma_name]] = u_pos[roles.right_toe]
            placed[soma_idx[soma_name]] = True


def fit_soma30_positions(
    unirig_names: list[str],
    unirig_positions: np.ndarray,
    unirig_parents: np.ndarray | None = None,
) -> SomaFitResult:
    """Map SOMA-30 joints onto UniRig via rigid snap, neck interpolation, and rest offsets."""
    soma_names, rest, soma_parents = _load_soma_rest()
    u_pos = np.asarray(unirig_positions, dtype=np.float64)
    u_parents = (
        np.asarray(unirig_parents, dtype=np.int64)
        if unirig_parents is not None
        else np.full(len(unirig_names), -1, dtype=np.int64)
    )

    roles = _resolve_unirig_roles(unirig_names, u_pos, u_parents)
    soma_idx = {name: i for i, name in enumerate(soma_names)}
    pos = np.zeros((len(soma_names), 3), dtype=np.float64)
    placed = np.zeros(len(soma_names), dtype=bool)

    _snap_direct(pos, placed, soma_names, roles, u_pos)
    _place_neck_interp(pos, placed, soma_idx, roles, u_pos, u_parents)

    for joint_name, anchor_name in SOMA_HEAD_ACCESSORY_ANCHOR.items():
        _apply_rest_offset(pos, rest, soma_idx, anchor_name, joint_name)

    _snap_end_effectors(pos, placed, soma_idx, roles, u_pos)

    for thumb_name, hand_name, middle_name in SOMA_THUMB_END_INFER:
        _infer_thumb_end(pos, rest, soma_idx, thumb_name, hand_name, middle_name)

    return SomaFitResult(
        names=soma_names,
        joints=pos.astype(np.float32),
        parents=soma_parents,
    )


# --- Backward-compatible helpers (used by tests / legacy callers) ---


def _infer_up_vector(
    u_pos: np.ndarray,
    u_parents: np.ndarray,
    unirig_names: list[str],
    hips_idx: int,
) -> np.ndarray:
    """Vertical axis from pelvis toward chest/spine (fallback Y-up)."""
    name_to_idx = {name: i for i, name in enumerate(unirig_names)}
    for chest_key in ("Spine2", "Chest", "Spine1", "Spine"):
        if chest_key in name_to_idx:
            vec = u_pos[name_to_idx[chest_key]] - u_pos[hips_idx]
            n = float(np.linalg.norm(vec))
            if n > 1e-6:
                return vec / n
    topo = _detect_topology(u_pos, u_parents)
    if topo is not None:
        _, chest, _, _, _, _, _ = topo
        vec = u_pos[chest] - u_pos[hips_idx]
        n = float(np.linalg.norm(vec))
        if n > 1e-6:
            return vec / n
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _kabsch_rotation(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid rotation R (3x3) mapping src landmark directions to dst (no scale)."""
    if src.shape[0] < 3:
        return np.eye(3, dtype=np.float64)
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
    return r


def _unirig_landmark_indices(
    unirig_names: list[str],
    u_pos: np.ndarray,
    u_parents: np.ndarray,
) -> dict[str, int]:
    """Map SOMA landmark name -> UniRig joint index."""
    out: dict[str, int] = {}
    name_to_idx = {name: i for i, name in enumerate(unirig_names)}

    for unirig_name, soma_name in UNIRIG_TO_SOMA.items():
        if soma_name in _SOMA_LANDMARKS and unirig_name in name_to_idx:
            out[soma_name] = name_to_idx[unirig_name]

    topo = _detect_topology(u_pos, u_parents)
    if topo is None:
        return out
    root, chest, head, l_hand, r_hand, l_foot, r_foot = topo
    defaults = {
        "Hips": root,
        "Chest": chest,
        "Head": head,
        "LeftHand": l_hand,
        "RightHand": r_hand,
        "LeftFoot": l_foot,
        "RightFoot": r_foot,
    }
    for soma_name, u_idx in defaults.items():
        out.setdefault(soma_name, u_idx)
    return out


def _estimate_pelvis_similarity(
    rest: np.ndarray,
    soma_names: list[str],
    u_pos: np.ndarray,
    unirig_names: list[str],
    u_parents: np.ndarray,
) -> tuple[float, np.ndarray, int, np.ndarray]:
    """Estimate scale + rotation with UniRig pelvis as the fixed anchor."""
    soma_idx = {name: i for i, name in enumerate(soma_names)}
    hips_idx = _resolve_pelvis_index(unirig_names, u_pos, u_parents)
    uni_hips = u_pos[hips_idx]
    rest_hips = rest[soma_idx["Hips"]] if "Hips" in soma_idx else rest[0]

    uni_landmarks = _unirig_landmark_indices(unirig_names, u_pos, u_parents)

    ratios: list[float] = []
    src_vecs: list[np.ndarray] = []
    dst_vecs: list[np.ndarray] = []
    for soma_name in _SOMA_LANDMARKS:
        if soma_name == "Hips":
            continue
        if soma_name not in soma_idx or soma_name not in uni_landmarks:
            continue
        rv = rest[soma_idx[soma_name]] - rest_hips
        uv = u_pos[uni_landmarks[soma_name]] - uni_hips
        nr = float(np.linalg.norm(rv))
        nu = float(np.linalg.norm(uv))
        if nr > 1e-6 and nu > 1e-6:
            ratios.append(nu / nr)
            src_vecs.append(rv)
            dst_vecs.append(uv)

    scale = float(np.median(ratios)) if ratios else 1.0
    if len(src_vecs) < 3:
        rotation = np.eye(3, dtype=np.float64)
    else:
        src_scaled = np.stack([scale * v for v in src_vecs])
        rotation = _kabsch_rotation(src_scaled, np.stack(dst_vecs))

    up = _infer_up_vector(u_pos, u_parents, unirig_names, hips_idx)
    return scale, rotation, hips_idx, up


def _estimate_global_rotation(
    rest: np.ndarray,
    soma_names: list[str],
    u_pos: np.ndarray,
    unirig_names: list[str],
    u_parents: np.ndarray,
) -> np.ndarray:
    """Backward-compatible wrapper returning pelvis-anchored rotation only."""
    _, rotation, _, _ = _estimate_pelvis_similarity(
        rest, soma_names, u_pos, unirig_names, u_parents
    )
    return rotation
