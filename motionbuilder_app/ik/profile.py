from __future__ import annotations

import hashlib
import json
from typing import Any

from scipy.spatial.transform import Rotation
import numpy as np

from ..meshpreps.bind_math import BindingResult, joint_positions_from_bind_matrices


IK_PROFILE_SCHEMA = "kairyth.humanoid_ik_rig"
IK_PROFILE_VERSION = 2
IK_SOLVER = "TwoBoneIK"

FACE_JOINTS = ("Jaw", "LeftEye", "RightEye")

_CHAIN_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "left_leg",
        "label": "Left Leg",
        "joint_names": ("LeftLeg", "LeftShin", "LeftFoot"),
        "goal_name": "LeftFootGoal",
        "pole_name": "LeftKneePole",
        "kind": "leg",
        "side": "left",
        "color": "#49d8ff",
        "fallback_pole": (0.0, 0.0, 1.0),
    },
    {
        "id": "right_leg",
        "label": "Right Leg",
        "joint_names": ("RightLeg", "RightShin", "RightFoot"),
        "goal_name": "RightFootGoal",
        "pole_name": "RightKneePole",
        "kind": "leg",
        "side": "right",
        "color": "#ff5ecb",
        "fallback_pole": (0.0, 0.0, 1.0),
    },
    {
        "id": "left_arm",
        "label": "Left Arm",
        "joint_names": ("LeftArm", "LeftForeArm", "LeftHand"),
        "goal_name": "LeftHandGoal",
        "pole_name": "LeftElbowPole",
        "kind": "arm",
        "side": "left",
        "color": "#7dff8a",
        "fallback_pole": (0.0, 0.0, -1.0),
    },
    {
        "id": "right_arm",
        "label": "Right Arm",
        "joint_names": ("RightArm", "RightForeArm", "RightHand"),
        "goal_name": "RightHandGoal",
        "pole_name": "RightElbowPole",
        "kind": "arm",
        "side": "right",
        "color": "#ffd34d",
        "fallback_pole": (0.0, 0.0, -1.0),
    },
)


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length > 1e-8:
        return vector / length
    fallback_length = float(np.linalg.norm(fallback))
    if fallback_length <= 1e-8:
        raise ValueError("cannot normalize zero-length IK direction")
    return fallback / fallback_length


def _pole_position(
    start: np.ndarray,
    middle: np.ndarray,
    end: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = end - start
    target_dir = _unit(target, np.asarray([0.0, -1.0, 0.0], dtype=np.float64))
    projected = start + target_dir * float(np.dot(middle - start, target_dir))
    bend = middle - projected
    if float(np.linalg.norm(bend)) <= 1e-6:
        bend = fallback - target_dir * float(np.dot(fallback, target_dir))
    if float(np.linalg.norm(bend)) <= 1e-6:
        alternate = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        bend = alternate - target_dir * float(np.dot(alternate, target_dir))
    bend_dir = _unit(bend, fallback)
    chain_length = float(np.linalg.norm(middle - start) + np.linalg.norm(end - middle))
    distance = max(chain_length * 0.6, 0.05)
    return middle + bend_dir * distance, bend_dir


def _fingerprint(names: list[str], parents: np.ndarray, joints: np.ndarray) -> str:
    payload = {
        "names": names,
        "parents": np.asarray(parents, dtype=np.int64).tolist(),
        "joints": np.round(np.asarray(joints, dtype=np.float64), 6).tolist(),
    }
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def build_humanoid_ik_profile(binding: BindingResult) -> dict[str, Any]:
    """Build the deterministic SOMA-30 Two-Bone IK control profile.

    The profile is a control layer over the deformation skeleton. It never adds
    weighted joints and deliberately excludes jaw and eye joints from IK.
    """

    names = [str(name) for name in binding.joint_names]
    parents = np.asarray(binding.joint_parents, dtype=np.int64)
    joints = joint_positions_from_bind_matrices(binding.bind_matrices).astype(np.float64)
    if len(names) != len(parents) or joints.shape != (len(names), 3):
        raise ValueError("invalid skeleton arrays for IK binding")

    name_to_index = {name: index for index, name in enumerate(names)}
    required = {"Hips"}
    for spec in _CHAIN_SPECS:
        required.update(spec["joint_names"])
    missing = sorted(required.difference(name_to_index))
    if missing:
        raise ValueError(f"humanoid IK joints missing: {', '.join(missing)}")

    chains: list[dict[str, Any]] = []
    for order, spec in enumerate(_CHAIN_SPECS):
        chain_names = list(spec["joint_names"])
        indices = [name_to_index[name] for name in chain_names]
        if int(parents[indices[1]]) != indices[0] or int(parents[indices[2]]) != indices[1]:
            raise ValueError(f"invalid parent hierarchy for IK chain {spec['id']}")

        start, middle, end = (joints[index] for index in indices)
        upper_length = float(np.linalg.norm(middle - start))
        lower_length = float(np.linalg.norm(end - middle))
        if min(upper_length, lower_length) <= 1e-6:
            raise ValueError(f"zero-length bone in IK chain {spec['id']}")

        fallback = np.asarray(spec["fallback_pole"], dtype=np.float64)
        pole_position, preferred_bend = _pole_position(start, middle, end, fallback)
        chains.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "order": order,
                "solver": IK_SOLVER,
                "kind": spec["kind"],
                "side": spec["side"],
                "color": spec["color"],
                "joint_names": chain_names,
                "joints": indices,
                "bone_lengths": [upper_length, lower_length],
                "goal": {
                    "name": spec["goal_name"],
                    "joint": indices[2],
                    "position": end.tolist(),
                    "rotation": Rotation.from_matrix(binding.bind_matrices[indices[2], :3, :3]).as_quat().tolist(),
                },
                "pole": {
                    "name": spec["pole_name"],
                    "joint": indices[1],
                    "position": pole_position.tolist(),
                },
                "preferred_bend": preferred_bend.tolist(),
                "softness": 0.02,
                "weight": 1.0,
            }
        )

    present_face_joints = [name for name in FACE_JOINTS if name in name_to_index]
    height = float(np.ptp(joints[:, 1])) if len(joints) else 1.0
    effector_specs = (
        ("head", "Head", "HeadGoal", "#f5f7ff", 0.85),
        ("left_hand", "LeftHand", "LeftHandGoal", "#7dff8a", 1.0),
        ("right_hand", "RightHand", "RightHandGoal", "#ffd34d", 1.0),
        ("left_foot", "LeftFoot", "LeftFootGoal", "#49d8ff", 1.0),
        ("right_foot", "RightFoot", "RightFootGoal", "#ff5ecb", 1.0),
    )
    effectors = []
    for effector_id, bone_name, control_name, color, rotation_alpha in effector_specs:
        index = name_to_index[bone_name]
        effectors.append(
            {
                "id": effector_id,
                "bone": bone_name,
                "joint": index,
                "control": control_name,
                "color": color,
                "position": joints[index].tolist(),
                "rotation": Rotation.from_matrix(binding.bind_matrices[index, :3, :3]).as_quat().tolist(),
                "position_alpha": 1.0,
                "rotation_alpha": rotation_alpha,
                "strength_alpha": 1.0,
                "pull_chain_alpha": 1.0,
                "pin_rotation": rotation_alpha,
            }
        )

    stiffness = {
        "Hips": (0.8, 0.8),
        "Spine1": (0.62, 0.9),
        "Spine2": (0.5, 0.92),
        "Chest": (0.42, 0.94),
        "Neck1": (0.55, 0.96),
        "Neck2": (0.42, 0.94),
        "LeftShoulder": (0.82, 1.0),
        "RightShoulder": (0.82, 1.0),
        "LeftArm": (0.18, 1.0),
        "RightArm": (0.18, 1.0),
        "LeftForeArm": (0.05, 1.0),
        "RightForeArm": (0.05, 1.0),
        "LeftLeg": (0.22, 1.0),
        "RightLeg": (0.22, 1.0),
        "LeftShin": (0.05, 1.0),
        "RightShin": (0.05, 1.0),
    }
    bone_settings = []
    for bone_name, (rotation_stiffness, position_stiffness) in stiffness.items():
        if bone_name not in name_to_index:
            continue
        setting = {
            "bone": bone_name,
            "joint": name_to_index[bone_name],
            "rotation_stiffness": rotation_stiffness,
            "position_stiffness": position_stiffness,
            "limits": {"x": "free", "y": "free", "z": "free"},
        }
        matching_chain = next((chain for chain in chains if bone_name in chain["joint_names"][:2]), None)
        if matching_chain and bone_name == matching_chain["joint_names"][1]:
            setting["use_preferred_angles"] = True
            setting["preferred_bend"] = matching_chain["preferred_bend"]
        bone_settings.append(setting)

    full_body = {
        "id": "full_body",
        "solver": "FullBodyIK",
        "root": "Hips",
        "root_index": name_to_index["Hips"],
        "effectors": effectors,
        "bone_settings": bone_settings,
        "excluded_joints": present_face_joints,
        "excluded_joint_indices": [name_to_index[name] for name in present_face_joints],
        "settings": {
            "iterations": 20,
            "mass_multiplier": 1.0,
            "allow_stretch": False,
            "root_behavior": "pre_pull",
            "global_pull_chain_alpha": 1.0,
            "max_angle": 30.0,
            "over_relaxation": 1.3,
            "start_from_input_pose": True,
        },
    }
    return {
        "schema": IK_PROFILE_SCHEMA,
        "version": IK_PROFILE_VERSION,
        "solver": IK_SOLVER,
        "skeleton": "SOMA-30",
        "skeleton_fingerprint": _fingerprint(names, parents, joints),
        "retarget_root": "Hips",
        "retarget_root_index": name_to_index["Hips"],
        "up_axis": [0.0, 1.0, 0.0],
        "forward_axis": [0.0, 0.0, 1.0],
        "control_scale": max(height * 0.035, 0.025),
        "solver_stack": [chain["id"] for chain in chains],
        "chains": chains,
        "available_solvers": [IK_SOLVER, "FullBodyIK"],
        "solvers": {
            "two_bone": {"solver": IK_SOLVER, "chains": [chain["id"] for chain in chains]},
            "full_body": full_body,
        },
        "full_body": full_body,
        "excluded_joints": present_face_joints,
        "excluded_joint_indices": [name_to_index[name] for name in present_face_joints],
    }
