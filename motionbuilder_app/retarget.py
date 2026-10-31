from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .meshpreps.binding_io import load_binding
from .motion_math import axis_angle_to_matrix, local_from_world, world_from_local


SMPLX22_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
SMPLX22_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19], dtype=np.int64)
SOURCE_INDEX = {name: i for i, name in enumerate(SMPLX22_NAMES)}

TARGET_SOURCE = {
    "Hips": "pelvis",
    "Spine1": "spine1", "Spine2": "spine2", "Chest": "spine3",
    "Neck1": "neck", "Neck2": "neck", "Head": "head",
    "LeftShoulder": "left_collar", "LeftArm": "left_shoulder",
    "LeftForeArm": "left_elbow", "LeftHand": "left_wrist",
    "RightShoulder": "right_collar", "RightArm": "right_shoulder",
    "RightForeArm": "right_elbow", "RightHand": "right_wrist",
    "LeftLeg": "left_hip", "LeftShin": "left_knee", "LeftFoot": "left_ankle",
    "LeftToeBase": "left_foot", "RightLeg": "right_hip", "RightShin": "right_knee",
    "RightFoot": "right_ankle", "RightToeBase": "right_foot",
}
MAPPING_VERSION = "smplx22-soma30-v1"


def build_source_motion_bundle(
    extracted_npz: str | Path,
    output_npz: str | Path,
    *,
    root_mode: str = "world",
) -> Path:
    """Bake GVHMR SMPL-X22 output on its native, fixed-length skeleton."""
    if root_mode not in {"world", "locked"}:
        raise ValueError("root_mode must be world or locked")
    source = np.load(Path(extracted_npz), allow_pickle=False)
    body_pose = np.asarray(source["body_pose"], dtype=np.float32).reshape(-1, 21, 3)
    global_orient = np.asarray(source["global_orient"], dtype=np.float32).reshape(-1, 3)
    transl = np.asarray(source["transl"], dtype=np.float32).reshape(-1, 3)
    rest_joints = np.asarray(source["source_rest_joints"], dtype=np.float32).reshape(22, 3)
    fps = float(np.asarray(source["fps"]).reshape(-1)[0])
    frames = len(body_pose)

    bind_world = np.repeat(np.eye(4, dtype=np.float32)[None], 22, axis=0)
    bind_world[:, :3, 3] = rest_joints
    bind_local = local_from_world(bind_world, SMPLX22_PARENTS)
    local = np.repeat(bind_local[None], frames, axis=0)
    local[:, 0, :3, :3] = axis_angle_to_matrix(global_orient)
    local[:, 1:, :3, :3] = axis_angle_to_matrix(body_pose)
    if root_mode == "world":
        local[:, 0, :3, 3] = bind_local[0, :3, 3] + (transl - transl[[0]])
    else:
        local[:, 0, :3, 3] = bind_local[0, :3, 3]
    world = world_from_local(local, SMPLX22_PARENTS)

    output = Path(output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        world_motion=world.astype(np.float32),
        target_bind=bind_world.astype(np.float32),
        parents=SMPLX22_PARENTS,
        names=np.asarray(SMPLX22_NAMES, dtype="U64"),
        fps=np.asarray([fps], dtype=np.float32),
        num_frames=np.asarray([frames], dtype=np.int64),
        skeleton_type=np.asarray(["SMPL-X22"], dtype="U32"),
        root_mode=np.asarray([root_mode], dtype="U16"),
    )
    return output


def retarget_source_bundle_fixed(
    source_bundle_npz: str | Path,
    binding_npz: str | Path,
    output_npz: str | Path,
) -> Path:
    """Map local rotation deltas while preserving every target bind translation.

    Hips translation is fixed to the target bind pose. Child translations always
    come from the target binding, so retargeting can never stretch a bone.
    """
    source = np.load(Path(source_bundle_npz), allow_pickle=False)
    binding = load_binding(binding_npz)
    source_world = np.asarray(source["world_motion"], dtype=np.float32)
    source_bind = np.asarray(source["target_bind"], dtype=np.float32)
    source_parents = np.asarray(source["parents"], dtype=np.int64)
    frames = source_world.shape[0]

    target_names = list(binding.joint_names)
    target_index = {name: i for i, name in enumerate(target_names)}
    if len(target_names) != 30 or "Hips" not in target_index:
        raise ValueError("character binding must use the 30-joint SOMA skeleton")
    target_bind = np.asarray(binding.bind_matrices, dtype=np.float32)
    target_local_bind = local_from_world(target_bind, binding.joint_parents)
    source_local_bind = local_from_world(source_bind, source_parents)
    source_local = np.stack([local_from_world(frame, source_parents) for frame in source_world])
    target_local = np.repeat(target_local_bind[None], frames, axis=0)

    for target_name, source_name in TARGET_SOURCE.items():
        target_joint = target_index[target_name]
        source_joint = SOURCE_INDEX[source_name]
        bind_rotation = source_local_bind[source_joint, :3, :3]
        rotations = np.einsum(
            "ij,fjk->fik",
            np.linalg.inv(bind_rotation),
            source_local[:, source_joint, :3, :3],
        )
        if target_name in {"Neck1", "Neck2"}:
            rotvec = Rotation.from_matrix(rotations).as_rotvec() * 0.5
            rotations = Rotation.from_rotvec(rotvec).as_matrix().astype(np.float32)
        target_local[:, target_joint, :3, :3] = np.einsum(
            "fij,jk->fik", rotations, target_local_bind[target_joint, :3, :3]
        )

    # Deliberately keep all translations, including Hips, at target bind values.
    target_local[:, :, :3, 3] = target_local_bind[None, :, :3, 3]
    world_motion = world_from_local(target_local, binding.joint_parents)
    output = Path(output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        rest_vertices=binding.vertices.astype(np.float32),
        faces=binding.faces.astype(np.int64),
        lbs_indices=binding.lbs_indices.astype(np.int64),
        lbs_weights=binding.lbs_weights.astype(np.float32),
        target_bind=target_bind,
        world_motion=world_motion.astype(np.float32),
        parents=binding.joint_parents.astype(np.int64),
        names=np.asarray(target_names, dtype="U64"),
        fps=np.asarray(source["fps"], dtype=np.float32),
        num_frames=np.asarray([frames], dtype=np.int64),
        root_mode=np.asarray(["fixed_hips"], dtype="U16"),
        mapping_version=np.asarray(["smplx22-soma30-fixed-length-v1"], dtype="U64"),
        retarget_scale=np.asarray([1.0], dtype=np.float32),
        root_motion=(source_world[:, 0, :3, 3] - source_world[[0], 0, :3, 3]).astype(np.float32),
    )
    return output


def _retarget_same_name_soma30_fixed(
    source_world: np.ndarray,
    source_bind: np.ndarray,
    source_parents: np.ndarray,
    source_names: list[str],
    binding_npz: str | Path,
    output_npz: str | Path,
    *,
    fps: np.ndarray,
    source_kind: str,
    mapping_version: str,
    prompts: np.ndarray | None = None,
    root_motion_override: np.ndarray | None = None,
    extra_payload: dict[str, np.ndarray] | None = None,
) -> Path:
    """Retarget a SOMA-30 source by same-name local rotation deltas.

    Target bone translations, including Hips, always stay at the character bind
    values. This matches the app's no-stretch retargeting rule.
    """
    binding = load_binding(binding_npz)
    target_names = list(binding.joint_names)
    target_index = {name: i for i, name in enumerate(target_names)}
    source_index = {name: i for i, name in enumerate(source_names)}
    if len(target_names) != 30 or "Hips" not in target_index:
        raise ValueError("character binding must use the 30-joint SOMA skeleton")
    missing = [name for name in target_names if name not in source_index]
    if missing:
        raise ValueError(f"Kimodo SOMA source is missing target joints: {', '.join(missing[:6])}")

    frames = int(source_world.shape[0])
    target_bind = np.asarray(binding.bind_matrices, dtype=np.float32)
    target_local_bind = local_from_world(target_bind, binding.joint_parents)
    source_local_bind = local_from_world(source_bind, source_parents)
    source_local = np.stack([local_from_world(frame, source_parents) for frame in source_world])
    target_local = np.repeat(target_local_bind[None], frames, axis=0)

    for target_name in target_names:
        target_joint = target_index[target_name]
        source_joint = source_index[target_name]
        bind_rotation = source_local_bind[source_joint, :3, :3]
        rotations = np.einsum(
            "ij,fjk->fik",
            np.linalg.inv(bind_rotation),
            source_local[:, source_joint, :3, :3],
        )
        target_local[:, target_joint, :3, :3] = np.einsum(
            "fij,jk->fik", rotations, target_local_bind[target_joint, :3, :3]
        )

    target_local[:, :, :3, 3] = target_local_bind[None, :, :3, 3]
    world_motion = world_from_local(target_local, binding.joint_parents)
    source_root = source_index.get("Hips", 0)
    output = Path(output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    root_motion = (
        np.asarray(root_motion_override, dtype=np.float32)
        if root_motion_override is not None
        else (source_world[:, source_root, :3, 3] - source_world[[0], source_root, :3, 3]).astype(np.float32)
    )
    payload = {
        "rest_vertices": binding.vertices.astype(np.float32),
        "faces": binding.faces.astype(np.int64),
        "lbs_indices": binding.lbs_indices.astype(np.int64),
        "lbs_weights": binding.lbs_weights.astype(np.float32),
        "target_bind": target_bind,
        "world_motion": world_motion.astype(np.float32),
        "parents": binding.joint_parents.astype(np.int64),
        "names": np.asarray(target_names, dtype="U64"),
        "fps": np.asarray(fps, dtype=np.float32).reshape(-1),
        "num_frames": np.asarray([frames], dtype=np.int64),
        "root_mode": np.asarray(["fixed_hips"], dtype="U16"),
        "mapping_version": np.asarray([mapping_version], dtype="U96"),
        "source_kind": np.asarray([source_kind], dtype="U32"),
        "retarget_scale": np.asarray([1.0], dtype=np.float32),
        "root_motion": root_motion,
    }
    if prompts is not None:
        payload["prompts"] = prompts
    if extra_payload:
        payload.update(extra_payload)
    np.savez_compressed(output, **payload)
    return output


def retarget_kimodo_bundle_fixed(
    kimodo_bundle_npz: str | Path,
    binding_npz: str | Path,
    output_npz: str | Path,
) -> Path:
    """Retarget a MotionBuilder-local Kimodo native SOMA bundle to a character."""
    bundle_path = Path(kimodo_bundle_npz)
    with np.load(bundle_path, allow_pickle=False) as source:
        if {"world_motion", "target_bind", "parents", "names"}.issubset(set(source.files)):
            source_world = np.asarray(source["world_motion"], dtype=np.float32)
            source_bind = np.asarray(source["target_bind"], dtype=np.float32)
            source_parents = np.asarray(source["parents"], dtype=np.int64)
            source_names = [str(item) for item in source["names"].tolist()]
            fps = np.asarray(source["fps"], dtype=np.float32)
            prompts = source["prompts"] if "prompts" in source.files else None
            smooth_root_pos = np.asarray(source["smooth_root_pos"], dtype=np.float32) if "smooth_root_pos" in source.files else None
            global_root_heading = np.asarray(source["global_root_heading"], dtype=np.float32) if "global_root_heading" in source.files else None
            durations_sec = source["durations_sec"] if "durations_sec" in source.files else None
        elif {"posed_joints", "global_rot_mats"}.issubset(set(source.files)):
            from .kimodo_runtime import kimodo_raw_to_soma30_bundle

            converted = bundle_path.with_name("native_kimodo_soma30.npz")
            kimodo_raw_to_soma30_bundle(bundle_path, converted)
            return retarget_kimodo_bundle_fixed(converted, binding_npz, output_npz)
        else:
            raise ValueError("Kimodo motion bundle must contain SOMA-30 world_motion or raw Kimodo posed_joints/global_rot_mats")

    extra: dict[str, np.ndarray] = {}
    root_override = None
    if smooth_root_pos is not None and len(smooth_root_pos) == len(source_world):
        root_override = (smooth_root_pos - smooth_root_pos[[0]]).astype(np.float32)
        extra["kimodo_smooth_root_pos"] = smooth_root_pos.astype(np.float32)
        extra["kimodo_smooth_root_motion"] = root_override.astype(np.float32)
    if global_root_heading is not None:
        extra["kimodo_global_root_heading"] = np.asarray(global_root_heading, dtype=np.float32)
    if durations_sec is not None:
        extra["durations_sec"] = durations_sec

    return _retarget_same_name_soma30_fixed(
        source_world,
        source_bind,
        source_parents,
        source_names,
        binding_npz,
        output_npz,
        fps=fps,
        source_kind="kimodo",
        mapping_version="kimodo-soma30-fixed-length-v1",
        prompts=prompts,
        root_motion_override=root_override,
        extra_payload=extra,
    )


def _leg_scale(source_rest: np.ndarray, target_bind: np.ndarray, target_names: list[str]) -> float:
    target_index = {name: i for i, name in enumerate(target_names)}
    source_lengths = []
    target_lengths = []
    for side in ("left", "right"):
        source_lengths.append(np.linalg.norm(source_rest[SOURCE_INDEX[f"{side}_ankle"]] - source_rest[0]))
        target_lengths.append(np.linalg.norm(target_bind[target_index[f"{side.title()}Foot"], :3, 3] - target_bind[target_index["Hips"], :3, 3]))
    source = float(np.mean(source_lengths))
    target = float(np.mean(target_lengths))
    return target / max(source, 1e-6)


def _remove_accumulated_yaw(root_rotations: np.ndarray) -> np.ndarray:
    out = root_rotations.copy().astype(np.float64)
    yaw0 = float(np.arctan2(out[0, 0, 2], out[0, 2, 2]))
    for i, matrix in enumerate(out):
        yaw = float(np.arctan2(matrix[0, 2], matrix[2, 2]))
        correction = Rotation.from_euler("y", yaw0 - yaw).as_matrix()
        out[i] = correction @ matrix
    return out.astype(np.float32)


def retarget_extracted(
    extracted_npz: str | Path,
    binding_npz: str | Path,
    output_npz: str | Path,
    *,
    root_mode: str = "world",
) -> Path:
    if root_mode not in {"world", "inplace"}:
        raise ValueError("root_mode must be world or inplace")
    source = np.load(Path(extracted_npz), allow_pickle=False)
    binding = load_binding(binding_npz)
    body_pose = np.asarray(source["body_pose"], dtype=np.float32).reshape(-1, 21, 3)
    global_orient = np.asarray(source["global_orient"], dtype=np.float32).reshape(-1, 3)
    transl = np.asarray(source["transl"], dtype=np.float32).reshape(-1, 3)
    source_rest = np.asarray(source["source_rest_joints"], dtype=np.float32).reshape(22, 3)
    fps = float(np.asarray(source["fps"]).reshape(-1)[0])
    frames = len(body_pose)

    source_rot = np.zeros((frames, 22, 3, 3), dtype=np.float32)
    source_rot[:, 0] = axis_angle_to_matrix(global_orient)
    source_rot[:, 1:] = axis_angle_to_matrix(body_pose)

    target_names = list(binding.joint_names)
    target_index = {name: i for i, name in enumerate(target_names)}
    if len(target_names) != 30 or "Hips" not in target_index:
        raise ValueError("character binding must use the 30-joint SOMA skeleton")
    target_bind = np.asarray(binding.bind_matrices, dtype=np.float32)
    target_local_bind = local_from_world(target_bind, binding.joint_parents)
    local = np.repeat(target_local_bind[None], frames, axis=0)

    for target_name, source_name in TARGET_SOURCE.items():
        target_joint = target_index[target_name]
        source_joint = SOURCE_INDEX[source_name]
        rotations = source_rot[:, source_joint]
        if target_name in {"Neck1", "Neck2"}:
            rotvec = Rotation.from_matrix(rotations).as_rotvec() * 0.5
            rotations = Rotation.from_rotvec(rotvec).as_matrix().astype(np.float32)
        local[:, target_joint, :3, :3] = rotations

    root = target_index["Hips"]
    scale = _leg_scale(source_rest, target_bind, target_names)
    delta = (transl - transl[[0]]) * scale
    root_rotations = source_rot[:, 0]
    if root_mode == "inplace":
        delta[:, [0, 2]] = 0.0
        root_rotations = _remove_accumulated_yaw(root_rotations)
        local[:, root, :3, :3] = root_rotations
    local[:, root, :3, 3] = target_local_bind[root, :3, 3] + delta
    world_motion = world_from_local(local, binding.joint_parents)

    output = Path(output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        rest_vertices=binding.vertices.astype(np.float32),
        faces=binding.faces.astype(np.int64),
        lbs_indices=binding.lbs_indices.astype(np.int64),
        lbs_weights=binding.lbs_weights.astype(np.float32),
        target_bind=target_bind,
        world_motion=world_motion,
        parents=binding.joint_parents.astype(np.int64),
        names=np.asarray(target_names, dtype="U64"),
        fps=np.asarray([fps], dtype=np.float32),
        num_frames=np.asarray([frames], dtype=np.int64),
        root_mode=np.asarray([root_mode], dtype="U16"),
        mapping_version=np.asarray([MAPPING_VERSION], dtype="U64"),
        retarget_scale=np.asarray([scale], dtype=np.float32),
    )
    return output
