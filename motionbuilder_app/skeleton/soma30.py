from __future__ import annotations

import numpy as np

from ..assets import skeleton_asset_path


SOMA30_HIERARCHY: list[tuple[str, str | None]] = [
    ("Hips", None),
    ("Spine1", "Hips"), ("Spine2", "Spine1"), ("Chest", "Spine2"),
    ("Neck1", "Chest"), ("Neck2", "Neck1"), ("Head", "Neck2"),
    ("Jaw", "Head"), ("LeftEye", "Head"), ("RightEye", "Head"),
    ("LeftShoulder", "Chest"), ("LeftArm", "LeftShoulder"),
    ("LeftForeArm", "LeftArm"), ("LeftHand", "LeftForeArm"),
    ("LeftHandThumbEnd", "LeftHand"), ("LeftHandMiddleEnd", "LeftHand"),
    ("RightShoulder", "Chest"), ("RightArm", "RightShoulder"),
    ("RightForeArm", "RightArm"), ("RightHand", "RightForeArm"),
    ("RightHandThumbEnd", "RightHand"), ("RightHandMiddleEnd", "RightHand"),
    ("LeftLeg", "Hips"), ("LeftShin", "LeftLeg"),
    ("LeftFoot", "LeftShin"), ("LeftToeBase", "LeftFoot"),
    ("RightLeg", "Hips"), ("RightShin", "RightLeg"),
    ("RightFoot", "RightShin"), ("RightToeBase", "RightFoot"),
]


class SOMASkeleton30:
    """Small, inference-free SOMA-30 definition used by the binding tools."""

    name = "somaskel30"
    bone_order_names_with_parents = SOMA30_HIERARCHY

    def __init__(self, load: bool = True, **_: object) -> None:
        self.bone_order_names = [name for name, _ in SOMA30_HIERARCHY]
        index = {name: i for i, name in enumerate(self.bone_order_names)}
        self.joint_parents = np.asarray(
            [-1 if parent is None else index[parent] for _, parent in SOMA30_HIERARCHY],
            dtype=np.int64,
        )
        self.root_idx = 0
        self.nbjoints = len(self.bone_order_names)
        self.folder = str(skeleton_asset_path(self.name))
        if load:
            self.neutral_joints = np.load(skeleton_asset_path(self.name, "joints.npy")).astype(np.float32)
