# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA-30 retarget profile: bone name maps, chains, skin vs non-skin joints."""

from __future__ import annotations

# UniRig template bone -> SOMA-30 rig_joint_names entry
UNIRIG_TO_SOMA: dict[str, str] = {
    "Hips": "Hips",
    "Spine": "Spine1",
    "Spine1": "Spine2",
    "Spine2": "Chest",
    "Neck": "Neck1",
    "Head": "Head",
    "LeftShoulder": "LeftShoulder",
    "LeftArm": "LeftArm",
    "LeftForeArm": "LeftForeArm",
    "LeftHand": "LeftHand",
    "RightShoulder": "RightShoulder",
    "RightArm": "RightArm",
    "RightForeArm": "RightForeArm",
    "RightHand": "RightHand",
    "LeftUpLeg": "LeftLeg",
    "LeftLeg": "LeftShin",
    "LeftFoot": "LeftFoot",
    "RightUpLeg": "RightLeg",
    "RightLeg": "RightShin",
    "RightFoot": "RightFoot",
}

UNIRIG_NAME_ALIASES: dict[str, str] = {
    "hip": "Hips",
    "hips": "Hips",
    "pelvis": "Hips",
    "root": "Hips",
    "spine": "Spine",
    "spine1": "Spine1",
    "spine2": "Spine2",
    "chest": "Spine2",
    "neck": "Neck",
    "head": "Head",
    "leftshoulder": "LeftShoulder",
    "leftarm": "LeftArm",
    "leftforearm": "LeftForeArm",
    "lefthand": "LeftHand",
    "rightshoulder": "RightShoulder",
    "rightarm": "RightArm",
    "rightforearm": "RightForeArm",
    "righthand": "RightHand",
    "leftupleg": "LeftUpLeg",
    "leftupperleg": "LeftUpLeg",
    "leftleg": "LeftLeg",
    "leftfoot": "LeftFoot",
    "rightupleg": "RightUpLeg",
    "rightupperleg": "RightUpLeg",
    "rightleg": "RightLeg",
    "rightfoot": "RightFoot",
}

ALIGN_LANDMARKS: list[tuple[str, str]] = [
    ("Hips", "Hips"),
    ("Spine2", "Chest"),
    ("LeftShoulder", "LeftShoulder"),
    ("RightShoulder", "RightShoulder"),
    ("LeftFoot", "LeftFoot"),
    ("RightFoot", "RightFoot"),
]

SOMA_SPINE_CHAIN = ["Hips", "Spine1", "Spine2", "Chest"]
SOMA_NECK_CHAIN = ["Neck1", "Neck2", "Head"]
SOMA_LARM_CHAIN = ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"]
SOMA_RARM_CHAIN = ["RightShoulder", "RightArm", "RightForeArm", "RightHand"]
SOMA_LLEG_CHAIN = ["LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase"]
SOMA_RLEG_CHAIN = ["RightLeg", "RightShin", "RightFoot", "RightToeBase"]

# SOMA-30 bone -> UniRig template bone for rigid snap (inverse of UNIRIG_TO_SOMA).
# Neck1/Neck2 are interpolated on the chest->head path instead of snapping to Neck.
SOMA_DIRECT_SNAP: dict[str, str] = {
    soma: unirig for unirig, soma in UNIRIG_TO_SOMA.items() if unirig != "Neck"
}

# Spine bones above pelvis, in root-to-chest order.
SOMA_SPINE_SNAP_ORDER: list[str] = ["Hips", "Spine1", "Spine2", "Chest"]

# End bones placed relative to an anchor using SOMA rest local offset.
SOMA_HEAD_ACCESSORY_ANCHOR: dict[str, str] = {
    "Jaw": "Head",
    "LeftEye": "Head",
    "RightEye": "Head",
}

# (thumb_end, hand, middle_end) — thumb is inferred from hand + middle finger tip.
SOMA_THUMB_END_INFER: tuple[tuple[str, str, str], ...] = (
    ("LeftHandThumbEnd", "LeftHand", "LeftHandMiddleEnd"),
    ("RightHandThumbEnd", "RightHand", "RightHandMiddleEnd"),
)

SOMA_MIDDLE_END_JOINTS: tuple[str, str] = ("LeftHandMiddleEnd", "RightHandMiddleEnd")
SOMA_TOE_BASE_JOINTS: tuple[str, str] = ("LeftToeBase", "RightToeBase")

# UniRig anonymous skeleton index -> canonical humanoid bone name.
UNIRIG_ANONYMOUS_52: dict[int, str] = {
    0: "Hips",
    1: "Spine",
    2: "Spine1",
    3: "Spine2",
    4: "Neck",
    5: "Head",
    6: "LeftShoulder",
    7: "LeftArm",
    8: "LeftForeArm",
    9: "LeftHand",
    25: "RightShoulder",
    26: "RightArm",
    27: "RightForeArm",
    28: "RightHand",
    44: "LeftUpLeg",
    45: "LeftLeg",
    46: "LeftFoot",
    47: "LeftToe",
    48: "RightUpLeg",
    49: "RightLeg",
    50: "RightFoot",
    51: "RightToe",
}

UNIRIG_ANONYMOUS_40: dict[int, str] = {
    0: "Hips",
    1: "Spine",
    2: "Spine1",
    3: "Spine2",
    4: "Neck",
    5: "Head",
    6: "LeftShoulder",
    7: "LeftArm",
    8: "LeftForeArm",
    9: "LeftHand",
    19: "RightShoulder",
    20: "RightArm",
    21: "RightForeArm",
    22: "RightHand",
    32: "LeftUpLeg",
    33: "LeftLeg",
    34: "LeftFoot",
    35: "LeftToe",
    36: "RightUpLeg",
    37: "RightLeg",
    38: "RightFoot",
    39: "RightToe",
}

PRIMARY_SOMA_JOINTS: list[str] = [
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
]

# SOMA bones that keep skeleton but must not receive skin weights.
SOMA_NON_SKIN_JOINTS: frozenset[str] = frozenset(
    {
        "Jaw",
        "LeftEye",
        "RightEye",
        "LeftHandThumbEnd",
        "LeftHandMiddleEnd",
        "RightHandThumbEnd",
        "RightHandMiddleEnd",
    }
)

# SOMA-30 has no intermediate finger phalanges.
SOMA_FINGER_MID_JOINTS: frozenset[str] = frozenset()

SOMA_SKIN_JOINTS: frozenset[str] = frozenset(PRIMARY_SOMA_JOINTS)

SOMA_LEG_JOINTS: frozenset[str] = frozenset(SOMA_LLEG_CHAIN + SOMA_RLEG_CHAIN)

SOMA_ARM_JOINTS: frozenset[str] = frozenset(
    SOMA_LARM_CHAIN
    + SOMA_RARM_CHAIN
    + list(SOMA_MIDDLE_END_JOINTS)
    + [name for name, _, _ in SOMA_THUMB_END_INFER]
)

SOMA_WEIGHT_EXCLUDE: frozenset[str] = SOMA_NON_SKIN_JOINTS | SOMA_FINGER_MID_JOINTS
