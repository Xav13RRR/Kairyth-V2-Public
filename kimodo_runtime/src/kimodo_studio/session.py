# SPDX-FileCopyrightText: Copyright (c) 2026 MKA Project contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from kimodo.rigging.types import RemappedSOMABinding


class GenerateState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class ModelBundle:
    model: Any
    motion_rep: Any
    skeleton: Any
    model_fps: float
    model_name: str


@dataclass
class MotionResult:
    posed_joints: np.ndarray
    global_rot_mats: np.ndarray
    foot_contacts: np.ndarray | None
    fps: float
    prompts: list[str]
    raw_output: dict | None = None


@dataclass
class GenerateJob:
    state: GenerateState = GenerateState.IDLE
    message: str = ""
    error: str | None = None


@dataclass
class StudioSession:
    binding_path: Path
    remapped_binding: RemappedSOMABinding
    prompts: list[str] = field(default_factory=lambda: ["A person walks forward."])
    durations_sec: list[float] = field(default_factory=lambda: [6.0])
    constraints_json: list[dict] = field(default_factory=list)
    model_name: str = "Kimodo-SOMA-RP-v1.1"
    model_bundle: ModelBundle | None = None
    motion: MotionResult | None = None
    generate_job: GenerateJob = field(default_factory=GenerateJob)
    model_load_job: GenerateJob = field(default_factory=GenerateJob)
    custom_mesh_offset: np.ndarray = field(
        default_factory=lambda: np.array([1.5, 0.0, 0.0], dtype=np.float32)
    )
    motion_retargeted: bool = False
    device: str = "cuda:0"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def timeline_dict(self) -> dict:
        fps = self.model_bundle.model_fps if self.model_bundle else 30.0
        return {
            "prompts": list(self.prompts),
            "durations_sec": list(self.durations_sec),
            "model_fps": fps,
            "total_frames": self.total_frames(),
        }

    def total_frames(self) -> int:
        fps = self.model_bundle.model_fps if self.model_bundle else 30.0
        return sum(max(1, int(d * fps) - 1) for d in self.durations_sec)

    def num_frames_list(self) -> list[int]:
        fps = self.model_bundle.model_fps if self.model_bundle else 30.0
        return [max(1, int(d * fps) - 1) for d in self.durations_sec]

    def motion_dict(self) -> dict | None:
        if self.motion is None:
            return None
        m = self.motion
        return {
            "num_frames": int(m.posed_joints.shape[0]),
            "num_joints": int(m.posed_joints.shape[1]),
            "fps": float(m.fps),
            "prompts": list(m.prompts),
            "retargeted": self.motion_retargeted,
        }
