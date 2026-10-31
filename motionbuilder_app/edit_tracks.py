from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .motion_math import interpolate_quaternions, local_from_world, quaternion_weighted, world_from_local


@dataclass
class EditOperation:
    id: str
    joint: int
    scope: str
    start_frame: int
    end_frame: int
    blend_frames: int
    translation: list[float]
    rotation: list[float]
    blend_in_frames: int | None = None
    blend_out_frames: int | None = None
    enabled: bool = True
    created_at: float = 0.0

    @classmethod
    def create(cls, **values: object) -> "EditOperation":
        return cls(id=uuid.uuid4().hex[:12], created_at=time.time(), **values)


class EditProject:
    def __init__(self, path: Path, num_frames: int, num_joints: int) -> None:
        self.path = path
        self.num_frames = int(num_frames)
        self.num_joints = int(num_joints)
        self.operations: list[EditOperation] = []
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.operations = [EditOperation(**row) for row in payload.get("operations", [])]
            for op in self.operations:
                op.start_frame = int(np.clip(op.start_frame, 0, max(0, self.num_frames - 1)))
                op.end_frame = int(np.clip(op.end_frame, op.start_frame, max(0, self.num_frames - 1)))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "operations": [asdict(op) for op in self.operations]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, op: EditOperation) -> EditOperation:
        if op.scope not in {"frame", "range", "all"}:
            raise ValueError("scope must be frame, range or all")
        if not 0 <= op.joint < self.num_joints:
            raise ValueError("joint out of range")
        op.start_frame = int(np.clip(op.start_frame, 0, self.num_frames - 1))
        op.end_frame = int(np.clip(op.end_frame, op.start_frame, self.num_frames - 1))
        op.blend_frames = max(0, min(int(op.blend_frames), 30))
        op.blend_in_frames = max(0, min(int(op.blend_in_frames if op.blend_in_frames is not None else op.blend_frames), 120))
        op.blend_out_frames = max(0, min(int(op.blend_out_frames if op.blend_out_frames is not None else op.blend_frames), 120))
        self.operations.append(op)
        self.save()
        return op

    def delete(self, operation_id: str) -> bool:
        before = len(self.operations)
        self.operations = [op for op in self.operations if op.id != operation_id]
        if len(self.operations) != before:
            self.save()
            return True
        return False

    def update_blend(self, operation_id: str, blend_in_frames: int, blend_out_frames: int) -> EditOperation:
        for op in self.operations:
            if op.id == operation_id:
                op.blend_in_frames = max(0, min(int(blend_in_frames), 120))
                op.blend_out_frames = max(0, min(int(blend_out_frames), 120))
                self.save()
                return op
        raise KeyError(operation_id)

    def take(self, operation_id: str) -> EditOperation:
        for index, op in enumerate(self.operations):
            if op.id == operation_id:
                self.operations.pop(index)
                self.save()
                return op
        raise KeyError(operation_id)

    def clear(self) -> None:
        self.operations.clear()
        self.save()

    def _range_weight(self, op: EditOperation, frame: int) -> float:
        if op.scope == "all":
            return 1.0
        # The selected range is the fully repaired (stable) region. Transitions
        # live immediately before/after it, so they do not consume repair frames.
        # A frame operation is the same shape with a one-frame stable region.
        if op.start_frame <= frame <= op.end_frame:
            return 1.0
        blend_in = int(op.blend_in_frames or 0)
        if blend_in and op.start_frame - blend_in <= frame < op.start_frame:
            amount = (frame - (op.start_frame - blend_in)) / blend_in
            return float(0.5 - 0.5 * np.cos(np.pi * amount))
        blend_out = int(op.blend_out_frames or 0)
        if blend_out and op.end_frame < frame <= op.end_frame + blend_out:
            amount = (op.end_frame + blend_out - frame) / blend_out
            return float(0.5 - 0.5 * np.cos(np.pi * amount))
        return 0.0

    def apply(self, base_world: np.ndarray, parents: np.ndarray) -> np.ndarray:
        local = local_from_world(base_world, parents).astype(np.float64)
        frame_ops: dict[int, list[EditOperation]] = {}
        for op in self.operations:
            if not op.enabled:
                continue
            blend_in = int(op.blend_in_frames if op.blend_in_frames is not None else op.blend_frames)
            blend_out = int(op.blend_out_frames if op.blend_out_frames is not None else op.blend_frames)
            # Keep the original key-to-key interpolation for unblended frame
            # keys. Explicit transition frames instead form a local fade around
            # the selected frame and are evaluated by _range_weight below.
            if op.scope == "frame" and blend_in == 0 and blend_out == 0:
                frame_ops.setdefault(op.joint, []).append(op)
                continue
            for frame in range(self.num_frames):
                weight = self._range_weight(op, frame)
                if weight <= 0:
                    continue
                local[frame, op.joint, :3, 3] += np.asarray(op.translation, dtype=np.float64) * weight
                local[frame, op.joint, :3, :3] = local[frame, op.joint, :3, :3] @ quaternion_weighted(op.rotation, weight)

        identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for joint, operations in frame_ops.items():
            keyed = sorted(operations, key=lambda op: (op.start_frame, op.created_at))
            for index, left in enumerate(keyed):
                right = keyed[index + 1] if index + 1 < len(keyed) else None
                end = right.start_frame if right else left.start_frame
                for frame in range(left.start_frame, end + 1):
                    amount = 0.0 if right is None or end == left.start_frame else (frame - left.start_frame) / (end - left.start_frame)
                    q = interpolate_quaternions(left.rotation, right.rotation if right else left.rotation, amount)
                    tr = (1.0 - amount) * np.asarray(left.translation) + amount * np.asarray(right.translation if right else left.translation)
                    local[frame, joint, :3, 3] += tr
                    local[frame, joint, :3, :3] = local[frame, joint, :3, :3] @ quaternion_weighted(q if np.linalg.norm(q) else identity, 1.0)
        return world_from_local(local, parents)
