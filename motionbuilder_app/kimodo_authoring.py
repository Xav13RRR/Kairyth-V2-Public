from __future__ import annotations

import copy
import math
import uuid
from typing import Any


CONSTRAINT_TYPES = {"root2d", "fullbody"}
SOMA_JOINT_COUNTS = {30, 77}


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _vector(value: Any, size: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must contain {size} numbers")
    return [_finite(item, label) for item in value]


def _matrix(value: Any, rows: int, cols: int, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{label} row count must match frame_indices")
    return [_vector(row, cols, label) for row in value]


def normalize_constraints(
    raw: list[dict[str, Any]] | None,
    *,
    total_frames: int | None = None,
) -> list[dict[str, Any]]:
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError("Kimodo constraints must be a list")
    if len(raw) > 512:
        raise ValueError("Kimodo constraints exceed the 512 item limit")
    cleaned: list[dict[str, Any]] = []
    for constraint_index, source in enumerate(copy.deepcopy(raw)):
        if not isinstance(source, dict):
            raise ValueError(f"Kimodo constraint #{constraint_index + 1} must be an object")
        kind = str(source.get("type", "")).strip()
        if kind not in CONSTRAINT_TYPES:
            raise ValueError(f"Unsupported Kimodo constraint type: {kind or '(empty)'}")
        indices_raw = source.get("frame_indices")
        if not isinstance(indices_raw, list) or not indices_raw:
            raise ValueError(f"Kimodo constraint #{constraint_index + 1} has no frame indices")
        if len(indices_raw) > 18000:
            raise ValueError("A Kimodo constraint cannot exceed 18000 frames")
        indices = [int(item) for item in indices_raw]
        if any(item < 0 for item in indices):
            raise ValueError("Kimodo constraint frame indices cannot be negative")
        if total_frames is not None and any(item >= total_frames for item in indices):
            raise ValueError("Kimodo constraint frame lies outside the generated motion")
        if indices != sorted(set(indices)):
            raise ValueError("Kimodo constraint frame indices must be unique and ordered")
        item: dict[str, Any] = {"type": kind, "frame_indices": indices}
        count = len(indices)
        if kind == "root2d":
            item["smooth_root_2d"] = _matrix(source.get("smooth_root_2d"), count, 2, "smooth_root_2d")
            if source.get("global_root_heading") is not None:
                headings = _matrix(source["global_root_heading"], count, 2, "global_root_heading")
                item["global_root_heading"] = headings
        else:
            roots = _matrix(source.get("root_positions"), count, 3, "root_positions")
            rotations = source.get("local_joints_rot")
            if not isinstance(rotations, list) or len(rotations) != count:
                raise ValueError("local_joints_rot row count must match frame_indices")
            joint_count = len(rotations[0]) if rotations and isinstance(rotations[0], list) else 0
            if joint_count not in SOMA_JOINT_COUNTS:
                raise ValueError("local_joints_rot must use a SOMA-30 or SOMA-77 skeleton")
            clean_rotations: list[list[list[float]]] = []
            for frame in rotations:
                if not isinstance(frame, list) or len(frame) != joint_count:
                    raise ValueError("local_joints_rot must have a stable joint count")
                clean_rotations.append([_vector(rotation, 3, "local_joints_rot") for rotation in frame])
            item["root_positions"] = roots
            item["local_joints_rot"] = clean_rotations
            if source.get("smooth_root_2d") is not None:
                item["smooth_root_2d"] = _matrix(source["smooth_root_2d"], count, 2, "smooth_root_2d")
        cleaned.append(item)
    return cleaned


def normalize_preview_scene(raw: dict[str, Any] | None) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    rows = source.get("scene_objects", [])
    if not isinstance(rows, list):
        raise ValueError("preview_scene.scene_objects must be a list")
    if len(rows) > 100:
        raise ValueError("Kimodo preview scene cannot exceed 100 objects")
    objects: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Kimodo preview scene objects must be objects")
        kind = str(row.get("asset_kind", "")).strip()
        if kind not in {"static_mesh", "terrain"}:
            raise ValueError(f"Unsupported Kimodo preview asset kind: {kind}")
        source_id = str(row.get("source_asset_id", "")).strip()
        if not source_id:
            raise ValueError("Kimodo preview scene objects require source_asset_id")
        transform = row.get("transform") if isinstance(row.get("transform"), dict) else {}
        translation = _vector(transform.get("translation", [0, 0, 0]), 3, "translation")
        quaternion = _vector(transform.get("quaternion", [0, 0, 0, 1]), 4, "quaternion")
        length = math.sqrt(sum(value * value for value in quaternion))
        quaternion = [0.0, 0.0, 0.0, 1.0] if length < 1e-8 else [value / length for value in quaternion]
        scale_value = transform.get("uniform_scale", transform.get("scale", 1.0))
        if isinstance(scale_value, list):
            scale_value = scale_value[0] if scale_value else 1.0
        uniform_scale = min(1000.0, max(0.001, _finite(scale_value, "uniform_scale")))
        objects.append(
            {
                "id": str(row.get("id") or f"scene-{uuid.uuid4().hex[:10]}")[:80],
                "name": str(row.get("name") or kind)[:80],
                "asset_kind": kind,
                "source_asset_id": source_id[:128],
                "source_revision": max(0, int(row.get("source_revision", 0))),
                "transform": {
                    "translation": translation,
                    "quaternion": quaternion,
                    "uniform_scale": uniform_scale,
                },
                "editor_hidden": bool(row.get("editor_hidden", False)),
            }
        )
    return {"scene_objects": objects}
