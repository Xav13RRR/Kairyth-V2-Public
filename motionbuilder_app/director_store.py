from __future__ import annotations

import copy
import json
import math
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIRECTOR_PROJECTS_ROOT = PROJECT_ROOT / "data" / "director_projects"
DIRECTOR_EXPORTS_ROOT = PROJECT_ROOT / "data" / "director_exports"

ASPECT_PRESETS = {"16:9", "9:16", "1:1", "21:9", "custom"}
RESOLUTION_PRESETS = {"720p", "1080p", "custom"}
SCENE_ITEM_KINDS = {"skeletal_actor", "static_mesh", "terrain"}
INTERPOLATIONS = {"step", "linear", "cubic"}
CAMERA_PATH_KINDS = {"polyline", "catmull_rom"}
CAMERA_ORIENTATION_MODES = {"free", "rail_tangent", "look_at", "rail_level"}
LOOK_AT_KINDS = {"none", "actor", "bone", "object", "world"}
DIRECTOR_SEGMENT_KINDS = {"camera", "blackout"}
EXPORT_STATES = {"queued", "preparing", "rendering", "encoding", "packaging", "completed", "failed", "cancelled"}
EXPORT_PASSES = {"rgb", "depth", "normal", "segmentation"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any, default: float = 0.0) -> float:
    result = float(default if value is None else value)
    if not math.isfinite(result):
        raise ValueError("director values must be finite")
    return result


def _integer(value: Any, default: int = 0, minimum: int = 0, maximum: int = 2_000_000) -> int:
    result = int(default if value is None else value)
    if result < minimum or result > maximum:
        raise ValueError(f"director integer must be within {minimum}..{maximum}")
    return result


def _vector(value: Any, length: int, default: list[float]) -> list[float]:
    values = list(default if value is None else value)
    if len(values) != length:
        raise ValueError(f"director vector must contain {length} values")
    return [_finite(item) for item in values]


def _quaternion(value: Any) -> list[float]:
    values = _vector(value, 4, [0.0, 0.0, 0.0, 1.0])
    norm = math.sqrt(sum(item * item for item in values))
    if norm < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    return [item / norm for item in values]


def _identifier(value: Any, prefix: str) -> str:
    text = str(value or "").strip()
    return text[:128] if text else f"{prefix}-{uuid.uuid4().hex[:12]}"


def _name(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip()
    if not text:
        text = fallback
    return text[:120]


def _color(value: Any, fallback: str = "#d9e3f0") -> str:
    text = str(value or fallback).strip().lower()
    if len(text) == 7 and text.startswith("#") and all(char in "0123456789abcdef" for char in text[1:]):
        return text
    raise ValueError("director color must be #RRGGBB")


def _transform(value: Any) -> dict[str, Any]:
    source = value or {}
    scale = _finite(source.get("uniform_scale", source.get("scale", 1.0)), 1.0)
    if scale <= 0.0001 or scale > 1000.0:
        raise ValueError("director uniform scale must be within 0.0001..1000")
    return {
        "translation": _vector(source.get("translation"), 3, [0.0, 0.0, 0.0]),
        "quaternion": _quaternion(source.get("quaternion")),
        "uniform_scale": scale,
    }


def _range(value: Any, duration: int, *, allow_empty: bool = False) -> dict[str, int]:
    source = value or {}
    start = _integer(source.get("start_frame", 0), 0, 0, max(0, duration))
    minimum_end = start if allow_empty else start + 1
    end = _integer(source.get("end_frame", duration), duration, minimum_end, max(minimum_end, duration))
    return {"start_frame": start, "end_frame": end}


def _keyframes(values: Any, value_length: int | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in values or []:
        frame = _integer(raw.get("frame", 0), 0)
        value = raw.get("value")
        if value_length is not None:
            value = _vector(value, value_length, [0.0] * value_length)
            if value_length == 4:
                value = _quaternion(value)
        elif isinstance(value, (int, float)):
            value = _finite(value)
        interpolation = str(raw.get("interpolation", "linear")).lower()
        if interpolation not in INTERPOLATIONS:
            raise ValueError("director interpolation must be step, linear, or cubic")
        result.append({"frame": frame, "value": value, "interpolation": interpolation})
    result.sort(key=lambda item: item["frame"])
    deduped: dict[int, dict[str, Any]] = {item["frame"]: item for item in result}
    return [deduped[frame] for frame in sorted(deduped)]


def _normalize_scene_item(raw: dict[str, Any], duration: int) -> dict[str, Any]:
    kind = str(raw.get("kind", "static_mesh")).lower()
    if kind not in SCENE_ITEM_KINDS:
        raise ValueError("director scene item kind is invalid")
    source_asset_id = str(raw.get("source_asset_id") or raw.get("asset_id") or "").strip()
    if not source_asset_id:
        raise ValueError("director scene item source asset is required")
    source_animation_id = str(raw.get("source_animation_id") or raw.get("animation_id") or "").strip()
    if kind == "skeletal_actor" and not source_animation_id:
        raise ValueError("skeletal director items require an animation")
    timeline = _range(raw.get("timeline"), duration)
    source_range = raw.get("source_range") or {}
    source_start = _integer(source_range.get("start_frame", 0), 0)
    source_end = _integer(source_range.get("end_frame", source_start + 1), source_start + 1, source_start + 1)
    material_mode = str((raw.get("material_override") or {}).get("mode", "replace")).lower()
    if material_mode not in {"multiply", "replace"}:
        raise ValueError("director material mode must be multiply or replace")
    return {
        "id": _identifier(raw.get("id"), "item"),
        "name": _name(raw.get("name"), "Scene Item"),
        "kind": kind,
        "source_asset_id": source_asset_id,
        "source_animation_id": source_animation_id,
        "source_revision": str(raw.get("source_revision") or "")[:128],
        "transform": _transform(raw.get("transform")),
        "material_override": {
            "color": _color((raw.get("material_override") or {}).get("color"), "#e4ff4e" if kind != "terrain" else "#4287c7"),
            "mode": material_mode,
        },
        "timeline": timeline,
        "source_range": {"start_frame": source_start, "end_frame": source_end},
        "root_motion_enabled": bool(raw.get("root_motion_enabled", False)) if kind == "skeletal_actor" else False,
        "playback_rate": max(0.01, min(16.0, _finite(raw.get("playback_rate"), 1.0))),
        "end_behavior": str(raw.get("end_behavior", "hold")) if str(raw.get("end_behavior", "hold")) in {"hold", "loop", "hide"} else "hold",
        "visible": bool(raw.get("visible", True)),
        "locked": bool(raw.get("locked", False)),
        "transform_keys": {
            "translation": _keyframes((raw.get("transform_keys") or {}).get("translation"), 3),
            "rotation": _keyframes((raw.get("transform_keys") or {}).get("rotation"), 4),
            "scale": _keyframes((raw.get("transform_keys") or {}).get("scale")),
        },
        "visibility_ranges": [
            {**_range(item, duration), "visible": bool(item.get("visible", False))}
            for item in raw.get("visibility_ranges", [])
        ],
    }


def _normalize_look_at(value: Any) -> dict[str, Any]:
    source = value or {}
    kind = str(source.get("kind", "none")).lower()
    if kind not in LOOK_AT_KINDS:
        raise ValueError("director look-at kind is invalid")
    return {
        "enabled": bool(source.get("enabled", kind != "none")),
        "kind": kind,
        "target_id": str(source.get("target_id") or "")[:128],
        "bone_name": str(source.get("bone_name") or "")[:128],
        "world_position": _vector(source.get("world_position"), 3, [0.0, 1.0, 0.0]),
        "offset": _vector(source.get("offset"), 3, [0.0, 0.0, 0.0]),
        "strength": max(0.0, min(1.0, _finite(source.get("strength"), 1.0))),
        "keep_level": bool(source.get("keep_level", True)),
    }


def _normalize_camera(raw: dict[str, Any]) -> dict[str, Any]:
    fov = max(5.0, min(170.0, _finite(raw.get("fov"), 42.0)))
    near = max(0.001, min(1000.0, _finite(raw.get("near"), 0.01)))
    far = max(near + 0.01, min(100000.0, _finite(raw.get("far"), 100.0)))
    shake = raw.get("shake") or {}
    return {
        "id": _identifier(raw.get("id"), "camera"),
        "number": str(raw.get("number") or "")[:32],
        "name": _name(raw.get("name"), "Camera"),
        "notes": str(raw.get("notes") or "")[:4000],
        "transform": _transform(raw.get("transform")),
        "fov": fov,
        "near": near,
        "far": far,
        "look_at": _normalize_look_at(raw.get("look_at")),
        "path_id": str(raw.get("path_id") or "")[:128],
        "orientation_mode": str(raw.get("orientation_mode", "free")) if str(raw.get("orientation_mode", "free")) in CAMERA_ORIENTATION_MODES else "free",
        "keys": {
            "translation": _keyframes((raw.get("keys") or {}).get("translation"), 3),
            "rotation": _keyframes((raw.get("keys") or {}).get("rotation"), 4),
            "fov": _keyframes((raw.get("keys") or {}).get("fov")),
            "path_position": _keyframes((raw.get("keys") or {}).get("path_position")),
        },
        "shake": {
            "enabled": bool(shake.get("enabled", False)),
            "position_amplitude": _vector(shake.get("position_amplitude"), 3, [0.0, 0.0, 0.0]),
            "rotation_amplitude": _vector(shake.get("rotation_amplitude"), 3, [0.0, 0.0, 0.0]),
            "frequency": max(0.01, min(60.0, _finite(shake.get("frequency"), 1.5))),
            "seed": _integer(shake.get("seed", 1), 1, 0, 2_147_483_647),
            "start_frame": _integer(shake.get("start_frame", 0), 0),
            "end_frame": _integer(shake.get("end_frame", 0), 0),
            "blend_in_frames": _integer(shake.get("blend_in_frames", 0), 0, 0, 2400),
            "blend_out_frames": _integer(shake.get("blend_out_frames", 0), 0, 0, 2400),
        },
    }


def _normalize_camera_path(raw: dict[str, Any]) -> dict[str, Any]:
    kind = str(raw.get("kind", "catmull_rom")).lower()
    if kind not in CAMERA_PATH_KINDS:
        raise ValueError("director camera path kind is invalid")
    points = []
    for point in raw.get("points", []):
        points.append({
            "id": _identifier(point.get("id"), "path-point"),
            "position": _vector(point.get("position"), 3, [0.0, 1.0, 0.0]),
        })
    if len(points) < 2:
        raise ValueError("director camera path requires at least two points")
    return {
        "id": _identifier(raw.get("id"), "path"),
        "name": _name(raw.get("name"), "Camera Path"),
        "kind": kind,
        "closed": bool(raw.get("closed", False)),
        "points": points,
        "duration_frames": _integer(raw.get("duration_frames", 90), 90, 1),
        "speed_mode": str(raw.get("speed_mode", "duration")) if str(raw.get("speed_mode", "duration")) in {"duration", "constant", "keys"} else "duration",
        "speed": max(0.001, min(1000.0, _finite(raw.get("speed"), 1.0))),
    }


def _normalize_cut(raw: dict[str, Any], duration: int, camera_ids: set[str], order: int = 0) -> dict[str, Any]:
    bounds = _range(raw, duration)
    kind = str(raw.get("kind") or "camera").lower()
    if kind not in DIRECTOR_SEGMENT_KINDS:
        raise ValueError("director segment kind must be camera or blackout")
    camera_id = str(raw.get("camera_id") or "")
    if kind == "camera" and camera_id not in camera_ids:
        raise ValueError("director camera cut references a missing camera")
    if kind == "blackout":
        camera_id = ""
    default_name = "Blackout" if kind == "blackout" else "Shot"
    default_color = "#111111" if kind == "blackout" else "#e4ff4e"
    return {
        "id": _identifier(raw.get("id"), "shot"),
        "kind": kind,
        "order": _integer(raw.get("order", order), order, 0, 2_000_000),
        "number": str(raw.get("number") or "")[:32],
        "name": _name(raw.get("name"), default_name),
        "notes": str(raw.get("notes") or "")[:4000],
        "camera_id": camera_id,
        "color": _color(raw.get("color"), default_color),
        **bounds,
        "blend_frames": _integer(raw.get("blend_frames", 0), 0, 0, min(2400, bounds["end_frame"] - bounds["start_frame"])),
        "thumbnail_frame": _integer(raw.get("thumbnail_frame", bounds["start_frame"]), bounds["start_frame"], bounds["start_frame"], bounds["end_frame"] - 1),
        "time_warp_id": str(raw.get("time_warp_id") or "")[:128],
    }


def _normalize_project(payload: dict[str, Any], *, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    current = existing or {}
    duration = _integer(payload.get("duration_frames", current.get("duration_frames", 300)), 300, 1, 2_000_000)
    aspect = str(payload.get("aspect_preset", current.get("aspect_preset", "16:9")))
    resolution = str(payload.get("resolution_preset", current.get("resolution_preset", "1080p")))
    if aspect not in ASPECT_PRESETS:
        raise ValueError("director aspect preset is invalid")
    if resolution not in RESOLUTION_PRESETS:
        raise ValueError("director resolution preset is invalid")
    custom_resolution = payload.get("custom_resolution", current.get("custom_resolution", [1920, 1080]))
    custom_resolution = [_integer(item, 1, 1, 16384) for item in list(custom_resolution)]
    if len(custom_resolution) != 2:
        raise ValueError("director custom resolution requires width and height")
    scene_items = [_normalize_scene_item(item, duration) for item in payload.get("scene_items", current.get("scene_items", []))]
    cameras = [_normalize_camera(item) for item in payload.get("cameras", current.get("cameras", []))]
    camera_ids = {item["id"] for item in cameras}
    paths = [_normalize_camera_path(item) for item in payload.get("camera_paths", current.get("camera_paths", []))]
    cuts = [
        _normalize_cut(item, duration, camera_ids, index)
        for index, item in enumerate(payload.get("camera_cuts", current.get("camera_cuts", [])))
    ]
    cuts.sort(key=lambda item: (item["order"], item["start_frame"], item["end_frame"]))
    previous_end = 0
    for index, cut in enumerate(cuts):
        if cut["start_frame"] < previous_end:
            raise ValueError("director segments must not overlap")
        cut["order"] = index
        previous_end = cut["end_frame"]
    markers = []
    for raw in payload.get("markers", current.get("markers", [])):
        markers.append({
            "id": _identifier(raw.get("id"), "marker"),
            "frame": _integer(raw.get("frame", 0), 0, 0, duration - 1),
            "name": _name(raw.get("name"), "Marker"),
            "notes": str(raw.get("notes") or "")[:4000],
            "color": _color(raw.get("color"), "#e4ff4e"),
            "kind": str(raw.get("kind") or "general")[:32],
        })
    time_warps = []
    for raw in payload.get("time_warps", current.get("time_warps", [])):
        bounds = _range(raw, duration)
        mode = str(raw.get("mode", "rate"))
        if mode not in {"rate", "freeze", "curve"}:
            raise ValueError("director time warp mode is invalid")
        time_warps.append({
            "id": _identifier(raw.get("id"), "warp"),
            "name": _name(raw.get("name"), "Time Warp"),
            **bounds,
            "mode": mode,
            "rate": max(0.001, min(16.0, _finite(raw.get("rate"), 1.0))),
            "source_frame": _integer(raw.get("source_frame", bounds["start_frame"]), bounds["start_frame"]),
            "keys": _keyframes(raw.get("keys")),
            "camera_follows_warp": bool(raw.get("camera_follows_warp", False)),
        })
    guides = payload.get("guides", current.get("guides", {})) or {}
    export_settings = payload.get("export_settings", current.get("export_settings", {})) or {}
    now = time.time()
    result = {
        "id": str(current.get("id") or payload.get("id") or uuid.uuid4().hex[:12]),
        "schema_version": 2,
        "name": _name(payload.get("name", current.get("name")), "Untitled Director Project"),
        "description": str(payload.get("description", current.get("description", "")))[:4000],
        "fps": max(1.0, min(120.0, _finite(payload.get("fps", current.get("fps", 30.0)), 30.0))),
        "duration_frames": duration,
        "aspect_preset": aspect,
        "resolution_preset": resolution,
        "custom_resolution": custom_resolution,
        "preview_quality": max(0.25, min(1.0, _finite(payload.get("preview_quality", current.get("preview_quality", 0.5)), 0.5))),
        "scene_items": scene_items,
        "cameras": cameras,
        "camera_paths": paths,
        "camera_cuts": cuts,
        "markers": markers,
        "time_warps": time_warps,
        "guides": {
            "safe_frame": bool(guides.get("safe_frame", True)),
            "center": bool(guides.get("center", False)),
            "thirds": bool(guides.get("thirds", True)),
        },
        "export_settings": {
            "format": str(export_settings.get("format", "mp4")) if str(export_settings.get("format", "mp4")) in {"mp4", "webm"} else "mp4",
            "bitrate": _integer(export_settings.get("bitrate", 8_000_000), 8_000_000, 250_000, 200_000_000),
            "include_depth": bool(export_settings.get("include_depth", False)),
            "include_normal": bool(export_settings.get("include_normal", False)),
            "include_segmentation": bool(export_settings.get("include_segmentation", False)),
        },
        "created_at": float(current.get("created_at", now)),
        "updated_at": now,
        "revision": int(current.get("revision", 0)),
    }
    ids = [item["id"] for item in scene_items]
    ids += [item["id"] for item in cameras]
    ids += [item["id"] for item in paths]
    ids += [item["id"] for item in cuts]
    ids += [item["id"] for item in markers]
    ids += [item["id"] for item in time_warps]
    if len(ids) != len(set(ids)):
        raise ValueError("director object ids must be unique")
    return result


class DirectorProjectStore:
    def __init__(self, root: Path = DIRECTOR_PROJECTS_ROOT, export_root: Path = DIRECTOR_EXPORTS_ROOT) -> None:
        self.root = Path(root)
        self.export_root = Path(export_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.export_root.mkdir(parents=True, exist_ok=True)
        self._encoding_jobs: set[str] = set()
        self._encoding_lock = threading.Lock()

    def _project_path(self, project_id: str) -> Path:
        return self.root / Path(project_id).name / "manifest.json"

    def get(self, project_id: str) -> dict[str, Any]:
        path = self._project_path(project_id)
        if not path.is_file():
            raise KeyError(project_id)
        return _read_json(path)

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for path in self.root.glob("*/manifest.json"):
            try:
                row = _read_json(path)
                rows.append({
                    "id": row["id"],
                    "name": row.get("name", "Untitled Director Project"),
                    "description": row.get("description", ""),
                    "fps": row.get("fps", 30.0),
                    "duration_frames": row.get("duration_frames", 1),
                    "aspect_preset": row.get("aspect_preset", "16:9"),
                    "resolution_preset": row.get("resolution_preset", "1080p"),
                    "scene_item_count": len(row.get("scene_items", [])),
                    "camera_count": len(row.get("cameras", [])),
                    "shot_count": sum(1 for cut in row.get("camera_cuts", []) if cut.get("kind", "camera") == "camera"),
                    "blackout_count": sum(1 for cut in row.get("camera_cuts", []) if cut.get("kind") == "blackout"),
                    "revision": row.get("revision", 0),
                    "updated_at": row.get("updated_at", 0),
                    "created_at": row.get("created_at", 0),
                })
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: float(row.get("updated_at", 0)), reverse=True)

    def create(self, name: str = "Untitled Director Project", payload: dict[str, Any] | None = None) -> dict[str, Any]:
        source = dict(payload or {})
        source["name"] = name
        row = _normalize_project(source)
        row["revision"] = 1
        _write_json(self._project_path(row["id"]), row)
        return row

    def update(self, project_id: str, payload: dict[str, Any], expected_revision: int | None = None) -> dict[str, Any]:
        current = self.get(project_id)
        if expected_revision is not None and int(expected_revision) != int(current.get("revision", 0)):
            raise RuntimeError("director project revision conflict")
        row = _normalize_project(payload, existing=current)
        row["id"] = current["id"]
        row["created_at"] = current.get("created_at", row["created_at"])
        row["revision"] = int(current.get("revision", 0)) + 1
        _write_json(self._project_path(project_id), row)
        return row

    def duplicate(self, project_id: str, name: str | None = None) -> dict[str, Any]:
        source = copy.deepcopy(self.get(project_id))
        source.pop("id", None)
        source.pop("created_at", None)
        source.pop("updated_at", None)
        source.pop("revision", None)
        return self.create(name or f"{source.get('name', 'Director Project')} Copy", source)

    def delete(self, project_id: str) -> None:
        directory = self._project_path(project_id).parent.resolve()
        if directory.parent != self.root.resolve() or not directory.is_dir():
            raise KeyError(project_id)
        shutil.rmtree(directory)

    def create_snapshot(self, project_id: str, name: str = "Snapshot", kind: str = "manual") -> dict[str, Any]:
        project = self.get(project_id)
        snapshot_id = uuid.uuid4().hex[:12]
        row = {
            "id": snapshot_id,
            "name": _name(name, "Snapshot"),
            "kind": str(kind or "manual")[:32],
            "created_at": time.time(),
            "project_revision": int(project.get("revision", 0)),
            "project": project,
        }
        _write_json(self._project_path(project_id).parent / "snapshots" / f"{snapshot_id}.json", row)
        return {key: value for key, value in row.items() if key != "project"}

    def list_snapshots(self, project_id: str) -> list[dict[str, Any]]:
        self.get(project_id)
        rows = []
        root = self._project_path(project_id).parent / "snapshots"
        for path in root.glob("*.json"):
            try:
                row = _read_json(path)
                rows.append({key: value for key, value in row.items() if key != "project"})
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: float(row.get("created_at", 0)), reverse=True)

    def restore_snapshot(self, project_id: str, snapshot_id: str) -> dict[str, Any]:
        current = self.get(project_id)
        path = self._project_path(project_id).parent / "snapshots" / f"{Path(snapshot_id).name}.json"
        if not path.is_file():
            raise KeyError(snapshot_id)
        snapshot = _read_json(path)
        self.create_snapshot(project_id, f"Before restore {snapshot.get('name', snapshot_id)}", "pre_restore")
        restored = _normalize_project(snapshot.get("project") or {}, existing=current)
        restored["id"] = current["id"]
        restored["created_at"] = current.get("created_at", restored["created_at"])
        restored["revision"] = int(current.get("revision", 0)) + 1
        _write_json(self._project_path(project_id), restored)
        return restored

    def _job_path(self, project_id: str, job_id: str) -> Path:
        return self.export_root / Path(project_id).name / Path(job_id).name / "job.json"

    def create_export(self, project_id: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        project = self.get(project_id)
        job_id = uuid.uuid4().hex[:12]
        now = time.time()
        row = {
            "id": job_id,
            "project_id": project_id,
            "project_revision": project.get("revision", 0),
            "state": "queued",
            "stage": "queued",
            "progress": 0.0,
            "completed_frames": 0,
            "total_frames": int(project.get("duration_frames", 1)),
            "last_checkpoint_frame": -1,
            "settings": settings or project.get("export_settings", {}),
            "error": "",
            "created_at": now,
            "updated_at": now,
            "cancel_requested": False,
            "encoding_attempt": 0,
            "artifacts": [],
        }
        path = self._job_path(project_id, job_id)
        _write_json(path, row)
        _write_json(path.parent / "project.json", project)
        return row

    def get_export(self, project_id: str, job_id: str) -> dict[str, Any]:
        path = self._job_path(project_id, job_id)
        if not path.is_file():
            raise KeyError(job_id)
        return _read_json(path)

    def list_exports(self, project_id: str) -> list[dict[str, Any]]:
        self.get(project_id)
        rows = []
        root = self.export_root / Path(project_id).name
        for path in root.glob("*/job.json"):
            try:
                rows.append(_read_json(path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: float(row.get("updated_at", 0)), reverse=True)

    def update_export(self, project_id: str, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self.get_export(project_id, job_id)
        state = str(payload.get("state", row.get("state", "queued")))
        if state not in EXPORT_STATES:
            raise ValueError("director export state is invalid")
        row["state"] = state
        row["stage"] = str(payload.get("stage", row.get("stage", state)))[:64]
        row["progress"] = max(0.0, min(1.0, _finite(payload.get("progress", row.get("progress", 0.0)))))
        row["completed_frames"] = _integer(payload.get("completed_frames", row.get("completed_frames", 0)), 0)
        row["last_checkpoint_frame"] = int(payload.get("last_checkpoint_frame", row.get("last_checkpoint_frame", -1)))
        row["error"] = str(payload.get("error", row.get("error", "")))[:8000]
        if "encoding_attempt" in payload:
            row["encoding_attempt"] = _integer(payload.get("encoding_attempt"), int(row.get("encoding_attempt", 0)), 0)
        if "artifacts" in payload:
            row["artifacts"] = [str(value)[:256] for value in payload.get("artifacts", [])]
        row["updated_at"] = time.time()
        _write_json(self._job_path(project_id, job_id), row)
        return row

    def cancel_export(self, project_id: str, job_id: str) -> dict[str, Any]:
        row = self.get_export(project_id, job_id)
        if row.get("state") not in {"completed", "failed"}:
            row["state"] = "cancelled"
            row["stage"] = "cancelled"
            row["cancel_requested"] = True
            row["updated_at"] = time.time()
            _write_json(self._job_path(project_id, job_id), row)
        return row

    def resume_export(self, project_id: str, job_id: str) -> dict[str, Any]:
        row = self.get_export(project_id, job_id)
        if row.get("state") not in {"cancelled", "failed"}:
            raise ValueError("only cancelled or failed exports can be resumed")
        row["state"] = "queued"
        row["stage"] = "queued"
        row["cancel_requested"] = False
        # Invalidate a worker that may still be unwinding after cancellation.
        row["encoding_attempt"] = int(row.get("encoding_attempt", 0)) + 1
        row["error"] = ""
        row["updated_at"] = time.time()
        _write_json(self._job_path(project_id, job_id), row)
        return row

    def save_artifact(self, project_id: str, job_id: str, relative_name: str, data: bytes) -> dict[str, Any]:
        row = self.get_export(project_id, job_id)
        relative = Path(relative_name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) > 5:
            raise ValueError("invalid director export artifact path")
        if not data:
            raise ValueError("director export artifact is empty")
        root = self._job_path(project_id, job_id).parent.resolve()
        target = (root / "artifacts" / relative).resolve()
        if root not in target.parents:
            raise ValueError("invalid director export artifact path")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        artifact_name = relative.as_posix()
        artifacts = list(row.get("artifacts", []))
        if artifact_name not in artifacts:
            artifacts.append(artifact_name)
        return self.update_export(project_id, job_id, {**row, "artifacts": artifacts})

    def package_export(self, project_id: str, job_id: str) -> Path:
        row = self.get_export(project_id, job_id)
        root = self._job_path(project_id, job_id).parent
        package = root / f"{Path(project_id).name}_{Path(job_id).name}_reference.zip"
        with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            project = root / "project.json"
            if project.is_file():
                archive.write(project, "manifest.json")
            artifacts = root / "artifacts"
            if artifacts.is_dir():
                for path in artifacts.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(artifacts).as_posix())
            # RGB frames are represented by the director video. Optional control
            # passes stay lossless and are bundled for downstream generation tools.
            frames = root / "frames"
            if frames.is_dir():
                for pass_name in ("depth", "normal", "segmentation"):
                    pass_root = frames / pass_name
                    if not pass_root.is_dir():
                        continue
                    for path in pass_root.glob("*.*"):
                        if path.is_file():
                            archive.write(path, f"control_passes/{pass_name}/{path.name}")
        self.update_export(project_id, job_id, {**row, "state": "completed", "stage": "completed", "progress": 1.0})
        return package

    @staticmethod
    def output_resolution(project: dict[str, Any]) -> tuple[int, int]:
        aspect = str(project.get("aspect_preset", "16:9"))
        quality = str(project.get("resolution_preset", "1080p"))
        presets = {
            ("16:9", "720p"): (1280, 720),
            ("16:9", "1080p"): (1920, 1080),
            ("9:16", "720p"): (720, 1280),
            ("9:16", "1080p"): (1080, 1920),
            ("1:1", "720p"): (720, 720),
            ("1:1", "1080p"): (1080, 1080),
            ("21:9", "720p"): (1680, 720),
            ("21:9", "1080p"): (2520, 1080),
        }
        if aspect == "custom" or quality == "custom":
            custom = project.get("custom_resolution", [1920, 1080])
            return int(custom[0]), int(custom[1])
        return presets.get((aspect, quality), (1920, 1080))

    def save_frame(self, project_id: str, job_id: str, pass_name: str, frame: int, data: bytes) -> dict[str, Any]:
        row = self.get_export(project_id, job_id)
        pass_name = str(pass_name).lower()
        if pass_name not in EXPORT_PASSES:
            raise ValueError("director export pass is invalid")
        frame = _integer(frame, 0, 0, max(0, int(row.get("total_frames", 1)) - 1))
        if not data or not (data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8")):
            raise ValueError("director export frame must be PNG or JPEG")
        root = self._job_path(project_id, job_id).parent / "frames" / pass_name
        root.mkdir(parents=True, exist_ok=True)
        suffix = ".png" if data.startswith(b"\x89PNG") else ".jpg"
        target = root / f"{frame:08d}{suffix}"
        target.write_bytes(data)
        if pass_name == "rgb":
            total = max(1, int(row.get("total_frames", 1)))
            completed = len(list(root.glob("*.*")))
            row = self.update_export(project_id, job_id, {
                **row,
                "state": "rendering",
                "stage": "rendering",
                "completed_frames": completed,
                "last_checkpoint_frame": max(int(row.get("last_checkpoint_frame", -1)), frame),
                "progress": min(0.85, completed / total * 0.85),
            })
        return row

    def missing_frames(self, project_id: str, job_id: str, pass_name: str = "rgb") -> dict[str, Any]:
        row = self.get_export(project_id, job_id)
        pass_name = str(pass_name).lower()
        if pass_name not in EXPORT_PASSES:
            raise ValueError("director export pass is invalid")
        root = self._job_path(project_id, job_id).parent / "frames" / pass_name
        present = set()
        if root.is_dir():
            for path in root.glob("*.*"):
                try:
                    present.add(int(path.stem))
                except ValueError:
                    continue
        total = int(row.get("total_frames", 1))
        missing = [frame for frame in range(total) if frame not in present]
        return {"pass": pass_name, "total_frames": total, "completed_frames": total - len(missing), "missing_frames": missing}

    def start_encoding(self, project_id: str, job_id: str) -> dict[str, Any]:
        row = self.get_export(project_id, job_id)
        attempt = int(row.get("encoding_attempt", 0)) + 1
        row["encoding_attempt"] = attempt
        key = f"{Path(project_id).name}:{Path(job_id).name}:{attempt}"
        with self._encoding_lock:
            if key in self._encoding_jobs:
                return row
            self._encoding_jobs.add(key)
        self.update_export(project_id, job_id, {**row, "state": "encoding", "stage": "encoding", "progress": max(0.86, float(row.get("progress", 0)))})
        thread = threading.Thread(target=self._encode_worker, args=(project_id, job_id, key, attempt), daemon=True)
        thread.start()
        return self.get_export(project_id, job_id)

    def _encode_worker(self, project_id: str, job_id: str, key: str, attempt: int) -> None:
        try:
            import cv2

            row = self.get_export(project_id, job_id)
            project = _read_json(self._job_path(project_id, job_id).parent / "project.json")
            missing = self.missing_frames(project_id, job_id, "rgb")["missing_frames"]
            if missing:
                raise ValueError(f"director export is missing {len(missing)} RGB frames")
            frame_root = self._job_path(project_id, job_id).parent / "frames" / "rgb"
            frame_paths = sorted(frame_root.glob("*.*"))
            if not frame_paths:
                raise ValueError("director export contains no RGB frames")
            width, height = self.output_resolution(project)
            requested = str((row.get("settings") or {}).get("format", "mp4")).lower()
            suffix = ".webm" if requested == "webm" else ".mp4"
            codec = cv2.VideoWriter_fourcc(*("VP80" if suffix == ".webm" else "mp4v"))
            output = self._job_path(project_id, job_id).parent / "artifacts" / f"director{suffix}"
            output.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(output), codec, float(project.get("fps", 30.0)), (width, height))
            if not writer.isOpened():
                raise RuntimeError("video encoder could not be opened")
            try:
                total = len(frame_paths)
                for index, path in enumerate(frame_paths):
                    current = self.get_export(project_id, job_id)
                    if int(current.get("encoding_attempt", 0)) != attempt:
                        raise InterruptedError("director export encoding superseded")
                    if current.get("cancel_requested"):
                        raise InterruptedError("director export cancelled")
                    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError(f"could not decode export frame {path.name}")
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    writer.write(frame)
                    if index % 10 == 0 or index + 1 == total:
                        self.update_export(project_id, job_id, {
                            **current,
                            "state": "encoding",
                            "stage": "encoding",
                            "progress": 0.86 + (index + 1) / total * 0.1,
                        })
            finally:
                writer.release()
            current = self.get_export(project_id, job_id)
            if int(current.get("encoding_attempt", 0)) != attempt:
                raise InterruptedError("director export encoding superseded")
            artifacts = list(current.get("artifacts", []))
            name = output.relative_to(output.parent).as_posix()
            if name not in artifacts:
                artifacts.append(name)
            self.update_export(project_id, job_id, {
                **current,
                "state": "packaging",
                "stage": "packaging",
                "progress": 0.97,
                "artifacts": artifacts,
            })
            self.package_export(project_id, job_id)
        except InterruptedError:
            try:
                current = self.get_export(project_id, job_id)
                if int(current.get("encoding_attempt", 0)) == attempt:
                    self.cancel_export(project_id, job_id)
            except KeyError:
                pass
        except Exception as exc:
            try:
                row = self.get_export(project_id, job_id)
                self.update_export(project_id, job_id, {**row, "state": "failed", "stage": "failed", "error": str(exc)})
            except KeyError:
                pass
        finally:
            with self._encoding_lock:
                self._encoding_jobs.discard(key)
