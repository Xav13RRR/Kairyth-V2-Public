from __future__ import annotations

import argparse
import asyncio
import mimetypes
import json
import os
import shutil
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from pathlib import Path
from typing import Literal

import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .meshpreps.config import (
    DATA_ROOT,
    DEFAULT_BIND_TIMEOUT_S,
    DEFAULT_PORT,
    DEFAULT_SEED,
    SHUTDOWN_GRACE_S,
    STATIC_DIR,
    UPLOAD_DIR,
    is_pipeline_mode,
)
from .meshpreps.env_util import check_setup
from .meshpreps.mesh_align import soma30_reference_dict
from .meshpreps.session import SessionStore
from .edit_tracks import EditOperation, EditProject
from .exporter import export_fbx, find_blender, is_fbx_export_current
from .stores import CharacterStore, MotionStore, MOTIONS_ROOT
from .library_stores import AnimationStore, CompositionStore, RetargetPreviewStore, UpperLowerCompositePreviewStore, VirtualRootPreviewStore, ANIMATIONS_ROOT, COMPOSITIONS_ROOT
from .ik_constraints import apply_ik_constraints
from .additive_store import AdditiveCompositePreviewStore
from .gait_ik import GaitIKSettings
from .gait_store import GaitIKPreviewStore
from .foot_lock import FootLockSettings
from .foot_lock_store import FootLockPreviewStore
from .kimodo_store import KimodoMotionStore
from .kimodo_runtime import kimodo_default_soma_scene
from .pose_store import PoseStore
from .terrain_store import TerrainStore
from .terrain_routes import build_terrain_router
from .terrain_surface import heightfield_mesh
from .director_store import DirectorProjectStore
from .release_profile import capabilities, require_inference, InferenceUnavailableError

os.environ.setdefault("KAIRYTH_V2_INSTANCE_ID", uuid.uuid4().hex)

store = SessionStore()
app = FastAPI(title="MotionBuilder Studio", version="1.0.0", docs_url="/api/docs")


def _release_error_response(exc: InferenceUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": exc.detail["message"], **exc.detail,
                                                 "capabilities": exc.capabilities})


@app.exception_handler(InferenceUnavailableError)
async def inference_unavailable_handler(request: Request, exc: InferenceUnavailableError):
    return _release_error_response(exc)


@app.middleware("http")
async def research_release_guard(request: Request, call_next):
    # Run before multipart parsing/spooling: disabled uploads are not consumed.
    backend = {"/api/image-poses": "hybrik", "/api/motions": "gvhmr"}.get(request.url.path.rstrip("/"))
    if request.method == "POST" and backend:
        try: require_inference(backend)
        except InferenceUnavailableError as exc:
            return _release_error_response(exc)
    return await call_next(request)


def _require_inference_api(backend: str) -> None:
    require_inference(backend)
characters = CharacterStore()
motions = MotionStore(characters)
kimodo_motions = KimodoMotionStore()
poses = PoseStore()
terrains = TerrainStore()
animations = AnimationStore(characters, motions)
compositions = CompositionStore(animations)
retarget_previews = RetargetPreviewStore(characters, motions, animations, kimodo_motions)
virtual_root_previews = VirtualRootPreviewStore(animations)
upper_lower_previews = UpperLowerCompositePreviewStore(animations)
additive_previews = AdditiveCompositePreviewStore(animations)
gait_ik_previews = GaitIKPreviewStore(animations, terrains)
foot_lock_previews = FootLockPreviewStore(animations)
director_projects = DirectorProjectStore()
# Register concrete routes so app.routes remains a flat APIRoute list; the
# project's API contract tests and local route introspection rely on that.
app.router.routes.extend(build_terrain_router(terrains).routes)


def _download_format(request: Request, default: str = "binding") -> str:
    fmt = request.query_params.get("file_format") or request.query_params.get("format") or default
    return str(fmt).strip().lower()


def _binary_download(path: Path, filename: str, media_type: str) -> Response:
    if not path.is_file():
        raise HTTPException(404, "export file missing")
    data = path.read_bytes()
    if not data:
        raise HTTPException(500, "export file is empty")
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _assert_npz_bytes(data: bytes) -> None:
    if len(data) < 4 or data[:2] != b"PK":
        raise HTTPException(500, "invalid npz export")


def _assert_fbx_bytes(data: bytes) -> None:
    if len(data) < 21:
        raise HTTPException(500, "invalid fbx export")
    if not (data.startswith(b"Kaydara FBX Binary") or data.startswith(b"; FBX")):
        raise HTTPException(500, "invalid fbx export")


def _normalize_download_format(fmt: str) -> str:
    return str(fmt or "binding").strip().lower()


def _parse_bind_matrices_override(rows: list[list[float]] | None) -> np.ndarray | None:
    if rows is None:
        return None
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise HTTPException(400, f"bind_matrices must be [J, 16], got {arr.shape}")
    return arr.reshape(-1, 4, 4)


def _build_export_response(
    job,
    fmt: str,
    *,
    bind_matrices_override: np.ndarray | None = None,
) -> Response:
    if job.output_path is None or not job.output_path.is_file():
        raise HTTPException(404, "binding not found")

    fmt = _normalize_download_format(fmt)
    if fmt in ("binding", "npz", "binding-npz"):
        data = job.output_path.read_bytes()
        _assert_npz_bytes(data)
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": 'attachment; filename="binding.npz"',
                "Cache-Control": "no-store",
            },
        )

    if fmt in ("prepared", "prepared-npz", "prepared_binding"):
        out_path = job.output_path
        data = out_path.read_bytes()
        _assert_npz_bytes(data)
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": 'attachment; filename="prepared_binding.npz"',
                "Cache-Control": "no-store",
            },
        )

    if fmt in ("fbx", "binding-fbx", "binding_rest"):
        from .exporter import export_binding_fbx

        try:
            out_path = export_binding_fbx(
                job.output_path,
                bind_matrices_override=bind_matrices_override,
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc
        data = out_path.read_bytes()
        _assert_fbx_bytes(data)
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": 'attachment; filename="binding_rest.fbx"',
                "Cache-Control": "no-store",
            },
        )

    raise HTTPException(400, f"unsupported download format: {fmt}")


class AlignTransformBody(BaseModel):
    position: list[float] = Field(..., min_length=3, max_length=3)
    quaternion: list[float] = Field(..., min_length=4, max_length=4)
    scale: float = Field(..., gt=0)


class AlignMatrixBody(BaseModel):
    transform: AlignTransformBody | None = None
    matrix: list[list[float]] | None = Field(default=None, min_length=4, max_length=4)


class SomaJointsBody(BaseModel):
    joints: list[list[float]] = Field(..., min_length=1)
    bind_matrices: list[list[float]] | None = Field(default=None, min_length=1)


class BindingWeightsBody(BaseModel):
    lbs_indices: list[list[int]] = Field(..., min_length=1)
    lbs_weights: list[list[float]] = Field(..., min_length=1)


class ExportDownloadBody(BindingWeightsBody):
    file_format: str = Field(..., min_length=1)
    bind_matrices: list[list[float]] | None = Field(default=None, min_length=1)


class CharacterCreateBody(BaseModel):
    name: str = Field(default="未命名角色", min_length=1, max_length=80)


class CharacterRenameBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


class AnimationCreateBody(BaseModel):
    source_motion_id: str = ""
    source_kind: str = Field(default="gvhmr", min_length=1, max_length=32)
    kimodo_motion_id: str = ""
    character_id: str
    name: str = Field(default="重定向动作", min_length=1, max_length=80)


class RenameBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


class AnimationAdjustItem(BaseModel):
    joint: int = Field(..., ge=0)
    translation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)
    rotation: list[float] = Field(..., min_length=4, max_length=4)
    space: str = Field(default="local", max_length=16)


class AnimationAdjustBody(BaseModel):
    name: str = Field(default="Adjusted Motion", min_length=1, max_length=80)
    overwrite: bool = False
    adjustments: list[AnimationAdjustItem] = Field(default_factory=list)
    trim_start: int | None = Field(default=None, ge=0)
    trim_end: int | None = Field(default=None, ge=1)


class PoseEditSaveBody(BaseModel):
    name: str = Field(default="Edited Pose", min_length=1, max_length=80)
    world_motion: list[list[float]] = Field(..., min_length=1)


class PoseFromMotionBody(BaseModel):
    source_kind: Literal["video", "kimodo"]
    motion_id: str = Field(..., min_length=1, max_length=128)
    frame: int = Field(..., ge=0)
    name: str = Field(..., min_length=1, max_length=80)


class KimodoCreateBody(BaseModel):
    prompt: str = Field(default="", max_length=1200)
    segments: list[dict] = Field(default_factory=list)
    constraints: list[dict] = Field(default_factory=list)
    preview_scene: dict = Field(default_factory=dict)
    name: str = Field(default="Kimodo Motion", min_length=1, max_length=80)
    duration_sec: float = Field(default=6.0, ge=2.0, le=10.0)
    seed: int = Field(default=12345, ge=0)
    diffusion_steps: int = Field(default=100, ge=1, le=300)
    model_name: str = Field(default="Kimodo-SOMA-RP-v1.1", min_length=1, max_length=128)


class KimodoPreviewSceneBody(BaseModel):
    scene_objects: list[dict] = Field(default_factory=list)


class DirectorProjectCreateBody(BaseModel):
    name: str = Field(default="未命名导演工程", min_length=1, max_length=120)
    project: dict = Field(default_factory=dict)


class DirectorProjectUpdateBody(BaseModel):
    project: dict = Field(default_factory=dict)
    expected_revision: int | None = Field(default=None, ge=0)


class DirectorSnapshotBody(BaseModel):
    name: str = Field(default="手动快照", min_length=1, max_length=120)
    kind: str = Field(default="manual", min_length=1, max_length=32)


class DirectorExportCreateBody(BaseModel):
    settings: dict = Field(default_factory=dict)


class DirectorExportUpdateBody(BaseModel):
    state: str | None = Field(default=None, max_length=32)
    stage: str | None = Field(default=None, max_length=64)
    progress: float | None = Field(default=None, ge=0, le=1)
    completed_frames: int | None = Field(default=None, ge=0)
    last_checkpoint_frame: int | None = None
    error: str | None = Field(default=None, max_length=8000)
    artifacts: list[str] | None = None


class TimelineCutBody(BaseModel):
    start_frame: int = Field(..., ge=0)
    end_frame: int = Field(..., ge=0)


class TimelineSegmentBody(BaseModel):
    id: str | None = None
    animation_id: str
    start_frame: int = Field(default=0, ge=0)
    end_frame: int = Field(default=0, ge=0)
    speed: float = Field(default=1.0, ge=0.05, le=8.0)
    reverse: bool = False
    root_motion: bool = False
    mirror: bool = False
    group_id: str = ""
    deleted_ranges: list[TimelineCutBody] = Field(default_factory=list)
    absolute_root: bool = False
    root_curve_batch: str = ""
    root_curve_index: int = -1
    force_align_to_previous: bool = False
    display_name: str = Field(default="", max_length=80)


class DeleteFramesBody(BaseModel):
    start_frame: int = Field(..., ge=0)
    end_frame: int = Field(..., ge=0)


class SplitFrameBody(BaseModel):
    frame: int = Field(..., ge=0)


class RootCurveBody(BaseModel):
    points: list[list[float]] = Field(..., min_length=1)
    control_points: list[list[float]] | None = None
    still_threshold: float = Field(default=0.002, ge=0.0, le=0.25)
    exact_positions: bool = False
    exact_transforms: list[list[float]] | None = None


class UpperLowerPreviewBody(BaseModel):
    character_id: str
    upper_animation_id: str
    lower_animation_id: str
    duration_sec: float = Field(default=3.0, gt=0.0, le=120.0)
    name: str = Field(default="合成动画", min_length=1, max_length=80)

class AdditivePreviewBody(BaseModel):
    character_id: str
    base_animation_id: str
    additive_animation_id: str
    base_speed: float = Field(default=1.0, gt=0.05, le=4.0)
    additive_speed: float = Field(default=1.0, gt=0.05, le=4.0)
    weight: float = Field(default=1.0, ge=0.0, le=2.0)
    reference_type: Literal["first_frame", "animation_frame", "ref_pose"] = "first_frame"
    reference_frame: int = Field(default=0, ge=0)
    end_behavior: Literal["loop", "hold"] = "loop"
    name: str = Field(default="加法叠加动画", min_length=1, max_length=80)


class CompositionCreateBody(BaseModel):
    character_id: str
    name: str = Field(default="未命名合成", min_length=1, max_length=80)


class CompositionUpdateBody(BaseModel):
    name: str | None = Field(default=None, max_length=80)
    segments: list[TimelineSegmentBody]
    cuts: list[TimelineCutBody] = Field(default_factory=list)

class CompositionSceneObjectsBody(BaseModel):
    scene_objects: list[dict] = Field(default_factory=list)

class CompositionIKConstraintsBody(BaseModel):
    ik_constraints: list[dict] = Field(default_factory=list)


class CompositionSegmentRenameBody(BaseModel):
    segment_ids: list[str] = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=80)


class RootModeBody(BaseModel):
    root_mode: str


class EditBody(BaseModel):
    joint: int = Field(..., ge=0, lt=30)
    scope: str
    start_frame: int = Field(default=0, ge=0)
    end_frame: int = Field(default=0, ge=0)
    blend_frames: int = Field(default=5, ge=0, le=30)
    blend_in_frames: int | None = Field(default=None, ge=0, le=120)
    blend_out_frames: int | None = Field(default=None, ge=0, le=120)
    translation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)
    rotation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0], min_length=4, max_length=4)


class EditBlendBody(BaseModel):
    blend_in_frames: int = Field(..., ge=0, le=120)
    blend_out_frames: int = Field(..., ge=0, le=120)


class TransitionBody(BaseModel):
    left_frames: int = Field(..., ge=0)
    right_frames: int = Field(..., ge=0)


class TransitionCreateBody(BaseModel):
    left_segment_id: str
    right_segment_id: str


class OrientationCorrectionBody(BaseModel):
    start_frame: int = Field(..., ge=0)
    end_frame: int = Field(..., ge=0)
    mode: str = Field(default="range")


class OrientationCorrectionUpdateBody(BaseModel):
    blend_in_frames: int = Field(..., ge=0)
    blend_out_frames: int = Field(..., ge=0)


class VirtualRootPreviewBody(BaseModel):
    animation_id: str
    name: str = Field(default="VirtualRoot 动作", min_length=1, max_length=80)
    smoothing_frames: int = Field(default=9, ge=1, le=61)
    position_tolerance: float = Field(default=0.005, ge=0.0, le=0.2)

class GaitIKPreviewBody(BaseModel):
    animation_id: str
    playback_speed: float = Field(default=1.0, ge=0.05, le=4.0)
    name: str = Field(default="IK 步态动画", min_length=1, max_length=80)
    stride_scale: float = Field(default=1.0, ge=0.25, le=2.5)
    root_distance_scale: float = Field(default=1.0, ge=0.0, le=3.0)
    foot_lock_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    contact_height: float = Field(default=0.06, ge=0.001, le=0.25)
    contact_velocity: float = Field(default=0.25, ge=0.01, le=3.0)
    plant_blend_frames: int = Field(default=4, ge=0, le=30)
    sole_offset: float = Field(default=0.0, ge=-0.1, le=0.25)
    pelvis_max_correction: float = Field(default=0.12, ge=0.0, le=0.5)
    pelvis_smoothing_frames: int = Field(default=7, ge=1, le=61)
    ground_rotation_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    terrain_smoothing_frames: int = Field(default=5, ge=1, le=61)
    max_slope_degrees: float = Field(default=55.0, ge=1.0, le=89.0)
    terrain_id: str = Field(default="", max_length=64)
    initial_translation: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3
    )
    initial_yaw_degrees: float = Field(default=0.0, ge=-360.0, le=360.0)


class GaitIKCommitBody(BaseModel):
    include_terrain: bool = False

class FootLockPreviewBody(BaseModel):
    animation_id: str
    name: str = Field(default="Foot Lock Animation", min_length=1, max_length=80)
    reference_frame: int = Field(default=0, ge=0)
    foot_mode: Literal["left", "right", "both"] = "both"
    lock_rotation: bool = False
    solver: Literal["pbik", "jacobian_dls", "fabrik"] = "pbik"
    root_behavior: Literal["pin", "pre_pull", "foot_priority", "free"] = "foot_priority"
    iterations: int = Field(default=24, ge=1, le=100)
    tolerance: float = Field(default=0.002, ge=0.00001, le=0.05)


_ws_clients: set[WebSocket] = set()
_had_client = False
_shutdown_timer: threading.Timer | None = None
_shutdown_event = threading.Event()


def _cancel_shutdown_timer() -> None:
    global _shutdown_timer
    if _shutdown_timer is not None:
        _shutdown_timer.cancel()
        _shutdown_timer = None


def _request_shutdown() -> None:
    if _shutdown_event.is_set():
        return
    _shutdown_event.set()
    store.on_shutdown()


def _schedule_shutdown() -> None:
    global _shutdown_timer

    def _fire() -> None:
        if _ws_clients:
            return
        _request_shutdown()

    _cancel_shutdown_timer()
    _shutdown_timer = threading.Timer(SHUTDOWN_GRACE_S, _fire)
    _shutdown_timer.daemon = True
    _shutdown_timer.start()


@app.on_event("startup")
async def _startup() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ok, msg = check_setup()
    if not ok:
        print(f"[Meshpreps] WARNING: {msg}")


@app.on_event("shutdown")
def _shutdown_children() -> None:
    poses.shutdown()
    motions.shutdown()
    kimodo_motions.shutdown()
    store.on_shutdown()


@app.get("/api/health")
def health() -> dict:
    from .backend_config import research_backend_status
    ok, msg = check_setup()
    optional = research_backend_status()
    try:
        blender = str(find_blender())
    except FileNotFoundError:
        blender = ""
    return {
        "instance_root": str(Path(__file__).resolve().parents[1]),
        "instance_id": os.environ.get("KAIRYTH_V2_INSTANCE_ID", ""),
        "ok": ok,
        "service_ready": True,
        "release_profile": optional["release_profile"],
        "capabilities": optional["capabilities"],
        "rigging": {"state": "ready" if ok else "not_ready", "ready": ok, "message": msg,
                    "runtime_verified": False, "rights_verified": False},
        "optional_backends": {key: optional[key] for key in ("hybrik", "gvhmr")},
        "message": msg,
        "export_post": True,
        "blender": blender,
        "characters": len(characters.list()),
        "motions": len(motions.list()),
    }


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/api/shutdown")
def shutdown_api() -> dict:
    threading.Thread(target=_request_shutdown, daemon=True).start()
    return {"ok": True}


@app.post("/api/upload")
async def upload_mesh(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(400, "缺少文件名")
    suffix = Path(file.filename).suffix.lower() or ".glb"
    if suffix not in {".glb", ".gltf", ".obj", ".fbx", ".ply", ".stl"}:
        raise HTTPException(400, f"不支持的格式: {suffix}")
    dest = UPLOAD_DIR / f"{int(time.time() * 1000)}_{Path(file.filename).name}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        session = store.add_mesh(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    preview = store.mesh_preview(session.mesh_id)
    return {"mesh_id": session.mesh_id, "preview": preview}


@app.get("/api/mesh/{mesh_id}/preview")
def mesh_preview(mesh_id: str) -> dict:
    preview = store.mesh_preview(mesh_id)
    if preview is None:
        raise HTTPException(404, "mesh not found")
    return preview


@app.get("/api/soma30/reference")
def soma30_reference() -> dict:
    return soma30_reference_dict()


@app.get("/api/mesh/{mesh_id}/align/suggest")
def suggest_mesh_align(mesh_id: str) -> dict:
    try:
        matrix = store.suggest_mesh_align(mesh_id)
    except KeyError as exc:
        raise HTTPException(404, "mesh not found") from exc
    return {"matrix": matrix.tolist()}


@app.post("/api/mesh/{mesh_id}/align")
def confirm_mesh_align(mesh_id: str, body: AlignMatrixBody) -> dict:
    import numpy as np

    matrix = None
    translation = None
    quaternion = None
    scale = None
    if body.transform is not None:
        translation = body.transform.position
        quaternion = body.transform.quaternion
        scale = body.transform.scale
    elif body.matrix is not None:
        rows = body.matrix
        if any(len(row) != 4 for row in rows):
            raise HTTPException(400, "matrix must be 4x4")
        matrix = np.asarray(rows, dtype=np.float64)
    else:
        raise HTTPException(400, "transform or matrix required")
    try:
        store.apply_mesh_align(
            mesh_id,
            matrix=matrix,
            translation=translation,
            quaternion=quaternion,
            scale=scale,
        )
    except KeyError as exc:
        raise HTTPException(404, "mesh not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    result = store.mesh_align_result(mesh_id)
    if result is None:
        raise HTTPException(404, "mesh not found")
    return result


@app.get("/api/mesh/{mesh_id}/status")
def mesh_align_status(mesh_id: str) -> dict:
    status = store.mesh_align_status(mesh_id)
    if status is None:
        raise HTTPException(404, "mesh not found")
    return status


@app.post("/api/mesh/{mesh_id}/align/reset")
def reset_mesh_align(mesh_id: str) -> dict:
    try:
        store.reset_mesh_align(mesh_id)
    except KeyError as exc:
        raise HTTPException(404, "mesh not found") from exc
    result = store.mesh_align_result(mesh_id)
    if result is None:
        raise HTTPException(404, "mesh not found")
    return result


@app.get("/api/mesh/{mesh_id}/file")
def mesh_file(mesh_id: str):
    session = store.get_mesh(mesh_id)
    if session is None:
        raise HTTPException(404, "mesh not found")
    return FileResponse(session.upload_path, filename=session.upload_path.name)


@app.post("/api/bind/{mesh_id}")
def start_bind(mesh_id: str, seed: int = DEFAULT_SEED, timeout: float = DEFAULT_BIND_TIMEOUT_S) -> dict:
    try:
        job = store.start_bind(mesh_id, seed=seed, timeout_s=timeout)
    except KeyError as exc:
        raise HTTPException(404, "mesh not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return store.job_status(job.job_id) or {}


@app.post("/api/bind/{mesh_id}/auto")
def start_auto_bind(mesh_id: str, seed: int = DEFAULT_SEED, timeout: float = DEFAULT_BIND_TIMEOUT_S) -> dict:
    try:
        job = store.start_auto_bind(mesh_id, seed=seed, timeout_s=timeout)
    except KeyError as exc:
        raise HTTPException(404, "mesh not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return store.job_status(job.job_id) or {}


@app.get("/api/job/{job_id}")
def job_status(job_id: str) -> dict:
    status = store.job_status(job_id)
    if status is None:
        raise HTTPException(404, "job not found")
    return status


@app.get("/api/job/{job_id}/preview")
def binding_preview(job_id: str) -> dict:
    preview = store.binding_preview(job_id)
    if preview is None:
        raise HTTPException(404, "binding not ready")
    return preview


@app.get("/api/job/{job_id}/skeleton/preview")
def skeleton_preview(job_id: str) -> dict:
    preview = store.skeleton_preview(job_id)
    if preview is None:
        raise HTTPException(404, "skeleton not ready")
    return preview


@app.post("/api/job/{job_id}/skeleton/accept")
def accept_skeleton(job_id: str) -> dict:
    try:
        job = store.accept_skeleton(job_id)
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    preview = store.soma_preview(job_id)
    if preview is None:
        raise HTTPException(500, "soma preview unavailable")
    status = store.job_status(job_id) or {}
    return {**status, "preview": preview}


@app.get("/api/job/{job_id}/soma/preview")
def soma_preview(job_id: str) -> dict:
    preview = store.soma_preview(job_id)
    if preview is None:
        raise HTTPException(404, "soma edit not ready")
    return preview


@app.put("/api/job/{job_id}/soma")
def update_soma_joints(job_id: str, body: SomaJointsBody) -> dict:
    import numpy as np

    bind_mats = None
    if body.bind_matrices is not None:
        arr = np.asarray(body.bind_matrices, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 16:
            raise HTTPException(400, f"bind_matrices must be [J, 16], got {arr.shape}")
        bind_mats = arr.reshape(-1, 4, 4)

    try:
        store.update_soma_joints(
            job_id,
            np.asarray(body.joints, dtype=np.float32),
            bind_matrices=bind_mats,
        )
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    preview = store.soma_preview(job_id)
    if preview is None:
        raise HTTPException(500, "soma preview unavailable")
    return preview


@app.put("/api/job/{job_id}/binding/weights")
def save_binding_weights(job_id: str, body: BindingWeightsBody) -> dict:
    import numpy as np

    try:
        result = store.save_binding_weights(
            job_id,
            np.asarray(body.lbs_indices, dtype=np.int64),
            np.asarray(body.lbs_weights, dtype=np.float32),
        )
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return result


@app.post("/api/job/{job_id}/skin/complete")
def complete_skin(job_id: str, body: BindingWeightsBody) -> dict:
    import numpy as np

    try:
        result = store.complete_skin(
            job_id,
            np.asarray(body.lbs_indices, dtype=np.int64),
            np.asarray(body.lbs_weights, dtype=np.float32),
        )
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(500, str(exc)) from exc
    _cancel_shutdown_timer()
    return result


@app.post("/api/job/{job_id}/ik/auto")
def auto_ik_for_job(job_id: str) -> dict:
    try:
        return store.auto_ik(job_id)
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

@app.get("/api/job/{job_id}/download")
def download_binding(job_id: str, request: Request):
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return _build_export_response(job, _download_format(request))


@app.post("/api/job/{job_id}/export/download")
def export_download(job_id: str, body: ExportDownloadBody):
    import numpy as np

    try:
        store.save_binding_weights(
            job_id,
            np.asarray(body.lbs_indices, dtype=np.int64),
            np.asarray(body.lbs_weights, dtype=np.float32),
        )
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    bind_override = _parse_bind_matrices_override(body.bind_matrices)
    return _build_export_response(job, body.file_format, bind_matrices_override=bind_override)


@app.post("/api/job/{job_id}/download")
def download_binding_post(job_id: str, body: ExportDownloadBody):
    import numpy as np

    try:
        store.save_binding_weights(
            job_id,
            np.asarray(body.lbs_indices, dtype=np.int64),
            np.asarray(body.lbs_weights, dtype=np.float32),
        )
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    bind_override = _parse_bind_matrices_override(body.bind_matrices)
    return _build_export_response(job, body.file_format, bind_matrices_override=bind_override)


@app.post("/api/job/{job_id}/confirm")
def confirm_binding(job_id: str, timeout: float = DEFAULT_BIND_TIMEOUT_S) -> dict:
    try:
        job = store.confirm_and_skin(job_id, timeout_s=timeout)
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return store.job_status(job.job_id) or {}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    global _had_client
    await ws.accept()
    _ws_clients.add(ws)
    _had_client = True
    _cancel_shutdown_timer()
    try:
        while not _shutdown_event.is_set():
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws.send_text("ping")
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)
        if _had_client and not _ws_clients and not is_pipeline_mode():
            _schedule_shutdown()


@app.get("/api/characters")
def list_characters() -> list[dict]:
    return characters.list()


@app.post("/api/characters/from-job/{job_id}")
def create_character(job_id: str, body: CharacterCreateBody) -> dict:
    job = store.get_job(job_id)
    if job is None or job.output_path is None or not job.output_path.is_file():
        raise HTTPException(404, "binding not ready")
    try:
        return characters.create(body.name, job.output_path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/characters/static", status_code=201)
async def create_static_character(model: UploadFile = File(...), name: str = Form(default="")) -> dict:
    try:
        data = await model.read()
        return characters.create_static(name or Path(model.filename or "Static Asset").stem, model.filename or "model.obj", data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/characters/{character_id}")
def get_character(character_id: str) -> dict:
    try:
        return characters.get(character_id)
    except KeyError as exc:
        raise HTTPException(404, "character not found") from exc


@app.patch("/api/characters/{character_id}")
def rename_character(character_id: str, body: CharacterRenameBody) -> dict:
    try:
        return characters.rename(character_id, body.name)
    except KeyError as exc:
        raise HTTPException(404, "character not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/characters/{character_id}/scene")
def character_scene(character_id: str) -> dict:
    try:
        return characters.preview(character_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "character not found") from exc


@app.post("/api/characters/{character_id}/ik/auto")
def auto_ik_for_character(character_id: str) -> dict:
    try:
        character = characters.auto_ik(character_id)
        return {"ok": True, "character": character}
    except KeyError as exc:
        raise HTTPException(404, "character not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, "character binding not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

@app.get("/api/characters/{character_id}/exports/{file_format}")
def export_character(character_id: str, file_format: str) -> FileResponse:
    try:
        binding_path = characters.binding_path(character_id)
        if file_format == "npz":
            path = binding_path
        elif file_format == "fbx":
            path = characters.export_path(character_id, "fbx")
            if not is_fbx_export_current(binding_path, path, bind_pose_only=True):
                export_fbx(binding_path, path, bind_pose_only=True)
        else:
            raise HTTPException(400, "format must be fbx or npz")
        if not path.is_file():
            raise FileNotFoundError(path)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "character export not ready") from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    filename = f"{Path(character_id).name}_binding.{file_format}"
    return FileResponse(path, media_type="application/octet-stream", filename=filename)


@app.delete("/api/characters/{character_id}")
def delete_character(character_id: str) -> dict:
    try:
        for preview in retarget_previews.list(character_id):
            retarget_previews.delete(preview["id"])
        for composition in compositions.list(character_id):
            compositions.delete(composition["id"])
        for animation in animations.list(character_id, include_hidden=True):
            animations.delete(animation["id"])
        characters.delete(character_id)
    except KeyError as exc:
        raise HTTPException(404, "character not found") from exc
    return {"ok": True}


@app.get("/api/motions")
def list_motions() -> list[dict]:
    return motions.list()


@app.post("/api/motions", status_code=201)
async def create_motion(
    video: UploadFile = File(...),
    root_mode: str = Form("world"),
    static_camera: bool = Form(False),
    focal_length: str = Form(""),
) -> dict:
    _require_inference_api("gvhmr")
    focal: int | None = None
    if focal_length.strip():
        try:
            focal = int(focal_length)
        except ValueError as exc:
            raise HTTPException(400, "焦距必须是整数") from exc
        if not 1 <= focal <= 1000:
            raise HTTPException(400, "焦距范围应为 1–1000 mm")
    try:
        return motions.create(
            video.filename or "input.mp4", video.file,
            root_mode=root_mode, static_camera=static_camera, focal_length=focal,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        await video.close()


@app.get("/api/motions/{motion_id}")
def get_motion(motion_id: str) -> dict:
    try:
        return motions.public(motions.get(motion_id))
    except KeyError as exc:
        raise HTTPException(404, "motion not found") from exc


@app.patch("/api/motions/{motion_id}")
def rename_motion(motion_id: str, body: RenameBody) -> dict:
    try:
        return motions.rename(motion_id, body.name)
    except KeyError as exc:
        raise HTTPException(404, "motion not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/motions/{motion_id}")
def delete_motion(motion_id: str) -> dict:
    try:
        motions.delete(motion_id)
    except KeyError as exc:
        raise HTTPException(404, "motion not found") from exc
    return {"ok": True}


@app.post("/api/motions/{motion_id}/cancel")
def cancel_motion(motion_id: str) -> dict:
    try:
        return motions.cancel(motion_id)
    except KeyError as exc:
        raise HTTPException(404, "motion not found") from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/api/motions/{motion_id}/video")
def motion_video(motion_id: str) -> FileResponse:
    try:
        path = Path(motions.get(motion_id)["input_path"])
    except KeyError as exc:
        raise HTTPException(404, "motion not found") from exc
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")


@app.get("/api/motions/{motion_id}/scene")
def motion_scene(motion_id: str) -> dict:
    try:
        job = motions.get(motion_id)
        data = np.load(motions.source_bundle_path(job), allow_pickle=False)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "motion scene not ready") from exc
    try:
        return {
            "target_bind": data["target_bind"].reshape(-1, 16).tolist(),
            "parents": data["parents"].tolist(), "names": data["names"].tolist(),
            "fps": float(data["fps"].reshape(-1)[0]), "num_frames": int(data["world_motion"].shape[0]),
            "skeleton_type": "SMPL-X22", "native_skeleton": True,
            "root_mode": str(data["root_mode"].reshape(-1)[0]) if "root_mode" in data.files else job.get("root_mode", "world"),
        }
    finally:
        data.close()


@app.get("/api/motions/{motion_id}/frames")
def motion_frames(motion_id: str, from_frame: int = 0, to_frame: int = 1, edited: bool = True) -> dict:
    try:
        job = motions.get(motion_id)
        data = np.load(motions.source_bundle_path(job), allow_pickle=False)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "motion not ready") from exc
    try:
        count = int(data["world_motion"].shape[0])
        start = max(0, min(int(from_frame), count - 1))
        end = max(start + 1, min(int(to_frame), count, start + 240))
        world = data["world_motion"]
        return {"from": start, "to": end, "world": world[start:end].reshape(end - start, -1, 16).tolist()}
    finally:
        data.close()


@app.put("/api/motions/{motion_id}/root-mode")
def set_root_mode(motion_id: str, body: RootModeBody) -> dict:
    try:
        return motions.switch_root_mode(motion_id, body.root_mode)
    except KeyError as exc:
        raise HTTPException(404, "motion not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/motions/{motion_id}/edits")
def list_edits(motion_id: str) -> dict:
    try:
        job = motions.get(motion_id)
        project = motions.edit_project(job)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "motion not ready") from exc
    return {"operations": [vars(op) for op in project.operations]}


@app.post("/api/motions/{motion_id}/edits", status_code=201)
def add_edit(motion_id: str, body: EditBody) -> dict:
    try:
        job = motions.get(motion_id)
        project = motions.edit_project(job)
        op = project.add(EditOperation.create(**body.model_dump()))
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "motion not ready") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return vars(op)


@app.delete("/api/motions/{motion_id}/edits/{operation_id}")
def delete_edit(motion_id: str, operation_id: str) -> dict:
    try:
        project = motions.edit_project(motions.get(motion_id))
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "motion not ready") from exc
    if not project.delete(operation_id):
        raise HTTPException(404, "edit not found")
    return {"ok": True}


@app.delete("/api/motions/{motion_id}/edits")
def clear_edits(motion_id: str) -> dict:
    try:
        project = motions.edit_project(motions.get(motion_id))
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "motion not ready") from exc
    project.clear()
    return {"ok": True}


@app.post("/api/motions/{motion_id}/edits/bake")
def bake_edits(motion_id: str) -> dict:
    try:
        return motions.bake(motion_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "motion not ready") from exc


@app.post("/api/motions/{motion_id}/edits/undo-bake")
def undo_bake(motion_id: str) -> dict:
    try:
        return motions.undo_bake(motion_id)
    except KeyError as exc:
        raise HTTPException(404, "motion not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/motions/{motion_id}/exports/{file_format}")
def export_motion(motion_id: str, file_format: str) -> FileResponse:
    try:
        job = motions.get(motion_id)
        if file_format == "npz":
            path = motions.source_bundle_path(job)
        elif file_format == "pt":
            path = Path(job.get("pt_path", ""))
        else:
            raise HTTPException(400, "format must be npz or pt")
        if not path.is_file():
            raise FileNotFoundError(path)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "export not ready") from exc
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


def _pose_scene(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "target_bind": data["target_bind"].reshape(-1, 16).tolist(),
            "parents": data["parents"].tolist(),
            "names": data["names"].tolist(),
            "fps": float(data["fps"].reshape(-1)[0]),
            "num_frames": int(data["world_motion"].shape[0]),
            "skeleton_type": str(data["skeleton_type"].reshape(-1)[0]) if "skeleton_type" in data.files else "KIMODO-SOMA30-POSE",
            "native_skeleton": True,
            "source_kind": "hybrik_pose",
            "root_mode": "pose",
        }


@app.get("/api/hybrik-runtime/status")
def hybrik_runtime_status() -> dict:
    return poses.status()


@app.get("/api/research-backends/status")
def research_backends_status() -> dict:
    from .backend_config import research_backend_status
    return research_backend_status()


@app.get("/api/capabilities")
def release_capabilities() -> dict:
    return capabilities()


@app.post("/api/image-poses/{pose_id}/cancel")
def cancel_image_pose(pose_id: str) -> dict:
    try: return poses.cancel(pose_id)
    except KeyError: raise HTTPException(status_code=404, detail="Pose not found")


@app.get("/api/image-poses")
def list_image_pose_jobs() -> list[dict]:
    return poses.list_jobs()


@app.post("/api/image-poses", status_code=201)
async def create_image_pose(image: UploadFile = File(...), name: str = Form(default="")) -> dict:
    _require_inference_api("hybrik")
    try:
        data = await image.read()
        if not data:
            raise ValueError("图片为空")
        return poses.create(image_bytes=data, filename=image.filename or "image.png", name=name or Path(image.filename or "image").stem)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/image-poses/{pose_id}/confirm")
def confirm_image_pose(pose_id: str, body: RenameBody | None = None) -> dict:
    try:
        return poses.confirm(pose_id, name=body.name if body else None)
    except KeyError as exc:
        raise HTTPException(404, "Pose not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.delete("/api/image-poses/{pose_id}")
def delete_image_pose_job(pose_id: str) -> dict:
    try:
        poses.delete_job(pose_id)
    except KeyError as exc:
        raise HTTPException(404, "Pose job not found") from exc
    return {"ok": True}


@app.get("/api/image-poses/{pose_id}/image")
def image_pose_source_image(pose_id: str) -> FileResponse:
    try:
        path = poses.image_path(pose_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "source image not found") from exc
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream", filename=path.name)


@app.get("/api/image-poses/{pose_id}/scene")
def image_pose_scene(pose_id: str) -> dict:
    try:
        return _pose_scene(poses.asset_path(pose_id))
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "Pose scene not ready") from exc


@app.get("/api/image-poses/{pose_id}/frames")
def image_pose_frames(pose_id: str, from_frame: int = 0, to_frame: int = 1) -> dict:
    try:
        path = poses.asset_path(pose_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "Pose frames not ready") from exc
    with np.load(path, allow_pickle=False) as data:
        world = data["world_motion"]
        count = int(world.shape[0])
        start = max(0, min(int(from_frame), count - 1))
        end = max(start + 1, min(int(to_frame), count))
        return {"from": start, "to": end, "world": world[start:end].reshape(end - start, -1, 16).tolist()}


@app.get("/api/poses")
def list_pose_library() -> list[dict]:
    return poses.list_library()


@app.post("/api/poses", status_code=201)
async def import_pose_library_item(pose: UploadFile = File(...), name: str = Form(default="")) -> dict:
    try:
        data = await pose.read()
        return poses.import_file(file_bytes=data, filename=pose.filename or "pose.npz", name=name or Path(pose.filename or "pose").stem)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/poses/from-motion", status_code=201)
def create_pose_library_item_from_motion(body: PoseFromMotionBody) -> dict:
    try:
        if body.source_kind == "video":
            source = motions.get(body.motion_id)
            if source.get("status") != "completed":
                raise ValueError("Video motion has not completed")
            bundle = motions.source_bundle_path(source)
            source_name = source.get("name") or Path(source.get("filename") or body.motion_id).stem
        else:
            source = kimodo_motions.get(body.motion_id)
            if source.get("status") != "completed":
                raise ValueError("Text motion has not completed")
            bundle = kimodo_motions.bundle_path(source)
            source_name = source.get("name") or body.motion_id
        return poses.create_from_motion_bundle(
            bundle,
            source_kind=body.source_kind,
            source_id=body.motion_id,
            source_name=source_name,
            frame=body.frame,
            name=body.name,
        )
    except KeyError as exc:
        raise HTTPException(404, "Motion not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, "Motion data not ready") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/poses/{pose_id}")
def get_pose_library_item(pose_id: str) -> dict:
    try:
        job = poses.get(pose_id)
    except KeyError as exc:
        raise HTTPException(404, "Pose not found") from exc
    return poses.public(job)


@app.patch("/api/poses/{pose_id}")
def rename_pose_library_item(pose_id: str, body: RenameBody) -> dict:
    try:
        return poses.rename(pose_id, body.name)
    except KeyError as exc:
        raise HTTPException(404, "Pose not found") from exc


@app.post("/api/poses/{pose_id}/edited", status_code=201)
def save_edited_pose_library_item(pose_id: str, body: PoseEditSaveBody) -> dict:
    try:
        return poses.create_edited(pose_id, name=body.name, world_motion=body.world_motion)
    except KeyError as exc:
        raise HTTPException(404, "Pose not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/poses/{pose_id}")
def delete_pose_library_item(pose_id: str) -> dict:
    try:
        poses.delete(pose_id)
    except KeyError as exc:
        raise HTTPException(404, "Pose not found") from exc
    return {"ok": True}


@app.get("/api/poses/{pose_id}/scene")
def pose_library_scene(pose_id: str) -> dict:
    try:
        return _pose_scene(poses.asset_path(pose_id))
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "Pose scene not ready") from exc


@app.get("/api/poses/{pose_id}/frames")
def pose_library_frames(pose_id: str, from_frame: int = 0, to_frame: int = 1) -> dict:
    try:
        path = poses.asset_path(pose_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "Pose frames not ready") from exc
    with np.load(path, allow_pickle=False) as data:
        world = data["world_motion"]
        count = int(world.shape[0])
        start = max(0, min(int(from_frame), count - 1))
        end = max(start + 1, min(int(to_frame), count))
        return {"from": start, "to": end, "world": world[start:end].reshape(end - start, -1, 16).tolist()}


@app.get("/api/poses/{pose_id}/constraints")
def pose_library_constraints(pose_id: str) -> list[dict]:
    try:
        return poses.constraints(pose_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "Pose constraints not ready") from exc
    except ValueError as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/api/poses/{pose_id}/exports/constraints")
def export_pose_constraints(pose_id: str) -> FileResponse:
    try:
        path = poses.constraints_path(pose_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "Pose constraints not ready") from exc
    return FileResponse(path, media_type="application/json", filename=f"{pose_id}_kimodo_constraints.json")


@app.get("/api/kimodo-runtime/status")
def kimodo_runtime_status() -> dict:
    return kimodo_motions.status()


@app.get("/api/kimodo-runtime/default-scene")
def kimodo_runtime_default_scene() -> dict:
    try:
        return kimodo_default_soma_scene()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ModuleNotFoundError as exc:
        # Missing optional dependencies are not a core-service outage. Other
        # ImportError/runtime programming failures remain visible as failures.
        raise HTTPException(503, {"code": "optional-backend-not-ready", "backend": "kimodo",
                                  "missing_dependency": exc.name}) from exc


@app.get("/api/kimodo-motions")
def list_kimodo_motions() -> list[dict]:
    return kimodo_motions.list()


@app.post("/api/kimodo-motions", status_code=201)
def create_kimodo_motion(body: KimodoCreateBody) -> dict:
    try:
        return kimodo_motions.create(
            prompt=body.prompt,
            segments=body.segments,
            constraints=body.constraints,
            name=body.name,
            duration_sec=body.duration_sec,
            seed=body.seed,
            diffusion_steps=body.diffusion_steps,
            model_name="Kimodo-SOMA-RP-v1.1",
            preview_scene=body.preview_scene,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.patch("/api/kimodo-motions/{motion_id}")
def rename_kimodo_motion(motion_id: str, body: RenameBody) -> dict:
    try:
        return kimodo_motions.rename(motion_id, body.name)
    except KeyError as exc:
        raise HTTPException(404, "Kimodo motion not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/api/kimodo-motions/{motion_id}/preview-scene")
def update_kimodo_preview_scene(motion_id: str, body: KimodoPreviewSceneBody) -> dict:
    try:
        return kimodo_motions.update_preview_scene(motion_id, body.model_dump())
    except KeyError as exc:
        raise HTTPException(404, "Kimodo motion not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/kimodo-motions/{motion_id}/cancel")
def cancel_kimodo_motion(motion_id: str) -> dict:
    try:
        return kimodo_motions.cancel(motion_id)
    except KeyError as exc:
        raise HTTPException(404, "Kimodo motion not found") from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


@app.delete("/api/kimodo-motions/{motion_id}")
def delete_kimodo_motion(motion_id: str) -> dict:
    try:
        kimodo_motions.delete(motion_id)
    except KeyError as exc:
        raise HTTPException(404, "Kimodo motion not found") from exc
    return {"ok": True}


@app.get("/api/kimodo-motions/{motion_id}/scene")
def kimodo_motion_scene(motion_id: str) -> dict:
    try:
        job = kimodo_motions.get(motion_id)
        data = np.load(kimodo_motions.bundle_path(job), allow_pickle=False)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "Kimodo motion scene not ready") from exc
    try:
        return {
            "target_bind": data["target_bind"].reshape(-1, 16).tolist(),
            "parents": data["parents"].tolist(),
            "names": data["names"].tolist(),
            "fps": float(data["fps"].reshape(-1)[0]),
            "num_frames": int(data["world_motion"].shape[0]),
            "skeleton_type": "KIMODO-SOMA30",
            "native_skeleton": True,
            "source_kind": "kimodo",
            "root_mode": "native",
            "preview_scene": job.get("preview_scene", {"scene_objects": []}),
        }
    finally:
        data.close()


@app.get("/api/kimodo-motions/{motion_id}/frames")
def kimodo_motion_frames(motion_id: str, from_frame: int = 0, to_frame: int = 1) -> dict:
    try:
        job = kimodo_motions.get(motion_id)
        data = np.load(kimodo_motions.bundle_path(job), allow_pickle=False)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "Kimodo motion not ready") from exc
    try:
        count = int(data["world_motion"].shape[0])
        start = max(0, min(int(from_frame), count - 1))
        end = max(start + 1, min(int(to_frame), count, start + 240))
        world = data["world_motion"]
        return {"from": start, "to": end, "world": world[start:end].reshape(end - start, -1, 16).tolist()}
    finally:
        data.close()


@app.get("/api/kimodo-motions/{motion_id}/exports/npz")
def export_kimodo_motion(motion_id: str) -> FileResponse:
    try:
        path = kimodo_motions.bundle_path(motion_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "Kimodo export not ready") from exc
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


def _rig_scene(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        scene = {
            "vertices": data["rest_vertices"].tolist(),
            "faces": data["faces"].tolist(),
            "lbs_indices": data["lbs_indices"].tolist(),
            "lbs_weights": data["lbs_weights"].tolist(),
            "target_bind": data["target_bind"].reshape(-1, 16).tolist(),
            "parents": data["parents"].tolist(),
            "names": data["names"].tolist(),
            "fps": float(data["fps"].reshape(-1)[0]),
            "num_frames": int(data["world_motion"].shape[0]),
            "root_mode": "fixed_hips",
            "current_up_axis": str(data["current_up_axis"].reshape(-1)[0]) if "current_up_axis" in data.files else "Y",
        }
        if "virtual_root_motion" in data.files and len(data["virtual_root_motion"]) == int(data["world_motion"].shape[0]):
            root = np.asarray(data["virtual_root_motion"], dtype=np.float32)
            scene["virtual_root_transforms"] = root.reshape(-1, 16).tolist()
            scene["virtual_root_path"] = root[:, :3, 3].tolist()
        elif "root_motion" in data.files and len(data["root_motion"]) == int(data["world_motion"].shape[0]):
            root_path = np.asarray(data["root_motion"], dtype=np.float32)
            if root_path.ndim == 2 and root_path.shape[1] == 3:
                scene["root_motion_path"] = root_path.tolist()
        elif int(data["world_motion"].shape[0]) > 1:
            names = [str(item).lower() for item in data["names"].tolist()]
            root_index = names.index("hips") if "hips" in names else 0
            positions = np.asarray(data["world_motion"][:, root_index, :3, 3], dtype=np.float32)
            inferred_path = positions - positions[[0]]
            if float(np.max(np.linalg.norm(inferred_path, axis=1))) > 1e-4:
                scene["root_motion_path"] = inferred_path.tolist()
        if "root_curve_edit_control_points" in data.files:
            scene["root_curve_control_points"] = np.asarray(data["root_curve_edit_control_points"], dtype=np.float32).tolist()
        elif "root_curve_control_points" in data.files and len(data["root_curve_control_points"]) <= 80:
            scene["root_curve_control_points"] = np.asarray(data["root_curve_control_points"], dtype=np.float32).tolist()
        if "terrain_heights" in data.files and "terrain_size" in data.files:
            heights = np.asarray(data["terrain_heights"], dtype=np.float32)
            size = np.asarray(data["terrain_size"], dtype=np.float32).reshape(-1)
            vertices, faces = heightfield_mesh(heights, float(size[0]), float(size[1]))
            terrain_translation = np.asarray(data["terrain_translation"], dtype=np.float32).reshape(-1).tolist() if "terrain_translation" in data.files else [0.0, 0.0, 0.0]
            terrain_yaw = float(data["terrain_yaw_degrees"].reshape(-1)[0]) if "terrain_yaw_degrees" in data.files else 0.0
            terrain_scale = float(data["terrain_uniform_scale"].reshape(-1)[0]) if "terrain_uniform_scale" in data.files else 1.0
            scene["terrain"] = {
                "kind": "heightfield",
                "vertices": vertices.tolist(),
                "faces": faces.tolist(),
                "heights": heights.tolist(),
                "size_x": float(size[0]),
                "size_z": float(size[1]),
                "transform": {"translation": terrain_translation[:3], "yaw_degrees": terrain_yaw, "uniform_scale": terrain_scale},
                "source_terrain_id": str(data["terrain_source_id"].reshape(-1)[0])
                if "terrain_source_id" in data.files else "",
                "source_terrain_name": str(data["terrain_source_name"].reshape(-1)[0])
                if "terrain_source_name" in data.files else "",
                "source_revision": int(data["terrain_source_revision"].reshape(-1)[0])
                if "terrain_source_revision" in data.files else 0,
                "display_by_default": True,
            }
        return scene


def _frame_payload(world: np.ndarray, start_value: int, end_value: int) -> dict:
    count = len(world)
    start = max(0, min(int(start_value), count - 1))
    end = max(start + 1, min(int(end_value), count, start + 240))
    return {"from": start, "to": end, "world": world[start:end].reshape(end - start, -1, 16).tolist()}


@app.post("/api/retarget-previews", status_code=201)
def create_retarget_preview(body: AnimationCreateBody) -> dict:
    try:
        return retarget_previews.create(
            body.source_motion_id,
            body.character_id,
            body.name,
            source_kind=body.source_kind,
            kimodo_motion_id=body.kimodo_motion_id,
        )
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "source motion or character not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/retarget-previews/{preview_id}/scene")
def retarget_preview_scene(preview_id: str) -> dict:
    try:
        return _rig_scene(retarget_previews.bundle_path(preview_id))
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "retarget preview not found") from exc


@app.get("/api/retarget-previews/{preview_id}/frames")
def retarget_preview_frames(preview_id: str, from_frame: int = 0, to_frame: int = 1) -> dict:
    try:
        with np.load(retarget_previews.bundle_path(preview_id), allow_pickle=False) as data:
            return _frame_payload(data["world_motion"], from_frame, to_frame)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "retarget preview not found") from exc


@app.post("/api/retarget-previews/{preview_id}/commit", status_code=201)
def commit_retarget_preview(preview_id: str) -> dict:
    try:
        return retarget_previews.commit(preview_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "retarget preview not found") from exc


@app.delete("/api/retarget-previews/{preview_id}")
def delete_retarget_preview(preview_id: str) -> dict:
    try:
        retarget_previews.delete(preview_id)
    except KeyError as exc:
        raise HTTPException(404, "retarget preview not found") from exc
    return {"ok": True}


@app.post("/api/virtual-root-previews", status_code=201)
def create_virtual_root_preview(body: VirtualRootPreviewBody) -> dict:
    try:
        return virtual_root_previews.create(
            body.animation_id,
            body.name,
            smoothing_frames=body.smoothing_frames,
            position_tolerance=body.position_tolerance,
        )
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "animation not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/virtual-root-previews/{preview_id}/scene")
def virtual_root_preview_scene(preview_id: str) -> dict:
    try:
        path = virtual_root_previews.bundle_path(preview_id)
        scene = _rig_scene(path)
        with np.load(path, allow_pickle=False) as data:
            if "virtual_root_motion" in data.files:
                scene["virtual_root_path"] = data["virtual_root_motion"][:, :3, 3].tolist()
                scene["virtual_root_ground_height"] = float(data["virtual_root_ground_height"].reshape(-1)[0]) if "virtual_root_ground_height" in data.files else 0.0
                if "virtual_root_ground_heights" in data.files:
                    scene["virtual_root_ground_heights"] = data["virtual_root_ground_heights"].tolist()
        scene["root_mode"] = "virtual_root_preview"
        return scene
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "virtual root preview not found") from exc


@app.get("/api/virtual-root-previews/{preview_id}/frames")
def virtual_root_preview_frames(preview_id: str, from_frame: int = 0, to_frame: int = 1) -> dict:
    try:
        with np.load(virtual_root_previews.bundle_path(preview_id), allow_pickle=False) as data:
            world = data["preview_world_motion"] if "preview_world_motion" in data.files else data["world_motion"]
            return _frame_payload(world, from_frame, to_frame)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "virtual root preview not found") from exc


@app.post("/api/virtual-root-previews/{preview_id}/commit", status_code=201)
def commit_virtual_root_preview(preview_id: str) -> dict:
    try:
        return virtual_root_previews.commit(preview_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "virtual root preview not found") from exc


@app.post("/api/virtual-root-previews/{preview_id}/replace", status_code=201)
def replace_virtual_root_source(preview_id: str) -> dict:
    try:
        return virtual_root_previews.replace_source(preview_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "virtual root preview or source animation not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/virtual-root-previews/{preview_id}")
def delete_virtual_root_preview(preview_id: str) -> dict:
    try:
        virtual_root_previews.delete(preview_id)
    except KeyError as exc:
        raise HTTPException(404, "virtual root preview not found") from exc
    return {"ok": True}

@app.post("/api/gait-ik-previews", status_code=201)
def create_gait_ik_preview(body: GaitIKPreviewBody) -> dict:
    try:
        settings = GaitIKSettings(
            playback_speed=body.playback_speed,
            stride_scale=body.stride_scale,
            root_distance_scale=body.root_distance_scale,
            foot_lock_strength=body.foot_lock_strength,
            contact_height=body.contact_height,
            contact_velocity=body.contact_velocity,
            plant_blend_frames=body.plant_blend_frames,
            sole_offset=body.sole_offset,
            pelvis_max_correction=body.pelvis_max_correction,
            pelvis_smoothing_frames=body.pelvis_smoothing_frames,
            ground_rotation_strength=body.ground_rotation_strength,
            terrain_smoothing_frames=body.terrain_smoothing_frames,
            max_slope_degrees=body.max_slope_degrees,
        )
        return gait_ik_previews.create(
            body.animation_id,
            body.name,
            settings,
            terrain_id=body.terrain_id,
            initial_translation=body.initial_translation,
            initial_yaw_degrees=body.initial_yaw_degrees,
        )
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "animation or terrain not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/gait-ik-previews/{preview_id}/scene")
def gait_ik_preview_scene(preview_id: str) -> dict:
    try:
        path = gait_ik_previews.bundle_path(preview_id)
        scene = _rig_scene(path)
        row = gait_ik_previews.get(preview_id)
        scene["gait_metrics"] = row.get("metrics", {})
        scene["gait_settings"] = row.get("settings", {})
        scene["grounded_root"] = True
        scene["root_mode"] = "gait_ik_preview"
        with np.load(path, allow_pickle=False) as data:
            if "gait_contacts" in data.files:
                scene["gait_contacts"] = data["gait_contacts"].tolist()
            if "gait_contact_weights" in data.files:
                scene["gait_contact_weights"] = data["gait_contact_weights"].tolist()
        return scene
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "gait IK preview not found") from exc


@app.get("/api/gait-ik-previews/{preview_id}/frames")
def gait_ik_preview_frames(preview_id: str, from_frame: int = 0, to_frame: int = 1) -> dict:
    try:
        with np.load(gait_ik_previews.bundle_path(preview_id), allow_pickle=False) as data:
            world = data["preview_world_motion"] if "preview_world_motion" in data.files else data["world_motion"]
            return _frame_payload(world, from_frame, to_frame)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "gait IK preview not found") from exc


@app.post("/api/gait-ik-previews/{preview_id}/commit", status_code=201)
def commit_gait_ik_preview(preview_id: str, body: GaitIKCommitBody | None = None) -> dict:
    try:
        return gait_ik_previews.commit(preview_id, include_terrain=bool(body and body.include_terrain))
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "gait IK preview not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/gait-ik-previews/{preview_id}")
def delete_gait_ik_preview(preview_id: str) -> dict:
    try:
        gait_ik_previews.delete(preview_id)
    except KeyError as exc:
        raise HTTPException(404, "gait IK preview not found") from exc
    return {"ok": True}



@app.post("/api/foot-lock-previews", status_code=201)
def create_foot_lock_preview(body: FootLockPreviewBody) -> dict:
    try:
        return foot_lock_previews.create(
            body.animation_id,
            body.name,
            FootLockSettings(
                reference_frame=body.reference_frame,
                foot_mode=body.foot_mode,
                lock_rotation=body.lock_rotation,
                solver=body.solver,
                root_behavior=body.root_behavior,
                iterations=body.iterations,
                tolerance=body.tolerance,
            ),
        )
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "animation not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/foot-lock-previews/{preview_id}/scene")
def foot_lock_preview_scene(preview_id: str) -> dict:
    try:
        path = foot_lock_previews.bundle_path(preview_id)
        scene = _rig_scene(path)
        row = foot_lock_previews.get(preview_id)
        scene["foot_lock_metrics"] = row.get("metrics", {})
        scene["foot_lock_settings"] = row.get("settings", {})
        scene["foot_lock_reference_targets"] = row.get("reference_targets", {})
        with np.load(path, allow_pickle=False) as data:
            if "foot_lock_residuals" in data.files:
                scene["foot_lock_residuals"] = data["foot_lock_residuals"].tolist()
            if "foot_lock_reached" in data.files:
                scene["foot_lock_reached"] = data["foot_lock_reached"].tolist()
        return scene
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "foot lock preview not found") from exc


@app.get("/api/foot-lock-previews/{preview_id}/frames")
def foot_lock_preview_frames(preview_id: str, from_frame: int = 0, to_frame: int = 1) -> dict:
    try:
        with np.load(foot_lock_previews.bundle_path(preview_id), allow_pickle=False) as data:
            world = data["preview_world_motion"] if "preview_world_motion" in data.files else data["world_motion"]
            return _frame_payload(world, from_frame, to_frame)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "foot lock preview not found") from exc


@app.post("/api/foot-lock-previews/{preview_id}/commit", status_code=201)
def commit_foot_lock_preview(preview_id: str) -> dict:
    try:
        return foot_lock_previews.commit(preview_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "foot lock preview not found") from exc


@app.delete("/api/foot-lock-previews/{preview_id}")
def delete_foot_lock_preview(preview_id: str) -> dict:
    try:
        foot_lock_previews.delete(preview_id)
    except KeyError as exc:
        raise HTTPException(404, "foot lock preview not found") from exc
    return {"ok": True}


@app.post("/api/upper-lower-previews", status_code=201)
def create_upper_lower_preview(body: UpperLowerPreviewBody) -> dict:
    try:
        return upper_lower_previews.create(
            body.character_id,
            body.upper_animation_id,
            body.lower_animation_id,
            body.duration_sec,
            body.name,
        )
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "animation or character not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/upper-lower-previews/{preview_id}/scene")
def upper_lower_preview_scene(preview_id: str) -> dict:
    try:
        scene = _rig_scene(upper_lower_previews.bundle_path(preview_id))
        scene["root_mode"] = "upper_lower_composite_preview"
        return scene
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "upper/lower composite preview not found") from exc


@app.get("/api/upper-lower-previews/{preview_id}/frames")
def upper_lower_preview_frames(preview_id: str, from_frame: int = 0, to_frame: int = 1) -> dict:
    try:
        with np.load(upper_lower_previews.bundle_path(preview_id), allow_pickle=False) as data:
            return _frame_payload(data["world_motion"], from_frame, to_frame)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "upper/lower composite preview not found") from exc


@app.post("/api/upper-lower-previews/{preview_id}/commit", status_code=201)
def commit_upper_lower_preview(preview_id: str, body: RenameBody) -> dict:
    try:
        return upper_lower_previews.commit(preview_id, body.name)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "upper/lower composite preview not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/upper-lower-previews/{preview_id}")
def delete_upper_lower_preview(preview_id: str) -> dict:
    try:
        upper_lower_previews.delete(preview_id)
    except KeyError as exc:
        raise HTTPException(404, "upper/lower composite preview not found") from exc
    return {"ok": True}

@app.post("/api/additive-previews", status_code=201)
def create_additive_preview(body: AdditivePreviewBody) -> dict:
    try:
        return additive_previews.create(
            body.character_id,
            body.base_animation_id,
            body.additive_animation_id,
            base_speed=body.base_speed,
            additive_speed=body.additive_speed,
            weight=body.weight,
            reference_type=body.reference_type,
            reference_frame=body.reference_frame,
            end_behavior=body.end_behavior,
            name=body.name,
        )
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "animation or character not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/additive-previews/{preview_id}/scene")
def additive_preview_scene(preview_id: str) -> dict:
    try:
        scene = _rig_scene(additive_previews.bundle_path(preview_id))
        scene["root_mode"] = "additive_composite_preview"
        return scene
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "additive composite preview not found") from exc


@app.get("/api/additive-previews/{preview_id}/frames")
def additive_preview_frames(preview_id: str, from_frame: int = 0, to_frame: int = 1) -> dict:
    try:
        with np.load(additive_previews.bundle_path(preview_id), allow_pickle=False) as data:
            return _frame_payload(data["world_motion"], from_frame, to_frame)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "additive composite preview not found") from exc


@app.post("/api/additive-previews/{preview_id}/commit", status_code=201)
def commit_additive_preview(preview_id: str, body: RenameBody) -> dict:
    try:
        return additive_previews.commit(preview_id, body.name)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "additive composite preview not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/additive-previews/{preview_id}")
def delete_additive_preview(preview_id: str) -> dict:
    try:
        additive_previews.delete(preview_id)
    except KeyError as exc:
        raise HTTPException(404, "additive composite preview not found") from exc
    return {"ok": True}


@app.get("/api/animations")
def list_animations(character_id: str | None = None) -> list[dict]:
    return animations.list(character_id)


@app.get("/api/animations/{animation_id}")
def get_animation(animation_id: str) -> dict:
    try:
        return animations.get(animation_id)
    except KeyError as exc:
        raise HTTPException(404, "animation not found") from exc


@app.patch("/api/animations/{animation_id}")
def rename_animation(animation_id: str, body: RenameBody) -> dict:
    try:
        return animations.rename(animation_id, body.name)
    except KeyError as exc:
        raise HTTPException(404, "animation not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/animations/{animation_id}/adjust", status_code=201)
def adjust_animation(animation_id: str, body: AnimationAdjustBody) -> dict:
    try:
        return animations.adjust_whole_animation(
            animation_id,
            name=body.name,
            adjustments=[item.model_dump() for item in body.adjustments],
            overwrite=body.overwrite,
            trim_start=body.trim_start,
            trim_end=body.trim_end,
        )
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "animation not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/animations/{animation_id}")
def delete_animation(animation_id: str) -> dict:
    try:
        for composition in compositions.list():
            filtered = [segment for segment in composition["segments"] if segment["animation_id"] != animation_id]
            if len(filtered) != len(composition["segments"]):
                compositions.update(composition["id"], name=None, segments=filtered, cuts=[])
        animations.delete(animation_id)
    except KeyError as exc:
        raise HTTPException(404, "animation not found") from exc
    return {"ok": True}


@app.get("/api/animations/{animation_id}/scene")
def animation_scene(animation_id: str) -> dict:
    try:
        return _rig_scene(animations.bundle_path(animation_id))
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "animation not ready") from exc


@app.get("/api/animations/{animation_id}/frames")
def animation_frames(animation_id: str, from_frame: int = 0, to_frame: int = 1, edited: bool = True) -> dict:
    try:
        data, world = animations.effective_world(animation_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "animation not ready") from exc
    try:
        if not edited:
            world = data["world_motion"]
        return _frame_payload(world, from_frame, to_frame)
    finally:
        data.close()


@app.get("/api/animations/{animation_id}/edits")
def list_animation_edits(animation_id: str) -> dict:
    try:
        project = animations.edit_project(animation_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "animation not ready") from exc
    return {"operations": [vars(op) for op in project.operations]}


@app.post("/api/animations/{animation_id}/edits", status_code=201)
def add_animation_edit(animation_id: str, body: EditBody) -> dict:
    try:
        op = animations.edit_project(animation_id).add(EditOperation.create(**body.model_dump()))
        return vars(op)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "animation not ready") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/animations/{animation_id}/edits/{operation_id}")
def delete_animation_edit(animation_id: str, operation_id: str) -> dict:
    try:
        project = animations.edit_project(animation_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "animation not ready") from exc
    if not project.delete(operation_id):
        raise HTTPException(404, "edit not found")
    return {"ok": True}


@app.patch("/api/animations/{animation_id}/edits/{operation_id}")
def update_animation_edit(animation_id: str, operation_id: str, body: EditBlendBody) -> dict:
    try:
        return vars(animations.edit_project(animation_id).update_blend(operation_id, body.blend_in_frames, body.blend_out_frames))
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "edit not found") from exc


@app.post("/api/animations/{animation_id}/edits/{operation_id}/apply")
def apply_animation_edit(animation_id: str, operation_id: str) -> dict:
    try:
        operation = animations.edit_project(animation_id).take(operation_id)
        animations.baked_edit_project(animation_id).add(operation)
        return {"ok": True}
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "edit not found") from exc


@app.delete("/api/animations/{animation_id}/edits")
def clear_animation_edits(animation_id: str) -> dict:
    try:
        animations.edit_project(animation_id).clear()
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "animation not ready") from exc
    return {"ok": True}


def _effective_animation_export(animation_id: str) -> Path:
    data, world = animations.effective_world(animation_id)
    try:
        if not animations.edit_project(animation_id).operations and not animations.baked_edit_project(animation_id).operations:
            return animations.bundle_path(animation_id)
        payload = {key: data[key] for key in data.files}
        payload["world_motion"] = world
    finally:
        data.close()
    path = ANIMATIONS_ROOT / Path(animation_id).name / "edited_export.npz"
    np.savez_compressed(path, **payload)
    return path


@app.get("/api/animations/{animation_id}/exports/{file_format}")
def export_animation(animation_id: str, file_format: str) -> FileResponse:
    try:
        source = _effective_animation_export(animation_id)
        if file_format == "npz":
            path = source
        elif file_format == "fbx":
            path = ANIMATIONS_ROOT / Path(animation_id).name / "animation.fbx"
            if not is_fbx_export_current(source, path):
                export_fbx(source, path)
        else:
            raise HTTPException(400, "format must be fbx or npz")
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "animation export not ready") from exc
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@app.get("/api/compositions")
def list_compositions(character_id: str | None = None) -> list[dict]:
    return compositions.list(character_id)


@app.post("/api/compositions", status_code=201)
def create_composition(body: CompositionCreateBody) -> dict:
    try:
        return compositions.create(body.character_id, body.name)
    except KeyError as exc:
        raise HTTPException(404, "character not found") from exc


@app.put("/api/compositions/{composition_id}")
def update_composition(composition_id: str, body: CompositionUpdateBody) -> dict:
    try:
        row = compositions.update(
            composition_id,
            name=body.name,
            segments=[segment.model_dump() for segment in body.segments],
            cuts=[cut.model_dump() for cut in body.cuts],
        )
        if row["segments"]:
            with np.load(compositions.compile(composition_id), allow_pickle=False) as data:
                row = {**row, "num_frames": int(data["world_motion"].shape[0])}
        return row
    except KeyError as exc:
        raise HTTPException(404, "composition or animation not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

@app.put("/api/compositions/{composition_id}/scene-objects")
def update_composition_scene_objects(composition_id: str, body: CompositionSceneObjectsBody) -> dict:
    try:
        return compositions.update_scene_objects(composition_id, body.scene_objects)
    except KeyError as exc:
        raise HTTPException(404, "composition not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/api/compositions/{composition_id}/ik-constraints")
def update_composition_ik_constraints(composition_id: str, body: CompositionIKConstraintsBody) -> dict:
    try:
        return compositions.update_ik_constraints(composition_id, body.ik_constraints)
    except KeyError as exc:
        raise HTTPException(404, "composition not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc



@app.patch("/api/compositions/{composition_id}/segments/rename")
def rename_composition_segments(composition_id: str, body: CompositionSegmentRenameBody) -> dict:
    try:
        return compositions.rename_segments(composition_id, body.segment_ids, body.name)
    except KeyError as exc:
        raise HTTPException(404, "composition or segment not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/compositions/{composition_id}/commit", status_code=201)
def commit_composition_to_animation(composition_id: str, body: RenameBody) -> dict:
    try:
        return compositions.commit_to_animation(composition_id, body.name)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "composition not ready") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/compositions/{composition_id}/delete-frames")
def delete_composition_frames(composition_id: str, body: DeleteFramesBody) -> dict:
    try:
        row = compositions.delete_frames(composition_id, body.start_frame, body.end_frame)
        with np.load(compositions.compile(composition_id), allow_pickle=False) as data:
            row["num_frames"] = int(data["world_motion"].shape[0])
        return row
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "composition not ready") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/compositions/{composition_id}/split-frame")
def split_composition_frame(composition_id: str, body: SplitFrameBody) -> dict:
    try:
        row = compositions.split_at_frame(composition_id, body.frame)
        with np.load(compositions.compile(composition_id), allow_pickle=False) as data:
            row["num_frames"] = int(data["world_motion"].shape[0])
        return row
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "composition not ready") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/compositions/{composition_id}/root-curve")
def apply_composition_root_curve(composition_id: str, body: RootCurveBody) -> dict:
    try:
        row = compositions.apply_root_curve(
            composition_id,
            body.points,
            body.still_threshold,
            body.control_points,
            body.exact_positions,
            body.exact_transforms,
        )
        with np.load(compositions.compile(composition_id), allow_pickle=False) as data:
            row["num_frames"] = int(data["world_motion"].shape[0])
        return row
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "composition not ready") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/compositions/{composition_id}")
def delete_composition(composition_id: str) -> dict:
    try:
        compositions.delete(composition_id)
    except KeyError as exc:
        raise HTTPException(404, "composition not found") from exc
    return {"ok": True}


@app.patch("/api/compositions/{composition_id}/transitions/{transition_id}")
def update_composition_transition(composition_id: str, transition_id: str, body: TransitionBody) -> dict:
    try:
        row = compositions.update_transition(composition_id, transition_id, body.left_frames, body.right_frames)
        with np.load(compositions.compile(composition_id), allow_pickle=False) as data:
            row["num_frames"] = int(data["world_motion"].shape[0])
        return row
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "transition not found") from exc


@app.post("/api/compositions/{composition_id}/transitions", status_code=201)
def create_composition_transition(composition_id: str, body: TransitionCreateBody) -> dict:
    try:
        row = compositions.add_transition(composition_id, body.left_segment_id, body.right_segment_id)
        with np.load(compositions.compile(composition_id), allow_pickle=False) as data:
            row["num_frames"] = int(data["world_motion"].shape[0])
        return row
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "composition not ready") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/compositions/{composition_id}/transitions/{transition_id}")
def delete_composition_transition(composition_id: str, transition_id: str) -> dict:
    try:
        return compositions.delete_transition(composition_id, transition_id)
    except KeyError as exc:
        raise HTTPException(404, "transition not found") from exc


@app.post("/api/compositions/{composition_id}/transitions/{transition_id}/apply")
def apply_composition_transition(composition_id: str, transition_id: str) -> dict:
    try:
        return compositions.apply_transition(composition_id, transition_id)
    except KeyError as exc:
        raise HTTPException(404, "transition not found") from exc


@app.post("/api/compositions/{composition_id}/orientation-corrections", status_code=201)
def create_composition_orientation_correction(composition_id: str, body: OrientationCorrectionBody) -> dict:
    try:
        row = compositions.add_orientation_correction(composition_id, body.start_frame, body.end_frame, body.mode)
        with np.load(compositions.compile(composition_id), allow_pickle=False) as data:
            row["num_frames"] = int(data["world_motion"].shape[0])
        return row
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "composition not ready") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.patch("/api/compositions/{composition_id}/orientation-corrections/{correction_id}")
def update_composition_orientation_correction(composition_id: str, correction_id: str, body: OrientationCorrectionUpdateBody) -> dict:
    try:
        row = compositions.update_orientation_correction(composition_id, correction_id, body.blend_in_frames, body.blend_out_frames)
        with np.load(compositions.compile(composition_id), allow_pickle=False) as data:
            row["num_frames"] = int(data["world_motion"].shape[0])
        return row
    except KeyError as exc:
        raise HTTPException(404, "orientation correction not found") from exc


@app.delete("/api/compositions/{composition_id}/orientation-corrections/{correction_id}")
def delete_composition_orientation_correction(composition_id: str, correction_id: str) -> dict:
    try:
        row = compositions.delete_orientation_correction(composition_id, correction_id)
        with np.load(compositions.compile(composition_id), allow_pickle=False) as data:
            row["num_frames"] = int(data["world_motion"].shape[0])
        return row
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "orientation correction not found") from exc


@app.post("/api/compositions/{composition_id}/orientation-corrections/{correction_id}/apply")
def apply_composition_orientation_correction(composition_id: str, correction_id: str) -> dict:
    try:
        row = compositions.apply_orientation_correction(composition_id, correction_id)
        with np.load(compositions.compile(composition_id), allow_pickle=False) as data:
            row["num_frames"] = int(data["world_motion"].shape[0])
        return row
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "orientation correction not found") from exc


@app.get("/api/compositions/{composition_id}/scene")
def composition_scene(composition_id: str) -> dict:
    try:
        scene = _rig_scene(compositions.compile(composition_id))
        row = compositions.get(composition_id)
        scene["scene_schema_version"] = int(row.get("schema_version", 2))
        scene["scene_objects"] = row.get("scene_objects", [])
        scene["ik_constraints"] = row.get("ik_constraints", [])
        return scene
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "composition not ready") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _composition_world(composition_id: str) -> tuple[np.lib.npyio.NpzFile, np.ndarray, EditProject]:
    path = compositions.compile(composition_id)
    data = np.load(path, allow_pickle=False)
    try:
        frames, joints = data["world_motion"].shape[:2]
        baked = EditProject(COMPOSITIONS_ROOT / Path(composition_id).name / "baked_edits.json", frames, joints)
        project = EditProject(COMPOSITIONS_ROOT / Path(composition_id).name / "edits.json", frames, joints)
        world = baked.apply(data["world_motion"], data["parents"]) if baked.operations else data["world_motion"]
        world = project.apply(world, data["parents"]) if project.operations else world
        row = compositions.get(composition_id)
        if row.get("ik_constraints"):
            names = [str(item) for item in data["names"].tolist()]
            world = apply_ik_constraints(world, data["parents"], names, row["ik_constraints"])
        return data, np.asarray(world, dtype=np.float32), project
    except BaseException:
        data.close()
        raise


@app.get("/api/compositions/{composition_id}/frames")
def composition_frames(composition_id: str, from_frame: int = 0, to_frame: int = 1) -> dict:
    try:
        data, world, _ = _composition_world(composition_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "composition not ready") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        return _frame_payload(world, from_frame, to_frame)
    finally:
        data.close()


@app.get("/api/compositions/{composition_id}/edits")
def list_composition_edits(composition_id: str) -> dict:
    try:
        data, _, project = _composition_world(composition_id)
        data.close()
        return {"operations": [vars(op) for op in project.operations]}
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, "composition not ready") from exc


@app.post("/api/compositions/{composition_id}/edits", status_code=201)
def add_composition_edit(composition_id: str, body: EditBody) -> dict:
    try:
        data, _, project = _composition_world(composition_id)
        data.close()
        return vars(project.add(EditOperation.create(**body.model_dump())))
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "composition not ready") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/compositions/{composition_id}/edits/{operation_id}")
def delete_composition_edit(composition_id: str, operation_id: str) -> dict:
    try:
        data, _, project = _composition_world(composition_id)
        data.close()
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, "composition not ready") from exc
    if not project.delete(operation_id):
        raise HTTPException(404, "edit not found")
    return {"ok": True}


@app.patch("/api/compositions/{composition_id}/edits/{operation_id}")
def update_composition_edit(composition_id: str, operation_id: str, body: EditBlendBody) -> dict:
    try:
        data, _, project = _composition_world(composition_id)
        data.close()
        return vars(project.update_blend(operation_id, body.blend_in_frames, body.blend_out_frames))
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, "edit not found") from exc


@app.post("/api/compositions/{composition_id}/edits/{operation_id}/apply")
def apply_composition_edit(composition_id: str, operation_id: str) -> dict:
    try:
        data, _, project = _composition_world(composition_id)
        frames, joints = data["world_motion"].shape[:2]
        data.close()
        operation = project.take(operation_id)
        baked = EditProject(COMPOSITIONS_ROOT / Path(composition_id).name / "baked_edits.json", frames, joints)
        baked.add(operation)
        return {"ok": True}
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, "edit not found") from exc


@app.delete("/api/compositions/{composition_id}/edits")
def clear_composition_edits(composition_id: str) -> dict:
    try:
        data, _, project = _composition_world(composition_id)
        data.close()
        project.clear()
        return {"ok": True}
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, "composition not ready") from exc


def _composition_export_payload(data: np.lib.npyio.NpzFile, world: np.ndarray) -> dict[str, np.ndarray]:
    """Package effective global frames using the same local/root convention as commit.

    Compilation/preview owns global frames; portable animation bundles and the
    Blender bridge reconstruct them from root @ local. Never mutate the source
    animation or compile cache, and include baked, pending and IK layers alike.
    """
    payload = {key: np.asarray(data[key]).copy() for key in data.files}
    effective = np.asarray(world, dtype=np.float32)
    if "virtual_root_motion" in payload:
        root = np.asarray(payload["virtual_root_motion"], dtype=np.float32)
        if len(root) != len(effective):
            raise ValueError("virtual root frame count does not match composition")
        effective = np.einsum(
            "fab,fjbc->fjac", np.linalg.inv(root.astype(np.float64)), effective.astype(np.float64)
        ).astype(np.float32)
    payload["world_motion"] = effective.copy()
    return payload


@app.get("/api/compositions/{composition_id}/exports/{file_format}")
def export_composition(composition_id: str, file_format: str) -> FileResponse:
    try:
        data, world, _ = _composition_world(composition_id)
        try:
            payload = _composition_export_payload(data, world)
            source = COMPOSITIONS_ROOT / Path(composition_id).name / "final_composed.npz"
            np.savez_compressed(source, **payload)
        finally:
            data.close()
        if file_format == "npz":
            path = source
        elif file_format == "fbx":
            path = COMPOSITIONS_ROOT / Path(composition_id).name / "final_composed.fbx"
            if not is_fbx_export_current(source, path):
                export_fbx(source, path)
        else:
            raise HTTPException(400, "format must be fbx or npz")
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "composition export not ready") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@app.get("/api/director-projects")
def list_director_projects() -> list[dict]:
    return director_projects.list()


@app.post("/api/director-projects", status_code=201)
def create_director_project(body: DirectorProjectCreateBody) -> dict:
    try:
        return director_projects.create(body.name, body.project)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/director-projects/{project_id}")
def get_director_project(project_id: str) -> dict:
    try:
        return director_projects.get(project_id)
    except KeyError as exc:
        raise HTTPException(404, "director project not found") from exc


@app.put("/api/director-projects/{project_id}")
def update_director_project(project_id: str, body: DirectorProjectUpdateBody) -> dict:
    try:
        return director_projects.update(project_id, body.project, body.expected_revision)
    except KeyError as exc:
        raise HTTPException(404, "director project not found") from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/director-projects/{project_id}/duplicate", status_code=201)
def duplicate_director_project(project_id: str, body: RenameBody) -> dict:
    try:
        return director_projects.duplicate(project_id, body.name)
    except KeyError as exc:
        raise HTTPException(404, "director project not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/director-projects/{project_id}")
def delete_director_project(project_id: str) -> dict:
    try:
        director_projects.delete(project_id)
        return {"ok": True}
    except KeyError as exc:
        raise HTTPException(404, "director project not found") from exc


@app.get("/api/director-projects/{project_id}/snapshots")
def list_director_snapshots(project_id: str) -> list[dict]:
    try:
        return director_projects.list_snapshots(project_id)
    except KeyError as exc:
        raise HTTPException(404, "director project not found") from exc


@app.post("/api/director-projects/{project_id}/snapshots", status_code=201)
def create_director_snapshot(project_id: str, body: DirectorSnapshotBody) -> dict:
    try:
        return director_projects.create_snapshot(project_id, body.name, body.kind)
    except KeyError as exc:
        raise HTTPException(404, "director project not found") from exc


@app.post("/api/director-projects/{project_id}/snapshots/{snapshot_id}/restore")
def restore_director_snapshot(project_id: str, snapshot_id: str) -> dict:
    try:
        return director_projects.restore_snapshot(project_id, snapshot_id)
    except KeyError as exc:
        raise HTTPException(404, "director project or snapshot not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/director-projects/{project_id}/exports")
def list_director_exports(project_id: str) -> list[dict]:
    try:
        return director_projects.list_exports(project_id)
    except KeyError as exc:
        raise HTTPException(404, "director project not found") from exc


@app.post("/api/director-projects/{project_id}/exports", status_code=201)
def create_director_export(project_id: str, body: DirectorExportCreateBody) -> dict:
    try:
        return director_projects.create_export(project_id, body.settings)
    except KeyError as exc:
        raise HTTPException(404, "director project not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/director-projects/{project_id}/exports/{job_id}")
def get_director_export(project_id: str, job_id: str) -> dict:
    try:
        return director_projects.get_export(project_id, job_id)
    except KeyError as exc:
        raise HTTPException(404, "director export not found") from exc


@app.patch("/api/director-projects/{project_id}/exports/{job_id}")
def update_director_export(project_id: str, job_id: str, body: DirectorExportUpdateBody) -> dict:
    try:
        payload = body.model_dump(exclude_none=True)
        return director_projects.update_export(project_id, job_id, payload)
    except KeyError as exc:
        raise HTTPException(404, "director export not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/director-projects/{project_id}/exports/{job_id}/cancel")
def cancel_director_export(project_id: str, job_id: str) -> dict:
    try:
        return director_projects.cancel_export(project_id, job_id)
    except KeyError as exc:
        raise HTTPException(404, "director export not found") from exc


@app.post("/api/director-projects/{project_id}/exports/{job_id}/resume")
def resume_director_export(project_id: str, job_id: str) -> dict:
    try:
        return director_projects.resume_export(project_id, job_id)
    except KeyError as exc:
        raise HTTPException(404, "director export not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/api/director-projects/{project_id}/exports/{job_id}/artifacts/{artifact_path:path}")
async def upload_director_export_artifact(project_id: str, job_id: str, artifact_path: str, file: UploadFile = File(...)) -> dict:
    try:
        return director_projects.save_artifact(project_id, job_id, artifact_path, await file.read())
    except KeyError as exc:
        raise HTTPException(404, "director export not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/api/director-projects/{project_id}/exports/{job_id}/frames/{pass_name}/{frame}")
async def upload_director_export_frame(project_id: str, job_id: str, pass_name: str, frame: int, file: UploadFile = File(...)) -> dict:
    try:
        return director_projects.save_frame(project_id, job_id, pass_name, frame, await file.read())
    except KeyError as exc:
        raise HTTPException(404, "director export not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/director-projects/{project_id}/exports/{job_id}/missing-frames")
def director_export_missing_frames(project_id: str, job_id: str, pass_name: str = "rgb") -> dict:
    try:
        return director_projects.missing_frames(project_id, job_id, pass_name)
    except KeyError as exc:
        raise HTTPException(404, "director export not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/director-projects/{project_id}/exports/{job_id}/encode")
def encode_director_export(project_id: str, job_id: str) -> dict:
    try:
        return director_projects.start_encoding(project_id, job_id)
    except KeyError as exc:
        raise HTTPException(404, "director export not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/director-projects/{project_id}/exports/{job_id}/package")
def package_director_export(project_id: str, job_id: str) -> dict:
    try:
        path = director_projects.package_export(project_id, job_id)
        return {"ok": True, "filename": path.name}
    except KeyError as exc:
        raise HTTPException(404, "director export not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/director-projects/{project_id}/exports/{job_id}/download")
def download_director_export(project_id: str, job_id: str) -> FileResponse:
    try:
        root = director_projects._job_path(project_id, job_id).parent
        job = director_projects.get_export(project_id, job_id)
        path = root / f"{Path(project_id).name}_{Path(job_id).name}_reference.zip"
        settings = job.get("settings") or {}
        passes = settings.get("passes") or ["rgb"]
        if settings.get("kind") == "video" and passes == ["rgb"]:
            video_format = "webm" if settings.get("format") == "webm" else "mp4"
            video = root / "artifacts" / f"director.{video_format}"
            if not video.is_file():
                raise FileNotFoundError(video)
            media_type = "video/webm" if video_format == "webm" else "video/mp4"
            return FileResponse(video, media_type=media_type, filename=f"director_{job_id}.{video_format}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return FileResponse(path, media_type="application/zip", filename=path.name)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "director export package not ready") from exc


@app.get("/", response_class=HTMLResponse)
def studio_index() -> HTMLResponse:
    index = Path(__file__).resolve().parent / "static" / "index.html"
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.get("/rig-editor", response_class=HTMLResponse)
def rig_editor() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="rig-static")

APP_STATIC_DIR = Path(__file__).resolve().parent / "static"
if APP_STATIC_DIR.is_dir():
    app.mount("/app-static", StaticFiles(directory=str(APP_STATIC_DIR)), name="app-static")


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _probe_meshpreps(port: int) -> bool:
    url = f"http://127.0.0.1:{port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            payload = json.load(resp)
            return (resp.status == 200 and
                    Path(payload.get("instance_root", "")).resolve() == Path(__file__).resolve().parents[1] and
                    payload.get("instance_id") == os.environ["KAIRYTH_V2_INSTANCE_ID"])
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, TypeError):
        return False


def _open_browser(url: str) -> None:
    if sys.platform == "win32":
        import os
        import subprocess

        try:
            os.startfile(url)  # noqa: S606
            return
        except OSError:
            pass
        try:
            subprocess.Popen(["cmd", "/c", "start", "", url], close_fds=True)  # noqa: S603
            return
        except OSError:
            pass
    webbrowser.open(url)


def run_server(*, port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    import uvicorn

    url = f"http://127.0.0.1:{port}/"
    if _port_in_use("127.0.0.1", port):
        if _probe_meshpreps(port):
            print(f"[MotionBuilder] 已在运行: {url}")
            if open_browser:
                _open_browser(url)
            return
        raise SystemExit(f"端口 {port} 已被其他程序占用，MotionBuilder 无法启动。")

    @app.on_event("startup")
    async def _open_browser_on_startup() -> None:
        if open_browser:
            try:
                _open_browser(url)
            except OSError as exc:
                print(f"[MotionBuilder] 无法自动打开浏览器: {exc}")
                print(f"[MotionBuilder] 请手动访问: {url}")

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
    server = uvicorn.Server(config)

    def _watch_shutdown() -> None:
        while not _shutdown_event.is_set():
            time.sleep(0.2)
        server.should_exit = True

    threading.Thread(target=_watch_shutdown, name="meshpreps-shutdown", daemon=True).start()
    store.register_shutdown_hook(_request_shutdown)
    server.run()
    store.on_shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    run_server(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
