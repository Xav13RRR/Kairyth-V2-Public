# SPDX-FileCopyrightText: Copyright (c) 2026 MKA Project contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.skeleton import SOMASkeleton77

from .config import (
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_DURATION_SEC,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_SEED,
    MAX_DURATION_SEC,
    MIN_DURATION_SEC,
    STATIC_DIR,
)
from .generation import (
    ensure_model,
    run_generation,
    start_generation_async,
    start_model_load_async,
)
from .lifecycle import cancel_shutdown_timer, request_shutdown, schedule_shutdown, shutdown_event
from .session import GenerateState, StudioSession
from .skin_frames import (
    custom_binding_preview_dict,
    default_soma_preview_dict,
    motion_joint_frames,
    scene_dict,
    skin_motion_frames,
)

app = FastAPI(title="Kimodo Studio", version="1.0.0")
_session: StudioSession | None = None
_ws_clients: set[WebSocket] = set()
_had_client = False


class TimelineBody(BaseModel):
    prompts: list[str] = Field(..., min_length=1)
    durations_sec: list[float] = Field(..., min_length=1)


class ConstraintsBody(BaseModel):
    constraints: list[dict] = Field(default_factory=list)


class GenerateBody(BaseModel):
    seed: int = DEFAULT_SEED
    diffusion_steps: int = DEFAULT_DIFFUSION_STEPS
    async_mode: bool = True


class ModelBody(BaseModel):
    model_name: str = DEFAULT_MODEL


class OffsetBody(BaseModel):
    offset: list[float] = Field(..., min_length=3, max_length=3)


def init_session(binding_path: Path) -> StudioSession:
    global _session
    from kimodo.rigging.binding_io import load_remapped_binding

    path = Path(binding_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"binding not found: {path}")
    binding = load_remapped_binding(path)
    _session = StudioSession(
        binding_path=path,
        remapped_binding=binding,
        prompts=[DEFAULT_PROMPT],
        durations_sec=[DEFAULT_DURATION_SEC],
        model_name=DEFAULT_MODEL,
    )
    return _session


def get_session() -> StudioSession:
    if _session is None:
        raise HTTPException(503, "studio session not initialized")
    return _session


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
def health() -> dict:
    sess = get_session()
    return {
        "ok": True,
        "binding": str(sess.binding_path),
        "model_loaded": sess.model_bundle is not None,
        "has_motion": sess.motion is not None,
    }


@app.get("/api/studio/binding")
def studio_binding() -> dict:
    return custom_binding_preview_dict(get_session())


@app.get("/api/studio/scene")
def studio_scene() -> dict:
    return scene_dict(get_session())


@app.put("/api/studio/custom/offset")
def put_custom_offset(body: OffsetBody) -> dict:
    sess = get_session()
    sess.custom_mesh_offset = np.array(body.offset, dtype=np.float32)
    return {"custom_offset": sess.custom_mesh_offset.astype(np.float32).tolist()}


@app.get("/api/studio/timeline")
def get_timeline() -> dict:
    return get_session().timeline_dict()


@app.put("/api/studio/timeline")
def put_timeline(body: TimelineBody) -> dict:
    if len(body.prompts) != len(body.durations_sec):
        raise HTTPException(400, "prompts and durations_sec length mismatch")
    for d in body.durations_sec:
        if d < MIN_DURATION_SEC or d > MAX_DURATION_SEC:
            raise HTTPException(400, f"duration must be in [{MIN_DURATION_SEC}, {MAX_DURATION_SEC}]")
    sess = get_session()
    sess.prompts = list(body.prompts)
    sess.durations_sec = [float(d) for d in body.durations_sec]
    return sess.timeline_dict()


@app.get("/api/studio/constraints")
def get_constraints() -> dict:
    sess = get_session()
    return {"constraints": list(sess.constraints_json)}


@app.put("/api/studio/constraints")
def put_constraints(body: ConstraintsBody) -> dict:
    sess = get_session()
    sess.constraints_json = list(body.constraints)
    return {"constraints": list(sess.constraints_json)}


@app.post("/api/studio/model/load")
async def load_studio_model(request: Request) -> dict:
    sess = get_session()
    body = ModelBody()
    raw = await request.body()
    if raw:
        try:
            body = ModelBody.model_validate_json(raw)
        except Exception:
            pass
    sess.model_name = body.model_name
    if sess.model_bundle is not None and sess.model_load_job.state != GenerateState.RUNNING:
        bundle = sess.model_bundle
        return {
            "started": False,
            "state": GenerateState.DONE.value,
            "model_name": bundle.model_name,
            "model_fps": bundle.model_fps,
            "default_preview": default_soma_preview_dict(sess),
        }
    start_model_load_async(sess)
    return {"started": True, "state": GenerateState.RUNNING.value}


@app.get("/api/studio/model/load/status")
def model_load_status() -> dict:
    sess = get_session()
    job = sess.model_load_job
    out = {
        "state": job.state.value,
        "message": job.message,
        "error": job.error,
    }
    if job.state == GenerateState.DONE and sess.model_bundle is not None:
        bundle = sess.model_bundle
        out["model_name"] = bundle.model_name
        out["model_fps"] = bundle.model_fps
        out["default_preview"] = default_soma_preview_dict(sess)
    return out


@app.post("/api/studio/generate")
def studio_generate(body: GenerateBody | None = None) -> dict:
    body = body or GenerateBody()
    sess = get_session()
    if body.async_mode:
        start_generation_async(
            sess,
            seed=body.seed,
            diffusion_steps=body.diffusion_steps,
        )
        return {"started": True, "state": GenerateState.RUNNING.value}
    run_generation(sess, seed=body.seed, diffusion_steps=body.diffusion_steps)
    return {"started": False, "state": GenerateState.DONE.value, "motion": sess.motion_dict()}


@app.get("/api/studio/generate/status")
def generate_status() -> dict:
    sess = get_session()
    job = sess.generate_job
    out = {
        "state": job.state.value,
        "message": job.message,
        "error": job.error,
    }
    if job.state == GenerateState.DONE and sess.motion is not None:
        out["motion"] = sess.motion_dict()
    return out


@app.get("/api/studio/motion")
def studio_motion() -> dict:
    sess = get_session()
    meta = sess.motion_dict()
    if meta is None:
        raise HTTPException(404, "no motion generated")
    return meta


@app.get("/api/studio/motion/frames")
def motion_frames(
    from_frame: int = 0,
    to_frame: int = 1,
    target: str = "custom",
) -> dict:
    sess = get_session()
    if sess.motion is None:
        raise HTTPException(404, "no motion generated")
    if target not in {"default", "custom"}:
        raise HTTPException(400, "target must be 'default' or 'custom'")
    if sess.model_bundle is None:
        ensure_model(sess)
    frames = skin_motion_frames(sess, sess.motion, from_frame, to_frame, target=target)
    return {
        "from": from_frame,
        "to": to_frame,
        "target": target,
        "frames": frames,
    }


@app.get("/api/studio/motion/joints")
def motion_joints(
    from_frame: int = 0,
    to_frame: int = 1,
    target: str = "custom",
) -> dict:
    sess = get_session()
    if sess.motion is None:
        raise HTTPException(404, "no motion generated")
    if target not in {"default", "custom"}:
        raise HTTPException(400, "target must be 'default' or 'custom'")
    if sess.model_bundle is None:
        ensure_model(sess)
    joints = motion_joint_frames(sess, sess.motion, from_frame, to_frame, target=target)
    binding = sess.remapped_binding
    if target == "custom" and binding.has_soma30_rig():
        parents = np.asarray(binding.soma30_joint_parents, dtype=np.int64).tolist()
        names = list(binding.soma30_joint_names or [])
    else:
        skel77 = SOMASkeleton77(load=True)
        parents = skel77.joint_parents.detach().cpu().numpy().astype(np.int64).tolist()
        names = [n for n, _ in skel77.bone_order_names_with_parents]
    return {
        "from": from_frame,
        "to": to_frame,
        "target": target,
        "parents": parents,
        "names": names,
        "frames": joints,
    }


@app.post("/api/studio/animedits/launch")
def launch_animedits() -> dict:
    from .animedits_launch import animedits_launch_status, start_animedits_launch

    sess = get_session()
    if sess.motion is None:
        raise HTTPException(404, "no motion generated")
    if not sess.remapped_binding.has_soma30_rig():
        raise HTTPException(400, "binding has no SOMA-30 rig")
    try:
        start_animedits_launch(sess)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return animedits_launch_status()


@app.get("/api/studio/animedits/status")
def animedits_launch_status_endpoint() -> dict:
    from .animedits_launch import animedits_launch_status

    return animedits_launch_status()


@app.post("/api/studio/export/npz")
def export_npz() -> FileResponse:
    sess = get_session()
    if sess.motion is None or sess.motion.raw_output is None:
        raise HTTPException(404, "no motion to export")
    tmp = Path(tempfile.mkdtemp()) / "motion.npz"
    save_kimodo_npz(str(tmp), sess.motion.raw_output)
    return FileResponse(tmp, filename="kimodo_motion.npz", media_type="application/octet-stream")


@app.post("/api/studio/export/fbx")
def export_fbx() -> FileResponse:
    from .export_fbx import export_custom_mesh_fbx

    sess = get_session()
    if sess.motion is None:
        raise HTTPException(404, "no motion to export")
    if sess.model_bundle is None:
        ensure_model(sess)
    try:
        out_path = export_custom_mesh_fbx(sess)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    return FileResponse(out_path, filename="kimodo_custom_anim.fbx", media_type="application/octet-stream")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    global _had_client
    await ws.accept()
    _ws_clients.add(ws)
    _had_client = True
    cancel_shutdown_timer()
    try:
        while not shutdown_event().is_set():
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws.send_text("ping")
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)
        if _had_client and not _ws_clients:
            schedule_shutdown(had_client=_had_client, client_count=len(_ws_clients))


if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
