# SPDX-FileCopyrightText: Copyright (c) 2026 MKA Project contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from kimodo import load_model
from kimodo.constraints import load_constraints_lst
from kimodo.skeleton import SOMASkeleton30
from kimodo.tools import seed_everything

from .config import DEFAULT_CFG_WEIGHT, DEFAULT_DIFFUSION_STEPS, DEFAULT_SEED
from .session import GenerateState, ModelBundle, MotionResult, StudioSession


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def ensure_model(session: StudioSession) -> ModelBundle:
    if session.model_bundle is not None:
        return session.model_bundle

    from kimodo.demo.text_encoder_runtime import (
        ensure_text_encoder_running,
        get_text_encoder_start_error,
    )

    if not ensure_text_encoder_running():
        detail = get_text_encoder_start_error()
        msg = "Text encoder 未能启动。"
        if detail:
            raise RuntimeError(f"{msg} {detail}")
        raise RuntimeError(
            f"{msg} 请检查 Hugging Face token，或手动运行 text-encoder 服务。"
        )

    device = _resolve_device()
    session.device = device
    model = load_model(modelname=session.model_name, device=device)
    skeleton = model.motion_rep.skeleton
    if isinstance(skeleton, SOMASkeleton30):
        skeleton = skeleton.somaskel77.to(model.device)

    bundle = ModelBundle(
        model=model,
        motion_rep=model.motion_rep,
        skeleton=skeleton,
        model_fps=float(model.fps),
        model_name=session.model_name,
    )
    session.model_bundle = bundle
    return bundle


def run_generation(
    session: StudioSession,
    *,
    seed: int = DEFAULT_SEED,
    diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
    num_samples: int = 1,
    cfg_weight: list[float] | None = None,
) -> MotionResult:
    bundle = ensure_model(session)
    seed_everything(seed)

    prompts = list(session.prompts)
    num_frames = session.num_frames_list()
    cfg_weight = cfg_weight or DEFAULT_CFG_WEIGHT

    model_constraints = []
    if session.constraints_json:
        model_constraints = load_constraints_lst(
            session.constraints_json,
            bundle.model.skeleton,
            device=bundle.model.device,
        )

    with torch.inference_mode():
        output = bundle.model(
            prompts,
            num_frames,
            diffusion_steps,
            multi_prompt=True,
            constraint_lst=model_constraints,
            cfg_weight=cfg_weight,
            num_samples=num_samples,
            cfg_type="separated",
            return_numpy=True,
        )

    posed = output["posed_joints"][0]
    rots = output["global_rot_mats"][0]
    foot = output.get("foot_contacts")
    foot_np = foot[0] if foot is not None else None

    if hasattr(posed, "cpu"):
        posed = posed.detach().cpu().numpy()
    if hasattr(rots, "cpu"):
        rots = rots.detach().cpu().numpy()
    if foot_np is not None and hasattr(foot_np, "cpu"):
        foot_np = foot_np.detach().cpu().numpy()

    export_output = {
        k: (v[0] if hasattr(v, "__getitem__") and hasattr(v, "shape") and len(v.shape) > 0 else v)
        for k, v in output.items()
    }

    motion = MotionResult(
        posed_joints=posed.astype("float32"),
        global_rot_mats=rots.astype("float32"),
        foot_contacts=foot_np.astype("float32") if foot_np is not None else None,
        fps=bundle.model_fps,
        prompts=prompts,
        raw_output=export_output,
    )
    session.motion = motion
    session.motion_retargeted = True
    return motion


def start_model_load_async(session: StudioSession) -> None:
    with session._lock:
        if session.model_load_job.state == GenerateState.RUNNING:
            raise RuntimeError("已有模型加载任务进行中")
        session.model_load_job.state = GenerateState.RUNNING
        session.model_load_job.message = "加载模型中（首次较慢）…"
        session.model_load_job.error = None

    def _run() -> None:
        try:
            ensure_model(session)
            with session._lock:
                session.model_load_job.state = GenerateState.DONE
                session.model_load_job.message = "模型已加载"
        except Exception as exc:
            with session._lock:
                session.model_load_job.state = GenerateState.ERROR
                session.model_load_job.error = str(exc)
                session.model_load_job.message = str(exc)[:400]

    import threading

    threading.Thread(target=_run, name="kimodo-studio-model-load", daemon=True).start()


def start_generation_async(session: StudioSession, **kwargs) -> None:
    with session._lock:
        if session.generate_job.state == GenerateState.RUNNING:
            raise RuntimeError("已有生成任务进行中")
        session.generate_job.state = GenerateState.RUNNING
        session.generate_job.message = "生成中…"
        session.generate_job.error = None

    def _run() -> None:
        try:
            run_generation(session, **kwargs)
            with session._lock:
                session.generate_job.state = GenerateState.DONE
                session.generate_job.message = "生成完成"
        except Exception as exc:
            with session._lock:
                session.generate_job.state = GenerateState.ERROR
                session.generate_job.error = str(exc)
                session.generate_job.message = str(exc)[:400]

    import threading

    threading.Thread(target=_run, name="kimodo-studio-generate", daemon=True).start()
