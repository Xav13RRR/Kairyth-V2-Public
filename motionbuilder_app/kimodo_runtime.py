from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KIMODO_RUNTIME_ROOT = PROJECT_ROOT / "kimodo_runtime"
KIMODO_RUNTIME_SRC = KIMODO_RUNTIME_ROOT / "src"


def install_runtime_paths() -> None:
    """Expose only MotionBuilder-local Kimodo runtime folders to Python."""
    for path in (KIMODO_RUNTIME_SRC, KIMODO_RUNTIME_ROOT):
        if path.is_dir():
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)


def runtime_status() -> dict[str, Any]:
    candidates = [KIMODO_RUNTIME_SRC / "kimodo", KIMODO_RUNTIME_ROOT / "kimodo"]
    return {
        "runtime_root": str(KIMODO_RUNTIME_ROOT),
        "installed": any(path.is_dir() for path in candidates),
        "candidates": [str(path) for path in candidates],
    }


def _require_runtime() -> None:
    install_runtime_paths()
    if not runtime_status()["installed"]:
        raise RuntimeError(
            "Kimodo runtime is missing from Kairyth/kimodo_runtime/src. "
            "Restore the project-local Kimodo source and run scripts/verify_kimodo_runtime.ps1."
        )


def _soma30_names_parents() -> tuple[list[str], np.ndarray]:
    from kimodo.skeleton import SOMASkeleton30

    skel30 = SOMASkeleton30(load=True)
    pairs = list(skel30.bone_order_names_with_parents)
    names = [name for name, _parent in pairs]
    parents = np.asarray([-1 if parent is None else names.index(parent) for _name, parent in pairs], dtype=np.int64)
    return names, parents


def kimodo_raw_to_soma30_bundle(
    raw_npz: str | Path,
    output_npz: str | Path,
) -> Path:
    """Convert Kimodo raw posed joints / rotations to a SOMA-30 native bundle."""
    _require_runtime()
    from kimodo.rigging.soma30_retarget import extract_source_world_mats, source_bind_matrices

    raw_path = Path(raw_npz)
    output = Path(output_npz)
    with np.load(raw_path, allow_pickle=False) as data:
        posed = np.asarray(data["posed_joints"], dtype=np.float32)
        rots = np.asarray(data["global_rot_mats"], dtype=np.float32)
        fps = np.asarray(data["fps"], dtype=np.float32).reshape(-1)
        prompts = data["prompts"] if "prompts" in data.files else np.asarray([""], dtype="U256")
        durations_sec = data["durations_sec"] if "durations_sec" in data.files else np.asarray([], dtype=np.float32)
        constraints_json = data["constraints_json"] if "constraints_json" in data.files else None
        foot_contacts = np.asarray(data["foot_contacts"], dtype=np.float32) if "foot_contacts" in data.files else None
        smooth_root_pos = np.asarray(data["smooth_root_pos"], dtype=np.float32) if "smooth_root_pos" in data.files else None
        root_positions = np.asarray(data["root_positions"], dtype=np.float32) if "root_positions" in data.files else None
        global_root_heading = np.asarray(data["global_root_heading"], dtype=np.float32) if "global_root_heading" in data.files else None
        source_world = extract_source_world_mats(posed, rots).astype(np.float32)
        source_bind = source_bind_matrices().astype(np.float32)
    names, parents = _soma30_names_parents()
    payload: dict[str, Any] = {
        "world_motion": source_world,
        "target_bind": source_bind,
        "parents": parents,
        "names": np.asarray(names, dtype="U64"),
        "fps": fps,
        "num_frames": np.asarray([source_world.shape[0]], dtype=np.int64),
        "skeleton_type": np.asarray(["KIMODO-SOMA30"], dtype="U32"),
        "source_kind": np.asarray(["kimodo"], dtype="U32"),
        "prompts": prompts,
        "durations_sec": durations_sec,
        "posed_joints": posed,
        "global_rot_mats": rots,
    }
    if constraints_json is not None:
        payload["constraints_json"] = constraints_json
    if foot_contacts is not None:
        payload["foot_contacts"] = foot_contacts
    if smooth_root_pos is not None:
        payload["smooth_root_pos"] = smooth_root_pos
    if root_positions is not None:
        payload["root_positions"] = root_positions
    if global_root_heading is not None:
        payload["global_root_heading"] = global_root_heading
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return output


def kimodo_default_soma_scene() -> dict[str, Any]:
    """Return a skeleton-only default SOMA-30 scene for Kimodo UI previews."""
    _require_runtime()
    from kimodo.rigging.soma30_retarget import source_bind_matrices

    names, parents = _soma30_names_parents()
    bind = source_bind_matrices().astype(np.float32)
    return {
        "target_bind": bind.reshape(-1, 16).tolist(),
        "parents": parents.tolist(),
        "names": names,
        "fps": 30.0,
        "num_frames": 1,
        "skeleton_type": "KIMODO-SOMA30",
        "native_skeleton": True,
        "source_kind": "kimodo",
        "root_mode": "native_default",
    }


def generate_kimodo_native_bundle(
    output_npz: str | Path,
    *,
    prompt: str | None = None,
    prompts: list[str] | None = None,
    durations_sec: list[float] | None = None,
    duration_sec: float = 6.0,
    seed: int = 12345,
    diffusion_steps: int = 100,
    cfg_weight: list[float] | None = None,
    model_name: str = "Kimodo-SOMA-RP-v1.1",
    constraints: list[dict[str, Any]] | None = None,
) -> Path:
    """Generate a Kimodo native SOMA motion using the MotionBuilder-local runtime."""
    _require_runtime()
    import torch
    from kimodo import load_model
    from kimodo.constraints import load_constraints_lst
    from kimodo.demo.text_encoder_runtime import ensure_text_encoder_running, get_text_encoder_start_error
    from kimodo.tools import seed_everything

    if not ensure_text_encoder_running():
        detail = get_text_encoder_start_error()
        raise RuntimeError(f"Kimodo text encoder failed to start. {detail or ''}".strip())

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(modelname=model_name, device=device)
    fps = float(getattr(model, "fps", 30.0))
    prompt_list = [str(item).strip() for item in (prompts or ([prompt] if prompt is not None else [])) if str(item).strip()]
    if not prompt_list:
        raise ValueError("Kimodo prompt list cannot be empty")
    duration_list = [float(item) for item in (durations_sec or [duration_sec] * len(prompt_list))]
    if len(duration_list) != len(prompt_list):
        raise ValueError("Kimodo prompts and durations length mismatch")
    num_frames = [max(1, int(float(duration) * fps) - 1) for duration in duration_list]
    clean_constraints = constraints or []
    model_constraints = load_constraints_lst(clean_constraints, model.skeleton, device=device) if clean_constraints else []
    seed_everything(int(seed))
    with torch.inference_mode():
        output = model(
            prompt_list,
            num_frames,
            int(diffusion_steps),
            multi_prompt=True,
            constraint_lst=model_constraints,
            cfg_weight=cfg_weight or [2.0, 2.0],
            num_samples=1,
            cfg_type="separated",
            return_numpy=True,
        )

    posed = output["posed_joints"][0]
    rots = output["global_rot_mats"][0]
    foot = output.get("foot_contacts")
    if hasattr(posed, "detach"):
        posed = posed.detach().cpu().numpy()
    if hasattr(rots, "detach"):
        rots = rots.detach().cpu().numpy()
    foot_np = foot[0] if foot is not None else None
    if foot_np is not None and hasattr(foot_np, "detach"):
        foot_np = foot_np.detach().cpu().numpy()
    extra_arrays: dict[str, np.ndarray] = {}
    for key in ("smooth_root_pos", "root_positions", "global_root_heading"):
        if key in output:
            value = output[key][0]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            extra_arrays[key] = np.asarray(value, dtype=np.float32)

    temp_raw = Path(output_npz).with_name("kimodo_raw.npz")
    payload: dict[str, Any] = {
        "posed_joints": np.asarray(posed, dtype=np.float32),
        "global_rot_mats": np.asarray(rots, dtype=np.float32),
        "fps": np.asarray([fps], dtype=np.float32),
        "prompts": np.asarray(prompt_list, dtype="U512"),
        "durations_sec": np.asarray(duration_list, dtype=np.float32),
        "model_name": np.asarray([model_name], dtype="U128"),
    }
    if clean_constraints:
        constraints_text = json.dumps(clean_constraints, ensure_ascii=False)
        payload["constraints_json"] = np.asarray([constraints_text], dtype=f"U{max(1, len(constraints_text))}")
    payload.update(extra_arrays)
    if foot_np is not None:
        payload["foot_contacts"] = np.asarray(foot_np, dtype=np.float32)
    temp_raw.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temp_raw, **payload)
    return kimodo_raw_to_soma30_bundle(temp_raw, output_npz)
