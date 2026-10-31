# SPDX-License-Identifier: Apache-2.0
# Original generic IPC client only; external programs retain their own terms.
"""Plain-file HybrIK worker client. No model imports or in-process inference."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import zipfile
import math

import numpy as np

from .backend_config import POSE_PROTOCOL, HybrikBackend
from .skeleton.soma30 import SOMA30_HIERARCHY
from .release_profile import require_inference

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024


def _reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result): raise ValueError("Nonfinite JSON number")
    return result


def read_json(path: Path):
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Backend JSON exceeds size limit")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_finite_float)


def check_npz(path: Path) -> None:
    """Bound ZIP expansion and NPY allocations before NumPy reads any array."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if not 1 <= len(entries) <= 64 or len({e.filename for e in entries}) != len(entries):
            raise ValueError("Invalid NPZ member count/names")
        if sum(e.file_size for e in entries) > MAX_ARTIFACT_BYTES:
            raise ValueError("NPZ expanded data exceeds limit")
        for entry in entries:
            if not entry.filename.endswith(".npy") or "/" in entry.filename or "\\" in entry.filename:
                raise ValueError("Invalid NPZ entry name")
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported NPY header version")
                size = math.prod(shape) * dtype.itemsize
                if any(dim < 0 for dim in shape) or dtype.hasobject or dtype.kind not in "fiuUb" or size > MAX_ARTIFACT_BYTES or size != entry.file_size - stream.tell():
                    raise ValueError("Unsafe NPZ dtype/shape/declared byte size")


def write_request(root: Path, image: Path, backend: HybrikBackend, *, name: str, source_name: str) -> Path:
    root = root.resolve()
    image = image.resolve()
    image.relative_to(root)
    payload = {"schema": POSE_PROTOCOL, "output_dir": str(root), "input_image": str(image),
               "runtime_root": str(backend.root), "name": name, "source_image_name": source_name}
    path = root / "backend_request.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)
    return path


def worker_environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "WANDB_API_KEY"): env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
               YOLO_AUTOINSTALL="False", WANDB_MODE="disabled", GRADIO_ANALYTICS_ENABLED="False")
    for key, suffix in {"TEMP":"temp", "TMP":"temp", "TMPDIR":"temp", "HF_HOME":"hf", "HF_HUB_CACHE":"hf/hub",
        "MPLCONFIGDIR":"mpl", "TORCH_HOME":"torch", "TORCH_EXTENSIONS_DIR":"extensions", "CUDA_CACHE_PATH":"cuda",
        "TORCHINDUCTOR_CACHE_DIR":"inductor", "TRITON_CACHE_DIR":"triton", "XDG_CACHE_HOME":"xdg",
        "XDG_CONFIG_HOME":"config", "YOLO_CONFIG_DIR":"yolo", "PYTORCH_KERNEL_CACHE_PATH":"kernels",
        "HUGGINGFACE_HUB_CACHE":"hf/hub", "TRANSFORMERS_CACHE":"hf/hub"}.items():
        directory = root / "backend_cache" / suffix
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    return env


def start_worker(root: Path, backend: HybrikBackend, request: Path) -> subprocess.Popen[str]:
    require_inference("hybrik")
    flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) | int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if os.name == "nt" else 0
    return subprocess.Popen([str(backend.python), "-B", str(backend.worker), "--request", str(request)],
        cwd=str(root), env=worker_environment(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", creationflags=flags, start_new_session=os.name != "nt")


def parse_event(line: str) -> dict | None:
    try: event = json.loads(line)
    except (ValueError, TypeError): return None
    if not isinstance(event, dict) or event.get("schema") != POSE_PROTOCOL: return None
    if event.get("kind") not in {"stage", "error", "completed"}: return None
    result = {"kind": event["kind"], "message": str(event.get("message", ""))[:1000]}
    if event["kind"] == "stage":
        value = event.get("progress")
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 100: return None
        result["progress"] = int(value)
    # Deliberately do not accept any path, command, PID or state from stdout.
    return result


def _owned_file(root: Path, name: str) -> Path:
    path = root / name
    resolved = path.resolve()
    if path.is_symlink() or resolved.parent != root.resolve() or not resolved.is_file():
        raise ValueError("Backend artifact is not a regular file directly owned by this job: " + name)
    if resolved.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("Backend artifact exceeds size limit")
    return resolved


def validate_result(root: Path) -> dict:
    root = root.resolve()
    result = read_json(_owned_file(root, "backend_result.json"))
    expected = {"raw": "hybrik_raw.npz", "asset": "pose_soma30.npz", "constraints": "constraints.json"}
    if not isinstance(result, dict) or result.get("schema") != POSE_PROTOCOL or any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Backend result protocol/artifact names mismatch")
    paths = {key: _owned_file(root, name) for key, name in expected.items()}
    for key in ("raw", "asset"): check_npz(paths[key])
    with np.load(paths["raw"], allow_pickle=False) as data:
        for key in data.files:
            value = data[key]
            if value.dtype.kind in "fiu" and not np.isfinite(value).all():
                raise ValueError("Nonfinite raw array: " + key)
        for key, shape in {"pred_thetas":(216,), "pred_xyz_29":(29,3), "pred_xyz_24_struct":(24,3), "pred_betas":(10,), "transl":(3,)}.items():
            values = data[key]
            if values.shape != shape or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
                raise ValueError("Invalid numeric raw backend array: " + key)
    with np.load(paths["asset"], allow_pickle=False) as data:
        for key, shape in {"world_motion":(1,30,4,4), "target_bind":(30,4,4), "parents":(30,), "fps":(1,)}.items():
            values = data[key]
            if values.shape != shape or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
                raise ValueError("Invalid numeric pose backend array: " + key)
        names, parents, fps = data["names"], data["parents"], float(data["fps"][0])
        if names.shape != (30,) or names.dtype.kind != "U" or len(set(names.tolist())) != 30:
            raise ValueError("Invalid pose skeleton names")
        if parents.dtype.kind not in "iu" or parents[0] != -1 or any(not 0 <= int(parent) < index for index,parent in enumerate(parents[1:],1)):
            raise ValueError("Invalid pose skeleton hierarchy")
        expected_names = [name for name, _ in SOMA30_HIERARCHY]
        expected_parents = [-1 if parent is None else expected_names.index(parent) for _, parent in SOMA30_HIERARCHY]
        if names.tolist() != expected_names or parents.tolist() != expected_parents:
            raise ValueError("Backend skeleton does not match SOMA30 protocol")
        if fps <= 0: raise ValueError("Invalid pose fps")
        for key in data.files:
            value = data[key]
            if value.dtype.kind not in "fiuUb" or (value.dtype.kind in "fiu" and not np.isfinite(value).all()):
                raise ValueError("Unsafe/object/nonfinite pose array: " + key)
    constraints = read_json(paths["constraints"])
    if not isinstance(constraints, list) or len(constraints) != 1 or not isinstance(constraints[0], dict):
        raise ValueError("Invalid pose constraints JSON")
    constraint = constraints[0]
    if constraint.get("type") != "fullbody" or constraint.get("frame_indices") != [0] or type(constraint["frame_indices"][0]) is not int:
        raise ValueError("Invalid fullbody frame indices")
    for key, shape in {"local_joints_rot": (1,30,3), "root_positions": (1,3), "smooth_root_2d": (1,2)}.items():
        value = np.asarray(constraint.get(key))
        if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError("Invalid fullbody array: " + key)
    return {**expected, "frames": 1, "joints": 30, "fps": fps}


def terminate_worker(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None: return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=15,
                           creationflags=int(getattr(subprocess,"CREATE_NO_WINDOW",0)), check=False)
        except (OSError, subprocess.TimeoutExpired):
            # Fallback is still confined to the process this client created.
            if process.poll() is None: process.kill()
    else:
        try: os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError: pass
    try: process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait(timeout=5)
