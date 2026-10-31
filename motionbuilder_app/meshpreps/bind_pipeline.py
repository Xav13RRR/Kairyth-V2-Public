from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .bind_math import build_binding, normalize_parents, parents_to_array
from .binding_io import save_binding
from .config import (
    BINDING_METHOD,
    LOG_FILE,
    SKIN_INFER_SCRIPT,
    SKIN_MODEL_CKPT,
    SKELETON_TASK,
)
from .env_util import SKINTOKENS_ROOT, UNIRIG_ROOT, init_rig_env, skintokens_python, unirig_python
from .mesh_util import MeshData, write_obj, write_raw_data_npz

SKIN_INPUT = "motionbuilder_skin_input.npz"
SKIN_OUTPUT = "motionbuilder_skin_output.npz"


def _log(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {message}\n")


def _python_cmd(py: Path, *args: str) -> list[str]:
    py_str = str(py)
    if py_str.endswith("python") or py_str.endswith("python.exe"):
        return [py_str, *args]
    return py_str.split() + list(args)


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: float,
    stage: str,
    on_log: Callable[[str], None] | None = None,
    proc_registry: list[subprocess.Popen] | None = None,
) -> None:
    def emit(msg: str) -> None:
        _log(msg)
        if on_log:
            on_log(msg)

    emit(f"exec [{stage}]: {' '.join(cmd)} (cwd={cwd})")
    env = os.environ.copy()
    env.setdefault("TRANSFORMERS_ATTN_IMPLEMENTATION", "sdpa")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8:replace")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if proc_registry is not None:
        proc_registry.append(proc)
    start = time.time()
    stdout = ""
    stderr = ""
    try:
        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                proc.kill()
                proc.communicate()
                raise RuntimeError(f"{stage} 超时（>{int(timeout)}s）")
            try:
                stdout, stderr = proc.communicate(timeout=min(60.0, remaining))
                break
            except subprocess.TimeoutExpired:
                emit(f"{stage} still running ({int(time.time() - start)}s)...")
    finally:
        if proc_registry is not None and proc in proc_registry:
            proc_registry.remove(proc)
    if stdout:
        emit(f"stdout tail: {stdout[-1500:]}")
    if stderr:
        emit(f"stderr tail: {stderr[-1500:]}")
    if proc.returncode != 0:
        err = (stderr or stdout or "").strip()
        tail = err[-3000:] if len(err) > 3000 else err
        raise RuntimeError(f"{stage} 失败 (exit {proc.returncode}):\n{tail}")


def _find_npz(work_dir: Path, name: str) -> Path:
    matches = sorted(work_dir.rglob(name))
    if not matches:
        raise FileNotFoundError(f"未找到 {name}（{work_dir}）")
    return matches[-1]


def _load_npz_dict(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key][()] for key in data.files}


def write_skin_input_npz(work_dir: Path, skel_npz_path: Path) -> Path:
    payload = _load_npz_dict(skel_npz_path)
    names = payload.get("names")
    if names is None:
        names = [f"bone_{i}" for i in range(np.asarray(payload["joints"]).shape[0])]
    return write_soma_skin_input_npz(
        work_dir,
        vertices=np.asarray(payload["vertices"], dtype=np.float32),
        faces=np.asarray(payload["faces"], dtype=np.int64),
        joints=np.asarray(payload["joints"], dtype=np.float32),
        parents=normalize_parents(payload["parents"]),
        names=[str(n) for n in list(names)],
    )


def write_soma_skin_input_npz(
    work_dir: Path,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    names: list[str],
) -> Path:
    out = Path(work_dir) / SKIN_INPUT
    np.savez(
        out,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        joints=np.asarray(joints, dtype=np.float32),
        parents=normalize_parents(parents),
        names=np.array(list(names), dtype=object),
    )
    return out


def _run_skintokens_infer(
    work_dir: Path,
    *,
    timeout_s: float,
    on_log: Callable[[str], None] | None,
    proc_registry: list[subprocess.Popen] | None,
) -> np.ndarray:
    init_rig_env()
    skin_root = SKINTOKENS_ROOT.resolve()
    work_dir = Path(work_dir)
    skin_py = skintokens_python()
    infer_script = (skin_root / SKIN_INFER_SCRIPT).resolve()
    skin_cmd = _python_cmd(
        skin_py,
        str(infer_script),
        f"--input={(work_dir / SKIN_INPUT).as_posix()}",
        f"--output={(work_dir / SKIN_OUTPUT).as_posix()}",
        f"--model_ckpt={SKIN_MODEL_CKPT}",
    )
    run_cmd(
        skin_cmd,
        cwd=skin_root,
        timeout=timeout_s,
        stage="skintokens_skin",
        on_log=on_log,
        proc_registry=proc_registry,
    )
    return parse_skin_output(work_dir)


def parse_skin_output(work_dir: Path) -> np.ndarray:
    path = work_dir / SKIN_OUTPUT
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["skin"], dtype=np.float32)


def parse_skeleton(work_dir: Path, source_path: Path | None) -> dict:
    skel = _load_npz_dict(_find_npz(work_dir, "predict_skeleton.npz"))
    names_raw = skel.get("names")
    if names_raw is None:
        names_raw = [f"bone_{i}" for i in range(np.asarray(skel["joints"]).shape[0])]
    return {
        "vertices": np.asarray(skel["vertices"], dtype=np.float32),
        "faces": np.asarray(skel["faces"], dtype=np.int64),
        "joint_positions": np.asarray(skel["joints"], dtype=np.float32),
        "joint_parents": parents_to_array(skel.get("parents")),
        "joint_names": _normalize_joint_names([str(n) for n in list(names_raw)]),
        "source_path": source_path,
    }


def _normalize_joint_names(names: list[str]) -> list[str]:
    import sys

    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from ..rigging.joint_names import normalize_joint_names

    return normalize_joint_names(names)


def _prepare_mesh_inputs(mesh: MeshData, work_dir: Path) -> tuple[Path, Path]:
    obj_path = work_dir / "mesh.obj"
    npz_dir = work_dir / "npz"
    raw_npz = npz_dir / "mesh" / "raw_data.npz"
    write_obj(mesh, obj_path)
    write_raw_data_npz(mesh, raw_npz)
    return obj_path, npz_dir


def run_skeleton(
    mesh: MeshData,
    work_dir: Path,
    *,
    seed: int = 12345,
    timeout_s: float = 7200.0,
    on_log: Callable[[str], None] | None = None,
    proc_registry: list[subprocess.Popen] | None = None,
) -> dict:
    init_rig_env()
    unirig_root = UNIRIG_ROOT.resolve()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    obj_path, npz_dir = _prepare_mesh_inputs(mesh, work_dir)
    unirig_py = unirig_python()
    skel_cmd = _python_cmd(
        unirig_py,
        "run.py",
        f"--task={SKELETON_TASK}",
        f"--input={obj_path.as_posix()}",
        f"--npz_dir={npz_dir.as_posix()}",
        f"--output_dir={work_dir.as_posix()}",
        f"--seed={seed}",
    )
    run_cmd(skel_cmd, cwd=unirig_root, timeout=timeout_s, stage="skeleton", on_log=on_log, proc_registry=proc_registry)
    return parse_skeleton(work_dir, mesh.source_path)


def run_skin(
    work_dir: Path,
    *,
    timeout_s: float = 7200.0,
    on_log: Callable[[str], None] | None = None,
    proc_registry: list[subprocess.Popen] | None = None,
) -> np.ndarray:
    work_dir = Path(work_dir)
    skel_path = _find_npz(work_dir, "predict_skeleton.npz")
    write_skin_input_npz(work_dir, skel_path)
    return _run_skintokens_infer(
        work_dir,
        timeout_s=timeout_s,
        on_log=on_log,
        proc_registry=proc_registry,
    )


def run_skin_on_skeleton(
    work_dir: Path,
    skel: dict,
    *,
    timeout_s: float = 7200.0,
    on_log: Callable[[str], None] | None = None,
    proc_registry: list[subprocess.Popen] | None = None,
) -> np.ndarray:
    """Run SkinTokens weight inference on a provided mesh + skeleton."""
    work_dir = Path(work_dir)
    write_soma_skin_input_npz(
        work_dir,
        vertices=skel["vertices"],
        faces=skel["faces"],
        joints=skel["joint_positions"],
        parents=skel["joint_parents"],
        names=list(skel["joint_names"]),
    )
    return _run_skintokens_infer(
        work_dir,
        timeout_s=timeout_s,
        on_log=on_log,
        proc_registry=proc_registry,
    )


def run_bind(
    mesh: MeshData,
    output_npz: Path,
    *,
    work_dir: Path | None = None,
    seed: int = 12345,
    timeout_s: float = 7200.0,
    on_log: Callable[[str], None] | None = None,
    proc_registry: list[subprocess.Popen] | None = None,
) -> Path:
    cleanup = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="meshpreps_bind_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        skel = run_skeleton(
            mesh,
            work_dir,
            seed=seed,
            timeout_s=timeout_s,
            on_log=on_log,
            proc_registry=proc_registry,
        )
        skin = run_skin(work_dir, timeout_s=timeout_s, on_log=on_log, proc_registry=proc_registry)
        binding = build_binding(skel, skin, method=BINDING_METHOD, source_path=mesh.source_path)
        save_binding(output_npz, binding)
        _log(f"binding saved: {output_npz}")
        return output_npz
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)
