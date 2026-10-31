# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess orchestration for VAST-AI UniRig skeleton + SkinTokens skin inference."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .humanoid_template import build_bind_matrices
from .joint_names import normalize_joint_names
from .skintokens_runner import (
    check_skintokens_setup,
    parse_skin_output,
    run_skintokens_skin,
    write_skin_input_npz,
)
from .types import ImportedMesh, UniRigBinding

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNIRIG_ROOT = PROJECT_ROOT / "third_party" / "UniRig"
BINDING_LOG = PROJECT_ROOT / ".kimodo" / "binding.log"

SKELETON_TASK = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
BINDING_METHOD = "unirig + skintokens"
FACES_TARGET_COUNT = 50_000
DEFAULT_SEED = 12_345
DEFAULT_TIMEOUT_S = 60 * 60


def unirig_root() -> Path:
    env = os.environ.get("UNIRIG_ROOT")
    if env:
        return Path(env).resolve()
    return DEFAULT_UNIRIG_ROOT.resolve()


def _prefix_python(name: str) -> Path | None:
    path = PROJECT_ROOT / "third_party" / "envs" / name / "python.exe"
    return path if path.is_file() else None


def _conda_unirig_python() -> Path | None:
    found = _prefix_python("unirig")
    if found is not None:
        return found
    conda = shutil.which("conda")
    if conda is None:
        return None
    try:
        proc = subprocess.run(
            [conda, "run", "-n", "unirig", "python", "-c", "import sys; print(sys.executable)"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not line:
        return None
    path = Path(line)
    return path if path.is_file() else None


def unirig_python() -> Path:
    env = os.environ.get("UNIRIG_PYTHON")
    if env:
        if env.endswith("python") or env.endswith("python.exe"):
            return Path(env.split()[0] if " " in env else env)
        # e.g. "conda run -n unirig python" — use conda invocation via wrapper
        return Path(env)
    found = _conda_unirig_python()
    if found is not None:
        return found
    raise FileNotFoundError(
        "未找到 UniRig Python 环境。请设置 UNIRIG_PYTHON 或运行 .\\scripts\\setup_unirig.ps1"
    )


def _python_cmd(*args: str) -> list[str]:
    py = unirig_python()
    py_str = str(py)
    if py_str.endswith("python") or py_str.endswith("python.exe"):
        return [py_str, *args]
    # UNIRIG_PYTHON may be a full shell command prefix
    return py_str.split() + list(args)


def check_unirig_setup(*, fast: bool = False) -> tuple[bool, str]:
    root = unirig_root()
    if not (root / "run.py").is_file():
        return False, f"UniRig 未克隆：{root}（运行 .\\scripts\\setup_unirig.ps1）"
    try:
        py = unirig_python()
    except FileNotFoundError as exc:
        return False, str(exc)

    skel_ckpt = root / "experiments/skeleton/articulation-xl_quantization_256/model.ckpt"
    if not skel_ckpt.is_file():
        return False, f"UniRig 骨骼权重缺失：{skel_ckpt.relative_to(root).as_posix()}（运行 setup_unirig.ps1 下载）"

    st_ok, st_msg = check_skintokens_setup(fast=fast)
    if not st_ok:
        return False, st_msg

    if fast:
        return True, f"UniRig + SkinTokens 路径就绪：{root}"

    ok, runtime_msg = _verify_unirig_runtime(py)
    if not ok:
        return False, runtime_msg
    return True, f"UniRig + SkinTokens 就绪：{root}（{runtime_msg}；{st_msg}）"


def _verify_unirig_runtime(py: Path) -> tuple[bool, str]:
    """Ensure unirig env can import packages required by run.py."""
    script = (
        "import importlib\n"
        "mods = ['torch', 'lightning', 'spconv', 'torch_scatter', 'numpy', 'trimesh', 'transformers', 'pyrender']\n"
        "missing = []\n"
        "for name in mods:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except ImportError:\n"
        "        missing.append(name)\n"
        "if missing:\n"
        "    raise SystemExit('missing:' + ','.join(missing))\n"
        "import torch\n"
        "print(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')\n"
    )
    try:
        proc = subprocess.run(
            [str(py), "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"UniRig 环境检测失败：{exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if err.startswith("missing:"):
            pkgs = err.split(":", 1)[1]
            return False, f"UniRig 依赖未安装：{pkgs}（请重新运行 .\\scripts\\setup_unirig.ps1）"
        tail = err[-500:] if len(err) > 500 else err
        return False, f"UniRig 环境异常：{tail}"
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "runtime ok"
    return True, line


def _write_obj(mesh: ImportedMesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    tm.export(path)


def _write_raw_data_npz(mesh: ImportedMesh, npz_path: Path, target_count: int = FACES_TARGET_COUNT) -> None:
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    vertices = np.asarray(tm.vertices, dtype=np.float32)
    faces = np.asarray(tm.faces, dtype=np.int64)
    if faces.shape[0] > target_count:
        try:
            simplified = tm.simplify_quadric_decimation(target_count)
            vertices = np.asarray(simplified.vertices, dtype=np.float32)
            faces = np.asarray(simplified.faces, dtype=np.int64)
            tm = simplified
        except Exception:
            pass
    payload = {
        "vertices": vertices,
        "vertex_normals": np.asarray(tm.vertex_normals, dtype=np.float32),
        "faces": faces,
        "face_normals": np.asarray(tm.face_normals, dtype=np.float32),
        "joints": None,
        "tails": None,
        "skin": None,
        "no_skin": None,
        "parents": None,
        "names": None,
        "matrix_local": None,
        "path": None,
        "cls": None,
    }
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **payload)


def _log_bind(message: str) -> None:
    import time

    BINDING_LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with BINDING_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {message}\n")


def _run_cmd(cmd: list[str], *, cwd: Path, timeout: float, stage: str = "infer") -> None:
    import time

    _log_bind(f"exec: {' '.join(cmd)} (cwd={cwd})")
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
    start = time.time()
    stdout = ""
    stderr = ""
    try:
        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                proc.kill()
                proc.communicate()
                _log_bind(f"failed timeout after {int(timeout)}s ({stage})")
                raise RuntimeError(f"UniRig {stage} 超时（>{int(timeout)}s）")
            try:
                stdout, stderr = proc.communicate(timeout=min(60.0, remaining))
                break
            except subprocess.TimeoutExpired:
                elapsed = int(time.time() - start)
                _log_bind(f"{stage} still running ({elapsed}s)...")
    except RuntimeError:
        raise
    if stdout:
        _log_bind(f"stdout tail: {stdout[-2000:]}")
    if stderr:
        _log_bind(f"stderr tail: {stderr[-2000:]}")
    if proc.returncode != 0:
        err = (stderr or stdout or "").strip()
        tail = err[-4000:] if len(err) > 4000 else err
        _log_bind(f"failed exit={proc.returncode}")
        raise RuntimeError(f"UniRig 命令失败 (exit {proc.returncode}):\n{tail}")


def _reskin(
    sampled_vertices: np.ndarray,
    vertices: np.ndarray,
    parents: list[int | None],
    faces: np.ndarray,
    sampled_skin: np.ndarray,
) -> np.ndarray:
    """Port of UniRig SkinWriter reskin (median + diffusion)."""
    nearest_samples = 7
    iter_steps = 1
    threshold = 0.03
    alpha = 2.0

    n_verts = vertices.shape[0]
    n_joints = sampled_skin.shape[1]
    tree = cKDTree(sampled_vertices)
    _dis, nearest = tree.query(vertices, k=nearest_samples, p=2)
    skin = np.median(sampled_skin[nearest], axis=1)

    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.concatenate([edges, edges[:, [1, 0]]], axis=0)

    for _ in range(iter_steps):
        sum_skin = skin.copy()
        for i in reversed(range(n_joints)):
            p = parents[i]
            if p is None or p < 0:
                continue
            sum_skin[:, p] += sum_skin[:, i]
        mask = sum_skin[edges[:, 1]] < sum_skin[edges[:, 0]]
        neighbor_skin = np.zeros_like(sum_skin)
        neighbor_co = np.zeros((n_verts, n_joints), dtype=np.float32)
        edge_dis = np.sqrt(((vertices[edges[:, 1]] - vertices[edges[:, 0]]) ** 2).sum(axis=1, keepdims=True))
        co = np.exp(-edge_dis * alpha)
        neighbor_skin[edges[:, 1]] += sum_skin[edges[:, 0]] * co * mask
        neighbor_co[edges[:, 1]] += co * mask
        sum_skin = (sum_skin + neighbor_skin) / (1.0 + neighbor_co)
        for i in range(n_joints):
            p = parents[i]
            if p is None or p < 0:
                continue
            sum_skin[:, p] -= sum_skin[:, i]
            skin = sum_skin / np.maximum(sum_skin.sum(axis=-1, keepdims=True), 1e-8)

    mask = (skin >= threshold).any(axis=-1, keepdims=True)
    skin[(skin < threshold) & mask] = 0.0
    skin = skin / np.maximum(skin.sum(axis=-1, keepdims=True), 1e-8)
    return skin.astype(np.float32)


def _parents_to_array(parents_obj) -> np.ndarray:
    if parents_obj is None:
        return np.array([], dtype=np.int64)
    arr = []
    for p in parents_obj:
        if p is None:
            arr.append(-1)
        else:
            arr.append(int(p))
    return np.array(arr, dtype=np.int64)


def _parents_list(parents_arr: np.ndarray) -> list[int | None]:
    out: list[int | None] = []
    for p in parents_arr:
        out.append(None if int(p) < 0 else int(p))
    return out


def _top_k_skin(skin: np.ndarray, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    n_verts, n_joints = skin.shape
    k = min(k, n_joints)
    top_idx = np.argsort(skin, axis=1)[:, -k:]
    top_w = np.take_along_axis(skin, top_idx, axis=1)
    top_w = top_w / np.maximum(top_w.sum(axis=1, keepdims=True), 1e-8)
    return top_idx.astype(np.int64), top_w.astype(np.float32)


def _find_npz(work_dir: Path, name: str) -> Path:
    matches = sorted(work_dir.rglob(name))
    if not matches:
        raise FileNotFoundError(f"未找到 {name}（工作目录 {work_dir}）")
    return matches[-1]


def _load_npz_dict(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key][()] for key in data.files}


def _parse_skeleton(work_dir: Path, source_path: Path | None) -> dict:
    skel_path = _find_npz(work_dir, "predict_skeleton.npz")
    skel = _load_npz_dict(skel_path)

    vertices = np.asarray(skel["vertices"], dtype=np.float32)
    faces = np.asarray(skel["faces"], dtype=np.int64)
    joints = np.asarray(skel["joints"], dtype=np.float32)
    parents = _parents_to_array(skel.get("parents"))
    names_raw = skel.get("names")
    if names_raw is None:
        names_raw = [f"bone_{i}" for i in range(joints.shape[0])]
    names = normalize_joint_names([str(n) for n in list(names_raw)])

    return {
        "vertices": vertices,
        "faces": faces,
        "joint_positions": joints,
        "joint_parents": parents,
        "joint_names": names,
        "source_path": source_path,
        "skel_path": skel_path,
    }


def _build_binding(skel: dict, skin: np.ndarray, *, method: str = BINDING_METHOD) -> UniRigBinding:
    vertices = skel["vertices"]
    joints = skel["joint_positions"]
    parents = skel["joint_parents"]

    if skin.ndim != 2:
        raise ValueError(f"skin 维度异常：{skin.shape}")
    if skin.shape[0] == joints.shape[0] and skin.shape[1] != joints.shape[0]:
        skin = skin.T
    if skin.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"skin 顶点数 {skin.shape[0]} 与 mesh {vertices.shape[0]} 不一致"
        )
    if skin.shape[1] != joints.shape[0]:
        raise ValueError(
            f"skin 骨骼数 {skin.shape[1]} 与 skeleton {joints.shape[0]} 不一致"
        )

    lbs_indices, lbs_weights = _top_k_skin(skin, k=4)
    bind_matrices = build_bind_matrices(joints, parents)

    return UniRigBinding(
        vertices=vertices,
        faces=skel["faces"],
        joint_positions=joints,
        joint_parents=parents,
        joint_names=skel["joint_names"],
        lbs_indices=lbs_indices,
        lbs_weights=lbs_weights,
        bind_matrices=bind_matrices,
        source_path=skel.get("source_path"),
        method=method,
    )


def run_unirig_bind(
    mesh: ImportedMesh,
    *,
    seed: int = DEFAULT_SEED,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    work_dir: Path | None = None,
) -> UniRigBinding:
    ok, msg = check_unirig_setup()
    if not ok:
        raise RuntimeError(msg)

    root = unirig_root()
    cleanup = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="kimodo_unirig_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    stem = "mesh"
    obj_path = work_dir / f"{stem}.obj"
    npz_dir = work_dir / "npz"
    sample_dir = npz_dir / stem
    raw_npz = sample_dir / "raw_data.npz"

    try:
        _log_bind(f"start bind seed={seed} work_dir={work_dir}")
        _write_obj(mesh, obj_path)
        _write_raw_data_npz(mesh, raw_npz)
        if not raw_npz.is_file():
            raise RuntimeError(f"raw_data.npz 未生成：{raw_npz}")
        _log_bind(f"raw_data.npz ready: {raw_npz}")

        skel_cmd = _python_cmd(
            "run.py",
            f"--task={SKELETON_TASK}",
            f"--input={obj_path.as_posix()}",
            f"--npz_dir={npz_dir.as_posix()}",
            f"--output_dir={work_dir.as_posix()}",
            f"--seed={seed}",
        )
        _run_cmd(skel_cmd, cwd=root, timeout=timeout_s, stage="skeleton")

        skel_path = _find_npz(work_dir, "predict_skeleton.npz")
        write_skin_input_npz(work_dir, skel_path)
        run_skintokens_skin(work_dir, timeout_s=timeout_s, run_cmd=_run_cmd)

        skel = _parse_skeleton(work_dir, mesh.source_path)
        skin = parse_skin_output(work_dir)
        return _build_binding(skel, skin)
    except Exception as exc:
        _log_bind(f"bind error: {exc}")
        raise
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)
