# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess orchestration for SkinTokens skin inference (Kimodo bridge)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKINTOKENS_ROOT = PROJECT_ROOT / "third_party" / "SkinTokens"

SKIN_INPUT_NPZ = "kimodo_skin_input.npz"
SKIN_OUTPUT_NPZ = "kimodo_skin_output.npz"
INFER_SCRIPT = "scripts/kimodo_skin_infer.py"
DEFAULT_MODEL_CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
DEFAULT_TIMEOUT_S = 60 * 60


def _normalize_parents(parents_obj) -> np.ndarray:
    """UniRig predict_skeleton.npz uses None for root bone parent."""
    if parents_obj is None:
        return np.array([], dtype=np.int64)
    out: list[int] = []
    for p in np.asarray(parents_obj, dtype=object).ravel():
        if p is None:
            out.append(-1)
        else:
            out.append(int(p))
    return np.array(out, dtype=np.int64)


def skintokens_root() -> Path:
    env = os.environ.get("SKINTOKENS_ROOT")
    if env:
        return Path(env).resolve()
    return DEFAULT_SKINTOKENS_ROOT.resolve()


def _prefix_python(name: str) -> Path | None:
    path = PROJECT_ROOT / "third_party" / "envs" / name / "python.exe"
    return path if path.is_file() else None


def _conda_skintokens_python() -> Path | None:
    found = _prefix_python("skintokens")
    if found is not None:
        return found
    conda = shutil.which("conda")
    if conda is None:
        return None
    try:
        proc = subprocess.run(
            [conda, "run", "-n", "skintokens", "python", "-c", "import sys; print(sys.executable)"],
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


def skintokens_python() -> Path:
    env = os.environ.get("SKINTOKENS_PYTHON")
    if env:
        if env.endswith("python") or env.endswith("python.exe"):
            return Path(env.split()[0] if " " in env else env)
        return Path(env)
    found = _conda_skintokens_python()
    if found is not None:
        return found
    raise FileNotFoundError(
        "未找到 SkinTokens Python 环境。请设置 SKINTOKENS_PYTHON 或运行 .\\scripts\\setup_skintokens.ps1"
    )


def _python_cmd(*args: str) -> list[str]:
    py = skintokens_python()
    py_str = str(py)
    if py_str.endswith("python") or py_str.endswith("python.exe"):
        return [py_str, *args]
    return py_str.split() + list(args)


def check_skintokens_setup(*, fast: bool = False) -> tuple[bool, str]:
    root = skintokens_root()
    infer = root / INFER_SCRIPT
    ckpt = root / DEFAULT_MODEL_CKPT
    vae = root / "experiments/skin_vae_2_10_32768/last.ckpt"

    if not infer.is_file():
        return False, f"SkinTokens 未克隆或缺少桥接脚本：{infer}（运行 .\\scripts\\setup_skintokens.ps1）"
    if not ckpt.is_file() or not vae.is_file():
        missing = []
        if not ckpt.is_file():
            missing.append(ckpt.relative_to(root).as_posix())
        if not vae.is_file():
            missing.append(vae.relative_to(root).as_posix())
        return False, f"SkinTokens 权重缺失：{', '.join(missing)}（运行 setup_skintokens.ps1 下载）"

    try:
        py = skintokens_python()
    except FileNotFoundError as exc:
        return False, str(exc)

    if fast:
        return True, f"SkinTokens 路径就绪：{root}"

    ok, runtime_msg = _verify_skintokens_runtime(py)
    if not ok:
        return False, runtime_msg
    return True, f"SkinTokens 就绪：{root}（{runtime_msg}）"


def _verify_skintokens_runtime(py: Path) -> tuple[bool, str]:
    script = (
        "import importlib\n"
        "mods = ['torch', 'lightning', 'numpy', 'scipy', 'trimesh', 'transformers', 'einops']\n"
        "missing = []\n"
        "for name in mods:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except ImportError:\n"
        "        missing.append(name)\n"
        "if missing:\n"
        "    raise SystemExit('missing:' + ','.join(missing))\n"
        "import torch\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit('no_cuda')\n"
        "print(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')\n"
    )
    try:
        proc = subprocess.run(
            [str(py), "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"SkinTokens 环境检测失败：{exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if err.startswith("missing:"):
            pkgs = err.split(":", 1)[1]
            return False, f"SkinTokens 依赖未安装：{pkgs}（请重新运行 .\\scripts\\setup_skintokens.ps1）"
        if "no_cuda" in err:
            return False, "SkinTokens 需要 NVIDIA CUDA GPU（推荐 >=14GB 显存）"
        tail = err[-500:] if len(err) > 500 else err
        return False, f"SkinTokens 环境异常：{tail}"
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "runtime ok"
    return True, line


def write_skin_input_npz(work_dir: Path, skel_npz_path: Path) -> Path:
    """Write kimodo_skin_input.npz from UniRig predict_skeleton.npz."""
    data = np.load(skel_npz_path, allow_pickle=True)
    payload = {key: data[key][()] for key in data.files}
    required = ("vertices", "faces", "joints", "parents")
    for key in required:
        if key not in payload:
            raise KeyError(f"predict_skeleton.npz 缺少字段 {key}")

    out = Path(work_dir) / SKIN_INPUT_NPZ
    names = payload.get("names")
    if names is None:
        names = [f"bone_{i}" for i in range(np.asarray(payload["joints"]).shape[0])]

    np.savez(
        out,
        vertices=np.asarray(payload["vertices"], dtype=np.float32),
        faces=np.asarray(payload["faces"], dtype=np.int64),
        joints=np.asarray(payload["joints"], dtype=np.float32),
        parents=_normalize_parents(payload["parents"]),
        names=np.array(list(names), dtype=object),
    )
    return out


def run_skintokens_skin(
    work_dir: Path,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    run_cmd,
) -> Path:
    """Run SkinTokens bridge script; returns output npz path."""
    root = skintokens_root()
    input_path = Path(work_dir) / SKIN_INPUT_NPZ
    output_path = Path(work_dir) / SKIN_OUTPUT_NPZ
    if not input_path.is_file():
        raise FileNotFoundError(f"SkinTokens 输入缺失：{input_path}")

    infer_script = (root / INFER_SCRIPT).resolve()
    cmd = _python_cmd(
        str(infer_script),
        f"--input={input_path.as_posix()}",
        f"--output={output_path.as_posix()}",
        f"--model_ckpt={DEFAULT_MODEL_CKPT}",
    )
    run_cmd(cmd, cwd=root, timeout=timeout_s, stage="skintokens_skin")
    if not output_path.is_file():
        raise RuntimeError(f"SkinTokens 未生成输出：{output_path}")
    return output_path


def parse_skin_output(work_dir: Path) -> np.ndarray:
    path = Path(work_dir) / SKIN_OUTPUT_NPZ
    if not path.is_file():
        raise FileNotFoundError(f"未找到 {SKIN_OUTPUT_NPZ}（{work_dir}）")
    data = np.load(path, allow_pickle=True)
    skin = np.asarray(data["skin"], dtype=np.float32)
    if skin.ndim != 2:
        raise ValueError(f"kimodo_skin_output.npz skin 维度异常：{skin.shape}")
    return skin
