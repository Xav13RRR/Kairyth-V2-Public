from __future__ import annotations

import os
from pathlib import Path

from .config import PROJECT_ROOT, SKINTOKENS_ROOT, UNIRIG_ROOT

UNIRIG_PREFIX_PYTHON = PROJECT_ROOT / "third_party" / "envs" / "unirig" / "python.exe"
SKINTOKENS_PREFIX_PYTHON = PROJECT_ROOT / "third_party" / "envs" / "skintokens" / "python.exe"


def _under_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
        return True
    except ValueError:
        return False


def _resolve_rig_python(env_key: str, prefix_py: Path, env_name: str) -> None:
    # V2 never auto-discovers or executes global/old Conda environments.
    # External executable paths require both an explicit path and opt-in.
    current = os.environ.get(env_key, "").strip()
    if current:
        cur_path = Path(current)
        if cur_path.is_file() and (
            _under_project(cur_path) or os.environ.get("KAIRYTH_V2_ALLOW_EXTERNAL_ENV") == "1"
        ):
            return
        os.environ.pop(env_key, None)
    if prefix_py.is_file():
        os.environ[env_key] = str(prefix_py.resolve())
        return


def init_rig_env() -> None:
    os.environ["UNIRIG_ROOT"] = str(UNIRIG_ROOT.resolve())
    os.environ["SKINTOKENS_ROOT"] = str(SKINTOKENS_ROOT.resolve())
    _resolve_rig_python("UNIRIG_PYTHON", UNIRIG_PREFIX_PYTHON, "unirig")
    _resolve_rig_python("SKINTOKENS_PYTHON", SKINTOKENS_PREFIX_PYTHON, "skintokens")


def unirig_python() -> Path:
    init_rig_env()
    py = os.environ.get("UNIRIG_PYTHON")
    if not py:
        raise FileNotFoundError("未找到 UniRig Python。请按 docs/DEPLOYMENT.md 配置 third_party/envs/unirig。")
    return Path(py)


def skintokens_python() -> Path:
    init_rig_env()
    py = os.environ.get("SKINTOKENS_PYTHON")
    if not py:
        raise FileNotFoundError("未找到 SkinTokens Python。请按 docs/DEPLOYMENT.md 配置 third_party/envs/skintokens。")
    return Path(py)


def check_setup() -> tuple[bool, str]:
    init_rig_env()
    missing = []
    if not (UNIRIG_ROOT / "run.py").is_file():
        missing.append(f"UniRig: {UNIRIG_ROOT}")
    if not (SKINTOKENS_ROOT / "scripts/motionbuilder_skin_infer.py").is_file():
        missing.append(f"SkinTokens bridge: {SKINTOKENS_ROOT}")
    ckpt = SKINTOKENS_ROOT / "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
    if not ckpt.is_file():
        missing.append(f"SkinTokens ckpt: {ckpt}")

    py_unirig = os.environ.get("UNIRIG_PYTHON")
    py_skin = os.environ.get("SKINTOKENS_PYTHON")
    if not py_unirig or not Path(py_unirig).is_file():
        try:
            unirig_python()
        except FileNotFoundError as exc:
            return False, str(exc)
    if not py_skin or not Path(py_skin).is_file():
        try:
            skintokens_python()
        except FileNotFoundError as exc:
            return False, str(exc)

    if missing:
        return False, "缺失: " + "; ".join(missing)
    return True, f"Meshpreps 就绪（UniRig + SkinTokens @ {PROJECT_ROOT}）"
