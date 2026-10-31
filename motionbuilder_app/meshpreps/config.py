from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MESHPREPS_ROOT = Path(__file__).resolve().parent
STATIC_DIR = MESHPREPS_ROOT / "static"
DATA_ROOT = PROJECT_ROOT / "data" / "rigging"
UPLOAD_DIR = DATA_ROOT / "uploads"
OUTPUT_DIR = DATA_ROOT / "output"
WORK_DIR = DATA_ROOT / "work"
LOG_FILE = DATA_ROOT / "meshpreps.log"

# Runtime isolation is intentional: this application must never fall back to or
# mutate another project's model checkout through inherited environment values.
UNIRIG_ROOT = PROJECT_ROOT / "third_party" / "UniRig"
SKINTOKENS_ROOT = PROJECT_ROOT / "third_party" / "SkinTokens"

DEFAULT_PORT = int(os.environ.get("MESHPREPS_PORT", "7862"))
SHUTDOWN_GRACE_S = float(os.environ.get("MESHPREPS_SHUTDOWN_GRACE", "5"))


def is_pipeline_mode() -> bool:
    return os.environ.get("MKA_PIPELINE", "").strip() in ("1", "true", "yes")
DEFAULT_BIND_TIMEOUT_S = float(os.environ.get("MESHPREPS_BIND_TIMEOUT", "7200"))
DEFAULT_SEED = 12345

SKELETON_TASK = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
SKIN_INFER_SCRIPT = "scripts/motionbuilder_skin_infer.py"
SKIN_MODEL_CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
BINDING_METHOD = "SOMA-30 + SkinTokens"
FACES_TARGET_COUNT = 50_000
