# SPDX-FileCopyrightText: Copyright (c) 2026 MKA Project contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
STUDIO_PORT = int(os.environ.get("STUDIO_PORT", os.environ.get("SERVER_PORT", "7860")))
STUDIO_HOST = os.environ.get("STUDIO_HOST", "127.0.0.1")
DEFAULT_MODEL = os.environ.get("KIMODO_MODEL", "Kimodo-SOMA-RP-v1.1")
DEFAULT_PROMPT = "A person walks forward."
DEFAULT_DURATION_SEC = 6.0
MIN_DURATION_SEC = 2.0
MAX_DURATION_SEC = 10.0
DEFAULT_DIFFUSION_STEPS = 100
DEFAULT_SEED = 12345
DEFAULT_CFG_WEIGHT = [2.0, 2.0]
DEFAULT_CUSTOM_MESH_OFFSET = [1.5, 0.0, 0.0]
BLENDER_EXE = os.environ.get("BLENDER_EXE", "")
SHUTDOWN_GRACE_S = float(os.environ.get("KIMODO_STUDIO_SHUTDOWN_GRACE", os.environ.get("KIMODO_SHUTDOWN_GRACE", "5")))
ANIMEDITS_PORT = int(os.environ.get("ANIMEDITS_PORT", "7861"))
ANIMEDITS_BUNDLE_DIR = PROJECT_ROOT / ".kimodo" / "animedits_bundles"


def is_pipeline_mode() -> bool:
    return os.environ.get("MKA_PIPELINE", "").strip() in ("1", "true", "yes")
