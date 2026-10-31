# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize UniRig / Mixamo-style joint names to the Kimodo humanoid template."""

from __future__ import annotations

import re
from typing import Sequence

from .soma_retarget_profile import UNIRIG_NAME_ALIASES, UNIRIG_TO_SOMA

_CANONICAL = {name.lower(): name for name in UNIRIG_TO_SOMA}


def normalize_joint_names(names: Sequence[str]) -> list[str]:
    out: list[str] = []
    for raw in names:
        name = str(raw).strip()
        if not name:
            out.append("Hips")
            continue
        if name in UNIRIG_TO_SOMA:
            out.append(name)
            continue
        stripped = re.sub(r"^(mixamorig:|mixamo:|bip01\s+|joint_)", "", name, flags=re.IGNORECASE).strip()
        if stripped in UNIRIG_TO_SOMA:
            out.append(stripped)
            continue
        key = re.sub(r"[_\-\s]", "", stripped).lower()
        if key in UNIRIG_NAME_ALIASES:
            out.append(UNIRIG_NAME_ALIASES[key])
            continue
        if key in _CANONICAL:
            out.append(_CANONICAL[key])
            continue
        out.append(stripped or name)
    return out
