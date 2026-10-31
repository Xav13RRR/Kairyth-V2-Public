# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimodo demo utilities (text-encoder runtime for kimodo_studio)."""

from .text_encoder_runtime import ensure_text_encoder_running, stop_text_encoder_subprocess

__all__ = [
    "ensure_text_encoder_running",
    "stop_text_encoder_subprocess",
]
