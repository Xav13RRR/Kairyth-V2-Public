# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted for Kairyth's limited HybrIK compatibility shim.
# Upstream provenance and modifications: ../NOTICE.md; licence: ../LICENSE.
from __future__ import annotations

import torch


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive = x > 0
    ret[positive] = torch.sqrt(x[positive])
    return ret


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotations to rotation matrices.

    Compatible subset of PyTorch3D's function used by HybrIK.
    """
    orig_shape = axis_angle.shape
    vec = axis_angle.reshape(-1, 3)
    angle = torch.linalg.norm(vec, dim=1, keepdim=True)
    axis = vec / torch.clamp(angle, min=1e-8)
    x, y, z = axis.unbind(dim=1)
    ca = torch.cos(angle).squeeze(1)
    sa = torch.sin(angle).squeeze(1)
    one = torch.ones_like(ca)
    zero = torch.zeros_like(ca)
    k = torch.stack(
        [
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ],
        dim=1,
    ).reshape(-1, 3, 3)
    eye = torch.eye(3, dtype=vec.dtype, device=vec.device).expand(vec.shape[0], 3, 3)
    outer = axis[:, :, None] * axis[:, None, :]
    mat = ca[:, None, None] * eye + (one - ca)[:, None, None] * outer + sa[:, None, None] * k
    small = angle.squeeze(1) < 1e-8
    if small.any():
        mat[small] = eye[small]
    return mat.reshape(*orig_shape[:-1], 3, 3)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(*matrix.shape[:-2], 4)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.norm(quaternions[..., 1:], dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2.0 * half_angles
    sin_half_angles_over_angles = torch.empty_like(angles)
    small_angles = angles.abs() < 1e-6
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle rotations."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

