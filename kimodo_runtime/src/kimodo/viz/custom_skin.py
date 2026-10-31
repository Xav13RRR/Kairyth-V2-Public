# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kimodo.rigging.types import RemappedSOMABinding
from kimodo.skeleton import SOMASkeleton77, batch_rigid_transform, global_rots_to_local_rots


class CustomMeshSkin:
    """LBS skinning for a user mesh remapped onto SOMA-77."""

    def __init__(self, skeleton, binding: RemappedSOMABinding):
        from kimodo.skeleton import SOMASkeleton30

        self.skeleton_input = skeleton
        assert isinstance(skeleton, (SOMASkeleton30, SOMASkeleton77))
        assert skeleton.neutral_joints is not None

        device = skeleton.neutral_joints.device
        skel_path = Path(skeleton.folder).parent / "somaskel77"
        self.skeleton_skin = SOMASkeleton77(str(skel_path)).to(device)

        bind_rig_np = np.array(binding.bind_rig_transform, dtype=np.float32)
        self.bind_rig_transform = torch.from_numpy(bind_rig_np).to(device=device, dtype=torch.float)
        bind_rig_inv_np = np.linalg.inv(bind_rig_np)
        self.bind_rig_transform_inv = torch.from_numpy(bind_rig_inv_np).to(device=device, dtype=torch.float)

        verts = binding.vertices.astype(np.float32) * float(binding.soma_scale)
        if binding.soma_root_offset is not None:
            verts = verts + binding.soma_root_offset.astype(np.float32)

        self.bind_vertices = torch.tensor(verts, device=device, dtype=torch.float)
        self.faces = torch.tensor(binding.faces, device=device, dtype=torch.long)
        self.lbs_indices = torch.tensor(binding.lbs_indices, device=device, dtype=torch.long)
        self.lbs_weights = torch.tensor(binding.lbs_weights, device=device, dtype=torch.float)

    def lbs(self, posed_transform: torch.Tensor) -> torch.Tensor:
        bind_rig_transform_inv = self.bind_rig_transform_inv
        bind_vertices = self.bind_vertices
        lbs_weights = self.lbs_weights
        for _ in range(posed_transform.dim() - 3):
            bind_rig_transform_inv = bind_rig_transform_inv.unsqueeze(0)
            bind_vertices = bind_vertices.unsqueeze(0)
            lbs_weights = lbs_weights.unsqueeze(0)

        affine_mat = (posed_transform @ bind_rig_transform_inv)[..., :3, :]
        vs = (
            affine_mat[..., self.lbs_indices, :, :]
            @ torch.concat([bind_vertices, torch.ones_like(bind_vertices[..., 0:1])], dim=-1)[..., None, :, None]
        )
        ws = lbs_weights[..., None, None]
        return (vs * ws).sum(dim=-3).squeeze(-1)

    def skin(self, joint_rotmat, joint_pos, rot_is_global: bool = False):
        nF, nJ = joint_pos.shape[:2]
        device = joint_rotmat.device

        if nJ != self.skeleton_skin.nbjoints:
            assert nJ == 30
            if rot_is_global:
                local_joint_rots_mats_subset = global_rots_to_local_rots(joint_rotmat, self.skeleton_input)
            else:
                local_joint_rots_mats_subset = joint_rotmat
            local_joint_rots_mats = self.skeleton_input.to_SOMASkeleton77(local_joint_rots_mats_subset)
            neutral_joints_seq = self.skeleton_skin.neutral_joints[None].repeat((nF, 1, 1)).to(device)
            new_joint_pos, joint_rotmat = batch_rigid_transform(
                local_joint_rots_mats,
                neutral_joints_seq,
                self.skeleton_skin.joint_parents.to(device),
                self.skeleton_skin.root_idx,
            )
            joint_pos = new_joint_pos + joint_pos[:, self.skeleton_input.root_idx : self.skeleton_input.root_idx + 1]
            rot_is_global = True

        fk_transform = torch.eye(4, device=device)[None, None].repeat(nF, self.skeleton_skin.nbjoints, 1, 1)
        fk_transform[..., :3, 3] = joint_pos
        if rot_is_global:
            fk_transform[..., :3, :3] = joint_rotmat
        else:
            neutral_joints_seq = self.skeleton_skin.neutral_joints[None].repeat((nF, 1, 1)).to(device)
            _, global_rots = batch_rigid_transform(
                joint_rotmat,
                neutral_joints_seq,
                self.skeleton_skin.joint_parents.to(device),
                self.skeleton_skin.root_idx,
            )
            fk_transform[..., :3, :3] = global_rots

        return self.lbs(fk_transform)
