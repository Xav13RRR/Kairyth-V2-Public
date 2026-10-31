# SPDX-FileCopyrightText: Copyright (c) 2026 MKA Project contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from kimodo.rigging.soma30_retarget import retarget_world_motion
from kimodo.viz.soma30_mesh_skin import Soma30MeshSkin

from .config import ANIMEDITS_BUNDLE_DIR, PROJECT_ROOT
from .session import StudioSession


def export_animedits_bundle(session: StudioSession, out_dir: Path | None = None) -> Path:
    if session.motion is None:
        raise ValueError("no motion generated")
    binding = session.remapped_binding
    if not binding.has_soma30_rig():
        raise ValueError("binding has no SOMA-30 rig")

    motion = session.motion
    world = retarget_world_motion(binding, motion.posed_joints, motion.global_rot_mats)
    skin = Soma30MeshSkin(binding)
    target_bind = skin.target_bind
    parents = np.asarray(binding.soma30_joint_parents, dtype=np.int64)
    names = np.array(binding.soma30_joint_names or [], dtype=object)

    bundle_dir = Path(out_dir or ANIMEDITS_BUNDLE_DIR)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    out_path = bundle_dir / f"animedits_{int(time.time() * 1000)}.npz"
    np.savez_compressed(
        out_path,
        rest_vertices=skin.rest_vertices_mesh.astype(np.float32),
        faces=np.asarray(binding.faces, dtype=np.int64),
        lbs_indices=skin.lbs_indices.astype(np.int64),
        lbs_weights=skin.lbs_weights.astype(np.float32),
        target_bind=target_bind.astype(np.float32),
        world_motion=world.astype(np.float32),
        parents=parents,
        names=names,
        fps=np.float32(motion.fps),
        num_frames=np.int32(world.shape[0]),
        binding_path=str(session.binding_path.resolve()),
        studio_root=str(PROJECT_ROOT),
    )
    return out_path.resolve()
