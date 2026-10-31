from __future__ import annotations

from ..motion_math import apply_lbs_numpy, world_to_skin_matrices

from .session import AnimEditsSession


def mesh_frames(session: AnimEditsSession, start: int, end: int, *, use_overlay: bool = True) -> list[list[list[float]]]:
    n_frames = session.num_frames
    start = max(0, start)
    end = min(n_frames, end)
    if start >= end:
        return []
    bind = session.target_bind
    out: list[list[list[float]]] = []
    for fi in range(start, end):
        world = (
            session.effective_world(fi)[None]
            if use_overlay
            else session.world_motion[fi][None]
        )
        skin_mats = world_to_skin_matrices(world, bind)[0]
        verts = apply_lbs_numpy(
            session.rest_vertices,
            session.lbs_indices,
            session.lbs_weights,
            skin_mats,
        )
        out.append(verts.tolist())
    return out


def joint_world_mats(
    session: AnimEditsSession,
    start: int,
    end: int,
    *,
    use_overlay: bool = True,
) -> list[list[list[float]]]:
    n_frames = session.num_frames
    start = max(0, start)
    end = min(n_frames, end)
    if start >= end:
        return []
    if use_overlay:
        return [session.effective_world(fi).reshape(-1).tolist() for fi in range(start, end)]
    return [session.world_motion[fi].reshape(-1).tolist() for fi in range(start, end)]
