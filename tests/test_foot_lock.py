from __future__ import annotations

import numpy as np

from motionbuilder_app.foot_lock import FootLockSettings, process_foot_lock
from motionbuilder_app.ik.full_body import IKTarget, solve_full_body_frame
from motionbuilder_app.meshpreps.bind_math import build_bind_matrices
from motionbuilder_app.server import app
from motionbuilder_app.skeleton import SOMASkeleton30


def _soma_motion(frames: int = 18) -> tuple[np.ndarray, np.ndarray, list[str]]:
    skeleton = SOMASkeleton30(load=True)
    names = list(skeleton.bone_order_names)
    parents = np.asarray(skeleton.joint_parents, dtype=np.int64)
    bind = build_bind_matrices(np.asarray(skeleton.neutral_joints, dtype=np.float32), parents).astype(np.float32)
    world = np.repeat(bind[None], frames, axis=0)
    world[:, :, 0, 3] += np.linspace(0.0, 0.35, frames, dtype=np.float32)[:, None]
    return world, parents, names


def test_foot_lock_holds_reference_feet_and_builds_constant_midpoint_root() -> None:
    world, parents, names = _soma_motion()
    reference = 4
    result = process_foot_lock(
        world,
        parents,
        names,
        settings=FootLockSettings(reference_frame=reference, foot_mode="both", solver="pbik"),
    )
    left = names.index("LeftFoot")
    right = names.index("RightFoot")
    np.testing.assert_allclose(
        result.preview_world_motion[:, left, :3, 3], np.repeat(world[[reference], left, :3, 3], len(world), axis=0), atol=2e-3
    )
    np.testing.assert_allclose(
        result.preview_world_motion[:, right, :3, 3], np.repeat(world[[reference], right, :3, 3], len(world), axis=0), atol=2e-3
    )
    midpoint = 0.5 * (world[reference, left, :3, 3] + world[reference, right, :3, 3])
    np.testing.assert_allclose(result.virtual_root_motion[:, :3, 3], np.repeat(midpoint[None], len(world), axis=0), atol=1e-7)
    reconstructed = np.einsum(
        "fab,fjbc->fjac", result.virtual_root_motion, result.local_world_motion
    )
    np.testing.assert_allclose(reconstructed, result.preview_world_motion, atol=2e-5)
    assert result.metrics["virtual_root_position_drift"] <= 1e-7


def test_all_full_body_solver_modes_reduce_a_reachable_effector_error() -> None:
    world, parents, names = _soma_motion(frames=1)
    left = names.index("LeftFoot")
    hips = names.index("Hips")
    target = world[0, left, :3, 3] + np.asarray([0.035, 0.0, 0.02], dtype=np.float32)
    before = float(np.linalg.norm(world[0, left, :3, 3] - target))
    for solver in ("pbik", "jacobian_dls", "fabrik"):
        result = solve_full_body_frame(
            world[0],
            parents,
            hips,
            [IKTarget(left, target)],
            solver=solver,
            root_behavior="pre_pull",
            iterations=30,
            tolerance=1e-3,
        )
        assert float(result.residuals[0]) < before
        assert np.isfinite(result.world).all()


def test_foot_lock_routes_are_exposed() -> None:
    paths = app.openapi()["paths"]
    assert "/api/foot-lock-previews" in paths
    assert "/api/foot-lock-previews/{preview_id}/scene" in paths
    assert "/api/foot-lock-previews/{preview_id}/frames" in paths
    assert "/api/foot-lock-previews/{preview_id}/commit" in paths
