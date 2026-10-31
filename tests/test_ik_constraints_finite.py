"""Fail explicitly at the application boundary; do not change IK algorithms."""
from types import SimpleNamespace

import numpy as np
import pytest

from motionbuilder_app import ik_constraints


def _input():
    names = ["Hips", "LeftLeg", "LeftShin", "LeftFoot"]
    parents = np.asarray([-1, 0, 1, 2])
    world = np.repeat(np.eye(4, dtype=np.float32)[None, None], 4, axis=1)
    world[0, :, 1, 3] = [1.2, 1.1, .6, .1]
    return world, parents, names


def _constraint(solver="jacobian_dls"):
    return dict(joint_name="LeftFoot", solver=solver, root_behavior="pin",
                target_position=[5., 1.2, 0.], start_frame=0, end_frame=0)


@pytest.mark.parametrize("field", ["world", "residuals"])
def test_nonfinite_full_body_output_is_rejected_without_mutating_input(monkeypatch, field):
    world, parents, names = _input()
    original = world.copy()
    output = SimpleNamespace(world=world[0].copy(), residuals=np.asarray([1.], dtype=np.float32))
    getattr(output, field).flat[0] = np.nan
    monkeypatch.setattr(ik_constraints, "solve_full_body_frame", lambda *a, **kw: output)
    with pytest.raises(ValueError, match="jacobian_dls.*frame 0.*LeftFoot"):
        ik_constraints.apply_ik_constraints(world, parents, names, [_constraint()])
    np.testing.assert_array_equal(world, original)


def test_nonfinite_two_bone_output_is_rejected_without_mutating_input(monkeypatch):
    world, parents, names = _input()
    original = world.copy()
    failed = world[0].copy()
    failed[0, 0, 0] = np.inf
    monkeypatch.setattr(ik_constraints, "_solve_two_bone", lambda *a, **kw: failed)
    with pytest.raises(ValueError, match="two_bone.*frame 0.*LeftFoot"):
        ik_constraints.apply_ik_constraints(world, parents, names, [_constraint("two_bone")])
    np.testing.assert_array_equal(world, original)


def test_finite_solver_output_is_returned_exactly_and_parameters_unchanged(monkeypatch):
    world, parents, names = _input()
    expected = world[0].copy()
    expected[3, 0, 3] = .125
    calls = []

    def solve(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(world=expected, residuals=np.asarray([.001]))

    monkeypatch.setattr(ik_constraints, "solve_full_body_frame", solve)
    result = ik_constraints.apply_ik_constraints(world, parents, names, [_constraint()])
    np.testing.assert_array_equal(result, expected[None])
    assert calls == [dict(solver="jacobian_dls", iterations=20, tolerance=.002, root_behavior="pin")]
    np.testing.assert_array_equal(world, _input()[0])


def test_nonfinite_input_is_rejected_even_without_constraints():
    world, parents, names = _input()
    world[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="IK input contains non-finite"):
        ik_constraints.apply_ik_constraints(world, parents, names, [])


def test_finite_no_constraints_is_an_exact_independent_copy():
    world, parents, names = _input()
    result = ik_constraints.apply_ik_constraints(world, parents, names, [])
    np.testing.assert_array_equal(result, world)
    assert not np.shares_memory(result, world)
