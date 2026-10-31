from types import SimpleNamespace

import numpy as np
import pytest

import motionbuilder_app.server as server


class _OpenBundle(dict):
    closed = False

    def close(self):
        self.closed = True


@pytest.mark.parametrize("failure_stage", ["edit", "ik", None])
def test_composition_bundle_closes_on_error_but_success_transfers_ownership(monkeypatch, failure_stage):
    data = _OpenBundle(world_motion=np.zeros((1, 1, 4, 4)), parents=np.array([-1]), names=np.array(["Hips"]))
    monkeypatch.setattr(server.np, "load", lambda *args, **kwargs: data)
    monkeypatch.setattr(server, "compositions", SimpleNamespace(
        compile=lambda _: "unused.npz", get=lambda _: {"ik_constraints": [{"enabled": True}]}
    ))

    def apply(world, parents):
        if failure_stage == "edit":
            raise ValueError("edit failed")
        return world

    monkeypatch.setattr(server, "EditProject", lambda *args: SimpleNamespace(operations=[object()], apply=apply))

    def ik(world, *args):
        if failure_stage == "ik":
            raise ValueError("non-finite solver output")
        return world

    monkeypatch.setattr(server, "apply_ik_constraints", ik)
    if failure_stage:
        with pytest.raises(ValueError):
            server._composition_world("test")
        assert data.closed
    else:
        returned, _, _ = server._composition_world("test")
        assert returned is data and not data.closed
        returned.close()
