import numpy as np
import pytest
from types import SimpleNamespace

from motionbuilder_app.server import _composition_export_payload
import motionbuilder_app.server as server


@pytest.mark.parametrize("with_root", [False, True])
def test_effective_composition_export_preserves_layers_and_root_convention(tmp_path, with_root):
    base = np.repeat(np.eye(4)[None, None], 3, axis=0).astype(np.float32)
    effective = base.copy()
    effective[:, 0, :3, 3] = [[.2, 1, 0], [.4, 1.1, .1], [.6, 1, .2]]
    payload = {"world_motion": base, "parents": np.array([-1]), "fps": np.array([30])}
    if with_root:
        root = np.repeat(np.eye(4)[None], 3, axis=0).astype(np.float32)
        for index, angle in enumerate([.2, .4, .6]):
            c, s = np.cos(angle), np.sin(angle)
            root[index, :3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        root[:, :3, 3] = [[1, 0, 0], [1.5, .1, 0], [2, 0, .5]]
        payload["virtual_root_motion"] = root
    source = tmp_path / "compile.npz"
    np.savez_compressed(source, **payload)
    original_bytes = source.read_bytes()
    with np.load(source, allow_pickle=False) as data:
        exported = _composition_export_payload(data, effective)
        assert np.array_equal(data["world_motion"], base)
    restored = exported["virtual_root_motion"][:, None] @ exported["world_motion"] if with_root else exported["world_motion"]
    assert np.allclose(restored, effective, atol=2e-7)
    assert source.read_bytes() == original_bytes
    assert np.array_equal(payload["world_motion"], base)


def test_composition_export_rejects_mismatched_root_length(tmp_path):
    source = tmp_path / "bad.npz"
    np.savez_compressed(source, world_motion=np.eye(4)[None, None], virtual_root_motion=np.repeat(np.eye(4)[None], 2, axis=0))
    with np.load(source, allow_pickle=False) as data:
        with pytest.raises(ValueError, match="frame count"):
            _composition_export_payload(data, data["world_motion"])


@pytest.mark.parametrize("route", ["character", "animation", "composition"])
@pytest.mark.parametrize("current", [False, True])
def test_fbx_routes_use_content_identity_with_correct_binding_flag(tmp_path, monkeypatch, route, current):
    source = tmp_path / "source.npz"
    np.savez_compressed(source, world_motion=np.eye(4)[None, None])
    root = tmp_path / "exports"
    (root / "test").mkdir(parents=True)
    expected_output = root / "test" / {
        "character": "binding.fbx", "animation": "animation.fbx", "composition": "final_composed.fbx"
    }[route]
    expected_output.write_bytes(b"existing cached export")
    monkeypatch.setattr(server, "ANIMATIONS_ROOT", root)
    monkeypatch.setattr(server, "COMPOSITIONS_ROOT", root)
    monkeypatch.setattr(server, "characters", SimpleNamespace(
        binding_path=lambda _: source, export_path=lambda *_: expected_output
    ))
    monkeypatch.setattr(server, "_effective_animation_export", lambda _: source)
    monkeypatch.setattr(server, "_composition_world", lambda _: (
        np.load(source, allow_pickle=False), np.eye(4)[None, None], SimpleNamespace(operations=[])
    ))
    checks, exports = [], []

    def is_current(source_path, output_path, *, bind_pose_only=False):
        checks.append((source_path, output_path, bind_pose_only))
        return current

    def export(source_path, output_path, *, bind_pose_only=False):
        exports.append((source_path, output_path, bind_pose_only))

    monkeypatch.setattr(server, "is_fbx_export_current", is_current)
    monkeypatch.setattr(server, "export_fbx", export)
    response = {"character": server.export_character, "animation": server.export_animation,
                "composition": server.export_composition}[route]("test", "fbx")
    assert response.path == expected_output
    assert len(checks) == 1 and checks[0][1] == expected_output
    assert checks[0][2] is (route == "character")
    assert len(exports) == (0 if current else 1)
    if exports:
        assert exports[0] == checks[0]
