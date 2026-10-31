"""Presence-only optional backend checks; no models or research source required."""
import pytest
import json
from motionbuilder_app import release_profile
from motionbuilder_app.backend_config import GvhmrBackend

GVHMR_AUXILIARY = ["hmr4d/utils/smplx_utils.py", *["hmr4d/utils/body_model/"+name for name in (
    "__init__.py", "body_model.py", "body_model_smplh.py", "body_model_smplx.py", "smplx_lite.py",
    "coco_aug_dict.pth", "smpl_coco17_J_regressor.pt", "smplx_verts437.pt", "smplx2smpl_sparse.pt", "smpl_neutral_J_regressor.pt")]]


@pytest.fixture
def declared_local_profile(tmp_path,monkeypatch):
    (tmp_path/'release-profile.json').write_text(json.dumps(dict(schema='kairyth-release/v1',profile='local-research')))
    monkeypatch.setattr(release_profile,'PROJECT_ROOT',tmp_path)


def physical_gvhmr_fixture(tmp_path, omit=None):
    root=tmp_path/'research with spaces'
    paths=[*GVHMR_AUXILIARY, 'tools/demo/demo.py','inputs/checkpoints/yolo/yolov8x.pt',
        'inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt','inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl',
        'inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz']
    for relative in paths:
        if relative==omit:continue
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('--prepare_only --tracking_file' if relative.endswith('demo.py') else 'synthetic presence only')
    extractor=tmp_path/'extractor.py';extractor.write_text('synthetic')
    detector=tmp_path/'detector';detector.mkdir();(detector/'cli.py').write_text('synthetic')
    return GvhmrBackend(root,'/research/python','Research',extractor,detector,'/detector/python','test')


@pytest.mark.parametrize('missing',GVHMR_AUXILIARY)
def test_known_required_auxiliary_missing_is_not_ready(tmp_path,missing,declared_local_profile):
    backend=physical_gvhmr_fixture(tmp_path,missing)
    status=backend.status(cached_pose=True,cached_features=True)
    assert status['ready'] is False and str(backend.root/missing) in status['missing']
    assert status['runtime_verified'] is False and status['rights_verified'] is False


def test_hmr2_mean_params_only_required_without_cached_features(tmp_path,declared_local_profile):
    backend=physical_gvhmr_fixture(tmp_path)
    assert backend.status(cached_pose=True,cached_features=True)['ready']
    status=backend.status(cached_pose=True,cached_features=False)
    assert str(backend.root/'hmr4d/network/hmr2/configs/smpl_mean_params.npz') in status['missing']
    assert str(backend.root/'inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt') in status['missing']
