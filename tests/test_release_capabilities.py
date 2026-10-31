"""Model-free release policy and direct-creation/worker barriers."""
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from motionbuilder_app import release_profile as release
from motionbuilder_app import backend_config, pose_backend_client


def set_profile(tmp_path, monkeypatch, profile='public-workbench'):
    marker=tmp_path/'release-profile.json'
    marker.write_text(json.dumps(dict(schema='kairyth-release/v1',profile=profile)),encoding='utf-8')
    monkeypatch.setattr(release,'PROJECT_ROOT',tmp_path)
    return marker


@pytest.mark.parametrize('profile',['public-workbench','local-research'])
def test_explicit_profile_not_environment_or_checkout_identity(tmp_path,monkeypatch,profile):
    set_profile(tmp_path,monkeypatch,profile)
    monkeypatch.setenv('KAIRYTH_RELEASE_PROFILE','local-research' if profile=='public-workbench' else 'public-workbench')
    monkeypatch.setenv('KAIRYTH_HYBRIK_ROOT',str(tmp_path/'research'))
    monkeypatch.setenv('KAIRYTH_GVHMR_ROOT',str(tmp_path/'research'))
    caps=release.capabilities()
    assert caps['release_profile']==profile and caps['profile_valid']
    assert all(row['enabled']==(profile=='local-research') for row in caps['features'].values())


@pytest.mark.parametrize('content',[None,'{}','{bad','[]',
    '{"schema":"kairyth-release/v1","profile":"public-workbench","profile":"local-research"}',
    '{"schema":"kairyth-release/v1","profile":"unknown"}',
    '{"schema":"kairyth-release/v1","profile":"local-research","enabled":true}',
    '['*1500+']'*1500,'x'*4097])
def test_missing_or_invalid_marker_fails_closed_without_breaking_core(tmp_path,monkeypatch,content):
    monkeypatch.setattr(release,'PROJECT_ROOT',tmp_path)
    if content is not None:(tmp_path/'release-profile.json').write_text(content,encoding='utf-8')
    caps=release.capabilities()
    assert caps['release_profile']=='unconfigured' and not caps['profile_valid']
    assert all(row['state']=='disabled_configuration' for row in caps['features'].values())
    with pytest.raises(release.InferenceUnavailableError):release.require_inference('hybrik')


def test_public_status_does_not_resolve_research_environment(tmp_path,monkeypatch):
    set_profile(tmp_path,monkeypatch)
    monkeypatch.setenv('KAIRYTH_HYBRIK_ROOT',str(tmp_path/'environment backend'))
    monkeypatch.setenv('KAIRYTH_GVHMR_ROOT',str(tmp_path/'environment backend'))
    fail=Mock(side_effect=AssertionError('Public status must not probe backend configuration'))
    monkeypatch.setattr(backend_config,'hybrik_backend',fail)
    monkeypatch.setattr(backend_config,'gvhmr_backend',fail)
    status=backend_config.research_backend_status()
    for name in ('hybrik','gvhmr'):
        assert status[name]['state']=='release-unavailable'
        assert status[name]['physical_check_performed'] is False
        assert status[name]['message_key']=='research.compatibility_pending'
    fail.assert_not_called()


def test_create_and_start_reject_before_writes_or_reads(tmp_path,monkeypatch):
    from motionbuilder_app import pose_store, stores
    set_profile(tmp_path,monkeypatch)
    # Bypass constructors solely to prove create's barrier precedes all state
    # access, UUID/job creation, stream reads, cache creation or process launch.
    pose=pose_store.PoseStore.__new__(pose_store.PoseStore)
    motion=stores.MotionStore.__new__(stores.MotionStore)
    stream=Mock()
    for call in (lambda:pose.create(image_bytes=b'x',filename='image.png',name='test'),
        lambda:motion.create('input.mp4',stream,root_mode='world',static_camera=True,focal_length=None),
        lambda:pose_backend_client.start_worker(tmp_path,None,None)):
        before=set(tmp_path.rglob('*'))
        with pytest.raises(release.InferenceUnavailableError):call()
        assert set(tmp_path.rglob('*'))==before
    stream.read.assert_not_called()


def test_direct_motion_process_and_run_cannot_bypass_release(tmp_path,monkeypatch):
    from motionbuilder_app import stores, pose_store
    set_profile(tmp_path,monkeypatch)
    monkeypatch.setattr(stores,'MOTIONS_ROOT',tmp_path/'motions')
    monkeypatch.setattr(pose_store,'POSES_ROOT',tmp_path/'poses')
    motion=stores.MotionStore(None)
    pose=pose_store.PoseStore()
    start=Mock(side_effect=AssertionError('process must not start'))
    monkeypatch.setattr(stores.subprocess,'Popen',start)
    monkeypatch.setattr(pose_store,'start_worker',start)
    for store,root,job in ((motion,stores.MOTIONS_ROOT,dict(id='m',status='queued',log=[],progress=1)),
                          (pose,pose_store.POSES_ROOT,dict(id='p',status='queued'))):
        (root/job['id']).mkdir()
        store.jobs[job['id']]=job
        store._run(job['id'])
        assert job['status'] in {'error','failed'}
    with pytest.raises(release.InferenceUnavailableError):
        motion._run_process(dict(id='m',status='running'),['bad'],tmp_path)
    start.assert_not_called()
