"""Offline boundary tests: no research models, weights or WSL startup."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest

from motionbuilder_app import backend_config as config
from motionbuilder_app import pose_backend_client as client
from motionbuilder_app.skeleton.soma30 import SOMA30_HIERARCHY


def declare_local_profile(tmp_path, monkeypatch):
    # This individual test explicitly exercises the maintenance profile, even
    # when collected from a public artifact. No global fail-open test default.
    from motionbuilder_app import release_profile
    profile=tmp_path/'declared local profile';profile.mkdir(exist_ok=True)
    (profile/'release-profile.json').write_text(json.dumps(dict(schema='kairyth-release/v1',profile='local-research')))
    monkeypatch.setattr(release_profile,'PROJECT_ROOT',profile)


def artifacts(root):
    names = [name for name, _ in SOMA30_HIERARCHY]
    parents = [-1 if parent is None else names.index(parent) for _, parent in SOMA30_HIERARCHY]
    np.savez(root/'hybrik_raw.npz', pred_thetas=np.zeros(216), pred_xyz_29=np.zeros((29,3)),
        pred_xyz_24_struct=np.zeros((24,3)), pred_betas=np.zeros(10), transl=np.zeros(3))
    pose = dict(world_motion=np.zeros((1,30,4,4)), target_bind=np.zeros((30,4,4)),
        parents=np.array(parents), names=np.array(names), fps=np.array([30.]))
    np.savez(root/'pose_soma30.npz', **pose)
    constraints = [dict(type='fullbody', frame_indices=[0], local_joints_rot=np.zeros((1,30,3)).tolist(),
        root_positions=[[0.,0.,0.]], smooth_root_2d=[[0.,0.]])]
    (root/'constraints.json').write_text(json.dumps(constraints))
    (root/'backend_result.json').write_text(json.dumps(dict(schema=config.POSE_PROTOCOL,
        raw='hybrik_raw.npz', asset='pose_soma30.npz', constraints='constraints.json')))
    return pose, constraints


def test_missing_configuration_never_imports_model_or_starts_wsl(tmp_path, monkeypatch):
    import sys
    before = set(sys.modules)
    for key in ('KAIRYTH_HYBRIK_ROOT','KAIRYTH_HYBRIK_WORKER','KAIRYTH_HYBRIK_PYTHON','KAIRYTH_GVHMR_ROOT'):
        monkeypatch.delenv(key, raising=False)
    assert not config.hybrik_backend(tmp_path).status()['ready']
    assert not config.gvhmr_backend(tmp_path).status()['ready']
    assert 'motionbuilder_app.pose_runtime' not in sys.modules
    assert not any(name == 'torch' or name.startswith(('torch.', 'hybrik.', 'hmr4d.')) for name in set(sys.modules)-before)


def test_explicit_roots_are_not_cwd_dependent(tmp_path, monkeypatch):
    monkeypatch.setenv('KAIRYTH_HYBRIK_ROOT','runtime with spaces')
    monkeypatch.setenv('KAIRYTH_GVHMR_ROOT','research with spaces')
    assert config.hybrik_backend(tmp_path).root == tmp_path/'runtime with spaces'
    assert config.gvhmr_backend(tmp_path).root == tmp_path/'research with spaces'


def test_event_does_not_inject_control_fields():
    event = client.parse_event(json.dumps(dict(schema=config.POSE_PROTOCOL,kind='stage',progress=30,
        message='test', command='bad', pid=123, asset='../bad', status='completed')))
    assert event == dict(kind='stage',progress=30,message='test')
    assert client.parse_event('ordinary warning') is None
    assert client.parse_event(json.dumps(dict(schema=config.POSE_PROTOCOL,kind='stage',progress=float('nan')))) is None


def test_valid_fixed_result(tmp_path):
    artifacts(tmp_path)
    assert client.validate_result(tmp_path)['joints'] == 30


@pytest.mark.parametrize('mutation',['path','list','nan','shape','parents','names','object','nonfinite'])
def test_rejects_invalid_results(tmp_path, mutation):
    pose, constraints = artifacts(tmp_path)
    if mutation == 'path':
        (tmp_path/'backend_result.json').write_text(json.dumps(dict(schema=config.POSE_PROTOCOL,raw='../x',asset='pose_soma30.npz',constraints='constraints.json')))
    elif mutation == 'list': (tmp_path/'backend_result.json').write_text('[]')
    elif mutation in ('nan','shape'):
        constraints[0]['root_positions'] = [[float('nan'),0,0]] if mutation=='nan' else [[0,0]]
        (tmp_path/'constraints.json').write_text(json.dumps(constraints))
    else:
        if mutation == 'parents': pose['parents'][1] = -1
        if mutation == 'names': pose['names'][1] = 'Wrong'
        if mutation == 'object': pose['evil'] = np.array([object()])
        if mutation == 'nonfinite': pose['world_motion'][0,0,0,0] = np.inf
        np.savez(tmp_path/'pose_soma30.npz',**pose)
    with pytest.raises((ValueError,TypeError)): client.validate_result(tmp_path)


def test_npz_declared_huge_shape_rejected_before_allocation(tmp_path):
    member = io.BytesIO()
    np.lib.format.write_array_header_1_0(member,dict(descr='<f8',fortran_order=False,shape=(2**40,)))
    with zipfile.ZipFile(tmp_path/'huge.npz','w') as archive: archive.writestr('x.npy',member.getvalue())
    with pytest.raises(ValueError,match='declared byte size'): client.check_npz(tmp_path/'huge.npz')


def test_npz_expansion_and_json_limits(tmp_path,monkeypatch):
    artifacts(tmp_path)
    monkeypatch.setattr(client,'MAX_ARTIFACT_BYTES',100)
    with pytest.raises(ValueError): client.check_npz(tmp_path/'pose_soma30.npz')
    monkeypatch.setattr(client,'MAX_JSON_BYTES',2)
    with pytest.raises(ValueError,match='size limit'): client.read_json(tmp_path/'constraints.json')


def make_pose_store(tmp_path, monkeypatch):
    declare_local_profile(tmp_path,monkeypatch)
    from motionbuilder_app import pose_store
    monkeypatch.setattr(pose_store,'POSES_ROOT',tmp_path/'poses')
    monkeypatch.setattr(pose_store,'require_backend',lambda status:None)
    return pose_store, pose_store.PoseStore()


def test_pose_completion_requires_validated_files(tmp_path,monkeypatch):
    from types import SimpleNamespace
    module,store = make_pose_store(tmp_path,monkeypatch)
    starts=[]
    def start(root,backend,request):
        assert store.lock._is_owned()
        starts.append(root)
        # A forged completed event without valid files must not create an asset.
        return SimpleNamespace(stdout=iter([json.dumps(dict(schema=config.POSE_PROTOCOL,kind='completed',message='fake',asset='../bad'))]),
            wait=lambda:0,poll=lambda:0)
    monkeypatch.setattr(module,'start_worker',start)
    job=store.create(image_bytes=b'fake-image',filename='image.png',name='test')
    thread=store.threads.get(job['id'])
    if thread:thread.join(5)
    actual=store.get(job['id'])
    assert starts and actual['status']=='failed' and actual['asset'] is None
    store.shutdown()


def test_pose_cancel_and_shutdown_only_own_workers(tmp_path,monkeypatch):
    import threading
    from types import SimpleNamespace
    module,store = make_pose_store(tmp_path,monkeypatch)
    started,release=threading.Event(),threading.Event()
    killed=[]
    def lines():
        release.wait(5)
        return
        yield ''
    process=SimpleNamespace(stdout=lines(),wait=lambda:130,poll=lambda:None if not release.is_set() else 130)
    def start(*args):started.set();return process
    def terminate(child):assert child is process;killed.append(child);release.set()
    monkeypatch.setattr(module,'start_worker',start)
    monkeypatch.setattr(module,'terminate_worker',terminate)
    job=store.create(image_bytes=b'fake',filename='a.png',name='test')
    assert started.wait(3)
    store.shutdown()
    assert killed==[process] and store.get(job['id'])['status']=='cancelled'
    assert (tmp_path/'poses'/job['id']/'backend.cancel').is_file()
    assert not store.processes and not store.threads


def make_motion_store(tmp_path,monkeypatch):
    declare_local_profile(tmp_path,monkeypatch)
    from motionbuilder_app import stores
    monkeypatch.setattr(stores,'MOTIONS_ROOT',tmp_path/'motions')
    store=stores.MotionStore(None)
    root=stores.MOTIONS_ROOT/'external-job';root.mkdir()
    source=root/'input.mp4';source.write_bytes(b'fixture')
    job=dict(id='external-job',input_path=str(source),status='queued',log=[],progress=1,
             static_camera=True,focal_length=35,root_mode='world')
    store.jobs[job['id']]=job
    return stores,store,job,root


def test_gvhmr_not_ready_starts_no_wsl(tmp_path,monkeypatch):
    from unittest.mock import Mock
    module,store,job,root=make_motion_store(tmp_path,monkeypatch)
    monkeypatch.setenv('KAIRYTH_GVHMR_ROOT',str(tmp_path/'missing backend'))
    run=Mock();monkeypatch.setattr(store,'_run_process',run)
    store._run(job['id'])
    assert job['status']=='error' and 'not ready' in job['error']
    run.assert_not_called()


def test_gvhmr_all_stages_use_selected_external_root(tmp_path,monkeypatch):
    module,store,job,root=make_motion_store(tmp_path,monkeypatch)
    selected=tmp_path/'external backend with spaces'
    detector=tmp_path/'detector with spaces'
    extractor=tmp_path/'external extractor.py'
    for key,value in dict(KAIRYTH_GVHMR_ROOT=str(selected),KAIRYTH_DETECTOR_ROOT=str(detector),
        KAIRYTH_GVHMR_EXTRACTOR=str(extractor),MOTIONBUILDER_WSL_DISTRO='SelectedResearch',
        MOTIONBUILDER_WSL_PYTHON='/selected/python',MOTIONBUILDER_DETECTOR_PYTHON='/detector/python').items():
        monkeypatch.setenv(key,value)
    monkeypatch.setattr(module,'require_backend',lambda status:None)
    commands=[]
    def run(current,command,cwd):
        commands.append(command)
        if len(commands)==3:
            path=root/'gvhmr/input/hmr4d_results.pt';path.parent.mkdir(parents=True);path.write_bytes(b'fake')
    monkeypatch.setattr(store,'_run_process',run)
    monkeypatch.setattr(module,'build_source_motion_bundle',lambda src,dst,**kwargs:dst.write_bytes(b'fake'))
    store._run(job['id'])
    assert job['status']=='completed' and len(commands)==4
    assert all(command[2]=='SelectedResearch' for command in commands)
    assert all('/selected/python' in commands[index] for index in (0,2,3))
    assert '/detector/python' in commands[1]
    assert module.windows_to_wsl(selected/'tools/demo/demo.py') in commands[0]
    assert module.windows_to_wsl(extractor) in commands[3]
    assert all('PYTHONPATH='+module.windows_to_wsl(selected) in commands[index] for index in (0,2,3))


def server_function(name, namespace):
    # Avoid importing a live server or creating its production data folders.
    import ast
    path=Path(__file__).resolve().parents[1]/'motionbuilder_app/server.py'
    node=next(node for node in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(node,ast.FunctionDef) and node.name==name)
    node.decorator_list=[]
    namespace['__name__']='motionbuilder_app.server'
    namespace['__package__']='motionbuilder_app'
    namespace['__file__']=str(path)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def test_health_preserves_legacy_ok_but_core_is_online(monkeypatch):
    import os
    from types import SimpleNamespace
    monkeypatch.setattr(config,'research_backend_status',lambda:dict(hybrik=dict(ready=False),gvhmr=dict(ready=False),
        release_profile='local-research',capabilities=dict(features={})))
    health=server_function('health',dict(Path=Path,os=os,check_setup=lambda:(False,'optional rigging missing'),
        find_blender=lambda:'',characters=SimpleNamespace(list=lambda:[]),motions=SimpleNamespace(list=lambda:[])))
    result=health()
    assert result['service_ready'] is True and result['ok'] is False
    assert result['rigging']['state']=='not_ready' and not result['optional_backends']['hybrik']['ready']


def test_kimodo_missing_dependency_is_503_not_core_error():
    class HTTPError(Exception):
        def __init__(self,code,detail):self.status_code=code;self.detail=detail
    def missing():raise ModuleNotFoundError("No module named torch",name='torch')
    function=server_function('kimodo_runtime_default_scene',dict(kimodo_default_soma_scene=missing,HTTPException=HTTPError))
    with pytest.raises(HTTPError) as caught:function()
    assert caught.value.status_code==503 and caught.value.detail['code']=='optional-backend-not-ready'
    def business_error():raise ImportError('unrelated implementation bug')
    function.__globals__['kimodo_default_soma_scene']=business_error
    with pytest.raises(ImportError,match='implementation bug'):function()


def test_actual_registered_worker_process_terminates(tmp_path,monkeypatch):
    declare_local_profile(tmp_path,monkeypatch)
    import sys
    worker=tmp_path/'sleeping worker.py'
    worker.write_text("import time\nprint('ready',flush=True)\ntime.sleep(60)\n")
    request=tmp_path/'backend_request.json';request.write_text('{}')
    backend=config.HybrikBackend(tmp_path,Path(sys.executable),worker,'test')
    process=client.start_worker(tmp_path,backend,request)
    try:
        assert process.stdout.readline().strip()=='ready'
        client.terminate_worker(process)
        assert process.poll() is not None
    finally:
        if process.poll() is None:client.terminate_worker(process)
