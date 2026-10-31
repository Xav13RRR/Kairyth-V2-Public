"""Real store/HTTP endpoint contracts stay usable without inference modules.

Compile the actual unchanged endpoint bodies into an isolated FastAPI app so
collecting this test never constructs the production server/data directories.
"""
import ast
import io
import json
from pathlib import Path
from typing import Literal
import numpy as np
import pytest
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from motionbuilder_app import pose_store, stores, release_profile as release
from motionbuilder_app.skeleton.soma30 import SOMA30_HIERARCHY


@pytest.mark.parametrize('profile',['public-workbench','missing','invalid'])
def test_existing_assets_remain_available_via_real_endpoint_bodies(tmp_path,monkeypatch,profile):
    monkeypatch.setattr(release,'PROJECT_ROOT',tmp_path)
    if profile!='missing':
        (tmp_path/'release-profile.json').write_text('{bad' if profile=='invalid' else
            json.dumps(dict(schema='kairyth-release/v1',profile=profile)))
    monkeypatch.setattr(pose_store,'POSES_ROOT',tmp_path/'poses')
    monkeypatch.setattr(stores,'MOTIONS_ROOT',tmp_path/'motions')
    poses=pose_store.PoseStore();motions=stores.MotionStore(None)
    app=FastAPI()
    namespace=dict(__name__=__name__,app=app,np=np,Path=Path,Literal=Literal,BaseModel=BaseModel,Field=Field,
        File=File,Form=Form,HTTPException=HTTPException,Request=Request,UploadFile=UploadFile,JSONResponse=JSONResponse,
        require_inference=release.require_inference,InferenceUnavailableError=release.InferenceUnavailableError,
        poses=poses,motions=motions)
    functions={'_release_error_response','research_release_guard','_pose_scene','confirm_image_pose',
        'import_pose_library_item','pose_library_scene','save_edited_pose_library_item',
        'create_pose_library_item_from_motion','motion_scene','motion_frames','list_pose_library'}
    classes={'RenameBody','PoseEditSaveBody','PoseFromMotionBody'}
    path=Path(__file__).resolve().parents[1]/'motionbuilder_app/server.py'
    nodes=[node for node in ast.parse(path.read_text(encoding='utf-8')).body
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.name in functions|classes]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    names=[name for name,_ in SOMA30_HIERARCHY]
    parents=np.array([-1 if parent is None else names.index(parent) for _,parent in SOMA30_HIERARCHY])
    bind=np.tile(np.eye(4,dtype=np.float32),(30,1,1));bind[:,1,3]=np.arange(30)*.02
    world=np.tile(bind,(2,1,1,1));world[1,:,0,3]+=.1
    data=dict(world_motion=world,target_bind=bind,parents=parents,names=np.array(names),fps=np.array([30.]))
    source=stores.MOTIONS_ROOT/'old-video';source.mkdir()
    np.savez(source/'source_smplx22.npz',**data)
    motions.jobs['old-video']=dict(id='old-video',status='completed',source_bundle='source_smplx22.npz',root_mode='world',name='Existing motion')
    payload=io.BytesIO();np.savez(payload,**dict(data,world_motion=world[:1]))
    with TestClient(app) as client:
        assert client.get('/api/motions/old-video/scene').status_code==200
        assert client.get('/api/motions/old-video/frames').status_code==200
        imported=client.post('/api/poses',files={'pose':('pose.npz',payload.getvalue())},data={'name':'Imported existing pose'})
        assert imported.status_code==201,imported.text
        pose_id=imported.json()['id']
        poses.jobs[pose_id]['confirmed']=False
        assert client.post('/api/image-poses/'+pose_id+'/confirm',json={'name':'Confirmed old output'}).status_code==200
        assert client.get('/api/poses/'+pose_id+'/scene').status_code==200
        edited=client.post('/api/poses/'+pose_id+'/edited',json={'name':'Edited old pose','world_motion':world[0].reshape(30,16).tolist()})
        assert edited.status_code==201,edited.text
        extracted=client.post('/api/poses/from-motion',json={'source_kind':'video','motion_id':'old-video','frame':1,'name':'Old frame'})
        assert extracted.status_code==201,extracted.text
        assert len(client.get('/api/poses').json())==3
    assert poses.processes=={} and motions.processes=={}
    poses.shutdown();motions.shutdown()
