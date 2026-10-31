"""Exercise the real source middleware before any multipart/body read."""
import ast
import asyncio
import json
from pathlib import Path
import pytest
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from motionbuilder_app import release_profile as release


def middleware_from_server():
    path=Path(__file__).resolve().parents[1]/'motionbuilder_app/server.py'
    nodes=[]
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in {'_release_error_response','research_release_guard'}:
            node.decorator_list=[];nodes.append(node)
    namespace=dict(Request=Request,JSONResponse=JSONResponse,require_inference=release.require_inference,
                   InferenceUnavailableError=release.InferenceUnavailableError)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['research_release_guard']


@pytest.mark.parametrize('mode',['public','missing','invalid'])
@pytest.mark.parametrize('path',['/api/image-poses','/api/image-poses/','/api/motions','/api/motions/'])
def test_rejected_inference_does_not_receive_body(tmp_path,monkeypatch,mode,path):
    monkeypatch.setattr(release,'PROJECT_ROOT',tmp_path)
    monkeypatch.setenv('KAIRYTH_RELEASE_PROFILE','local-research')
    if mode!='missing':
        (tmp_path/'release-profile.json').write_text('{bad' if mode=='invalid' else
            json.dumps(dict(schema='kairyth-release/v1',profile='public-workbench')))
    async def endpoint(request):raise AssertionError('Disabled route was called')
    app=Starlette(routes=[Route(path,endpoint,methods=['POST'])])
    app.add_middleware(BaseHTTPMiddleware,dispatch=middleware_from_server())
    events=[]
    async def receive():raise AssertionError('Disabled upload body was consumed')
    async def send(event):events.append(event)
    scope=dict(type='http',asgi=dict(version='3.0',spec_version='2.3'),http_version='1.1',method='POST',scheme='http',
        path=path,raw_path=path.encode(),root_path='',query_string=b'',headers=[(b'content-type',b'multipart/form-data; boundary=x')],
        server=('127.0.0.1',1),client=('127.0.0.1',2))
    asyncio.run(app(scope,receive,send))
    assert events[0]['status']==503
    body=json.loads(b''.join(event.get('body',b'') for event in events))
    assert body['code']=='release-unavailable' and isinstance(body['detail'],str)
    assert body['capabilities']['features']['hybrik_image_inference']['enabled'] is False
    assert body['capability_state']==('disabled_experimental' if mode=='public' else 'disabled_configuration')
