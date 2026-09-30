"""Policy job validation, immutable inputs, reference isolation and cancellation."""
import importlib
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from humanoid_lab.evaluation.policy import FluxJointPolicy


class Dataset:
    fps = 30
    joint_names = [f'joint-{i}' for i in range(28)]
    index = __import__('pathlib').Path('/dataset/index')

    def validate_frame(self, episode_id, frame):
        if episode_id != '893' or not 1 <= frame < 70:
            raise ValueError('invalid frame')

    def validate_window(self, episode_id, frame):
        self.validate_frame(episode_id, frame)
        if frame + 32 >= 70:
            raise ValueError('32 future frames required')

    def reference_length(self, episode_id, frame, steps):
        self.validate_frame(episode_id, frame)
        return min(steps, 69 - frame)

    def fingerprint(self, episode_id):
        return {'rows': {'bytes': 1000, 'modified_ns': 123}}

    def describe(self, episode_id):
        return {'id':episode_id, 'source_episode':181,'dataset':'PickApple','split':'validation',
                'prompt':'Dataset caption','camera':'observation.images.cam_left_high','fps':30,'length':70}

    def catalog(self):
        return [self.describe('893')]

    def observation(self, episode_id, frame):
        self.validate_frame(episode_id, frame)
        return {'image': np.full((192,256,3),frame,dtype=np.uint8),
                'state':np.full(28,frame,dtype=np.float32),'row_index':frame + 100,'video_frame':frame+200}

    def reference(self, episode_id, frame, steps):
        count = self.reference_length(episode_id,frame,steps)
        return {'frames':np.full((count,192,256,3),244,dtype=np.uint8),
                'actions':np.full((count,28),999,dtype=np.float32),'steps':count}


@pytest.fixture
def policy_api(tmp_path, monkeypatch):
    monkeypatch.setenv('FLUX_EVAL_OUTPUT',str(tmp_path/'outputs'))
    monkeypatch.setenv('FLUX_EVAL_DATASET',str(tmp_path/'absent'))
    import humanoid_lab.evaluation.app as module
    module = importlib.reload(module)
    module.POLICY_DATASET = Dataset()
    calls = []

    class Model:
        log = None
        def reset(self): pass
        def unload(self): pass
        def predict(self,image,state,prompt,*,variant,seed):
            calls.append((variant,image.copy(),state.copy(),prompt,seed))
            frames = np.full((32,192,256,3),len(calls)+40,dtype=np.uint8)
            actions = np.full((32,28),len(calls)+10,dtype=np.float32)
            return {'frames':frames,'actions':actions,'normalized_actions':actions/2,
                    'normalized_state':state/2,'latents':np.zeros((1,96,9,6,8),dtype=np.float32),
                    'timings':{'total_seconds':1.0},'metadata':{'prompt':prompt},'clipping':{}}

    monkeypatch.setattr(FluxJointPolicy,'load',classmethod(lambda cls,**kw: Model()))
    import torch
    for fn in ('empty_cache','reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda,fn,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    from humanoid_lab.evaluation import normalization
    monkeypatch.setattr(normalization,'normalize',lambda values,*args:(np.asarray(values)/2,{'clipped_values':0}))
    with TestClient(module.app) as client:
        yield module,client,calls
    module.WORKER.shutdown(wait=True)


def wait(client,run_id):
    for _ in range(300):
        run = client.get(f'/api/runs/{run_id}').json()
        if run['status'] not in ('running','queued'):
            return run
        time.sleep(.02)
    raise AssertionError('job timed out')


def test_policy_validation_preview_and_paths(policy_api):
    module,client,_ = policy_api
    payload = {'episode_id':'893','start_frame':5,'prompt':' exact\ncustom prompt '}
    preview = client.post('/api/policy/preview',json=payload)
    assert preview.status_code == 200
    info = preview.json()
    assert info['prompt'] == payload['prompt']
    assert info['state'] == [5] * 28
    assert info['provenance']['row_index'] == 105
    assert client.get(info['image_url']).status_code == 200
    assert client.get('/api/policy/catalog').json()['default_episode'] == '893'
    for invalid in ({'prompt':'  '},{'start_frame':70},{'models':['base','base']},{'models':['unknown']},{'chunks':21}):
        assert client.post('/api/policy/runs',json={**payload,**invalid}).status_code == 422
    edge = client.post('/api/policy/preview',json={**payload,'start_frame':69}).json()
    assert edge['reference_available'] is False and edge['validation_error']
    assert client.post('/api/policy/runs',json={**payload,'start_frame':69}).status_code == 422
    assert client.post('/api/policy/runs',json=payload,headers={'Origin':'https://evil.invalid'}).status_code == 403


def test_dream_independent_physical_feedback_and_artifacts(policy_api):
    module,client,calls = policy_api
    payload = {'task_mode':'policy_dream','episode_id':'893','start_frame':5,'prompt':' exact\ncaption ', 'chunks':2}
    response = client.post('/api/policy/runs',json=payload)
    assert response.status_code == 202
    run = wait(client,response.json()['id'])
    assert run['status'] == 'completed',run.get('error')
    assert run['input_snapshot']['prompt'] == payload['prompt']
    assert len(calls) == 4
    for first,second in (calls[:2],calls[2:]):
        np.testing.assert_array_equal(first[1],np.full((192,256,3),5))
        np.testing.assert_array_equal(first[2],np.full(28,5))
        assert second[1][0,0,0] != 244
        assert second[2][0] != 999
    np.testing.assert_array_equal(calls[1][2],np.full(28,11))
    np.testing.assert_array_equal(calls[3][2],np.full(28,13))
    assert all(call[3] == payload['prompt'] and call[4] == 0 for call in calls)
    assert set(run['results']) == {'base','ft'}
    for variant,result in run['results'].items():
        data = client.get(result['actions_url']).json()
        assert len(data['predicted']) == 64 and len(data['reference']) == 64
        assert len(result['chunks']) == 2
        assert client.get(run['artifacts'][f'{variant}/actions.npz']).status_code == 200
        assert client.get(run['artifacts'][f'{variant}/actions.csv']).status_code == 200
    persisted = json.loads((module.STORE/'runs'/run['id']/'run.json').read_text())
    assert persisted['config'] == payload | {'models':['base','ft']}


def test_queued_cancel_persists_without_loading(policy_api,monkeypatch):
    module,client,calls = policy_api
    monkeypatch.setattr(module.WORKER,'submit',lambda *args:None)
    response = client.post('/api/policy/runs',json={'episode_id':'893','start_frame':5,'prompt':'apple'})
    run_id = response.json()['id']
    assert client.post(f'/api/runs/{run_id}/cancel').json()['cancel_requested']
    module.execute_policy(run_id)
    assert client.get(f'/api/runs/{run_id}').json()['status'] == 'cancelled'
    assert not calls
    assert client.post('/api/runs/unknown/cancel').status_code == 404


def test_cancel_intent_survives_restart(policy_api, monkeypatch):
    module, client, _ = policy_api
    monkeypatch.setattr(module.WORKER,'submit',lambda *args:None)
    created = client.post('/api/policy/runs',json={'episode_id':'893','start_frame':5,'prompt':'apple'}).json()
    client.post(f"/api/runs/{created['id']}/cancel")
    module.WORKER.shutdown(wait=True)
    module = importlib.reload(module)
    assert module.RUNS[created['id']]['status'] == 'cancelled'
    stored = json.loads((module.STORE/'runs'/created['id']/'run.json').read_text())
    assert stored['status'] == 'cancelled' and stored['error'] is None


def test_changed_dataset_rejects_comparison_basis(policy_api, monkeypatch):
    module, client, calls = policy_api
    monkeypatch.setattr(module.WORKER,'submit',lambda *args:None)
    created = client.post('/api/policy/runs',json={'episode_id':'893','start_frame':5,'prompt':'apple'}).json()
    monkeypatch.setattr(module.POLICY_DATASET,'fingerprint',lambda episode:{'changed':True})
    module.execute_policy(created['id'])
    result = client.get(f"/api/runs/{created['id']}").json()
    assert result['status'] == 'failed' and 'changed after submission' in result['error']
    assert not calls


def test_normalized_comparison_uses_same_clip_for_both_trajectories(policy_api, monkeypatch):
    _, client, _ = policy_api
    from humanoid_lab.evaluation import normalization
    monkeypatch.setattr(normalization,'normalize',lambda values,*args:(np.clip(np.asarray(values)/2,-6,6),{'clipped_values':0}))
    created = client.post('/api/policy/runs',json={'episode_id':'893','start_frame':5,'prompt':'apple'}).json()
    result = wait(client,created['id'])
    assert result['status'] == 'completed',result['error']
    data = client.get(result['results']['ft']['actions_url']).json()
    assert np.max(np.abs(data['normalized_predicted'])) <= 6
    assert np.max(np.abs(data['normalized_reference'])) <= 6
    assert len(data['sampler_normalized_predicted']) == 32
