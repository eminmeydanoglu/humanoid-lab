from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from humanoid_lab.groot_inference_capture import GrootInferenceCapture, load_capture

class GrootCaptureTest(unittest.TestCase):
 def test_nested_arrays_round_trip_bit_exact(self):
  observation={'video':{'ego_view':np.arange(2*4*5*3,dtype=np.uint8).reshape(1,2,4,5,3)},'state':{'left_arm':np.linspace(-1,1,14,dtype=np.float32).reshape(1,2,7)},'language':{'annotation.human.task_description':[['task']]}}
  action={'motion_token':np.arange(40*64,dtype=np.float32).reshape(1,40,64),'left_hand_joints':np.ones((1,40,7),np.float32)}
  with tempfile.TemporaryDirectory() as tmp:
   cap=GrootInferenceCapture(Path(tmp),max_requests=1); token=cap.begin(observation,prompt='task',embodiment_tag='unitree_g1_sonic',options={'seed':3},source_stamps={'image_ns':11,'state_ns':None},checkpoint={'path':'/ckpt','processor_sha256':'abc'},send_id='send-7'); path=cap.finish(token,action,receive_id='reply-7')
   loaded=load_capture(path); meta=loaded['metadata']
  np.testing.assert_array_equal(loaded['observation']['video']['ego_view'],observation['video']['ego_view']); np.testing.assert_array_equal(loaded['observation']['state']['left_arm'],observation['state']['left_arm']); np.testing.assert_array_equal(loaded['action']['motion_token'],action['motion_token'])
  self.assertEqual(meta['send_id'],'send-7'); self.assertEqual(meta['receive_id'],'reply-7'); self.assertIsNone(meta['execute_id']); self.assertIsNone(meta['source_stamps']['state_ns'])
 def test_bound_disables_additional_capture(self):
  with tempfile.TemporaryDirectory() as tmp:
   cap=GrootInferenceCapture(Path(tmp),max_requests=1); one=cap.begin({'x':np.zeros(1)},prompt='p',embodiment_tag='e',options=None,source_stamps=None,checkpoint={}); cap.finish(one,{'a':np.zeros(1)}); self.assertIsNone(cap.begin({'x':np.zeros(1)},prompt='p',embodiment_tag='e',options=None,source_stamps=None,checkpoint={}))
 def test_metadata_is_never_lossy_array_json(self):
  with tempfile.TemporaryDirectory() as tmp:
   cap=GrootInferenceCapture(Path(tmp)); token=cap.begin({'image':np.zeros((4,5,3),np.uint8)},prompt='p',embodiment_tag='e',options=None,source_stamps=None,checkpoint={}); path=cap.finish(token,{'chunk':np.zeros((40,78),np.float32)}); metadata=json.loads((path/'metadata.json').read_text())
  self.assertEqual(metadata['observation_tree']['image']['dtype'],'|u1'); self.assertNotIn('[[',json.dumps(metadata['observation_tree']))

if __name__=='__main__': unittest.main()
