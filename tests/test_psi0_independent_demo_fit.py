import importlib.util
from pathlib import Path
import unittest
import numpy as np

SPEC=importlib.util.spec_from_file_location('fit',Path(__file__).parents[1]/'scripts/psi0-independent-demo-fit.py')
fit=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(fit)

class IndependentFitTest(unittest.TestCase):
 def test_alignment_masks_episode_end(self):
  x=np.arange(5*2).reshape(5,2)
  chunk,n=fit.aligned_chunk(x,3,4)
  self.assertEqual(n,2); np.testing.assert_array_equal(chunk,x[3:5])
 def test_hold_repeats_previous_recorded_action(self):
  x=np.arange(5*2).reshape(5,2)
  np.testing.assert_array_equal(fit.repeated_hold(x,3,3),np.repeat(x[2:3],3,axis=0))
 def test_fsq_quantization(self):
  np.testing.assert_allclose(fit.fsq(np.array([.1,.7,-.7])),[.125,.625,-.625])
 def test_no_aug_selects_center_not_last_jitter_row(self):
  window=np.arange(21*2).reshape(21,2)
  selected=fit.no_aug_state_row(window,temporal_jitter=10)
  np.testing.assert_array_equal(selected,window[10:11])

if __name__=='__main__': unittest.main()
