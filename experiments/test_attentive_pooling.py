"""Focused contracts for masking and fold isolation in supervised pooling."""
import unittest
import numpy as np
import torch
from evaluate_attentive_pooling import Scorer, fit_run


class AttentivePoolingContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_padding_does_not_change_predictions(self):
        model = Scorer().eval()
        x = torch.randn(2, 7, 1024)
        padded = torch.cat([x,torch.randn(2,5,1024)*100],dim=1)
        with torch.inference_mode():
            expected = model(x,torch.ones(2,7,dtype=torch.bool))
            actual = model(padded,torch.arange(12)[None,:].repeat(2,1)<7)
        torch.testing.assert_close(actual,expected)

    def test_heldout_labels_and_moments_do_not_affect_fit(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=(8,9,1024)).astype('float16')
        lengths = np.full(8,9)
        means = x.astype('float32').mean(1)
        squares = (x.astype('float32')**2).mean(1)
        y = np.linspace(1,4,6)
        tr, ids = np.arange(4),np.arange(8)
        first = fit_run(x,lengths,means,squares,y,tr,ids,42,(1,),'cpu')[1]
        y[4:] += 1000
        means[4:] += 1000
        squares[4:] += 1000000
        second = fit_run(x,lengths,means,squares,y,tr,ids,42,(1,),'cpu')[1]
        np.testing.assert_array_equal(first,second)
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(((first>=0)&(first<=5)).all())


if __name__=='__main__':
    unittest.main()
