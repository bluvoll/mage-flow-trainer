"""Regression for SDNQ quantized Kahan state initialization and resume."""
import copy
import os
import unittest

import torch

from trainer.training.config import OptimizerConfig
from trainer.training.optim import build_optimizer


@unittest.skipUnless(os.environ.get('MAGEFLOW_GPU_TESTS') == '1', 'GPU opt-in')
class QuantizedKahanTests(unittest.TestCase):
    def test_quantized_lora_state_and_resume(self):
        from sdnq.training import SDNQTensor
        for offload in (False, True):
            with self.subTest(offload=offload):
                p = torch.nn.Parameter(torch.randn(32, 1024, device='cuda', dtype=torch.bfloat16))
                cfg = OptimizerConfig(kind='adamw8bit', quantize_state=True,
                                      offload_state=offload, use_kahan=True)
                def make(param):
                    return build_optimizer([{'params': [param], 'lr': 1e-4}], cfg)
                opt = make(p)
                before = p.detach().clone()
                for _ in range(2):
                    p.grad = torch.randn_like(p)
                    opt.step()
                    opt.zero_grad()
                torch.cuda.synchronize()
                self.assertTrue(torch.isfinite(p).all())
                self.assertFalse(torch.equal(before, p))
                self.assertIsInstance(opt.state[p]['kahan_buffer'], SDNQTensor)
                self.assertNotIn('use_svd_quantization', opt.param_groups[0])
                saved = copy.deepcopy(opt.state_dict())
                q = torch.nn.Parameter(p.detach().clone())
                resumed = make(q)
                resumed.load_state_dict(saved)
                q.grad = torch.randn_like(q)
                resumed.step()
                torch.cuda.synchronize()
                self.assertEqual(resumed.state[q]['step'], 3)
                self.assertTrue(torch.isfinite(q).all())
