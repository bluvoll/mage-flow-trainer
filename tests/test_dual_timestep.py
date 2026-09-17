import copy
import os
import unittest

import torch

from trainer.modeling.mage_flow import MageFlow, MageFlowParams
from trainer.training.flow import (FlowConfig, prepare_flow_batch, prepare_training_flow_batch,
                                   effective_timesteps, hf_loss)


def model(rank=8):
    return MageFlow(MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], True,
                                  modulation_rank=rank))


class DualTimestepTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(8)

    def test_disabled_preserves_rng_and_outputs(self):
        latent = torch.randn(2, 128, 1, 3, 4)
        state = torch.get_rng_state()
        expected = prepare_flow_batch(latent, FlowConfig())
        after = torch.get_rng_state()
        torch.set_rng_state(state)
        actual = prepare_training_flow_batch(latent, FlowConfig())
        for a, b in zip(actual[:3], expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(actual[3:], (None, None))
        self.assertTrue(torch.equal(after, torch.get_rng_state()))

    def test_noising_hf_reconstruction_and_curriculum(self):
        latent = torch.randn(2, 128, 1, 8, 9)
        cfg = FlowConfig(dual_timestep=True)
        noisy, t, target, second, mask = prepare_training_flow_batch(latent, cfg, t_range=(.2, .7))
        self.assertTrue(((t >= .2) & (t <= .7)).all())
        self.assertTrue(((second >= .2) & (second <= .7)).all())
        self.assertTrue(mask.any() and (~mask).any())
        times = effective_timesteps(t, second, mask)
        torch.testing.assert_close(noisy - times*target, latent, atol=1e-6, rtol=1e-5)
        self.assertLess(hf_loss(target, noisy, latent, times, 1, 1.).item(), 1e-10)
        for bad in (0, -.1, .51, float('nan')):
            with self.assertRaises(ValueError):
                FlowConfig(dual_timestep_mask_ratio=bad)

    def test_uniform_parity_and_checkpoint_gradients(self):
        for rank in (0, 8):
            a = model(rank)
            image = torch.randn(2, 128, 1, 3, 4)
            t = torch.tensor([.2, .7])
            ctx = (torch.randn(2, 5, 24), torch.ones(2, 5, dtype=torch.bool))
            mask = torch.rand(2, 3, 4) < .25
            expected = a(image, t, ctx)[0]
            got = a(image, t, ctx, second_timestep=t, timestep_mask=mask)[0]
            torch.testing.assert_close(expected, got)
            b = copy.deepcopy(a)
            b.configure_execution(False)
            s = 1-t
            y = a(image, t, ctx, second_timestep=s, timestep_mask=mask)[0]
            z = b(image, t, ctx, second_timestep=s, timestep_mask=mask)[0]
            torch.testing.assert_close(y, z)
            y.square().mean().backward()
            z.square().mean().backward()
            for (name,p), (_,q) in zip(a.named_parameters(), b.named_parameters()):
                torch.testing.assert_close(p.grad, q.grad, msg=name)

    @unittest.skipUnless(os.environ.get('MAGE_GPU_TEST') == '1', 'GPU opt-in')
    def test_packed_varlen_and_compiled_quantized_lokr(self):
        from trainer.training.params import AdapterConfig, apply_adapter
        from trainer.training.quant import QuantConfig, quantize_module
        a = model().to('cuda', torch.bfloat16)
        a = quantize_module(a, QuantConfig(mode='frozen', weights_dtype='uint8'),
                            torch.device('cuda'), torch.bfloat16, False)
        apply_adapter(a, AdapterConfig(kind='lycoris_lora', lycoris_algo='lokr', rank=4, alpha=4))
        b = copy.deepcopy(a)
        a.configure_execution(False, attention_backend='sdpa')
        b.configure_execution(True, compile_mode='default', attention_backend='torch_varlen')
        images = [torch.randn(1,128,1,h,w,device='cuda',dtype=torch.bfloat16) for h,w in ((3,4),(2,3))]
        masks = [torch.rand(1,*im.shape[-2:],device='cuda') < .25 for im in images]
        t = torch.tensor([.2,.7],device='cuda',dtype=torch.bfloat16)
        s = 1-t
        ctx = (torch.randn(2,5,24,device='cuda',dtype=torch.bfloat16),
               torch.tensor([[1,1,0,0,0],[1,1,1,1,1]],device='cuda').bool())
        ys = [a(im,t[i:i+1],(ctx[0][i:i+1],ctx[1][i:i+1]),
                second_timestep=s[i:i+1],timestep_mask=masks[i])[0] for i,im in enumerate(images)]
        zs = b(images,t,ctx,second_timestep=s,timestep_mask=masks)[0]
        for y,z in zip(ys,zs):
            torch.testing.assert_close(y,z,atol=.02,rtol=.03)
        sum(y.float().square().mean() for y in ys).backward()
        sum(z.float().square().mean() for z in zs).backward()
        for (n,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=5e-4,rtol=.1,msg=n)


if __name__ == '__main__':
    unittest.main()
