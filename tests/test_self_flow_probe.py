import unittest

import torch

from trainer.modeling.mage_flow import MageFlow, MageFlowParams
from trainer.training.params import AdapterConfig, apply_adapter
from trainer.tools.probe_self_flow_lokr import AdapterEMA, forward_probe


class SelfFlowProbeTests(unittest.TestCase):
    def test_uniform_parity_teacher_isolation_and_dual_time_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        model = MageFlow(MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], True,
                                        modulation_rank=8))
        apply_adapter(model, AdapterConfig(kind='lycoris_lora', lycoris_algo='lokr',
                                           rank=10000, alpha=10000, lokr_factor=1,
                                           dtype='float32'))
        image = torch.randn(1, 128, 1, 3, 4)
        context = (torch.randn(1, 5, 24), torch.tensor([[1, 1, 1, 0, 0]]).bool())
        times = torch.tensor([.2, .8])
        model.eval()
        expected = model(image, times[:1], context)[0]
        actual, _ = forward_probe(model, image, times[:1], context, capture_layer=1)
        torch.testing.assert_close(actual, expected)

        ema = AdapterEMA(model, 2, 'cpu', .9)
        with torch.no_grad():
            _, reference = forward_probe(model, image, times[:1], context, stop_at=2)
            for name, p in model.named_parameters():
                if 'lokr_w2' in name:
                    p.add_(.1*torch.randn_like(p))
            snapshots = {n:p.detach().clone() for n,p in model.named_parameters()}
            _, teacher = forward_probe(model, image, times[:1], context, stop_at=2, ema=ema)
        torch.testing.assert_close(teacher, reference)
        self.assertFalse(teacher.requires_grad)
        for n,p in model.named_parameters():
            torch.testing.assert_close(p, snapshots[n], rtol=0, atol=0)
        name = next(iter(ema.params[0]))
        old = ema.shadow[0][name].clone()
        ema.update()
        torch.testing.assert_close(ema.shadow[0][name], old*.9 + ema.params[0][name]*.1)

        model.train()
        ids = torch.arange(12).reshape(3,4) % 2
        output, features = forward_probe(model, image, times, context,
                                         token_ids=ids, capture_layer=1)
        loss = output.square().mean() + (1-torch.nn.functional.cosine_similarity(
            features, teacher, dim=-1)).mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.parameters() if p.requires_grad))
        for p in model.parameters():
            if p.requires_grad:
                self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all())
            else:
                self.assertIsNone(p.grad)


if __name__ == '__main__':
    unittest.main()
