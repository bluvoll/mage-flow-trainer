import unittest

import torch

from trainer.modeling.mage_flow import MageFlow, MageFlowParams
from trainer.training.params import AdapterConfig, apply_adapter
from trainer.tools.probe_self_flow_lokr import AdapterEMA, forward_probe, fingerprint_indices


class SelfFlowProbeTests(unittest.TestCase):
    def test_packed_mixed_resolution_matches_independent_forward_and_backward(self):
        torch.set_num_threads(2)
        torch.manual_seed(37)
        model = MageFlow(MageFlowParams(128,128,24,32,4,2,[2,2,4],True,modulation_rank=8))
        model.configure_execution(gradient_checkpointing=True)
        model.eval()
        images = [torch.randn(1,128,1,3,4), torch.randn(1,128,1,2,3)]
        context = (torch.randn(2,5,24), torch.tensor([[1,1,1,0,0],[1,1,1,1,1]]).bool())
        times = torch.tensor([.2,.8,.4,.9])
        masks = [torch.arange(12).reshape(3,4)%2, torch.arange(6).reshape(2,3)%2]
        packed, features = forward_probe(model, images, times, context, token_ids=masks, capture_layer=1)
        independent = []
        for i in range(2):
            out, feat = forward_probe(model, images[i], times[2*i:2*i+2],
                (context[0][i:i+1],context[1][i:i+1]), token_ids=masks[i], capture_layer=1)
            torch.testing.assert_close(packed[i], out, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(features[i], feat, atol=2e-6, rtol=2e-5)
            independent.append(out)
        ema = AdapterEMA(model,2,'cpu',.99,full_finetune=True)
        _, teachers = forward_probe(model, images, times[[0,2]], context, stop_at=2, ema=ema)
        for i in range(2):
            _, ref = forward_probe(model,images[i],times[2*i:2*i+1],
                (context[0][i:i+1],context[1][i:i+1]),stop_at=2,ema=ema)
            torch.testing.assert_close(teachers[i],ref,atol=2e-6,rtol=2e-5)
        # Changing sample 1 must not affect sample 0's attention or modulation.
        changed = (context[0].clone(), context[1])
        changed[0][1].add_(10)
        other,_ = forward_probe(model,[images[0],images[1]*3],times,changed,token_ids=masks)
        torch.testing.assert_close(other[0],packed[0],atol=2e-6,rtol=2e-5)
        torch.stack([x.square().mean() for x in independent]).mean().backward()
        reference_grad = model.img_in.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        model.train()
        outputs,_ = forward_probe(model,images,times,context,token_ids=masks)
        torch.stack([x.square().mean() for x in outputs]).mean().backward()
        torch.testing.assert_close(model.img_in.weight.grad,reference_grad,atol=2e-6,rtol=2e-4)

    def test_training_config_rejects_unsupported_self_flow_combinations(self):
        from trainer.training.config import Config, validate_model_options
        from trainer.data.dataset import DatasetConfig
        cfg = Config(dataset=DatasetConfig(path='/tmp/self-flow-test', source='latents'))
        cfg.train.model_family = 'mage_flow'
        cfg.adapter.kind = 'none'
        cfg.train.batch_size = 1
        cfg.train.pack_resolutions = cfg.train.cache_text_embeddings = True
        cfg.flow.dual_timestep = cfg.self_flow.enabled = True
        validate_model_options(cfg)
        cfg.train.batch_size = 2
        validate_model_options(cfg)
        cfg.train.batch_size = 1
        cfg.rti.enabled = True
        with self.assertRaisesRegex(ValueError, 'RTI'):
            validate_model_options(cfg)

    def test_ema_resume_and_inference_export(self):
        import io
        import tempfile
        from pathlib import Path
        from safetensors import safe_open
        from trainer.training.config import Config
        from trainer.modeling.export import export_checkpoint
        model = MageFlow(MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], True,
                                        modulation_rank=8))
        model.self_flow_projector = torch.nn.Linear(32, 32)
        ema = AdapterEMA(model, 2, 'cpu', .99, dtype=torch.bfloat16,
                         full_finetune=True, adaln_fp32=True)
        buffer = io.BytesIO()
        torch.save(ema.state_dict(), buffer)
        reference = ema.shadow[0]['img_mod.1.weight'].clone()
        ema.shadow[0]['img_mod.1.weight'].zero_()
        buffer.seek(0)
        ema.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(ema.shadow[0]['img_mod.1.weight'], reference, rtol=0, atol=0)
        from trainer.data.dataset import DatasetConfig
        cfg = Config(dataset=DatasetConfig(path='/tmp/self-flow-test'))
        cfg.adapter.kind = 'none'
        cfg.train.save_native = True
        cfg.self_flow.enabled = True
        with tempfile.TemporaryDirectory() as d:
            export_checkpoint(model, d, 'test', cfg, torch.bfloat16, 1)
            with safe_open(str(Path(d)/'test.safetensors'), framework='pt') as f:
                self.assertFalse(any(k.startswith('self_flow_projector.') for k in f.keys()))
                self.assertIn('img_in.weight', f.keys())
        self.assertTrue(any(k.startswith('self_flow_projector.') for k in model.state_dict()))

    def test_fingerprint_indices_stay_in_bounds_for_large_weights(self):
        for size in (0, 1, 7, 2**24, 2**24+3, 100_000_000):
            indices = fingerprint_indices(size, 'cpu')
            self.assertEqual(indices.numel(), min(32, size))
            if size:
                self.assertEqual(indices[0].item(), 0)
                self.assertEqual(indices[-1].item(), size-1)
                self.assertTrue(((indices >= 0) & (indices < size)).all())

    def test_mixed_ema_keeps_all_modulation_in_fp32(self):
        from trainer.training.params import classify
        model = MageFlow(MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], True,
                                        modulation_rank=8))
        ema = AdapterEMA(model, 2, 'cpu', .99, dtype=torch.bfloat16,
                         stochastic_rounding=True, full_finetune=True, adaln_fp32=True)
        groups = [(f'transformer_blocks.{i}', group) for i, group in enumerate(ema.shadow[:2])]
        groups += [(name, ema.shadow[i]) for name, i in ema.input_indices.items()]
        for prefix, group in groups:
            for name, value in group.items():
                expected = torch.float32 if classify(f'{prefix}.{name}') == 'adaln' else torch.bfloat16
                self.assertEqual(value.dtype, expected)
        index = ema.input_indices['modulation_down']
        old = ema.shadow[index]['weight'].clone()
        with torch.no_grad():
            model.modulation_down.weight.add_(.001)
        ema.update()
        torch.testing.assert_close(ema.shadow[index]['weight'],
                                   old.lerp(model.modulation_down.weight.detach(), .01))

    def test_finetune_teacher_includes_input_and_shared_modulation(self):
        torch.manual_seed(21)
        model = MageFlow(MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], True,
                                        modulation_rank=8)).eval()
        image = torch.randn(1, 128, 1, 3, 4)
        context = (torch.randn(1, 5, 24), torch.ones(1, 5, dtype=torch.bool))
        times = torch.tensor([.3])
        ema = AdapterEMA(model, 2, 'cpu', .9, full_finetune=True)
        with torch.no_grad():
            _, reference = forward_probe(model, image, times, context, stop_at=2)
            for name, param in model.named_parameters():
                if name.startswith(('img_in.', 'txt_in.', 'time_text_embed.',
                                    'modulation_down.', 'transformer_blocks.')):
                    param.add_(torch.randn_like(param) * .02)
            _, student = forward_probe(model, image, times, context, stop_at=2)
            _, teacher = forward_probe(model, image, times, context, stop_at=2, ema=ema)
            _, student_after = forward_probe(model, image, times, context, stop_at=2)
        self.assertFalse(torch.allclose(student, reference))
        torch.testing.assert_close(teacher, reference)
        torch.testing.assert_close(student_after, student, rtol=0, atol=0)
        index = ema.input_indices['modulation_down']
        old = ema.shadow[index]['weight'].clone()
        ema.update()
        torch.testing.assert_close(ema.shadow[index]['weight'],
                                   old * .9 + model.modulation_down.weight * .1)

    def test_locon_teacher_uses_ema_without_mutating_student(self):
        torch.manual_seed(19)
        model = MageFlow(MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], True,
                                        modulation_rank=8))
        apply_adapter(model, AdapterConfig(kind='lycoris_lora', lycoris_algo='locon',
                                           rank=2, alpha=2, dtype='float32'))
        model.eval()
        image = torch.randn(1,128,1,3,4)
        context = (torch.randn(1,5,24), torch.ones(1,5,dtype=torch.bool))
        times = torch.tensor([.3])
        ema = AdapterEMA(model, 2, 'cpu', .99)
        with torch.no_grad():
            _, reference = forward_probe(model, image, times, context, stop_at=2)
            for n,p in model.named_parameters():
                if 'lycoris_adapter.lora_up' in n:
                    p.add_(torch.randn_like(p) * .1)
            _, student = forward_probe(model, image, times, context, stop_at=2)
            _, teacher = forward_probe(model, image, times, context, stop_at=2, ema=ema)
            _, student_after = forward_probe(model, image, times, context, stop_at=2)
        self.assertFalse(torch.allclose(student, reference))
        torch.testing.assert_close(teacher, reference)
        torch.testing.assert_close(student_after, student, rtol=0, atol=0)

    def test_bf16_stochastic_ema_preserves_sub_ulp_updates_in_expectation(self):
        torch.manual_seed(7)
        ema = AdapterEMA.__new__(AdapterEMA)
        ema.decay = .99
        ema.stochastic_rounding = True
        current = torch.nn.Parameter(torch.full((65536,), 1.0078125, dtype=torch.bfloat16))
        ema.params = [{'weight': current}]
        ema.shadow = [{'weight': torch.ones_like(current)}]
        ema.update()
        expected = 1.0 + .01 * .0078125
        self.assertLess(abs(ema.shadow[0]['weight'].float().mean().item() - expected), 1e-5)
        self.assertEqual(ema.shadow[0]['weight'].dtype, torch.bfloat16)
        self.assertTrue((current == 1.0078125).all())
        ema.stochastic_rounding = False
        ema.shadow[0]['weight'].fill_(1)
        ema.update()
        self.assertTrue((ema.shadow[0]['weight'] == 1).all())

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
