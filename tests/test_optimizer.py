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


class OptimizerFamilyTests(unittest.TestCase):
    def test_adafactor_first_moment_updates_and_resume(self):
        from trainer.training.optim import estimate_optimizer_bytes
        from trainer.modeling.checkpoint_metadata import optimizer_snapshot
        cfg = OptimizerConfig(kind='adafactor', lr=1e-4, weight_decay=0,
                              norm_mode='rms_clip', use_first_moment=True)
        p = torch.nn.Parameter(torch.zeros(16, 16))
        opt = build_optimizer([{'params': [p]}], cfg)
        opt.param_groups[0]['use_torch_compile'] = False
        p.grad = torch.ones_like(p)
        opt.step()
        # SDNQ smooths the normalized update with beta2; no bias correction.
        torch.testing.assert_close(opt.state[p]['exp_avg'], torch.full_like(p, 0.001))
        torch.testing.assert_close(p, torch.full_like(p, -1e-7), atol=1e-10, rtol=1e-5)
        saved = copy.deepcopy(opt.state_dict())
        q = torch.nn.Parameter(p.detach().clone())
        resumed = build_optimizer([{'params': [q]}], cfg)
        resumed.load_state_dict(saved)
        for param, optimizer in ((p, opt), (q, resumed)):
            param.grad = torch.full_like(param, 0.5)
            optimizer.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        self.assertTrue(optimizer_snapshot(opt)['optimizer_groups'][0]['settings']['use_first_moment'])
        off = OptimizerConfig(kind='adafactor', norm_mode='rms_clip')
        r = torch.nn.Parameter(torch.zeros(16, 16))
        baseline = build_optimizer([{'params': [r]}], off)
        baseline.param_groups[0]['use_torch_compile'] = False
        r.grad = torch.ones_like(r)
        baseline.step()
        self.assertNotIn('exp_avg', baseline.state[r])
        self.assertGreater(estimate_optimizer_bytes(100000, cfg), estimate_optimizer_bytes(100000, off))
        for kind in ('adamw8bit', 'came', 'optimi_adamw'):
            with self.assertRaisesRegex(ValueError, 'use_first_moment'):
                OptimizerConfig(kind=kind, use_first_moment=True)
        with self.assertRaisesRegex(ValueError, 'use_first_moment'):
            OptimizerConfig(kind='adafactor', use_first_moment='true')

    @unittest.skipUnless(os.environ.get('MAGEFLOW_GPU_TESTS') == '1', 'GPU opt-in')
    def test_adafactor_quantized_first_moment(self):
        from sdnq.training import SDNQTensor
        from trainer.training.quant import QuantConfig, quantize_module
        for mode in ('relative', 'rms_clip'):
            with self.subTest(mode=mode):
                model = torch.nn.Sequential(torch.nn.Linear(128, 256, bias=False)).to('cuda', torch.bfloat16)
                model = quantize_module(model, QuantConfig(mode='training', use_quantized_matmul=False),
                                        torch.device('cuda'), torch.bfloat16, False)
                p = model[0].weight
                self.assertIsInstance(p, SDNQTensor)
                cfg = OptimizerConfig(kind='adafactor', lr=1e-4, norm_mode=mode,
                                      use_first_moment=True, quantize_state=True)
                opt = build_optimizer([{'params': list(model.parameters())}], cfg)
                for _ in range(3):
                    model(torch.randn(8, 128, device='cuda', dtype=torch.bfloat16)).float().square().mean().backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                self.assertIsInstance(opt.state[p]['exp_avg'], SDNQTensor)
                self.assertEqual(opt.state[p]['row_var'].dtype, torch.float32)
                self.assertTrue(torch.isfinite(p.dequantize()).all())
                self.assertGreater(opt.state[p]['exp_avg'].dequantize().float().abs().sum().item(), 0)
                saved = copy.deepcopy(opt.state_dict())
                opt.load_state_dict(saved)
                model(torch.randn(8, 128, device='cuda', dtype=torch.bfloat16)).float().square().mean().backward()
                opt.step()
                self.assertEqual(opt.state[p]['step'], 4)

    def test_adafactor_zero_initialized_lora_updates(self):
        # LoRA up projections start at zero. SDNQ's relative normalization
        # limits their update norm to lr * 1e-3, regardless of matrix size.
        updates = {}
        for mode in (None, 'rms_clip'):
            cfg = OptimizerConfig(kind='adafactor', lr=1e-4, weight_decay=0,
                                  norm_mode=mode)
            p = torch.nn.Parameter(torch.zeros(128, 32))
            opt = build_optimizer([{'params': [p]}], cfg)
            opt.param_groups[0]['use_torch_compile'] = False
            p.grad = torch.ones_like(p)
            opt.step()
            updates[mode] = p.detach().clone()
        torch.testing.assert_close(updates['rms_clip'], torch.full((128, 32), -1e-4))
        self.assertGreater(updates['rms_clip'].norm().item(),
                           1000 * updates[None].norm().item())
        for kind, mode in [('adamw8bit', 'relative'), ('optimi_adamw', 'rms_clip'),
                           ('adafactor', 'invalid')]:
            with self.assertRaisesRegex(ValueError, 'norm_mode'):
                OptimizerConfig(kind=kind, norm_mode=mode)

    def test_upstream_defaults_and_steps(self):
        import optimi
        from trainer.training.optimizer_specs import OPTIMIZERS
        for kind, spec in OPTIMIZERS.items():
            with self.subTest(kind=kind):
                cfg = OptimizerConfig(kind=kind)
                p = torch.nn.Parameter(torch.ones(4, 4))
                opt = build_optimizer([{'params': [p]}], cfg)
                if spec.family == 'Optimi':
                    q = torch.nn.Parameter(p.detach().clone())
                    reference = getattr(optimi, spec.name)([q], lr=cfg.lr)
                    for key in ('beta1', 'beta2', 'beta3', 'eps', 'weight_decay', 'kahan_sum', 'momentum'):
                        if key in reference.defaults:
                            self.assertEqual(opt.defaults[key], reference.defaults[key])
                    for group in opt.param_groups + reference.param_groups:
                        group["triton"] = False
                    p.grad = torch.full_like(p, 0.5)
                    opt.step()
                    q.grad = torch.full_like(q, 0.5)
                    reference.step()
                    torch.testing.assert_close(p, q, rtol=0, atol=0)
                    saved = copy.deepcopy(opt.state_dict())
                    restored = build_optimizer([{'params': [torch.nn.Parameter(p.detach().clone())]}], cfg)
                    restored.load_state_dict(saved)
                    r = restored.param_groups[0]['params'][0]
                    p.grad = torch.full_like(p, 0.2)
                    r.grad = p.grad.clone()
                    opt.step()
                    restored.step()
                    torch.testing.assert_close(p, r, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(p).all())
                    self.assertFalse(torch.equal(p, torch.ones_like(p)))
                elif spec.family == 'SDNQ':
                    import sdnq.optim
                    reference = getattr(sdnq.optim, spec.name)([torch.nn.Parameter(torch.ones(4, 4))])
                    self.assertEqual(opt.param_groups[0]['betas'], reference.param_groups[0]['betas'])

    def test_custom_settings_and_invalid_combinations(self):
        cfg = OptimizerConfig(kind='optimi_adamw', betas=(0.8, 0.98), eps=1e-7, kahan_sum=False)
        p = torch.nn.Parameter(torch.ones(2))
        opt = build_optimizer([{'params': [p]}], cfg)
        self.assertEqual((opt.defaults['beta1'], opt.defaults['beta2']), (0.8, 0.98))
        self.assertEqual(opt.defaults['eps'], 1e-7)
        for kwargs in (dict(kind='adafactor', betas=(0.9, 0.99)),
                       dict(kind='optimi_adamw', quantize_state=True),
                       dict(kind='optimi_adamw', use_kahan=True),
                       dict(kind='adamw8bit', kahan_sum=True)):
            with self.assertRaises(ValueError):
                OptimizerConfig(**kwargs)

    def test_gui_switches_defaults_and_preserves_loaded_overrides(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        from trainer.gui.app import TrainingGUI
        from trainer.gui import bridge
        app = QApplication.instance() or QApplication([])
        gui = TrainingGUI()
        gui._apply(bridge.defaults() | {'dataset.path': '/images'})
        def choose(kind):
            editor = gui.editors['optimizer.kind']
            editor.widget.setCurrentIndex(editor.values.index(kind))
        choose('adafactor')
        self.assertEqual(gui.editors['optimizer.betas'].get(), [-0.8, 0.999])
        self.assertFalse(gui.editors['optimizer.eps'].widget.isEnabled())
        self.assertTrue(gui.editors['optimizer.norm_mode'].widget.isEnabled())
        self.assertTrue(gui.editors['optimizer.use_first_moment'].widget.isEnabled())
        gui.editors['optimizer.use_first_moment'].set(True)
        gui.editors['optimizer.norm_mode'].set('rms_clip')
        import toml
        self.assertEqual(toml.loads(bridge.dump_toml(gui.collect()))['optimizer']['norm_mode'], 'rms_clip')
        self.assertTrue(toml.loads(bridge.dump_toml(gui.collect()))['optimizer']['use_first_moment'])
        choose('came')
        self.assertFalse(gui.editors['optimizer.use_first_moment'].widget.isEnabled())
        self.assertFalse(gui.editors['optimizer.use_first_moment'].get())
        self.assertIsNone(gui.editors['optimizer.norm_mode'].get())
        self.assertEqual(len(gui.editors['optimizer.betas'].get()), 3)
        gui.editors['optimizer.quantize_state'].set(True)
        choose('optimi_adamw')
        self.assertFalse(gui.editors['optimizer.norm_mode'].widget.isEnabled())
        self.assertEqual(gui.editors['optimizer.eps'].get(), 1e-6)
        self.assertFalse(gui.editors['optimizer.quantize_state'].get())
        self.assertFalse(gui.editors['optimizer.use_kahan'].widget.isEnabled())
        self.assertTrue(gui.editors['optimizer.kahan_sum'].widget.isEnabled())
        choose('optimi_sgd')
        self.assertTrue(gui.editors['optimizer.momentum'].widget.isEnabled())
        self.assertFalse(gui.editors['optimizer.betas'].widget.isEnabled())
        custom = bridge.defaults('optimi_adamw') | {'optimizer.kind': 'optimi_adamw',
                    'optimizer.betas': [0.8, 0.95], 'optimizer.eps': 1e-9}
        gui._apply(custom)
        self.assertEqual(gui.editors['optimizer.betas'].get(), [0.8, 0.95])
        import toml
        self.assertEqual(toml.loads(bridge.dump_toml(gui.collect()))['optimizer']['eps'], 1e-9)
        gui.close()
        app.processEvents()

    def test_family_defaults_survive_toml(self):
        import tempfile
        from pathlib import Path
        from trainer.gui import bridge
        from trainer.training.config import load_config
        from trainer.training.optimizer_specs import OPTIMIZERS
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            for kind in OPTIMIZERS:
                flat = bridge.defaults(kind) | {"optimizer.kind": kind, "dataset.path": "/images"}
                bridge.write_toml(path, flat)
                self.assertEqual(load_config(path).optimizer, OptimizerConfig(kind=kind))
            flat = bridge.defaults("optimi_adan") | {"optimizer.kind": "optimi_adan",
                    "optimizer.kahan_sum": "auto", "dataset.path": "/images"}
            bridge.write_toml(path, flat)
            self.assertEqual(load_config(path).optimizer.kahan_sum, "auto")
            flat.update({"adapter.kind": "none", "quant.mode": "training"})
            bridge.write_toml(path, flat)
            with self.assertRaisesRegex(ValueError, "Optimi does not yet support"):
                load_config(path)
            flat = bridge.defaults('adafactor') | {'optimizer.kind': 'adafactor',
                    'optimizer.norm_mode': 'rms_clip', 'dataset.path': '/images'}
            bridge.write_toml(path, flat)
            self.assertEqual(load_config(path).optimizer.norm_mode, 'rms_clip')
            flat['optimizer.use_first_moment'] = True
            bridge.write_toml(path, flat)
            self.assertTrue(load_config(path).optimizer.use_first_moment)
            flat['adapter.kind'] = 'lycoris_lora'
            self.assertFalse(any('nearly stall' in msg for _, msg in bridge.advisories(flat)))
            flat['optimizer.norm_mode'] = None
            self.assertTrue(any('nearly stall' in msg for _, msg in bridge.advisories(flat)))
