import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from safetensors.torch import save_file

from test_mageflow import tiny
from trainer.data.dataset import DatasetConfig
from trainer.modeling.checkpoint_metadata import optimizer_snapshot, checkpoint_identity
from trainer.modeling.export import export_checkpoint
from trainer.tools.inspect_checkpoint import read_metadata
from trainer.training.config import Config, OptimizerConfig
from trainer.training.optim import build_optimizer
from trainer.training.params import AdapterConfig, apply_adapter


class CheckpointMetadataTests(unittest.TestCase):
    def test_settings_survive_all_export_formats_and_renaming(self):
        for kind, native in [('none', True), ('none', False), ('lora', True), ('lycoris_lora', True)]:
            with self.subTest(kind=kind, native=native), tempfile.TemporaryDirectory() as folder:
                cfg = Config(dataset=DatasetConfig(path='/example/dataset'))
                cfg.adapter = AdapterConfig(kind=kind, rank=2, alpha=2)
                cfg.train.save_native = native
                cfg.train.run_name = 'name-does-not-identify-optimizer'
                cfg.train.batch_size = 16
                cfg.dataset.path = '/example/dataset'
                cfg.optimizer = OptimizerConfig(kind='adafactor', norm_mode='rms_clip', lr=1e-4,
                                                use_first_moment=True)
                cfg.train.cache_text_embeddings = True
                model = tiny()
                if cfg.is_lora:
                    apply_adapter(model, cfg.adapter)
                param = torch.nn.Parameter(torch.ones(2, 2))
                optimizer = build_optimizer([{'params': [param]}], cfg.optimizer)
                # Resuming can restore a different LR/normalization than the input config.
                optimizer.param_groups[0]['lr'] = 7e-5
                optimizer.param_groups[0]['norm_mode'] = 'relative'
                runtime = optimizer_snapshot(SimpleNamespace(optimizer=optimizer))
                runtime.update(world_size=2, effective_batch_size=32, save_tag='epoch002')
                original_source = Path(folder) / 'original.safetensors'
                save_file({'modulation_down.weight': torch.ones(2, 2)}, str(original_source))
                identity = checkpoint_identity(original_source)
                runtime['source_checkpoint'] = identity
                export_checkpoint(model, folder, 'test', cfg, torch.float32, 96, runtime=runtime)
                source = Path(folder) / ('test.safetensors' if native else 'transformer/diffusion_pytorch_model.safetensors')
                renamed = source.with_name('arbitrary-name.safetensors')
                source.rename(renamed)
                metadata = read_metadata(renamed)
                self.assertEqual(metadata['step'], '96')
                self.assertEqual(metadata['training_metadata_version'], '2')
                self.assertEqual(metadata['source_checkpoint'], identity)
                self.assertEqual(identity['name'], 'original.safetensors')
                self.assertEqual(identity['model_type'], 'compressed')
                self.assertEqual(identity['md5'], hashlib.md5(original_source.read_bytes()).hexdigest())
                config = metadata['training_config']
                self.assertEqual(config['optimizer']['kind'], 'adafactor')
                self.assertEqual(config['optimizer']['norm_mode'], 'rms_clip')
                self.assertTrue(config['optimizer']['use_first_moment'])
                self.assertEqual(config['optimizer']['lr'], 1e-4)
                self.assertEqual(config['optimizer']['betas'], [-0.8, 0.999])
                self.assertEqual(config['adapter']['kind'], kind)
                self.assertEqual(config['dataset']['path'], '/example/dataset')
                self.assertTrue(config['train']['cache_text_embeddings'])
                state = metadata['training_state']
                self.assertEqual(state['global_step'], 96)
                self.assertEqual(state['effective_batch_size'], 32)
                self.assertIn('Adafactor', state['optimizer_class'])
                group = state['optimizer_groups'][0]
                self.assertEqual(group['settings']['lr'], 7e-5)
                self.assertEqual(group['settings']['norm_mode'], 'relative')
                self.assertTrue(group['settings']['use_first_moment'])
                self.assertEqual(group['parameter_count'], 4)
                self.assertNotIn('params', group['settings'])
                self.assertIn('torch', metadata['training_versions'])
                self.assertIn('adapter_config' if cfg.is_lora else 'model_config', metadata)

    def test_source_identity_types_and_renaming(self):
        for compressed, rti, kind in [(False, False, 'full_mage_flow'),
                                      (True, False, 'compressed'),
                                      (True, True, 'compressed_rti'),
                                      (False, True, 'full_mage_flow_rti')]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                weights = {'img_in.weight': torch.ones(2, 2)}
                if compressed:
                    weights['modulation_down.weight'] = torch.ones(2, 2)
                if rti:
                    weights['region_interface.proj.weight'] = torch.ones(2, 2)
                weights = {'diffusion_model.' + k: v for k, v in weights.items()}
                path = Path(folder) / 'diffusion_pytorch_model.safetensors'
                save_file(weights, str(path))
                identity = checkpoint_identity(folder)
                self.assertEqual(identity['model_type'], kind)
                renamed = path.with_name('base.safetensors')
                path.rename(renamed)
                renamed_identity = checkpoint_identity(renamed)
                self.assertEqual(renamed_identity['md5'], identity['md5'])
                self.assertEqual(renamed_identity['name'], 'base.safetensors')

    def test_legacy_metadata_is_readable_without_inventing_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.safetensors'
            save_file({'weight': torch.ones(1)}, str(path),
                      metadata={'run': 'old', 'step': '12', 'adapter_config': 'legacy text'})
            data = read_metadata(path)
            self.assertEqual(data, {'run': 'old', 'step': '12', 'adapter_config': 'legacy text'})

    def test_resume_retains_original_source_or_marks_legacy_uncertainty(self):
        from trainer.training.train import Trainer
        for original in (None, {'name': 'original.safetensors', 'md5': 'original-hash'}):
            with self.subTest(original=original), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / 'accelerator').mkdir()
                (root / 'state.json').write_text(json.dumps(
                    {'global_step': 20, 'source_checkpoint': original}))
                trainer = Trainer.__new__(Trainer)
                trainer.source_checkpoint = {'name': 'current-init.safetensors', 'md5': 'init-hash'}
                trainer.accelerator = SimpleNamespace(load_state=lambda path: None,
                                                      print=lambda *args: None)
                trainer.cfg = SimpleNamespace(train=SimpleNamespace(gradient_accumulation_steps=1))
                trainer.steps_per_epoch = 10
                trainer._resume(folder)
                if original is not None:
                    self.assertEqual(trainer.source_checkpoint, original)
                else:
                    self.assertFalse(trainer.source_checkpoint['original_training_source_verified'])
                    self.assertEqual(trainer.source_checkpoint['role'], 'initialization_before_legacy_resume')

    def test_optimizer_snapshot_excludes_tensor_state(self):
        p = torch.nn.Parameter(torch.ones(4))
        opt = torch.optim.AdamW([p], lr=2e-5)
        p.grad = torch.ones_like(p)
        opt.step()
        snapshot = optimizer_snapshot(opt)
        self.assertNotIn('exp_avg', json.dumps(snapshot))
        self.assertNotIn('params', snapshot['optimizer_groups'][0]['settings'])


if __name__ == '__main__':
    unittest.main()
