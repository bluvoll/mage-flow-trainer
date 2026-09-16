import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from safetensors.torch import save_file

from test_mageflow import tiny
from trainer.data.dataset import DatasetConfig
from trainer.modeling.checkpoint_metadata import optimizer_snapshot
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
                cfg.optimizer = OptimizerConfig(kind='adafactor', norm_mode='rms_clip', lr=1e-4)
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
                export_checkpoint(model, folder, 'test', cfg, torch.float32, 96, runtime=runtime)
                source = Path(folder) / ('test.safetensors' if native else 'transformer/diffusion_pytorch_model.safetensors')
                renamed = source.with_name('arbitrary-name.safetensors')
                source.rename(renamed)
                metadata = read_metadata(renamed)
                self.assertEqual(metadata['step'], '96')
                self.assertEqual(metadata['training_metadata_version'], '1')
                config = metadata['training_config']
                self.assertEqual(config['optimizer']['kind'], 'adafactor')
                self.assertEqual(config['optimizer']['norm_mode'], 'rms_clip')
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
                self.assertEqual(group['parameter_count'], 4)
                self.assertNotIn('params', group['settings'])
                self.assertIn('torch', metadata['training_versions'])
                self.assertIn('adapter_config' if cfg.is_lora else 'model_config', metadata)

    def test_legacy_metadata_is_readable_without_inventing_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.safetensors'
            save_file({'weight': torch.ones(1)}, str(path),
                      metadata={'run': 'old', 'step': '12', 'adapter_config': 'legacy text'})
            data = read_metadata(path)
            self.assertEqual(data, {'run': 'old', 'step': '12', 'adapter_config': 'legacy text'})

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
