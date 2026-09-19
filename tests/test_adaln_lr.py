import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
from pathlib import Path
import unittest

import torch
from PySide6.QtWidgets import QApplication

from trainer.gui.fields import TrainComponentEditor
from trainer.gui.bridge import dump_toml
from trainer.training.config import load_config
from trainer.training.params import ComponentLRs, build_param_groups
from trainer.training.optim import ScheduleConfig, build_scheduler


class AdaLNLearningRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_editor_toggle_preserves_override(self):
        editor = TrainComponentEditor("Train AdaLN")
        editor.set(None)
        self.assertTrue(editor.check.isChecked())
        self.assertIsNone(editor.get())
        editor.lr.setText("2e-6")
        editor.check.setChecked(False)
        self.assertEqual(editor.get(), 0)
        self.assertFalse(editor.lr.isEnabled())
        editor.check.setChecked(True)
        self.assertEqual(editor.get(), 2e-6)
        editor.set(0)
        self.assertFalse(editor.check.isChecked())
        editor.set(3e-6)
        self.assertEqual(editor.get(), 3e-6)
        editor.lr.clear()
        self.assertIsNone(editor.get())

    def test_gui_value_roundtrips_to_training_config(self):
        editor = TrainComponentEditor("Train AdaLN")
        editor.set(2e-6)
        flat = {"dataset.path": "/tmp", "adapter.kind": "none",
                "optimizer.lr": 1e-5, "component_lr.adaln": editor.get()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.toml'
            path.write_text(dump_toml(flat))
            cfg = load_config(path)
        self.assertEqual(cfg.component_lr.adaln, 2e-6)
        self.assertEqual(cfg.optimizer.lr, 1e-5)

    def test_invalid_override_is_not_silently_inherited(self):
        editor = TrainComponentEditor("Train AdaLN")
        editor.set(None)
        for text in ('oops', '-1e-6', 'nan', 'inf'):
            editor.lr.setText(text)
            with self.assertRaisesRegex(ValueError, 'component_lr.adaln'):
                ComponentLRs(adaln=editor.get())

    def test_shared_and_block_modulation_use_override_and_schedule(self):
        model = torch.nn.Module()
        model.modulation_down = torch.nn.Linear(2, 2)
        model.img_in = torch.nn.Linear(2, 2)
        block = torch.nn.Module()
        block.img_mod = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(2, 2))
        block.txt_mod = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(2, 2))
        model.transformer_blocks = torch.nn.ModuleList([block])
        report = build_param_groups(model, ComponentLRs(adaln=2e-6), 1e-5)
        adaln = next(g for g in report.groups if g['component'] == 'adaln')
        expected = list(model.modulation_down.parameters()) + list(block.parameters())
        self.assertEqual({id(p) for p in adaln['params']}, {id(p) for p in expected})
        opt = torch.optim.SGD(report.groups)
        scheduler = build_scheduler(opt, ScheduleConfig(kind='constant', warmup_steps=2), 10)
        for _ in range(4):
            opt.step()
            scheduler.step()
            rates = {g['component']: g['lr'] for g in opt.param_groups}
            self.assertAlmostEqual(rates['adaln'] / rates['base'], .2)


if __name__ == '__main__':
    unittest.main()
