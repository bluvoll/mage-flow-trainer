"""GUI coverage for the MageTrail features: editors, TOML round trip, rules, launches, signals.

Headless (Qt offscreen); constructs the real window like trainer/parity/test_gui.py does.
"""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtWidgets
except ImportError:  # pragma: no cover - GUI extra not installed
    QtWidgets = None

STAGES = [{"type": "linear", "end_lr": 7e-6, "percent": 0.1},
          {"type": "constant", "lr": 7e-6, "percent": 0.5},
          {"type": "rex", "max_val": 7e-6, "min_val": 0.0, "percent": 0.4}]


@unittest.skipIf(QtWidgets is None, "PySide6 not installed")
class MageTrailGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_stage_editor_round_trip_and_inactive_state(self):
        from trainer.gui import fields as F
        e = F.StageListEditor()
        e.set(STAGES)
        self.assertEqual(e.get(), STAGES)
        self.assertIn("100.000%", e.total.text())
        e.set_enabled(False)
        self.assertEqual(e.get(), [])
        self.assertEqual(e.table.rowCount(), 3, "rows survive another schedule kind")
        e.set_enabled(True)
        self.assertEqual(e.get(), STAGES)
        e.set([])
        e._add_row()
        self.assertEqual(e.get(), [{"type": "linear", "percent": 0.0}])
        self.assertIn("must be 100%", e.total.text())
        e._load_preset()
        self.assertEqual(e.get(), STAGES)

    def test_prompt_and_line_editors(self):
        from trainer.gui import fields as F
        p = F.PromptListEditor()
        p.set(["Drawn by emily, 1girl", "a castle"])
        self.assertEqual(p.get(), ["Drawn by emily, 1girl", "a castle"])
        p.widget.setPlainText('a girl\n{ prompt = "a castle, dusk", width = 1216, seed = 7 }')
        self.assertEqual(p.get(), [{"prompt": "a girl"},
                                   {"prompt": "a castle, dusk", "width": 1216, "seed": 7}])
        shown = p.widget.toPlainText()
        p.set(p.get())
        self.assertEqual(p.widget.toPlainText(), shown)
        p.widget.setPlainText("{ prompt = broken")
        self.assertIn("INVALID_TOML", p.get()[0])
        line = F.LineListEditor()
        line.set(["^drawn by\\s", "^art by\\s{1,3}"])
        self.assertEqual(line.get(), ["^drawn by\\s", "^art by\\s{1,3}"])

    def test_bridge_round_trip_validates(self):
        from trainer.gui import bridge
        from trainer.training.config import load_config
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            csv = root / "folders.csv"
            csv.write_text(f"path,repeats\n{root / 'a'},3\n", encoding="utf-8")
            flat = bridge.defaults() | {
                "dataset.path": str(root),       # left in the box; the subsets file wins
                "dataset.subsets_file": str(csv),
                "dataset.latent_dtype": "float16",
                "dataset.caption.attribution_patterns": ["^drawn by\\s"],
                "schedule.kind": "stage", "schedule.stages": STAGES,
                "tracking.backends": ["tensorboard", "wandb"], "tracking.wandb_mode": "offline",
                "sampling.prompts": [{"prompt": "a"}, {"prompt": "b", "seed": 3}],
                "sampling.every_n_steps": 500,
                "eval.path": str(root / "a"), "eval.every_n_steps": 250,
                "eval.quantiles": [0.25, 0.75],
            }
            ok, err = bridge.validate(flat)
            self.assertTrue(ok, err)
            path = root / "run.toml"
            bridge.write_toml(path, flat)
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(f'path = "{root.as_posix()}"', text.replace("\\\\", "/"))
            cfg = load_config(path)
            self.assertEqual(cfg.schedule.stages, STAGES)
            self.assertEqual(cfg.sampling.prompts[1], {"prompt": "b", "seed": 3})
            self.assertEqual(cfg.tracking.backends, ["tensorboard", "wandb"])
            self.assertEqual(len(cfg.dataset.effective_subsets()), 1)
            back = bridge.flatten(bridge.read_toml(path))
            for key in ("schedule.stages", "sampling.prompts", "tracking.backends",
                        "eval.quantiles", "dataset.caption.attribution_patterns"):
                self.assertEqual(back[key], flat[key], key)
            self.assertNotIn("dataset.path", back)

    def test_window_rules_launches_and_signals(self):
        from trainer.gui import bridge
        from trainer.gui.app import TrainingGUI
        from trainer.gui.process import cache_config_launch
        gui = TrainingGUI()
        try:
            flat = bridge.defaults() | {"dataset.path": "/data", "schedule.kind": "stage",
                                        "schedule.stages": STAGES}
            gui._apply(flat)
            self.assertEqual(gui.collect()["schedule.stages"], STAGES)
            gui.editors["schedule.kind"].set("cosine")
            gui._refresh()
            self.assertEqual(gui.collect()["schedule.stages"], [])
            self.assertTrue(bridge.validate(gui.collect())[0])
            gui.editors["schedule.kind"].set("stage")
            gui._refresh()
            self.assertEqual(gui.collect()["schedule.stages"], STAGES)
            self.assertFalse(gui.editors["sampling.steps"].widget.isEnabled())
            self.assertTrue(gui.save_now_btn.isHidden() and gui.save_quit_btn.isHidden())
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                gui._run_dir = Path(tmp) / "run"
                gui._signal("save_quit")
                self.assertTrue((Path(tmp) / "run" / "save_quit").exists())
        finally:
            gui.close()
        launch = cache_config_launch("c.toml", gpus="0,1", dry_run=True)
        self.assertEqual(launch.argv[3:], ["trainer.tools.cache_latents", "cache-config", "c.toml",
                                           "--devices", "0,1", "--dry-run"])
        self.assertFalse(launch.is_training)


if __name__ == "__main__":
    unittest.main()
