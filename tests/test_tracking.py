"""Experiment tracking: backends start, log and fail independently; runs resume."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from trainer.training.config import TrackingConfig
from trainer.training.tracking import Backend, Tracker, build_tracker


class Recording(Backend):
    name = "recording"

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def log(self, metrics, step):
        if self.fail_on == "log":
            raise RuntimeError("403 Forbidden")
        self.calls.append(("log", metrics, step))

    def log_images(self, key, entries, step):
        self.calls.append(("images", key, [n for n, _, _ in entries], step))


def config(tmp, **tracking):
    return SimpleNamespace(tracking=TrackingConfig(**tracking),
                           train=SimpleNamespace(run_name="unit"), to_json=None)


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_failing_backend_is_dropped_and_the_rest_keep_logging(self):
        good, bad = Recording(), Recording(fail_on="log")
        tracker = Tracker([bad, good])
        tracker.log({"train/loss": 0.5, "skip": None}, 3)
        tracker.log({"train/loss": 0.4}, 4)
        self.assertEqual(tracker.backends, [good])
        self.assertEqual(good.calls, [("log", {"train/loss": 0.5}, 3), ("log", {"train/loss": 0.4}, 4)])
        tracker.log_images("samples", [("00_a", Image.new("RGB", (8, 8)), "a")], 4)
        self.assertEqual(good.calls[-1], ("images", "samples", ["00_a"], 4))

    def test_inert_without_backends_or_off_main(self):
        cfg = config(self.out, backends=["tensorboard"])
        self.assertFalse(build_tracker(cfg, self.out, is_main=False).enabled)
        self.assertFalse(build_tracker(config(self.out), self.out, is_main=True).enabled)

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, "unknown backend"):
            TrackingConfig(backends=["mlflow"])
        with self.assertRaisesRegex(ValueError, "twice"):
            TrackingConfig(backends=["wandb", "wandb"])
        self.assertEqual(TrackingConfig(backends="TensorBoard").backends, ["tensorboard"])

    def _fake_cfg(self, **tracking):
        from trainer.training.config import Config, TrainConfig
        from trainer.data.dataset import DatasetConfig
        return Config(train=TrainConfig(run_name="unit"),
                      dataset=DatasetConfig(path=str(self.out)),
                      tracking=TrackingConfig(**tracking))

    def test_tensorboard_writes_scalars_and_images(self):
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            self.skipTest("tensorboard not installed")
        tracker = build_tracker(self._fake_cfg(backends=["tensorboard"]), self.out, True)
        self.assertTrue(tracker.enabled)
        tracker.log({"train/loss": 0.25}, 7)
        tracker.log_images("samples", [("00_a", Image.new("RGB", (16, 16), "red"), "a girl")], 7)
        tracker.finish()
        acc = EventAccumulator(str(self.out / "tracking" / "tensorboard"),
                               size_guidance={"images": 0, "scalars": 0})
        acc.Reload()
        self.assertEqual([(e.step, e.value) for e in acc.Scalars("train/loss")], [(7, 0.25)])
        self.assertIn("samples/00_a", acc.Tags()["images"])

    def test_wandb_offline_resumes_the_same_run_id(self):
        try:
            import wandb  # noqa: F401
        except ImportError:
            self.skipTest("wandb not installed")
        env = {"WANDB_SILENT": "true", "WANDB_DISABLE_GIT": "true"}
        with patch.dict(os.environ, env):
            cfg = self._fake_cfg(backends=["wandb"], wandb_mode="offline")
            first = build_tracker(cfg, self.out, True)
            self.assertTrue(first.enabled)
            first.log({"train/loss": 1.0}, 1)
            first.finish()
            run_id = json.loads((self.out / "tracking_run.json").read_text())["wandb_id"]
            second = build_tracker(cfg, self.out, True, resumed=True)
            self.assertEqual(second.backends[0].run.id, run_id)
            second.log({"train/loss": 0.9}, 2)
            second.finish()
            fresh = build_tracker(cfg, self.out, True, resumed=False)
            self.assertNotEqual(fresh.backends[0].run.id, run_id)
            fresh.finish()

    def test_trackio_local(self):
        try:
            import trackio  # noqa: F401
        except ImportError:
            self.skipTest("trackio not installed")
        with patch.dict(os.environ, {"TRACKIO_DIR": str(self.out / "trackio")}):
            tracker = build_tracker(self._fake_cfg(backends=["trackio"], project="unit-test"),
                                    self.out, True)
            if not tracker.enabled:
                self.skipTest("trackio could not start in this environment")
            tracker.log({"train/loss": 0.5}, 1)
            tracker.log_images("samples", [("00_a", Image.new("RGB", (16, 16)), "x")], 1)
            self.assertEqual(len(tracker.backends), 1)
            tracker.finish()

    def test_missing_package_is_a_warning_not_a_crash(self):
        import builtins
        real = builtins.__import__

        def fake(name, *a, **k):
            if name == "wandb":
                raise ImportError("No module named 'wandb'")
            return real(name, *a, **k)

        with patch("builtins.__import__", side_effect=fake):
            tracker = build_tracker(self._fake_cfg(backends=["wandb"]), self.out, True)
        self.assertFalse(tracker.enabled)


if __name__ == "__main__":
    unittest.main()
