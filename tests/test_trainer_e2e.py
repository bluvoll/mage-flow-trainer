"""End to end on CPU: a real Trainer, every MageTrail feature on, stopped by signal and resumed.

A tiny Mage-Flow checkpoint is written to disk and loaded through the normal loader; latents are
cached through the real cache path with a stand-in VAE. Only the Qwen3-VL text encoder and the
VAE decoder are faked (deterministic embeddings, a flat-colour decode), so everything the new
features touch -- text cache, StageLR, attribution, trackers, samples, eval, signals, resume --
runs as it would on a GPU.
"""

import json
import os
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from safetensors.torch import save_file

from trainer.data.cache import LatentCacher
from trainer.data.dataset import DatasetConfig
from trainer.modeling.mage_flow import MageFlow, MageFlowParams

PARAMS = dict(in_channels=128, out_channels=128, context_in_dim=24, hidden_size=32,
              num_heads=4, depth=2, axes_dim=[2, 2, 4])


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def encode(self, x):
        return torch.nn.functional.avg_pool2d(x.float(), 16).repeat(1, 43, 1, 1)[:, :128]


def fake_encode_prompts(components, prompts, device, max_length=512, **_):
    rows = []
    for p in prompts:
        g = torch.Generator().manual_seed(zlib.crc32(p.encode()))
        rows.append(torch.randn(1 + len(p.split()) % 6, 24, generator=g))
    hidden = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True).to(device)
    lengths = torch.tensor([r.shape[0] for r in rows], device=device)
    return hidden, torch.arange(hidden.shape[1], device=device)[None] < lengths[:, None]


class FakeDecoder:
    def __init__(self, *a):
        pass

    def __call__(self, latents, device):
        h, w = latents.shape[-2:]
        return Image.new("RGB", (w * 16, h * 16), (int(abs(latents.mean().item()) * 100) % 255,
                                                   64, 64))


def write_dataset(folder: Path, n: int, offset: int = 0):
    folder.mkdir(parents=True)
    cacher = LatentCacher(FakeVAE(), device="cpu", dtype=torch.float32,
                          storage_dtype=torch.float16)
    mgr = DatasetConfig(path=str(folder), resolution=128).build_bucket_manager()
    for i in range(n):
        size = (128, 128) if i % 4 else (128, 64)
        p = folder / f"img{i}.png"
        Image.new("RGB", size, ((37 * (i + offset)) % 255, 120, 200)).save(p)
        artist = ["emily (pure dream)", "alice"][i % 2]
        (folder / f"img{i}.txt").write_text(f"1girl, Drawn by {artist}, smile, tag{i}",
                                            encoding="utf-8")
        (folder / f"img{i}_nl.txt").write_text(f"A girl smiles. Drawn by {artist}.",
                                               encoding="utf-8")
        cacher.cache_image(p, mgr)


class TrainerEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.tmp.name)
        self.root = root
        model = root / "model" / "transformer"
        model.mkdir(parents=True)
        torch.manual_seed(0)
        net = MageFlow(MageFlowParams(**PARAMS, checkpoint=False))
        save_file({k: v.contiguous() for k, v in net.state_dict().items()},
                  str(model / "diffusion_pytorch_model.safetensors"))
        (model / "config.json").write_text(json.dumps(PARAMS))
        write_dataset(root / "data", 8)
        write_dataset(root / "held", 3, offset=50)
        self.env = patch.dict(os.environ, {"MAGE_FLOW_ALLOW_CPU": "1",
                                           "WANDB_SILENT": "true"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def write_config(self, extra_train=""):
        r = self.root.as_posix()
        path = self.root / "run.toml"
        path.write_text(f"""
[train]
model_path = "{r}/model"
output_dir = "{r}/out"
run_name = "e2e"
epochs = 2
batch_size = 2
dtype = "float32"
gradient_checkpointing = false
cache_text_embeddings = true
num_workers = 0
save_optimizer_state = true
progress = "plain"
seed = 1
{extra_train}
[dataset]
path = "{r}/data"
resolution = 128
[dataset.caption]
caption_mode = "tags_nl"
attribution_patterns = ['^drawn by\\s']
[optimizer]
kind = "adamw"
lr = 1e-4
[schedule]
kind = "stage"
stages = [
  {{ type = "linear", end_lr = 2e-4, percent = 0.5 }},
  {{ type = "rex", max_val = 2e-4, min_val = 0.0, percent = 0.5 }},
]
[tracking]
backends = ["tensorboard"]
[sampling]
prompts = ["Drawn by emily (pure dream), 1girl", {{ prompt = "a castle", width = 64, height = 128, cfg = 1.0 }}]
every_n_steps = 4
at_start = true
steps = 2
width = 64
height = 64
[eval]
path = "{r}/held"
every_n_steps = 2
quantiles = [0.25, 0.75]
batch_size = 2
""", encoding="utf-8")
        return path

    def run_trainer(self, path, signal_at=None):
        from trainer.training import train as train_mod
        from trainer.training.config import load_config

        real_load = train_mod.load_components

        def load(path_, *a, **kw):
            if kw.get("load_transformer") is False:      # the text encoder for the text cache
                return type("C", (), {"text_encoder": torch.nn.Identity(), "tokenizer": None})()
            return real_load(path_, *a, **kw)

        trainer_cls = train_mod.Trainer

        class Signalling(trainer_cls):
            def _periodic(self, step, epoch_end):
                # Drop the file the GUI's "Save & stop" button drops, from inside the run.
                if signal_at is not None and step == signal_at and epoch_end is None:
                    (self.out_dir / "save_quit").touch()
                return super()._periodic(step, epoch_end)

        with patch.object(train_mod, "load_components", side_effect=load), \
                patch.object(train_mod, "encode_prompts", side_effect=fake_encode_prompts), \
                patch("trainer.training.sampling.Decoder", FakeDecoder):
            cfg = load_config(path)
            trainer = Signalling(cfg, config_path=path)
            trainer.train()
        return trainer

    def scalars(self, tag):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        acc = EventAccumulator(str(self.root / "out/e2e/tracking/tensorboard"),
                               size_guidance={"scalars": 0, "images": 0})
        acc.Reload()
        return {e.step: e.value for e in acc.Scalars(tag)}

    def test_text_cache_only_builds_the_cache_without_the_transformer(self):
        from trainer.training import train as train_mod
        from trainer.training.config import load_config

        cache = self.root / "prep" / "captions.sqlite"
        path = self.write_config(f'caption_variations = 2\ncaption_cache_path = "{cache.as_posix()}"')
        loads = []

        def load(path_, *a, **kw):
            loads.append(kw.get("load_transformer", True))
            return type("C", (), {"text_encoder": torch.nn.Identity(), "tokenizer": None})()

        with patch.object(train_mod, "load_components", side_effect=load), \
                patch.object(train_mod, "encode_prompts", side_effect=fake_encode_prompts):
            train_mod.Trainer(load_config(path), config_path=path, text_cache_only=True)
        self.assertTrue(cache.is_file())
        self.assertEqual(loads, [False])          # the text encoder only

        # The same dataset under another absolute path (downloaded elsewhere) reuses every slot.
        import shutil
        shutil.copytree(self.root / "data", self.root / "elsewhere" / "data")
        text = path.read_text(encoding="utf-8").replace(f'path = "{self.root.as_posix()}/data"',
                                                        f'path = "{self.root.as_posix()}/elsewhere/data"')
        moved = self.root / "moved.toml"
        moved.write_text(text, encoding="utf-8")
        loads.clear()
        with patch.object(train_mod, "load_components", side_effect=load), \
                patch.object(train_mod, "encode_prompts", side_effect=fake_encode_prompts):
            train_mod.Trainer(load_config(moved), config_path=moved, text_cache_only=True)
        self.assertEqual(loads, [])               # nothing left to encode: the encoder never loads

    def test_full_run_then_signal_stop_and_resume(self):
        from trainer.training.stage_lr import StagePlan

        trainer = self.run_trainer(self.write_config(), signal_at=3)
        out = self.root / "out" / "e2e"
        # 8 images: 6 square + 2 wide -> 3 + 1 batches of 2 -> 4 steps/epoch, 8 total.
        self.assertEqual(trainer.total_steps, 8)
        # Stopped by save_quit at step 4 (file dropped after step 3): resumable state, no final.
        self.assertEqual(trainer.global_step, 4)
        self.assertTrue((out / "e2e-step000004-state" / "state.json").exists())
        self.assertTrue((out / "e2e-step000004.safetensors").exists())
        self.assertFalse((out / "e2e.safetensors").exists())
        self.assertFalse((out / "save_quit").exists(), "signal file consumed")

        # Attribution reached the text cache: every training caption leads with its artist and
        # carries it once despite tags_nl joining both fields.
        captions = [c for c in trainer.text_cache.embeddings if "smile" in c]
        self.assertTrue(captions)
        for c in captions:
            self.assertTrue(c.startswith("Drawn by "), c)
            self.assertEqual(c.count("Drawn by"), 1, c)

        # Samples: the step-0 baseline, both prompts, the castle at its own size. Step 4 would
        # sample too, but save_quit stops straight after its checkpoint -- a stop is a stop.
        files = sorted(p.name for p in (out / "samples" / "step000000").glob("*.png"))
        self.assertEqual(len(files), 2)
        self.assertFalse((out / "samples" / "step000004").exists())
        castle = next((out / "samples/step000000").glob("01_*.png"))
        self.assertEqual(Image.open(castle).size, (64, 128))

        # Trackers: eval baseline + every 2 steps, LR exactly the fitted StageLR plan.
        self.assertEqual(sorted(self.scalars("eval/loss")), [0, 2])
        plan = StagePlan(trainer.cfg.schedule.stages, 8, 0, 1e-4)
        lrs = self.scalars("train/lr")
        for step, lr in lrs.items():
            # Logged after the update, so the scheduler has advanced to the next update's LR.
            self.assertAlmostEqual(lr, plan.lr(step), delta=1e-10, msg=f"step {step}")
        self.assertIn(4, self.scalars("train/grad_norm"))
        self.assertIn(4, self.scalars("train/samples_per_second"))

        # Resume from the signal checkpoint and finish the run.
        resumed = self.run_trainer(self.write_config(
            f'resume_from = "{(out / "e2e-step000004-state").as_posix()}"'))
        self.assertEqual(resumed.global_step, 8)
        self.assertTrue((out / "e2e.safetensors").exists())
        lrs = self.scalars("train/lr")
        for step in range(5, 9):
            self.assertAlmostEqual(lrs[step], plan.lr(step), delta=1e-10, msg=f"step {step}")
        # The resumed run does not re-run the step-0 baselines.
        self.assertEqual(sorted(self.scalars("eval/loss")), [0, 2, 6, 8])
        self.assertEqual(len(list((out / "samples" / "step000008").glob("*.png"))), 2)


if __name__ == "__main__":
    unittest.main()
