"""Validation sampling and held-out eval, on a tiny CPU Mage-Flow."""

import json
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from trainer.data.cache import LatentCacher
from trainer.data.dataset import DatasetConfig
from trainer.modeling.mage_flow import MageFlow, MageFlowParams
from trainer.training.config import Config, EvalConfig, SamplingConfig, TrainConfig
from trainer.training.sampling import (
    ValidationSampler,
    build_sigmas,
    combine_cfg,
    denoise,
    sample_texts,
)


def tiny():
    torch.manual_seed(42)
    model = MageFlow(MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], False))
    model.configure_execution(False)
    return model.eval()


def fake_encode(texts, device="cpu", dtype=torch.float32):
    """Deterministic per-text embeddings of varying length, padded with a mask -- the shape
    `Trainer._encode` returns."""
    rows = []
    for t in texts:
        g = torch.Generator().manual_seed(zlib.crc32(t.encode()))
        rows.append(torch.randn(1 + len(t.split()) % 5, 24, generator=g))
    hidden = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True).to(device, dtype)
    lengths = torch.tensor([r.shape[0] for r in rows], device=device)
    return hidden, torch.arange(hidden.shape[1], device=device)[None] < lengths[:, None]


class Unfused(torch.nn.Module):
    """Same model, but with `region_interface` set so `denoise` takes the two-forward CFG path."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.in_channels = model.in_channels
        self.region_interface = object()

    def forward(self, **kw):
        return self.model(**kw)


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def encode(self, x):
        return torch.nn.functional.avg_pool2d(x.float(), 16).repeat(1, 43, 1, 1)[:, :128]


class SamplerTests(unittest.TestCase):
    def test_sigmas_match_the_reference_scheduler(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        for steps, shift in ((30, 6.0), (8, 3.0), (1, 6.0)):
            ref = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift,
                                                  use_dynamic_shifting=False)
            ref.set_timesteps(sigmas=torch.linspace(1.0, 1.0 / steps, steps).tolist())
            torch.testing.assert_close(build_sigmas(steps, shift).float(), ref.sigmas.float(),
                                       rtol=1e-5, atol=1e-6)

    def test_fused_cfg_equals_two_forwards_and_is_seeded(self):
        model = tiny()
        s = dict(prompt="Drawn by emily, 1girl", negative_prompt=" ", seed=7, width=64,
                 height=96, steps=4, cfg=4.0, shift=6.0, renormalize_cfg=False)
        fused = denoise(model, fake_encode, s, 7, "cpu", torch.float32)
        split = denoise(Unfused(model), fake_encode, s, 7, "cpu", torch.float32)
        self.assertEqual(tuple(fused.shape), (1, 128, 1, 6, 4))
        torch.testing.assert_close(fused, split, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(fused, denoise(model, fake_encode, s, 7, "cpu",
                                                  torch.float32))
        self.assertFalse(torch.equal(fused, denoise(model, fake_encode, s, 8, "cpu",
                                                    torch.float32)))

    def test_cfg_at_most_one_skips_the_negative(self):
        calls = []

        def counting(texts):
            calls.append(list(texts))
            return fake_encode(texts)

        s = dict(prompt="a", negative_prompt="bad", seed=0, width=64, height=64, steps=2,
                 cfg=1.0, shift=6.0, renormalize_cfg=False)
        denoise(tiny(), counting, s, 0, "cpu", torch.float32)
        self.assertEqual(calls, [["a"]])
        cfg = SamplingConfig(prompts=["a", {"prompt": "b", "cfg": 1.0}], every_n_steps=5,
                             negative_prompt="")
        self.assertEqual(sample_texts(cfg), ["a", " ", "b"])

    def test_renormalized_cfg_keeps_the_conditional_norm(self):
        cond, uncond = torch.randn(1, 128, 1, 2, 2), torch.randn(1, 128, 1, 2, 2)
        out = combine_cfg(cond, uncond, 7.0, renormalize=True)
        torch.testing.assert_close(out.norm(dim=1), cond.norm(dim=1), rtol=1e-4, atol=1e-4)

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, "no cadence"):
            SamplingConfig(prompts=["a"])
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            SamplingConfig(prompts=[{"prompt": "a", "width": 100}], at_start=True)
        with self.assertRaisesRegex(ValueError, "unknown key"):
            SamplingConfig(prompts=[{"prompt": "a", "sampler": "dpm"}], at_start=True)
        s = SamplingConfig(prompts=["Drawn by emily (pure dream), 1girl!"], at_start=True)
        self.assertEqual(s.resolved_prompts()[0]["label"], "Drawn_by_emily_pure_dream_1girl")

    def test_run_writes_pngs_manifest_and_tracker_images(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            images, logged = [], []
            tracker = SimpleNamespace(log_images=lambda k, e, s: images.append((k, e, s)),
                                      log=lambda m, s: logged.append((m, s)))
            cfg = SamplingConfig(prompts=["a girl", {"prompt": "a castle", "width": 64,
                                                     "height": 128, "seed": 3}],
                                 every_n_steps=2, steps=2, width=64, height=64)
            model = tiny()
            trainer = SimpleNamespace(
                cfg=SimpleNamespace(sampling=cfg, train=TrainConfig()),
                accelerator=SimpleNamespace(process_index=0, num_processes=1, is_main_process=True,
                                            device=torch.device("cpu")),
                transformer=model, _unwrap=lambda m: m, _encode=fake_encode,
                dtype=torch.float32, out_dir=Path(tmp), tracker=tracker, rti_interface=None,
                _print=lambda msg: None)

            class FakeDecoder:
                def __init__(self, *a):
                    pass

                def __call__(self, latents, device):
                    h, w = latents.shape[-2:]
                    return Image.new("RGB", (w * 16, h * 16), (int(latents.mean() * 10) % 255, 0, 0))

            sampler = ValidationSampler(trainer)
            self.assertTrue(sampler.due(2))
            self.assertFalse(sampler.due(3))
            with patch("trainer.training.sampling.Decoder", FakeDecoder):
                sampler.run(2, "step000002")
            self.assertFalse(sampler.due(2), "a step is sampled once even if epoch also hits it")
            out = Path(tmp) / "samples" / "step000002"
            files = sorted(p.name for p in out.glob("*.png"))
            self.assertEqual(files, ["00_a_girl_seed42.png", "01_a_castle_seed3.png"])
            self.assertEqual(Image.open(out / "01_a_castle_seed3.png").size, (64, 128))
            manifest = json.loads((out / "prompts.json").read_text())
            self.assertEqual([s["prompt"] for s in manifest["samples"]], ["a girl", "a castle"])
            self.assertEqual(images[0][0], "samples")
            self.assertEqual([n for n, _, _ in images[0][1]], ["00_a_girl", "01_a_castle"])
            self.assertIn("samples/seconds", logged[0][0])
            self.assertTrue(model.training is False)


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.tmp.name)
        self.data, self.held = root / "data", root / "held"
        for folder, n in ((self.data, 2), (self.held, 5)):
            folder.mkdir()
            for i in range(n):
                size = (128, 128) if i % 2 == 0 else (128, 64)
                Image.new("RGB", size, (30 * i, 90, 160)).save(folder / f"im{i}.png")
                (folder / f"im{i}.txt").write_text(f"tag{i}, Drawn by artist{i % 2}, smile")
            cfg = DatasetConfig(path=str(folder), resolution=128)
            cacher = LatentCacher(FakeVAE(), device="cpu", dtype=torch.float32)
            for p in sorted(folder.glob("*.png")):
                cacher.cache_image(p, cfg.build_bucket_manager())

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, **ev):
        return Config(dataset=DatasetConfig(path=str(self.data), resolution=128),
                      eval=EvalConfig(path=str(self.held), every_n_steps=1, **ev))

    def trainer(self, model):
        return SimpleNamespace(
            accelerator=SimpleNamespace(device=torch.device("cpu"), process_index=0,
                                        num_processes=1, is_main_process=True),
            transformer=model, _unwrap=lambda m: m, _encode=fake_encode, dtype=torch.float32,
            keep_fraction=1.0)

    def test_deterministic_and_independent_of_batch_size(self):
        from trainer.training.evaluation import Evaluator

        model = tiny()
        ev = Evaluator(self.config(quantiles=[0.2, 0.8], batch_size=1))
        self.assertEqual(len(ev), 5)
        a = ev.run(self.trainer(model), 0)
        b = ev.run(self.trainer(model), 1)
        self.assertEqual(a, b)
        c = Evaluator(self.config(quantiles=[0.2, 0.8], batch_size=4)).run(self.trainer(model), 0)
        for k in a:
            self.assertAlmostEqual(a[k], c[k], places=5)
        self.assertAlmostEqual(a["eval/loss"], (a["eval/loss_q0.20"] + a["eval/loss_q0.80"]) / 2)
        self.assertFalse(model.training)

    def test_max_samples_and_captions_are_unaugmented(self):
        from trainer.training.evaluation import Evaluator

        cfg = self.config(max_samples=3)
        cfg.dataset.caption.shuffle_tags = True
        cfg.dataset.caption.caption_mode = "mixed"
        ev = Evaluator(cfg)
        self.assertEqual(len(ev), 3)
        self.assertEqual(ev.indices, Evaluator(cfg).indices)
        self.assertTrue(all(c.startswith("tag") for c in ev.captions.values()))

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, "no cadence"):
            EvalConfig(path="x", at_start=False)
        with self.assertRaisesRegex(ValueError, r"\(0, 1\)"):
            EvalConfig(quantiles=[0.0, 0.5])


if __name__ == "__main__":
    unittest.main()
