import collections
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from trainer.data.caption import CaptionConfig
from trainer.data.caption_variations import CaptionVariationCache, digest


class CaptionVariationTests(unittest.TestCase):
    def test_dedup_slots_extension_resume_and_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.sqlite"
            entry = SimpleNamespace(
                path=Path(folder) / "image.png",
                tags="a, b, c, d",
                nl="A scene. Another sentence.",
            )
            cfg = CaptionConfig(shuffle_tags=True, caption_mode="mixed")
            cache = CaptionVariationCache(path, "encoder")
            cache.prepare([entry, entry], cfg, 42, 100)
            slots = (
                cache.reader()
                .execute("SELECT slot,caption FROM slots ORDER BY slot")
                .fetchall()
            )
            self.assertEqual(len(slots), 100)
            self.assertLess(len(set(c for _, c in slots)), 100)
            pending = list(cache.pending())
            for key, caption in pending:
                cache.add(
                    key,
                    torch.ones(1, 4, 3) * len(caption),
                    torch.tensor([[1, 1, 0, 0]]),
                )
            self.assertEqual(list(cache.pending()), [])
            sequence = [
                cache.caption(i, epoch) for epoch in range(50) for i in range(2)
            ]
            self.assertEqual(
                collections.Counter(map(digest, sequence)),
                collections.Counter(c for _, c in slots),
            )
            restored = pickle.loads(pickle.dumps(cache))
            self.assertEqual(restored.caption(1, 23), cache.caption(1, 23))
            hidden, mask = restored.get(sequence[:3], "cpu", torch.float32)
            self.assertEqual(hidden.shape, (3, 2, 3))
            self.assertTrue(mask.all())
            restored.close()
            cache.prepare([entry], cfg, 42, 120)
            self.assertEqual(
                cache.reader()
                .execute("SELECT slot,caption FROM slots WHERE slot<100 ORDER BY slot")
                .fetchall(),
                slots,
            )
            other = CaptionVariationCache(path, "changed-encoder")
            other.prepare([entry], cfg, 42, 120)
            self.assertTrue(list(other.pending()))
            with self.assertRaisesRegex(RuntimeError, "Missing cached"):
                other.get([sequence[0]], "cpu", torch.float32)
            other.close()
            cache.close()

    def test_no_augmentation_dedups_all_slots_and_dropout_reuses_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = CaptionVariationCache(Path(folder) / "cache.sqlite", "encoder")
            e = SimpleNamespace(path=Path(folder) / "a.png", tags="a, b", nl=None)
            cache.prepare(
                [e, SimpleNamespace(path=Path(folder) / "b.png", tags=e.tags, nl=e.nl)],
                CaptionConfig(caption_dropout_percent=1),
                5,
                100,
            )
            self.assertEqual(len(list(cache.pending())), 2)
            self.assertEqual(cache.caption(0, 50), "")
            cache.close()


class EncoderFingerprintTests(unittest.TestCase):
    def config(self, folder):
        from trainer.training.quant import QuantConfig

        return SimpleNamespace(
            train=SimpleNamespace(
                model_path=folder, dtype="bfloat16", max_text_tokens=512
            ),
            quant=QuantConfig(
                mode="frozen", quantize_text_encoder=True, skip_policy="all_adaln"
            ),
        )

    def test_transformer_settings_do_not_invalidate_encoder(self):
        from dataclasses import replace
        from trainer.data.caption_variations import encoder_fingerprint

        with tempfile.TemporaryDirectory() as folder:
            cfg = self.config(folder)
            original = encoder_fingerprint(cfg)
            cfg.quant = replace(
                cfg.quant,
                mode="training",
                skip_policy="mlp_down",
                extra_skip=["transformer_blocks.0"],
                use_quantized_matmul=True,
            )
            self.assertEqual(original, encoder_fingerprint(cfg))
            cfg.quant = replace(cfg.quant, mode="none")
            self.assertEqual(original, encoder_fingerprint(cfg))
            cfg.quant = replace(cfg.quant, group_size=64)
            self.assertNotEqual(original, encoder_fingerprint(cfg))

    def test_effective_encoder_settings_still_invalidate(self):
        from dataclasses import replace
        from trainer.data.caption_variations import encoder_fingerprint

        with tempfile.TemporaryDirectory() as folder:
            cfg = self.config(folder)
            original = encoder_fingerprint(cfg)
            for change in (
                {"weights_dtype": "fp8"},
                {"quantize_text_encoder": False},
            ):
                other = self.config(folder)
                other.quant = replace(other.quant, **change)
                self.assertNotEqual(original, encoder_fingerprint(other))
            cfg.train.max_text_tokens = 256
            self.assertNotEqual(original, encoder_fingerprint(cfg))
            cfg = self.config(folder)
            cfg.train.dtype = "float32"
            self.assertNotEqual(original, encoder_fingerprint(cfg))
            cfg = self.config(folder)
            (Path(folder) / "text_encoder").mkdir()
            (Path(folder) / "text_encoder" / "config.json").write_text("{}")
            self.assertNotEqual(original, encoder_fingerprint(cfg))

    def test_reuses_old_lora_cache_for_finetune_without_copying_embeddings(self):
        from dataclasses import asdict, replace
        from trainer.data.caption_variations import (
            encoder_fingerprint,
            _encoder_fingerprint_payload,
        )

        with tempfile.TemporaryDirectory() as folder:
            cfg = self.config(folder)
            old_quant = asdict(cfg.quant)
            old_quant.pop("text_encoder_weights_dtype", None)
            old_key = digest(
                {**_encoder_fingerprint_payload(cfg), "quant": old_quant}
            )
            path = Path(folder) / "cache.sqlite"
            entry = SimpleNamespace(path=Path(folder) / "a.png", tags="a", nl=None)
            cache = CaptionVariationCache(path, old_key)
            cache.prepare([entry], CaptionConfig(), 42, 1)
            for key, _ in cache.pending():
                cache.add(key, torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.bool))
            before = (
                cache.reader().execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            )
            cache.close()
            cfg.quant = replace(cfg.quant, mode="training", use_quantized_matmul=False)
            self.assertEqual(encoder_fingerprint(cfg, path), old_key)
            resumed = CaptionVariationCache(path, encoder_fingerprint(cfg, path))
            resumed.prepare([entry], CaptionConfig(), 42, 1)
            self.assertEqual(list(resumed.pending()), [])
            self.assertEqual(
                resumed.reader()
                .execute("SELECT COUNT(*) FROM embeddings")
                .fetchone()[0],
                before,
            )
            resumed.close()
            cfg.quant = replace(cfg.quant, weights_dtype="fp8")
            self.assertNotEqual(encoder_fingerprint(cfg, path), old_key)


def test_loaded_encoder_can_be_quantized_without_transformer(monkeypatch):
    from types import SimpleNamespace
    from trainer.training.train import Trainer
    from trainer.training.quant import QuantConfig
    calls = []
    encoder, quantized_encoder = object(), object()
    def quantize(module, config, *args):
        calls.append((module, config))
        return quantized_encoder
    monkeypatch.setattr('trainer.training.train.quantize_module', quantize)
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(quant=QuantConfig(mode='none', quantize_text_encoder=True)),
        accelerator=SimpleNamespace(device='cpu'), text_encoder=encoder,
        components=SimpleNamespace(text_encoder=encoder), dtype=None,
    )
    Trainer._quantize(trainer)
    assert trainer.text_encoder is quantized_encoder
    assert trainer.components.text_encoder is quantized_encoder
    assert trainer.quant_info is None
    assert len(calls) == 1 and calls[0][1].mode == 'frozen'


def test_encoder_dtype_override_keeps_cache_when_transformer_changes(tmp_path):
    from dataclasses import replace
    from trainer.data.caption_variations import encoder_fingerprint
    from trainer.training.quant import text_encoder_quant_config
    cfg = EncoderFingerprintTests().config(str(tmp_path))
    original = encoder_fingerprint(cfg)
    cfg.quant = replace(cfg.quant, weights_dtype='uint8', text_encoder_weights_dtype='int8')
    assert encoder_fingerprint(cfg) == original
    assert text_encoder_quant_config(cfg.quant).weights_dtype == 'int8'
    cfg.quant = replace(cfg.quant, text_encoder_weights_dtype='uint8')
    assert encoder_fingerprint(cfg) != original
