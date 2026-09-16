import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from trainer.data.caption import CaptionConfig
from trainer.data.text_cache import TextEmbeddingCache, validate_static_captions


class TextCacheTests(unittest.TestCase):
    def test_static_cache_batches_and_trims_each_caption(self):
        from trainer.training.config import Config, TrainConfig
        from trainer.training.train import Trainer
        from trainer.data.dataset import DatasetConfig
        from trainer.training.quant import QuantConfig
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = Config(train=TrainConfig(cache_text_embeddings=True, text_cache_batch_size=2),
                             dataset=DatasetConfig(path="/unused"), quant=QuantConfig(mode="none"))
        trainer.dtype = torch.float32
        captions = ["a", "bb", "ccc", "dddd", "eeeee"]
        trainer.dataset = SimpleNamespace(entries=[SimpleNamespace(tags=c, nl=None) for c in captions])
        trainer.accelerator = SimpleNamespace(device="cpu", is_main_process=True)
        batches = []

        def encode(components, captions, device, max_length):
            batches.append(list(captions))
            lengths = torch.tensor([len(c) for c in captions])
            hidden = lengths.float()[:, None, None].expand(-1, int(lengths.max()), 4)
            return hidden, torch.arange(hidden.shape[1])[None] < lengths[:, None]

        with patch("trainer.training.train.load_components", return_value=SimpleNamespace(text_encoder=torch.nn.Identity())), \
             patch("trainer.training.train.encode_prompts", side_effect=encode):
            trainer._build_text_cache()
        self.assertEqual([len(b) for b in batches], [2, 2, 1])
        for caption in captions:
            value = trainer.text_cache.embeddings[caption]
            self.assertEqual(value.shape, (len(caption), 4))
            self.assertTrue((value == len(caption)).all())

    def test_text_cache_batch_size_validation(self):
        from trainer.training.config import TrainConfig
        self.assertEqual(TrainConfig().text_cache_batch_size, 4)
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                TrainConfig(text_cache_batch_size=invalid)

    def test_padding_and_storage_are_independent(self):
        c = TextEmbeddingCache()
        hidden = torch.arange(24.0).reshape(1, 6, 4)
        c.add("a", hidden, torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool))
        c.add("b", hidden, torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool))
        hidden.zero_()
        batch, mask = c.get(["a", "b"], "cpu", torch.float32)
        self.assertEqual(batch.shape, (2, 4, 4))
        self.assertEqual(mask.sum(1).tolist(), [2, 4])
        self.assertEqual(batch[0, 1, 3].item(), 7)
        self.assertFalse(c.embeddings["a"].requires_grad)
        self.assertEqual(c.embeddings["a"].untyped_storage().nbytes(), 2 * 4 * 4)
        with self.assertRaises(KeyError):
            c.get(["unseen"], "cpu", torch.float32)

    def test_rejects_dynamic_captions(self):
        cfg = SimpleNamespace(
            dataset=SimpleNamespace(caption=CaptionConfig()),
            curriculum=SimpleNamespace(phases=[]),
            preserve=SimpleNamespace(enabled=False),
        )
        validate_static_captions(cfg)
        for key in (
            "shuffle_tags",
            "tag_dropout_percent",
            "caption_dropout_percent",
            "nl_shuffle_sentences",
        ):
            cfg.dataset.caption = CaptionConfig(
                **{key: True if "shuffle" in key else 0.1}
            )
            with self.assertRaises(ValueError):
                validate_static_captions(cfg)


if __name__ == "__main__":
    unittest.main()
