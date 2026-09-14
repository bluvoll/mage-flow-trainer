import unittest
from types import SimpleNamespace
import torch
from trainer.data.caption import CaptionConfig
from trainer.data.text_cache import TextEmbeddingCache, validate_static_captions


class TextCacheTests(unittest.TestCase):
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
