"""Guard the conditioning boundary and independent transformer/encoder settings."""
from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest

import torch

from trainer.modeling.mageflow_text import encode_text_hidden
from trainer.training.config import load_config
from trainer.training.quant import text_encoder_quant_config


class Encoder:
    def __init__(self, offset=0):
        self.offset = offset
        self.wrapper_calls = 0
        self.model = self.backbone

    def backbone(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=input_ids.float() + self.offset)

    def __call__(self, input_ids, **kwargs):
        self.wrapper_calls += 1
        return SimpleNamespace(hidden_states=[input_ids.float()])


class LoadedEncoderTests(unittest.TestCase):
    def test_equivalence_is_checked_once(self):
        encoder = Encoder()
        ids = torch.tensor([[1, 2]])
        for _ in range(2):
            actual = encode_text_hidden(encoder, ids, torch.ones_like(ids), embedding_only=True)
            self.assertTrue(torch.equal(actual, ids.float()))
        self.assertEqual(encoder.wrapper_calls, 1)

    def test_different_normalization_is_rejected(self):
        ids = torch.tensor([[1, 2]])
        encoder = Encoder(offset=1)
        with self.assertRaisesRegex(RuntimeError, 'text_encoder_embedding_only=false'):
            encode_text_hidden(encoder, ids, torch.ones_like(ids), embedding_only=True)
        self.assertFalse(getattr(encoder, '_mage_embedding_only_verified', False))
        self.assertTrue(torch.equal(encode_text_hidden(encoder, ids, torch.ones_like(ids)), ids.float()))

    def test_transformer_quantization_does_not_quantize_encoder(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.toml'
            path.write_text('''[dataset]
path = "/tmp"
[train]
dtype = "bfloat16"
compile_text_encoder = true
text_encoder_embedding_only = true
[quant]
mode = "training"
weights_dtype = "uint8"
quantize_text_encoder = false
[adapter]
kind = "none"
''')
            cfg = load_config(path)
        self.assertTrue(cfg.train.compile_text_encoder)
        self.assertTrue(cfg.train.text_encoder_embedding_only)
        self.assertIsNone(text_encoder_quant_config(cfg.quant))
        self.assertEqual(cfg.quant.mode, 'training')


if __name__ == '__main__':
    unittest.main()
