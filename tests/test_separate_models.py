"""Component selection and single-file loading without a full model repository."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from trainer.modeling.loader import load_components
from trainer.modeling.mage_flow import MageFlow, MageFlowParams
from trainer.training.config import Config
from trainer.data.dataset import DatasetConfig
from trainer.data.caption_variations import encoder_fingerprint
from trainer.gui.process import cache_launch


class SeparateModelTests(unittest.TestCase):
    def test_native_transformer_metadata_and_prefix(self):
        params = MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], False)
        original = MageFlow(params)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "finetune.safetensors"
            for prefix in ("", "diffusion_model.", "model.diffusion_model."):
                save_file({prefix + k: v for k, v in original.state_dict().items()}, str(path),
                          metadata={"model_config": json.dumps(asdict(params))})
                loaded = load_components("/missing-repo", transformer_path=path,
                                         load_vae=False, load_text_encoder=False,
                                         load_tokenizers=False, dtype=torch.float32)
                self.assertEqual(loaded.transformer.params, params)
                for key, value in original.state_dict().items():
                    self.assertTrue(torch.equal(value, loaded.transformer.state_dict()[key]), key)
                self.assertFalse(any(p.is_meta for p in loaded.transformer.parameters()))

    def test_cache_tracks_encoder_and_tokenizer_not_transformer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            encoder = root / "encoder.safetensors"
            encoder.write_bytes(b"weights")
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            asset = tokenizer / "tokenizer.json"
            asset.write_text("tokenizer")
            cfg = Config(dataset=DatasetConfig(path=folder))
            cfg.train.text_encoder_path = str(encoder)
            cfg.train.tokenizer_path = str(tokenizer)
            before = encoder_fingerprint(cfg)
            cfg.train.transformer_path = "/another/model.safetensors"
            cfg.train.model_path = "/another/repo"
            self.assertEqual(before, encoder_fingerprint(cfg))
            encoder.write_bytes(b"different weights")
            after = encoder_fingerprint(cfg)
            self.assertNotEqual(before, after)
            asset.write_text("different tokenizer")
            self.assertNotEqual(after, encoder_fingerprint(cfg))

    def test_cache_launch_passes_vae(self):
        launch = cache_launch("/images", "/unused", [1024], min_bucket_reso=256,
                              max_bucket_reso=4096, bucket_reso_steps=64, upscale=False,
                              multires_training=False, vae_path="/models/my vae.safetensors")
        self.assertEqual(launch.argv[launch.argv.index("--vae-path") + 1],
                         "/models/my vae.safetensors")


if __name__ == "__main__":
    unittest.main()
