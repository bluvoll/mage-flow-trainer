"""Exercise real concurrent SQLite writers and DDP barriers with a tiny CPU encoder."""
from datetime import timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from trainer.data.caption import CaptionConfig
from trainer.data.caption_variations import CaptionVariationCache, digest
from trainer.data.dataset import DatasetConfig
from trainer.training.config import Config, TrainConfig
from trainer.training.quant import QuantConfig
from trainer.training.train import Trainer


def worker(rank, world_size, folder, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=90))
    try:
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = Config(
            train=TrainConfig(cache_text_embeddings=True, caption_variations=5,
                              caption_cache_path=str(Path(folder) / "cache.sqlite")),
            dataset=DatasetConfig(path=folder, caption=CaptionConfig(shuffle_tags=True)),
            quant=QuantConfig(mode="none"),
        )
        trainer.dtype = torch.float32
        trainer.dataset = SimpleNamespace(entries=[SimpleNamespace(
            path=Path(folder) / f"image{i}.png", tags=f"image{i}, a, b, c, d", nl=None
        ) for i in range(40)])
        trainer.accelerator = SimpleNamespace(
            process_index=rank, num_processes=world_size, is_main_process=rank == 0,
            device=torch.device("cpu"), wait_for_everyone=dist.barrier, print=lambda *a: None,
        )
        calls = []

        def encode(components, captions, device, max_length):
            calls.extend(captions)
            return torch.full((1, 3, 4), float(len(captions[0]))), torch.tensor([[True, True, False]])

        with patch("accelerate.utils.broadcast_object_list", dist.broadcast_object_list), \
             patch("trainer.data.caption_variations.encoder_fingerprint", return_value="test"), \
             patch("trainer.training.train.load_components", return_value=SimpleNamespace(text_encoder=torch.nn.Identity())) as loader, \
             patch("trainer.training.train.encode_prompts", side_effect=encode):
            trainer._build_variation_cache()
            assert loader.call_count == 1
            cold_count = len(calls)
            assert cold_count > 0
            # Warm-cache startup must not load an encoder on any rank.
            loader.side_effect = AssertionError("Warm cache loaded an encoder")
            trainer._build_variation_cache()
            assert len(calls) == cold_count
            hidden, mask = trainer.text_cache.get([calls[0]], "cpu", torch.float32)
            assert hidden.shape == (1, 2, 4) and mask.all()
        (Path(folder) / f"rank{rank}.json").write_text(json.dumps(calls))
        trainer.text_cache.close()
    finally:
        dist.destroy_process_group()


class DistributedCaptionCacheTests(unittest.TestCase):
    def test_two_ranks_encode_unique_shards_and_skip_warm_encoder_load(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(worker, args=(2, folder, str(Path(folder) / "rendezvous")), nprocs=2)
            calls = [json.loads((Path(folder) / f"rank{i}.json").read_text()) for i in range(2)]
            self.assertFalse(set(calls[0]) & set(calls[1]))
            cache = CaptionVariationCache(Path(folder) / "cache.sqlite", "test")
            self.assertEqual(list(cache.pending()), [])
            stored = {row[0] for row in cache.reader().execute("SELECT caption FROM embeddings")}
            self.assertEqual(stored, {digest(c) for group in calls for c in group})
            self.assertIn(digest(""), stored)
            # Resume after missing entries are replanned with a different world size.
            with cache.writer() as db:
                db.execute("INSERT INTO pending SELECT encoder,caption FROM embeddings")
            shards = [set(k for k, _ in cache.pending(rank, 3)) for rank in range(3)]
            self.assertEqual(set.union(*shards), stored)
            self.assertEqual(sum(map(len, shards)), len(stored))
            cache.close()

    def test_process_group_timeout_is_thirty_minutes(self):
        self.assertEqual(Trainer._process_group_kwargs().timeout, timedelta(minutes=30))


if __name__ == "__main__":
    unittest.main()
