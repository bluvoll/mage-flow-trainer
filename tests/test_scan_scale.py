"""Dataset scanning at per-artist scale: threaded scan equals sequential, rank-0 scan is shared."""

import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from trainer.data.cache import LatentCacher
from trainer.data.dataset import DatasetConfig, MageFlowDataset


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def encode(self, x):
        return torch.nn.functional.avg_pool2d(x.float(), 16).repeat(1, 43, 1, 1)[:, :128]


def build_folders(root: Path, n_folders=12, per_folder=3, cache=True, delete_images=False):
    cacher = LatentCacher(FakeVAE(), device="cpu", dtype=torch.float32)
    rows = ["path,repeats"]
    for f in range(n_folders):
        folder = root / f"artist{f:03d}"
        folder.mkdir(parents=True)
        mgr = DatasetConfig(path=str(folder), resolution=128).build_bucket_manager()
        for i in range(per_folder):
            size = (128, 128) if (i + f) % 3 else (128, 64)
            p = folder / f"img{i}.png"
            Image.new("RGB", size, (f * 7 % 255, i * 40, 90)).save(p)
            (folder / f"img{i}.txt").write_text(f"Drawn by artist{f}, tag{i}", encoding="utf-8")
            if cache:
                cacher.cache_image(p, mgr)
            if delete_images:
                p.unlink()
        rows.append(f"{folder},{1 + f % 3}")
    manifest = root / "folders.csv"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def snapshot(ds):
    return ([(str(e.path), e.bucket, e.tags, e.resolution, e.texture_ok) for e in ds.entries],
            [(s.path, n) for s, n in ds.subset_report])


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def scan(self, manifest, workers, **kw):
        with patch.dict(os.environ, {"MAGEFLOW_SCAN_WORKERS": str(workers)}):
            return MageFlowDataset(DatasetConfig(subsets_file=str(manifest), resolution=128, **kw))

    def test_threaded_scan_is_identical_to_sequential(self):
        manifest = build_folders(self.root)
        seq = self.scan(manifest, 1)
        par = self.scan(manifest, 8)
        self.assertEqual(snapshot(seq), snapshot(par))
        self.assertEqual(len(seq), sum(3 * (1 + f % 3) for f in range(12)))

    def test_threaded_latents_only_scan(self):
        manifest = build_folders(self.root, delete_images=True)
        self.assertEqual(snapshot(self.scan(manifest, 1)), snapshot(self.scan(manifest, 6)))

    def test_errors_surface_from_worker_threads(self):
        manifest = build_folders(self.root, n_folders=9, cache=False)
        with self.assertRaisesRegex(RuntimeError, "no cached latent"):
            self.scan(manifest, 4)

    def test_dataset_survives_the_broadcast_pickle(self):
        manifest = build_folders(self.root)
        ds = self.scan(manifest, 4)
        clone = pickle.loads(pickle.dumps(ds, protocol=pickle.HIGHEST_PROTOCOL))
        self.assertEqual(snapshot(ds), snapshot(clone))
        self.assertTrue(torch.equal(ds[(0, 0)]["latents"], clone[(0, 0)]["latents"]))


class SharedScanTests(unittest.TestCase):
    """`Trainer._shared_dataset` on a simulated two-rank job."""

    def run_rank(self, rank, build, sent=None):
        from trainer.training.train import Trainer

        trainer = Trainer.__new__(Trainer)
        trainer.accelerator = SimpleNamespace(num_processes=2, is_main_process=rank == 0)

        def broadcast(box):
            if rank == 0:
                sent.append(box[0])
            else:
                box[0] = sent[0]

        with patch("accelerate.utils.broadcast_object_list", side_effect=broadcast):
            return trainer._shared_dataset(build)

    def test_only_rank_zero_builds_and_both_get_equal_results(self):
        calls, sent = [], []

        def build():
            calls.append(1)
            return {"entries": list(range(5))}

        a = self.run_rank(0, build, sent)
        b = self.run_rank(1, lambda: self.fail("rank 1 must not scan"), sent)
        self.assertEqual(calls, [1])
        self.assertEqual(a, b)

    def test_a_rank_zero_failure_is_raised_on_every_rank(self):
        sent = []

        def broken():
            raise FileNotFoundError("dataset path not found: /nope")

        for rank in (0, 1):
            with self.assertRaisesRegex(RuntimeError, "dataset path not found"):
                self.run_rank(rank, broken, sent)


if __name__ == "__main__":
    unittest.main()
