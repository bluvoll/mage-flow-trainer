"""Large-dataset plumbing: subset manifests, half-precision latent caches, one-process caching."""

import argparse
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from trainer.data.cache import LatentCacher, cache_path, load_cached_latent, storable
from trainer.data.dataset import DatasetConfig, load_subsets_file


class FakeVAE(torch.nn.Module):
    """Deterministic stand-in for Mage-VAE: 128 channels at 1/16 resolution."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def encode(self, x):
        pooled = torch.nn.functional.avg_pool2d(x.float(), 16)
        return (pooled.repeat(1, 43, 1, 1)[:, :128] * self.scale).to(x.dtype)


def write_images(folder: Path, n=3, size=(64, 64), caption="1girl, Drawn by emily"):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", size, (40 * i, 80, 120)).save(folder / f"img{i}.png")
        (folder / f"img{i}.txt").write_text(caption, encoding="utf-8")


class SubsetsFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        for name in ("emily", "alice", "empty"):
            (self.root / "images" / name).mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_studio_csv_reads_as_written(self):
        csv = self.root / "folders.csv"
        csv.write_text(
            "site,artist,tag,folder,path,images,planned_images,repeats\n"
            f"danbooru,emily,emily_(pure_dream),emily,{self.root / 'images/emily'},60,60,3\n"
            f"e621,alice,alice,alice,{self.root / 'images/alice'},40,60,5\n"
            f"danbooru,empty,empty,empty,{self.root / 'images/empty'},0,60,0\n",
            encoding="utf-8")
        subsets = load_subsets_file(csv)
        self.assertEqual([(Path(s.path).name, s.num_repeats) for s in subsets],
                         [("emily", 3), ("alice", 5)])
        cfg = DatasetConfig(subsets_file=str(csv))
        self.assertEqual(len(cfg.effective_subsets()), 2)
        # Only the file path is part of the config record, not thousands of expanded rows.
        self.assertEqual(asdict(cfg)["subsets"], [])
        self.assertEqual(asdict(cfg)["subsets_file"], str(csv))

    def test_json_toml_relative_paths_and_root_remap(self):
        js = self.root / "folders.json"
        js.write_text(json.dumps([{"path": "images/emily", "repeats": 2},
                                  {"path": "images/alice", "texture": "false"}]))
        subsets = load_subsets_file(js)
        self.assertEqual(Path(subsets[0].path), self.root / "images/emily")
        self.assertFalse(subsets[1].texture)
        toml = self.root / "dataset.toml"
        toml.write_text('[[directory]]\npath = "/home/jovyan/lib/images/emily"\nnum_repeats = 4\n'
                        '[[directory]]\npath = "C:\\\\lib\\\\images\\\\alice"\n')
        remapped = load_subsets_file(toml, self.root / "images")
        self.assertEqual([Path(s.path) for s in remapped],
                         [self.root / "images/emily", self.root / "images/alice"])
        self.assertEqual(remapped[0].num_repeats, 4)

    def test_mask_path_column_trains_masked_folders(self):
        """The Studio's `mask_path` column: relative to the manifest (also when `root` re-roots the
        image folders), blank = unmasked, and the dataset attaches each image's mask by stem."""
        write_images(self.root / "images/emily", n=2, size=(64, 96))
        write_images(self.root / "images/alice", n=1)
        masks = self.root / "masks/emily"
        masks.mkdir(parents=True)
        for i in range(2):
            mask = Image.new("L", (32, 48), 255)
            mask.paste(0, (0, 24, 32, 48))           # a big banner over the bottom half
            mask.save(masks / f"img{i}.png")
        csv = self.root / "folders.csv"
        csv.write_text("path,num_repeats,images,mask_path\n"
                       "images/emily,1,2,masks/emily\nimages/alice,1,1,\n", encoding="utf-8")
        subsets = load_subsets_file(csv)
        self.assertEqual([s.mask_path for s in subsets], [str(self.root / "masks/emily"), None])
        remapped = load_subsets_file(csv, self.root / "elsewhere")
        self.assertEqual(remapped[0].mask_path, str(self.root / "masks/emily"))
        cfg = DatasetConfig(subsets_file=str(csv), source="encode", resolution=64, min_bucket_reso=64, max_bucket_reso=128,
                            bucket_reso_steps=16)
        self.assertTrue(cfg.masked)
        from trainer.data.dataset import MageFlowDataset
        dataset = MageFlowDataset(cfg)
        by_folder = {e.path.parent.name: e for e in dataset.entries}
        self.assertIsNone(by_folder["alice"].mask)
        weights = by_folder["emily"].mask
        self.assertEqual(float(weights[..., -1, :].max()), 0.0)    # the banner carries no loss
        self.assertEqual(float(weights[..., 0, :].min()), 1.0)     # the rest trains normally

    def test_exclusivity_and_duplicates(self):
        csv = self.root / "f.csv"
        csv.write_text(f"path\n{self.root / 'images/emily'}\n")
        with self.assertRaisesRegex(ValueError, "not both"):
            DatasetConfig(path=str(self.root), subsets_file=str(csv))
        # The manifest is read when folders are first needed, like any dataset path: a config
        # naming a file that only exists on the training machine still loads elsewhere.
        both = DatasetConfig(subsets=[{"path": str(self.root / "images/emily")}],
                             subsets_file=str(csv))
        with self.assertRaisesRegex(ValueError, "twice"):
            both.effective_subsets()
        elsewhere = DatasetConfig(subsets_file=str(self.root / "not-here.csv"))
        with self.assertRaisesRegex(FileNotFoundError, "subsets_file not found"):
            elsewhere.effective_subsets()
        with self.assertRaisesRegex(ValueError, "subsets_root only"):
            DatasetConfig(path=str(self.root), subsets_root=str(self.root))
        bad = self.root / "bad.csv"
        bad.write_text("folder\nx\n")
        with self.assertRaisesRegex(ValueError, "path"):
            load_subsets_file(bad)


class LatentDtypeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_float16_cache_is_half_size_and_loads_as_float32(self):
        write_images(self.root / "a", n=2)
        write_images(self.root / "b", n=2)
        mgr = DatasetConfig(path=str(self.root / "a"), resolution=64).build_bucket_manager()
        full = LatentCacher(FakeVAE(), device="cpu", dtype=torch.float32)
        half = LatentCacher(FakeVAE(), device="cpu", dtype=torch.float32,
                            storage_dtype=torch.float16)
        out32, _ = full.cache_image(self.root / "a/img0.png", mgr)
        out16, _ = half.cache_image(self.root / "b/img0.png", mgr)
        self.assertLess(out16.stat().st_size, out32.stat().st_size * 0.6)
        self.assertEqual(load_cached_latent(out16).dtype, torch.float32)
        torch.testing.assert_close(load_cached_latent(out16), load_cached_latent(out32),
                                   rtol=1e-3, atol=1e-3)
        written = half.cache_batch([self.root / "b/img1.png"], mgr)
        self.assertEqual(len(written), 1)

    def test_overflow_falls_back_to_float32(self):
        big = torch.full((4,), 1e6)
        self.assertEqual(storable(big, torch.float16).dtype, torch.float32)
        self.assertEqual(storable(big / 1e6, torch.float16).dtype, torch.float16)

    def test_convert_rewrites_in_place(self):
        from trainer.tools.cache_latents import cmd_convert
        write_images(self.root / "a", n=2)
        mgr = DatasetConfig(path=str(self.root / "a"), resolution=64).build_bucket_manager()
        cacher = LatentCacher(FakeVAE(), device="cpu", dtype=torch.float32)
        paths = [cacher.cache_image(p, mgr)[0] for p in sorted((self.root / "a").glob("*.png"))]
        before = [load_cached_latent(p) for p in paths]
        sizes = [p.stat().st_size for p in paths]
        args = argparse.Namespace(path=[str(self.root / "a")], config=None, dtype="float16",
                                  dry_run=False)
        self.assertEqual(cmd_convert(args), 0)
        for p, ref, size in zip(paths, before, sizes):
            self.assertLess(p.stat().st_size, size)
            torch.testing.assert_close(load_cached_latent(p), ref, rtol=1e-3, atol=1e-3)
        self.assertFalse(list((self.root / "a").glob("*.tmp")))

    def test_cache_config_covers_manifest_and_eval_in_one_call(self):
        from trainer.tools import cache_latents
        for name in ("emily", "alice", "heldout"):
            write_images(self.root / name)
        csv = self.root / "folders.csv"
        csv.write_text(f"path,repeats\n{self.root / 'emily'},2\n{self.root / 'alice'},1\n")
        model = self.root / "model"
        (model / "transformer").mkdir(parents=True)
        cfg = self.root / "run.toml"
        cfg.write_text(f"""
[train]
model_path = "{model.as_posix()}"
[dataset]
subsets_file = "{csv.as_posix()}"
resolution = 64
latent_dtype = "float16"
[eval]
path = "{(self.root / 'heldout').as_posix()}"
every_n_steps = 10
""")
        seen = []
        args = argparse.Namespace(config=str(cfg), resolution=None, batch_size=2, devices="",
                                  device="cpu", latent_dtype=None, overwrite=False,
                                  dry_run=True, allow_missing_captions=False, shard_index=0,
                                  num_shards=1)
        with patch.object(cache_latents, "cmd_cache", side_effect=lambda a: seen.append(a) or 0):
            self.assertEqual(cache_latents.cmd_cache_config(args), 0)
        self.assertEqual(len(seen), 1)
        self.assertEqual([Path(p).name for p in seen[0].path], ["emily", "alice", "heldout"])
        self.assertEqual(seen[0].latent_dtype, "float16")
        self.assertEqual(seen[0].resolution, [64])
        # And the real planner accepts the folder list as one dry run.
        self.assertEqual(cache_latents.cmd_cache(seen[0]), 0)

    def test_latents_only_folders_are_not_an_error(self):
        from trainer.tools import cache_latents
        write_images(self.root / "a", n=1)
        mgr = DatasetConfig(path=str(self.root / "a"), resolution=64).build_bucket_manager()
        LatentCacher(FakeVAE(), device="cpu").cache_image(self.root / "a/img0.png", mgr)
        (self.root / "a/img0.png").unlink()
        args = argparse.Namespace(path=[str(self.root / "a")], batch_size=1, devices="",
                                  num_shards=1, shard_index=0)
        self.assertEqual(cache_latents.cmd_cache(args), 0)
        self.assertTrue(cache_path(self.root / "a/img0.png", (64, 64)).exists())


if __name__ == "__main__":
    unittest.main()
