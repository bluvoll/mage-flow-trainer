"""Mage-VAE latent caches: 128 channels, 16x downsampling, no latent normalization."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from .bucket import BucketManager
from .masks import center_crop_box

Image.MAX_IMAGE_PIXELS = None  # dataset images legitimately exceed the decompression-bomb guard

CACHE_SUFFIX = "_trainer.safetensors"
CACHE_VERSION = "mageflow-1"

# `<stem>_WWWWxHHHH_trainer.safetensors` -- the bucket is recoverable from the name alone, which is
# what lets a dataset run from latents after its source images have been deleted.
_CACHE_RE = re.compile(rf"^(?P<stem>.+)_(?P<w>\d{{4,}})x(?P<h>\d{{4,}}){re.escape(CACHE_SUFFIX)}$")


@dataclass
class CachedLatent:
    latents: torch.Tensor          # (128, 1, H/16, W/16)
    bucket: tuple[int, int]
    original_size: tuple[int, int]
    crop_ltrb: tuple[int, int, int, int]


def cache_path(image_path: str | Path, bucket: tuple[int, int]) -> Path:
    p = Path(image_path)
    return p.with_name(f"{p.stem}_{bucket[0]:04d}x{bucket[1]:04d}{CACHE_SUFFIX}")


def load_and_crop(image_path: str | Path, bucket_mgr: BucketManager) -> tuple[Image.Image, dict]:
    """Resize to the bucket's aspect then centre-crop. Never pads."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    bucket, resized, ar_error = bucket_mgr.select_bucket(w, h)

    if bucket[0] <= 0 or bucket[1] <= 0:
        raise ValueError(f"{image_path}: degenerate bucket {bucket} for source {w}x{h}")

    # The box is shared with `masks.load_mask`, which must crop a painted mask exactly like this.
    box = center_crop_box(resized, bucket)
    img = img.resize(resized, Image.LANCZOS).crop(box)

    return img, {
        "bucket": bucket,
        "original_size": (w, h),
        "crop_ltrb": box,
        "ar_error": ar_error,
    }


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> (1, 3, 1, H, W) in [-1, 1], the dataset image layout."""
    # np.array (not asarray): PIL exposes a read-only buffer, and torch.from_numpy on a
    # non-writable array warns and yields a tensor with undefined write semantics.
    x = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)
    x = x.float().div_(127.5).sub_(1.0)
    return x.unsqueeze(0).unsqueeze(2)


class LatentCacher:
    """Encode Mage-VAE posterior means without additional scaling or normalization."""

    def __init__(self, vae, device: torch.device | str = "cuda", dtype: torch.dtype = torch.bfloat16, flux2_vae: bool = False, provenance: dict | None = None, cache_validation: str = "warn"):
        self.vae = vae.to(device).eval()
        self.device = torch.device(device)
        self.dtype = dtype
        self.flux2_vae = flux2_vae
        self.provenance = (dict(provenance, encode_dtype=str(next(self.vae.parameters()).dtype).removeprefix("torch."))
                           if provenance is not None else None)
        self.cache_validation = cache_validation
    @torch.no_grad()
    def encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,1,H,W] pixels -> [B,128,1,H/16,W/16], without rescaling."""
        if x.ndim != 5 or x.shape[2] != 1:
            raise ValueError("Mage-VAE only supports image batches")
        encoded = self.vae.encode(x.squeeze(2).to(self.device, next(self.vae.parameters()).dtype))
        latents = encoded.latent_dist.mean if hasattr(encoded, "latent_dist") else encoded
        if self.flux2_vae and not getattr(self.vae, "outputs_packed_flux2", False):
            latents = torch.nn.functional.pixel_unshuffle(latents, 2)
            bn = getattr(self.vae, "bn", None)
            if bn is None: raise ValueError("FLUX.2 VAE must expose bn running statistics")
            latents = (latents - bn.running_mean[None, :, None, None].to(latents)) / (bn.running_var[None, :, None, None].to(latents) + bn.eps).sqrt()
        return latents.float().unsqueeze(2)

    def encode(self, img: Image.Image) -> torch.Tensor:
        return self.encode_tensor(image_to_tensor(img)).squeeze(0).to(torch.float32)  # (128, 1, h, w)

    def _metadata(self, image_path, meta):
        from .cache_provenance import file_sha256
        result = {
            "version": CACHE_VERSION,
            "bucket": f"{meta['bucket'][0]}x{meta['bucket'][1]}",
            "original_size": f"{meta['original_size'][0]}x{meta['original_size'][1]}",
            "crop_ltrb": ",".join(str(v) for v in meta["crop_ltrb"]),
            "source": hashlib.sha256(str(Path(image_path).name).encode()).hexdigest()[:16],
            "source_sha256": file_sha256(image_path),
        }
        if self.provenance is not None:
            result["cache_provenance"] = json.dumps(dict(self.provenance,
                encode_dtype=str(next(self.vae.parameters()).dtype).removeprefix('torch.')), sort_keys=True)
        return result

    def cache_image(self, image_path, bucket_mgr, overwrite=False):
        _, meta = load_and_crop(image_path, bucket_mgr)
        self.cache_batch([image_path], bucket_mgr, overwrite)
        return cache_path(image_path, meta["bucket"]), meta

    def cache_batch(self, image_paths, bucket_mgr, overwrite=False):
        """Cache same-bucket images; validate provenance before reusing any file."""
        from .cache_provenance import validate_cache, atomic_save
        prepared = []
        for image_path in image_paths:
            img, meta = load_and_crop(image_path, bucket_mgr)
            out = cache_path(image_path, meta["bucket"])
            if not overwrite and out.exists():
                validate_cache(out, self.provenance, self.cache_validation, source=image_path)
            else:
                prepared.append((Path(image_path), out, meta, image_to_tensor(img)))
        if not prepared:
            return []
        latents = self.encode_tensor(torch.cat([row[3] for row in prepared], dim=0))
        written = []
        for (image_path, out, meta, _), latent in zip(prepared, latents):
            atomic_save({"latents": latent.contiguous()}, out, self._metadata(image_path, meta))
            written.append((out, meta))
        return written


def load_cached_latent(path: str | Path) -> torch.Tensor:
    tensor = load_file(path)["latents"]
    parsed = parse_cache_filename(path)
    if tensor.ndim != 4 or tensor.shape[:2] != (128, 1):
        raise ValueError(f"{path}: expected Mage-VAE latents [128,1,h,w]; rebuild cache")
    if parsed and tensor.shape[-2:] != (parsed[1][1] // 16, parsed[1][0] // 16):
        raise ValueError(f"{path}: latent dimensions do not match the cache bucket")
    return tensor


def read_cache_metadata(path: str | Path) -> dict[str, str]:
    """Header-only read -- the tensor is never touched.

    `original_size` is what makes latents-only multi-resolution possible: with the source image
    gone, it is the only way to re-derive which tier explains a given cached bucket. It has been
    written since CACHE_VERSION 1, but callers must tolerate its absence rather than assume.
    """
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        return f.metadata() or {}


def parse_original_size(meta: dict[str, str]) -> tuple[int, int] | None:
    raw = meta.get("original_size")
    if not raw or "x" not in raw:
        return None
    w, _, h = raw.partition("x")
    try:
        return int(w), int(h)
    except ValueError:
        return None


def is_cached(image_path: str | Path, bucket: tuple[int, int]) -> bool:
    return cache_path(image_path, bucket).exists()


def parse_cache_filename(path: str | Path) -> tuple[str, tuple[int, int]] | None:
    """`<stem>_1024x0576_trainer.safetensors` -> ("<stem>", (1024, 576)). None if it doesn't match."""
    m = _CACHE_RE.match(Path(path).name)
    if not m:
        return None
    return m.group("stem"), (int(m.group("w")), int(m.group("h")))


def find_cached_latents(root: str | Path) -> dict[str, list[tuple[Path, tuple[int, int]]]]:
    """Map stem -> [(cache_path, bucket)]. A stem can have several if bucket config changed."""
    out: dict[str, list[tuple[Path, tuple[int, int]]]] = {}
    for p in sorted(Path(root).glob(f"*{CACHE_SUFFIX}")):
        parsed = parse_cache_filename(p)
        if parsed:
            stem, bucket = parsed
            out.setdefault(stem, []).append((p, bucket))
    return out


@dataclass
class CacheAudit:
    trainable: list[str]            # cache + caption both present
    missing_caption: list[str]      # cached but no .txt -- would be dropped
    uncached_images: list[str]      # image present with no cache -- would be lost on delete
    multi_bucket: dict[str, list[tuple[int, int]]]
    total_cache_bytes: int
    total_image_bytes: int

    @property
    def safe_to_delete_images(self) -> bool:
        return not self.uncached_images and not self.missing_caption


def audit_cache(root: str | Path, image_extensions: tuple[str, ...]) -> CacheAudit:
    """Check a directory before its images get deleted.

    Deleting source images is irreversible, so this reports exactly what would survive: any image
    without a cache, or any cache without a caption, is data that silently vanishes from training.
    """
    root = Path(root)
    cached = find_cached_latents(root)

    images = {
        p.stem: p
        for p in root.iterdir()
        if p.suffix.lower() in image_extensions and not p.name.endswith(CACHE_SUFFIX)
    }

    trainable, missing_caption = [], []
    for stem in cached:
        if (root / f"{stem}.txt").exists():
            trainable.append(stem)
        else:
            missing_caption.append(stem)

    uncached = sorted(set(images) - set(cached))
    multi = {s: [b for _, b in v] for s, v in cached.items() if len(v) > 1}

    cache_bytes = sum(p.stat().st_size for v in cached.values() for p, _ in v)
    image_bytes = sum(p.stat().st_size for p in images.values())

    return CacheAudit(
        trainable=sorted(trainable),
        missing_caption=sorted(missing_caption),
        uncached_images=uncached,
        multi_bucket=multi,
        total_cache_bytes=cache_bytes,
        total_image_bytes=image_bytes,
    )
