"""Painted loss masks: per-image control over where the flow loss looks.

Each training image may have a grayscale mask in a separate folder, matched by filename stem:
white means "learn this region", black means "do not", and grey is a soft weight in between -- the
convention most mask editors and segmentation tools export. A character LoRA then learns the
character without also learning the backgrounds it happened to be drawn on.

The mask never touches the forward pass. The whole image -- background included -- still
conditions the model; only the *target* is weighted. That is the difference from cropping: the
model keeps seeing the subject in context, but the background pixels stop being something it is
asked to reproduce.

Three properties are load-bearing:
  * the mask is cropped with exactly the image's own geometry (`select_bucket` then
    `center_crop_box`, the pair `load_and_crop` uses), so a cached latent and its mask line up
    without re-encoding anything and without storing the mask in the cache;
  * grow and feather run at *bucket* resolution, so their pixel units mean image pixels rather
    than latent cells, and only then is the mask reduced to the 16x latent grid by AREA -- a latent
    cell that is a quarter covered gets a quarter weight instead of whatever one sampled pixel said;
  * `outside_weight` is a lerp, not a max: masks are feathered, and a max would step across the
    soft edge and train exactly the seam the feathering exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
LATENT_SCALE = 16


def center_crop_box(resized: tuple[int, int], bucket: tuple[int, int]) -> tuple[int, int, int, int]:
    """The centre crop that takes a `resized` image onto its `bucket`, as (l, t, r, b)."""
    left = (resized[0] - bucket[0]) // 2
    top = (resized[1] - bucket[1]) // 2
    return left, top, left + bucket[0], top + bucket[1]


def find_masks(root: str | Path) -> dict[str, Path]:
    """`stem -> mask file` for the top level of `root`, like the image scan.

    Two files sharing a stem (`a.png` and `a.jpg`) are refused rather than resolved by sort order:
    which one wins would be an accident of the filesystem, and the loser would be silently ignored.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"mask folder not found: {root}")
    found: dict[str, Path] = {}
    for p in sorted(root.iterdir()):
        if not p.is_file() or p.suffix.lower() not in MASK_EXTENSIONS:
            continue
        if p.stem in found:
            raise ValueError(f"{root}: two masks share the stem {p.stem!r} "
                             f"({found[p.stem].name}, {p.name}); keep one")
        found[p.stem] = p
    return found


def grow_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Dilate (px > 0) or erode (px < 0) a [0, 1] mask by roughly `px` pixels.

    A mask painted tight to a silhouette leaves the edge pixels half-supervised after the latent
    reduction; a few pixels of slack hands them back. Rank filters are O(n * k^2), so the radius
    is applied in chunks of at most 4 to keep each kernel small.
    """
    px = int(round(px))
    if px == 0:
        return mask
    img = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    rank = ImageFilter.MaxFilter if px > 0 else ImageFilter.MinFilter
    remaining = abs(px)
    while remaining > 0:
        step = min(4, remaining)
        img = img.filter(rank(2 * step + 1))
        remaining -= step
    return np.asarray(img, dtype=np.float32) / 255.0


def feather_mask(mask: np.ndarray, px: float) -> np.ndarray:
    """Gaussian-blur the mask edge. A hard edge between "learn" and "ignore" trains as a seam."""
    if px <= 0:
        return mask
    img = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(float(px))), dtype=np.float32) / 255.0


def load_mask(
    path: str | Path,
    source_size: tuple[int, int],
    bucket: tuple[int, int],
    resized: tuple[int, int],
    *,
    dilate_px: int = 0,
    feather_px: float = 0.0,
) -> torch.Tensor:
    """Mask coverage for one (image, bucket): (1, bucket_h/16, bucket_w/16) in [0, 1].

    `source_size` is the image's size; the mask may be a different resolution of the same frame (a
    half-size export, say) but not a different frame, so its aspect ratio must match.
    """
    with Image.open(path) as im:
        mask = im.convert("L")
    w, h = source_size
    if abs(mask.width * h - mask.height * w) > 0.01 * mask.height * w:
        raise ValueError(
            f"{path}: mask is {mask.width}x{mask.height} but its image is {w}x{h}. A mask must be "
            f"the image's own frame (same aspect ratio) to be cropped like it."
        )
    mask = mask.resize(resized, Image.BILINEAR).crop(center_crop_box(resized, bucket))
    array = np.asarray(mask, dtype=np.float32) / 255.0
    array = feather_mask(grow_mask(array, dilate_px), feather_px)
    bw, bh = bucket
    cells = torch.from_numpy(np.ascontiguousarray(array))
    cells = cells.view(bh // LATENT_SCALE, LATENT_SCALE, bw // LATENT_SCALE, LATENT_SCALE)
    return cells.mean(dim=(1, 3)).clamp(0.0, 1.0).unsqueeze(0)


def loss_weights(coverage: torch.Tensor, outside_weight: float) -> torch.Tensor:
    """Coverage -> per-cell loss weight: 1 inside the mask, `outside_weight` outside, lerped."""
    return outside_weight + coverage * (1.0 - outside_weight)
