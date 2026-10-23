# Masked training

Painted loss masks control **where the flow loss looks** for each image. A character LoRA trained
on fan art otherwise learns every background the character was drawn on; with a mask over the
character, the background still conditions the model as context but is no longer a target.

This is ordinary LoRA / LyCORIS / full-finetune training with a per-pixel loss weight, not a
different objective.

## Masks

- One grayscale image per training image, in its own folder, matched by **filename stem**:
  `/data/character/img_001.png` pairs with `/data/character_masks/img_001.png` (any of
  `.png .jpg .jpeg .webp .bmp`).
- **White = learn, black = do not, grey = a soft weight in between.** Colour images are converted
  to grayscale; alpha is ignored.
- A mask must be the image's own frame: the same aspect ratio (any resolution). It is resized and
  centre-cropped with exactly the image's bucket geometry, so it lines up with cached latents
  without re-caching anything.
- Once a mask folder is set, **every image needs a mask**; the run refuses to start and names the
  missing stems. Paint an image fully white to train it unmasked.

## Config

```toml
[dataset]
path = "/data/character"
mask_path = "/data/character_masks"
mask_outside_weight = 0.0   # loss weight where the mask is black; 1.0 = no masking
mask_dilate_px = 0          # grow (negative: shrink) masks, in bucket pixels
mask_feather_px = 0.0       # Gaussian-soften mask edges, in bucket pixels
use_masks = true            # false trains unmasked without clearing the folders
```

With subsets, each subset names its own folder; subsets without one train at uniform weight:

```toml
[[dataset.subsets]]
path = "/data/character"
mask_path = "/data/character_masks"

[[dataset.subsets]]
path = "/data/regularization"
texture = false
```

In the GUI these are under **Dataset → Loss Masks**, and the subset table has a **Masks** column.

A `dataset.subsets_file` takes a `mask_path` column the same way (blank = unmasked; a relative path
resolves against the file's own folder). Illustration Scrapping Studio's training export fills it in:
it writes `masks/<group>/<artist>/` with black boxes over each image's watermarks, signatures and URLs
(from the captioner's `_moss.json` / `_layout.json` boxes) for every artist folder that has any, so
a watermark most of an artist's images share stops sticking to the style.

## How it is applied

1. The mask is resized/cropped like its image, then grown and feathered at bucket resolution, so
   the pixel units mean image pixels.
2. It is reduced to the 16x latent grid by area: a latent cell a quarter covered gets weight 0.25.
3. Coverage becomes a loss weight, `outside + coverage * (1 - outside)`.
4. The flow loss is the weight-normalised mean `sum(w * err^2) / sum(w)`: per batch on the bucketed
   path, per sample (then averaged) with `pack_resolutions`. The HF token loss uses the same mask.

Masks are built once at startup and kept in memory (a 1024x1024 bucket is a 64x64 map), so they add
nothing per step and work with `source = "images"`, `"latents"` (from the cache's
`original_size` metadata) and `"encode"`.

## Tips

- `mask_outside_weight = 0` is classic masked loss. A small value (0.1-0.3) keeps light
  supervision on the background if fully ignoring it lets backgrounds drift.
- A few pixels of `mask_dilate_px` with about half that in `mask_feather_px` keeps a tight
  silhouette's edge supervised without training a hard seam.

## Not supported together

Painted masks weight the per-cell flow loss only. The loader refuses them with paired texture loss,
Self-Flow, decoded perceptual loss and the eye adversarial loss, and with texture-crop curriculum
phases on a masked folder (texture crops are cut per step and carry their own mask; list masked
folders as subsets with `texture = false`).
