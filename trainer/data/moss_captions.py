"""MOSS-Image caption annotations: captions at three lengths and layouts with boxes.

The MageTrail captioning script (moss-caption.py, in MOSS-Image's own caption format) writes next to each image:

    <stem>_moss.json    {"caption": {"long", "medium", "short"}, "watermark": [...], "borders": [...]}
    <stem>_layout.json  {"summary": {"meta", "canvas", "caption"},
                         "items": [{"label", "caption", "text"?, "tags"?, "box_2d"}], ...}

Boxes are [y_min, x_min, y_max, x_max] normalized to 0-1000 of the ORIGINAL image, so one file serves every
training resolution. `compile_layout` turns a layout into the prompt MOSS-Image was trained on for one training
crop: boxes moved into the bucket's pixels (the dataset scales an image to cover its bucket, then centre-crops),
items mostly cut off by the crop dropped, and coordinates written as MOSS's `abs_centralized` / `yxyx`, exactly
as its prompt compiler does (floor the top/left edge, ceil the bottom/right, subtract half the canvas).

`caption_mode` / `mixed_weights` select them as the variants `short`, `medium`, `long` and `layout`; a sample
without the file a variant needs falls back to its NL caption, then to its tags.

Besides MOSS's `character` / `object` / `text` items, MageTrail layouts carry `panel` items (a comic or collage
panel with a caption of what happens in it), an extension the fine-tune learns; `layout_panels = false` leaves
them out, and `layout_panels_as_objects` turns them into `object` items captioned "panel: ...", the way a box
node that only draws objects sends a panel. Character details (a ribbon, a hat, a worn weapon) are `object` items with `part_of` (the index of their
character): the model sees plain objects; `layout_parts = false` leaves them out.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

ANNOTATION_VARIANTS = ("short", "medium", "long", "layout")
LAYOUT_LABELS = ("character", "object", "text", "panel")   # "panel": MageTrail's extension
MIN_VISIBLE = 0.5         # an item keeps its box when at least this share of it survives the crop...
MIN_VISIBLE_TEXT = 0.9    # ...text needs nearly all of it, or the box would claim text that is cut off


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Annotations:
    """One sample's MOSS files, read lazily. `source_size` (the original image's width, height) and `bucket`
    are needed only for layouts; (0, 0) means unknown, which disables the layout variant for the sample."""

    def __init__(self, base: Path, source_size: tuple[int, int] = (0, 0), bucket: tuple[int, int] = (0, 0)):
        self.base = Path(base)
        self.source_size = tuple(source_size or (0, 0))
        self.bucket = tuple(bucket or (0, 0))
        self._moss = self._layout = None
        self._loaded = False

    def _load(self):
        if not self._loaded:
            self._moss = _read(self.base.with_name(self.base.name + "_moss.json"))
            self._layout = _read(self.base.with_name(self.base.name + "_layout.json"))
            self._loaded = True

    def caption(self, kind: str) -> str | None:
        self._load()
        caption = (self._moss or {}).get("caption") if isinstance(self._moss, dict) else None
        text = caption.get(kind) if isinstance(caption, dict) else None
        return " ".join(text.split()) if isinstance(text, str) and text.strip() else None

    def layout(self, summary_prefix: str = "", rng: random.Random | None = None, item_dropout: float = 0.0,
               panels: bool = True, short_share: float = 0.0, parts: bool = True,
               panels_as_objects: bool = False) -> str | None:
        self._load()
        if not isinstance(self._layout, dict) or not all(self.source_size) or not all(self.bucket):
            return None
        return compile_layout(self._layout, self.source_size, self.bucket, summary_prefix, rng, item_dropout, panels,
                              short_share, parts, panels_as_objects)

    def available(self, panels: bool = True, parts: bool = True) -> set[str]:
        out = {kind for kind in ("short", "medium", "long") if self.caption(kind)}
        if self.layout(panels=panels, parts=parts) is not None:
            out.add("layout")
        return out

    def fingerprint(self) -> str:
        """Changes when either file or the crop changes: part of the caption-variation cache key."""
        h = hashlib.sha1(repr((self.source_size, self.bucket)).encode())
        for suffix in ("_moss.json", "_layout.json"):
            try:
                h.update(self.base.with_name(self.base.name + suffix).read_bytes())
            except OSError:
                h.update(b"-")
        return h.hexdigest()


def crop_transform(source_size, bucket):
    """(scale, left, top) of the dataset's cover-resize + centre-crop, in resized pixels."""
    width, height = source_size
    bucket_w, bucket_h = bucket
    scale = bucket_h / height if width / height > bucket_w / bucket_h else bucket_w / width
    return scale, (width * scale - bucket_w) / 2, (height * scale - bucket_h) / 2


def drop_items(items: list, rng: random.Random, item_dropout: float) -> list:
    """Layout granularity: this sample leaves out a share of the items drawn between 0 and `item_dropout`, so
    the model sees layouts from complete to sparse. Order is kept and at least one item always stays."""
    if item_dropout <= 0 or len(items) < 2:
        return items
    share = rng.uniform(0.0, item_dropout)
    kept = [item for item in items if rng.random() >= share]
    return kept or [items[rng.randrange(len(items))]]


def panel_object_caption(caption: str) -> str:
    """A panel's caption as an object box: "panel: <caption>", unless its first words already call it a panel
    ("panel: the two girls hugging", "Left panel: Miku waves.")."""
    caption = caption.strip()
    if "panel" in " ".join(caption.lower().split()[:3]):
        return caption
    return f"panel: {caption}" if caption else "panel"


def compile_layout(layout: dict, source_size, bucket, summary_prefix: str = "",
                   rng: random.Random | None = None, item_dropout: float = 0.0, panels: bool = True,
                   short_share: float = 0.0, parts: bool = True, panels_as_objects: bool = False) -> str | None:
    """MOSS's model-facing layout prompt for one training crop, or None when no item survives the crop.
    With `item_dropout` (and an `rng`), a random share of the surviving items is left out (`drop_items`). With
    `short_share`, that share of samples gives every item that has one its `short` caption ("Marisa holding a
    spoon") instead of the full sentence, the way people prompt boxes. With `panels_as_objects`, panels become
    `object` items captioned "panel: ..." (`panel_object_caption`)."""
    use_short = rng is not None and short_share > 0 and rng.random() < short_share
    width, height = source_size
    bucket_w, bucket_h = bucket
    scale, left, top = crop_transform(source_size, bucket)
    items = []
    for item in layout.get("items") or []:
        box = item.get("box_2d") if isinstance(item, dict) else None
        if not isinstance(box, list) or len(box) != 4 or item.get("label") not in LAYOUT_LABELS:
            continue
        if item["label"] == "panel" and not panels:
            continue
        if "part_of" in item and not parts:
            continue
        y0, x0, y1, x1 = (box[0] / 1000 * height * scale - top, box[1] / 1000 * width * scale - left,
                          box[2] / 1000 * height * scale - top, box[3] / 1000 * width * scale - left)
        full = max(0.0, y1 - y0) * max(0.0, x1 - x0)
        cy0, cx0, cy1, cx1 = max(0.0, y0), max(0.0, x0), min(float(bucket_h), y1), min(float(bucket_w), x1)
        seen = max(0.0, cy1 - cy0) * max(0.0, cx1 - cx0)
        if not full or seen / full < (MIN_VISIBLE_TEXT if item["label"] == "text" else MIN_VISIBLE):
            continue
        bbox = [math.floor(round(cy0, 9)) - bucket_h // 2, math.floor(round(cx0, 9)) - bucket_w // 2,
                math.ceil(round(cy1, 9)) - bucket_h // 2, math.ceil(round(cx1, 9)) - bucket_w // 2]
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            continue
        caption = item.get("short") if use_short and item.get("short") else item.get("caption", "")
        label = item["label"]
        if label == "panel" and panels_as_objects:
            label, caption = "object", panel_object_caption(caption)
        out = {"label": label, "caption": caption}
        if item.get("text"):
            out["text"] = item["text"]
        out["bbox"] = bbox
        if item.get("tags") and item["label"] == "character":
            out["tags"] = list(item["tags"])
        items.append(out)
    if not items:
        return None
    if rng is not None:
        items = drop_items(items, rng, item_dropout)
    raw = layout.get("summary") or {}
    summary = {}
    if raw.get("meta"):
        summary["meta"] = list(raw["meta"])
    summary["canvas"] = raw.get("canvas") or "single"
    summary["caption"] = (summary_prefix + raw.get("caption", "")).strip()
    return json.dumps({"summary": summary, "items": items, "bbox_coord_space": "abs_centralized",
                       "bbox_format": "yxyx"}, ensure_ascii=False, separators=(",", ":"))


def uses_annotations(caption_cfg) -> bool:
    """Whether the caption config can pick a variant that needs the MOSS files at all."""
    if caption_cfg.caption_mode in ANNOTATION_VARIANTS:
        return True
    return caption_cfg.caption_mode == "mixed" and any(
        caption_cfg.mixed_weights.get(kind, 0) > 0 for kind in ANNOTATION_VARIANTS)


def annotations_for(entry, caption_cfg) -> Annotations | None:
    """The entry's annotations, or None when the config never uses them (no file reads for other datasets)."""
    if not uses_annotations(caption_cfg):
        return None
    from .cache import parse_cache_filename

    parsed = parse_cache_filename(entry.path)
    stem = parsed[0] if parsed else Path(entry.path).stem
    return Annotations(Path(entry.path).with_name(stem), getattr(entry, "source_size", (0, 0)), entry.bucket)
