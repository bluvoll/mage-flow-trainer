"""Dataset and aspect-ratio batch sampler.

Every sample in a batch must share a bucket -- latents of different shapes cannot be stacked -- so
batching is bucket-first: the sampler groups indices by bucket, then yields fixed-size batches
from within each bucket.

Per-bucket micro-batch sizes are supported because cost is not linear in resolution. Attention is
O(tokens^2), and tokens = W*H/256, so a 1536px bucket costs roughly 5x a 1024px one. A single
global batch size either wastes VRAM at low resolution or OOMs at high.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from .bucket import BucketManager, verify_max_resolution
from .caption import CaptionConfig, build_caption, read_caption_files
from .moss_captions import annotations_for
from .texture import TextureConfig, fitting_canvases
from .masks import LATENT_SCALE, find_masks, load_mask, loss_weights
from .cache import (
    CACHE_SUFFIX,
    cache_path,
    find_cached_latents,
    image_to_tensor,
    load_and_crop,
    load_cached_latent,
    parse_cache_filename,
    parse_original_size,
    read_cache_metadata,
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _mix(*values: int) -> int:
    """Fold ints into one seed. splitmix64's finalizer, so adjacent (epoch, index) pairs -- which
    is exactly what we feed it -- decorrelate instead of producing neighbouring RNG streams."""
    h = 0
    for v in values:
        h = (h * 0x9E3779B97F4A7C15 + (v & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 30
        h = (h * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 27
        h = (h * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 31
    return h


@dataclass
class SubsetConfig:
    """One source directory, with its own repeat count and texture eligibility.

    `texture = false` is the load-bearing field and the reason this exists. Texture crops replace
    the image's caption with `texture.trigger` (see `_texture_sample`), which is right for detail
    crops of a captioned subject and catastrophic for a regularization set: a flat colour field
    trained with an empty caption teaches the UNCONDITIONAL branch that colour, and CFG then
    subtracts it -- `pred = uncond + scale*(cond - uncond)` -- so colour anchors delivered through
    texture mode make generations less colourful, not more. Marking a subset `texture = false`
    keeps it in fullres for the whole run, captions intact, whatever the curriculum is doing.

    It is also the honest thing for flat images on their own terms: `energy_map` of a constant
    image is exactly zero, so `choose_crop` falls back to a uniform position. Texture mode exists
    to find native-scale high-frequency detail, and these have none by construction.
    """

    path: str
    num_repeats: int = 1
    texture: bool = True
    # Folder of painted loss masks for this subset's images, matched by filename stem. See
    # `DatasetConfig.mask_path`; unset leaves the subset unmasked.
    mask_path: str | None = None

    def __post_init__(self):
        if self.num_repeats < 1:
            raise ValueError(
                f"subset {self.path!r}: num_repeats must be >= 1, got {self.num_repeats}"
            )


def _scan_workers(n_subsets: int) -> int:
    """Threads for scanning subsets: 1 (the original sequential scan) for a handful of folders,
    up to 16 for per-artist datasets. MAGEFLOW_SCAN_WORKERS overrides; 1 disables threading."""
    import os

    override = os.environ.get("MAGEFLOW_SCAN_WORKERS")
    if override:
        return max(1, int(override))
    if n_subsets < 8:
        return 1
    return max(1, min(16, n_subsets, (os.cpu_count() or 4) * 2))


def _check_duplicate_subsets(subsets) -> None:
    seen = set()
    for s in subsets:
        if s.path in seen:
            raise ValueError(
                f"dataset.subsets lists {s.path!r} twice; use num_repeats to weight it instead"
            )
        seen.add(s.path)


_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off", ""}


def _manifest_bool(value, where: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"{where}: texture must be true/false, got {value!r}")


def load_subsets_file(path: str | Path, root: str | Path | None = None) -> list[SubsetConfig]:
    """Read a subset manifest: CSV, JSON or TOML.

    * CSV: a header row with `path`, optionally `num_repeats` (or `repeats`), `texture` and
      `mask_path` (the folder's painted loss masks; blank = unmasked). Other columns are ignored,
      so the Studio's `folders.csv` (site, artist, tag, folder, path, images, planned_images,
      repeats, mask_path) reads as-is.
    * JSON: a list of objects with the same keys (the Studio's `folders.json`), or
      `{"subsets": [...]}`.
    * TOML: `[[subsets]]` or diffusion-pipe `[[directory]]` tables (the Studio's `dataset.toml`).

    A relative `mask_path` resolves against the manifest's own folder, also with `root` (which
    re-roots only the image folders), so masks shipped next to the manifest are found wherever the
    dataset is downloaded.

    Rows whose repeats are 0 -- the Studio writes that for a folder emptied by curation -- or whose
    `images` column is 0 are skipped rather than rejected: they are folders with nothing to train.
    """
    import csv
    import json
    import tomllib

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"dataset.subsets_file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        with p.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows and "path" not in rows[0]:
            raise ValueError(f"{p}: CSV needs a `path` column, found {list(rows[0])}")
    elif suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data.get("subsets", data.get("directory")) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError(f"{p}: expected a JSON list of folders or {{\"subsets\": [...]}}")
    elif suffix == ".toml":
        data = tomllib.loads(p.read_text(encoding="utf-8"))
        rows = data.get("subsets") or data.get("directory") or data.get("dataset", {}).get("subsets")
        if not isinstance(rows, list):
            raise ValueError(f"{p}: expected [[subsets]] or [[directory]] tables")
    else:
        raise ValueError(f"{p}: dataset.subsets_file must be .csv, .json or .toml")

    out: list[SubsetConfig] = []
    for i, row in enumerate(rows):
        where = f"{p.name} row {i + 1}"
        if not isinstance(row, dict) or not str(row.get("path") or "").strip():
            raise ValueError(f"{where}: missing `path`")
        raw = row.get("num_repeats", row.get("repeats", 1))
        try:
            repeats = int(float(raw)) if str(raw).strip() != "" else 1
            images = row.get("images")
            empty = images is not None and str(images).strip() != "" and int(float(images)) == 0
        except ValueError:
            raise ValueError(f"{where}: repeats/images must be numbers, got {raw!r}") from None
        if repeats == 0 or empty:
            continue
        folder = Path(str(row["path"]).strip())
        if root:
            # Manifest written elsewhere: keep the folder name, swap everything above it. Split on
            # both separators so a Windows-written path re-roots correctly on Linux and vice versa.
            folder = Path(root) / str(row["path"]).strip().replace("\\", "/").rstrip("/").split("/")[-1]
        elif not folder.is_absolute():
            folder = p.parent / folder
        texture = _manifest_bool(row.get("texture", True), where)
        mask = str(row.get("mask_path") or "").strip()
        if mask and not Path(mask).is_absolute():
            mask = str(p.parent / mask)
        out.append(SubsetConfig(path=str(folder), num_repeats=repeats, texture=texture,
                                mask_path=mask or None))
    if not out:
        raise ValueError(f"{p}: no folders with images listed")
    return out


@dataclass
class DatasetConfig:
    # Exactly one of `path` / `subsets`. `path` is the single-directory form and stays the default
    # so every existing config is untouched; `subsets` is the multi-source form.
    path: str | None = None
    subsets: list[SubsetConfig] = field(default_factory=list)
    # Subsets listed in a file instead of `[[dataset.subsets]]` blocks -- for datasets with
    # thousands of folders (one per artist), where inline tables make the TOML unreadable and the
    # GUI's subset grid unusable. Accepts Illustration Scrapping Studio's training-layout export
    # (`folders.csv` / `folders.json` / `dataset.toml`) as written, or any CSV/JSON with a `path`
    # column and optional `num_repeats` (or `repeats`) and `texture`. Relative paths resolve
    # against the file's folder. Combines with inline `subsets`; excludes `path`.
    subsets_file: str | None = None
    # Re-root every `subsets_file` folder under this directory, keeping only its folder name. For
    # a manifest written on another machine: `/home/jovyan/lib/images/emily` becomes
    # `<subsets_root>/emily`.
    subsets_root: str | None = None
    resolution: int = 1024                 # AREA budget: max_reso = (r, r)
    # Multi-resolution: train every image at every listed area budget. Unset -> [resolution], which
    # is bit-identical to single-tier behaviour. A tier is a repeat with a different resolution, so
    # N tiers cost N x the epoch and N x the cache.
    resolutions: list[int] | None = None
    # What to do when two tiers produce the same bucket for an image. See `_scan_images`.
    tier_collapse: str = "dedup"           # "dedup" | "repeat"
    # Drop sources smaller than this fraction of the SMALLEST tier's area budget. 0.0 = keep all.
    # Mage-Flow has no size micro-conditioning, so a tiny source contributes low-token supervision the
    # model cannot attribute to the source being small -- unlike SDXL, which conditions on
    # original_size for exactly this reason.
    min_source_area: float = 0.0
    min_bucket_reso: int = 256
    # User-selected per-side cap, independent of the resolution area budget.
    max_bucket_reso: int = 1920
    bucket_reso_steps: int = 64
    bucket_no_upscale: bool = True
    multires_training: bool = False
    num_repeats: int = 1
    caption: CaptionConfig = field(default_factory=CaptionConfig)

    # Painted loss masks (`masks.py`): a grayscale image per training image, matched by filename
    # stem, white = learn this region. `mask_path` pairs with `path`; with `subsets`, each subset
    # names its own. Once a folder is set every image in it needs a mask -- a stem mismatch that
    # silently left most of a dataset unmasked would look exactly like a run that worked.
    mask_path: str | None = None
    # Off trains unmasked without clearing the folders, for an A/B against the same config.
    use_masks: bool = True
    # Loss weight where the mask is black. 0 ignores those pixels entirely; 1 is unmasked training.
    mask_outside_weight: float = 0.0
    # Grow (negative: shrink) and Gaussian-soften every mask, in bucket pixels, before it is reduced
    # to the latent grid. Slack around a tight silhouette keeps its edge cells fully supervised.
    mask_dilate_px: int = 0
    mask_feather_px: float = 0.0

    # Train from cached latents alone, with the source images deleted. A sample is then
    # `<stem>_WWWWxHHHH_trainer.safetensors` + `<stem>.txt` (+ optional `<stem>_nl.txt`); buckets
    # come from the cache filenames rather than from measuring images. "auto" uses latents-only
    # when the directory contains caches but no images, so a run keeps working after a cleanup.
    # "encode" is the odd one out: it buckets from images exactly like "images", but yields *pixels*
    # and has the trainer run the VAE every step instead of reading a cache. It exists for modes
    # where the crop is chosen per step and so cannot be precomputed (see the texture curriculum);
    # for ordinary training it is strictly worse -- a VAE forward every step, forever, plus the VAE
    # resident in VRAM, to reproduce a tensor that never changes.
    cache_validation: str = "warn"  # warn permits legacy caches; strict requires provenance; off bypasses
    source: str = "auto"  # "images" | "latents" | "encode" | "auto"
    # Storage precision for latents cached from this config (`cache_latents cache-config`, the
    # GUI's Cache button). float16 halves the cache on disk; training reads either. Existing
    # caches keep their precision -- `cache_latents convert` rewrites them in place.
    latent_dtype: str = "float32"  # "float32" | "float16" | "bfloat16"

    # Only read when a curriculum phase has `mode = "texture"`. Nested under [dataset.texture].
    texture: TextureConfig = field(default_factory=TextureConfig)

    def __post_init__(self):
        if self.cache_validation not in ("warn", "strict", "off"):
            raise ValueError("dataset.cache_validation must be warn, strict or off")
        self.subsets = [
            s if isinstance(s, SubsetConfig) else SubsetConfig(**s) for s in self.subsets
        ]
        if self.subsets_root and not self.subsets_file:
            raise ValueError("dataset.subsets_root only applies to dataset.subsets_file")
        # The manifest is read on first use (`effective_subsets`), like every other dataset path:
        # a config naming a folder that is not on this machine yet still loads, which is what
        # lets configs travel between the laptop and the training box. Its rows are also kept off
        # the dataclass fields -- the config is embedded in every checkpoint's metadata, and ten
        # thousand expanded rows would be megabytes of it; the file path is recorded instead.
        self._manifest = None
        if bool(self.path) == bool(self.subsets or self.subsets_file):
            raise ValueError(
                "dataset needs exactly one of `path` (single directory) or `subsets` / "
                "`subsets_file` (a list of folders), not "
                + ("both" if self.path else "neither")
            )
        _check_duplicate_subsets(self.subsets)
        if self.latent_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"dataset.latent_dtype must be float32, float16 or bfloat16, "
                             f"got {self.latent_dtype!r}")
        if self.mask_path and self.subsets:
            raise ValueError(
                "dataset.mask_path pairs with dataset.path; with [[dataset.subsets]] set mask_path "
                "on each subset instead"
            )
        if not 0.0 <= self.mask_outside_weight <= 1.0:
            raise ValueError(
                f"dataset.mask_outside_weight must be in [0, 1], got {self.mask_outside_weight}"
            )
        if type(self.mask_dilate_px) is not int:
            raise ValueError("dataset.mask_dilate_px must be an integer number of pixels")
        if self.mask_feather_px < 0.0:
            raise ValueError(f"dataset.mask_feather_px must be >= 0, got {self.mask_feather_px}")
        if self.tier_collapse not in ("dedup", "repeat"):
            raise ValueError(
                f"dataset.tier_collapse must be 'dedup' or 'repeat', got {self.tier_collapse!r}"
            )
        if self.min_source_area < 0.0:
            raise ValueError(f"dataset.min_source_area must be >= 0, got {self.min_source_area}")
        if self.resolutions is not None:
            if not self.resolutions:
                raise ValueError("dataset.resolutions is empty; omit it for a single tier")
            for r in self.resolutions:
                if r <= 0:
                    raise ValueError(f"dataset.resolutions contains a non-positive tier: {r}")
                # A square image at tier r wants an r x r bucket, so a tier above the per-side cap
                # can never actually be reached -- it would silently behave as max_bucket_reso.
                if r > self.max_bucket_reso:
                    raise ValueError(
                        f"dataset.resolutions tier {r} exceeds max_bucket_reso "
                        f"{self.max_bucket_reso}; a square image could not reach it"
                    )

    def effective_subsets(self) -> list[SubsetConfig]:
        """Both config forms as one list, so scanning has a single code path.

        The single-`path` form is exactly a one-subset dataset carrying the top-level
        `num_repeats`, which is what keeps every existing config byte-identical in behaviour.
        """
        if self.subsets_file and self._manifest is None:
            self._manifest = load_subsets_file(self.subsets_file, self.subsets_root)
            _check_duplicate_subsets([*self.subsets, *self._manifest])
        if self.subsets or self._manifest:
            return [*self.subsets, *(self._manifest or [])]
        return [SubsetConfig(path=self.path, num_repeats=self.num_repeats, texture=True,
                             mask_path=self.mask_path)]

    @property
    def masked(self) -> bool:
        """Whether any sample carries a painted loss mask."""
        return self.use_masks and any(s.mask_path for s in self.effective_subsets())

    @property
    def tiers(self) -> list[int]:
        """Area budgets to train at, ascending. Buckets are monotone non-decreasing in tier, which
        is what lets `_scan_images` dedup by keeping the first occurrence."""
        return sorted(set(self.resolutions)) if self.resolutions else [self.resolution]

    def build_bucket_manager(self, resolution: int | None = None) -> BucketManager:
        bm = BucketManager(
            no_upscale=self.bucket_no_upscale,
            max_reso=((resolution or self.resolution),) * 2,
            min_size=self.min_bucket_reso,
            max_size=self.max_bucket_reso,
            reso_steps=self.bucket_reso_steps,
            multires_training=self.multires_training,
        )
        if not self.bucket_no_upscale:
            bm.make_buckets()
            verify_max_resolution(bm.predefined_resos)
        return bm

    def build_bucket_managers(self) -> dict[int, BucketManager]:
        """One manager per tier. Not shareable: `add_if_new_reso` accumulates per-instance state."""
        return {t: self.build_bucket_manager(t) for t in self.tiers}


@dataclass
class ImageEntry:
    """One trainable sample. `path` is the source image when it exists and is informational only;
    `latent_path` is what actually gets read, so entries stay valid after images are deleted.

    `resolution` is the tier this entry came from -- reporting only, since the bucket already
    determines everything about the sample."""

    path: Path
    latent_path: Path
    bucket: tuple[int, int]
    tags: str
    nl: str | None
    resolution: int = 0
    # The ORIGINAL image size, not the bucket. Only texture mode reads it, to pick a canvas the
    # source can hold; (0, 0) means unknown (latents-only mode, where there may be no image left)
    # and disables fit-aware selection for that entry rather than guessing from the bucket.
    source_size: tuple[int, int] = (0, 0)
    # False for entries from a `texture = false` subset: they stay fullres and keep their captions
    # even while the curriculum is in a texture phase. See `SubsetConfig`.
    texture_ok: bool = True
    # Per-cell loss weights from the painted mask, (1, h/16, w/16) on this entry's bucket. None for
    # an entry without one; it then trains at uniform weight.
    mask: torch.Tensor | None = None


@dataclass
class TierReport:
    """What the tier cross-product actually produced, so the cost is visible rather than implied."""

    tiers: list[int]
    entries_per_tier: dict[int, int]
    collapsed_per_tier: dict[int, int]
    dropped_small: list[str]
    n_sources: int

    def summary(self) -> str:
        if len(self.tiers) == 1 and not self.dropped_small:
            return ""
        lines = [f"tiers    {self.n_sources} sources x {len(self.tiers)} tier(s)"]
        for t in self.tiers:
            n, c = self.entries_per_tier[t], self.collapsed_per_tier[t]
            note = f"  ({c} collapsed onto a lower tier)" if c else ""
            lines.append(f"         {t:>5}: {n:>6} entries{note}")
        if self.dropped_small:
            lines.append(f"         dropped {len(self.dropped_small)} source(s) below "
                         f"min_source_area (e.g. {self.dropped_small[:3]})")
        return "\n".join(lines)


def _merge_tier_reports(reports: list[TierReport]) -> TierReport | None:
    """Fold per-subset reports into one, so the startup summary describes the run and not its
    last subset. Tiers are dataset-wide (they come from `DatasetConfig`), so the keys always
    agree and the merge is a plain sum."""
    if not reports:
        return None
    if len(reports) == 1:
        return reports[0]
    tiers = reports[0].tiers
    return TierReport(
        tiers=tiers,
        entries_per_tier={t: sum(r.entries_per_tier.get(t, 0) for r in reports) for t in tiers},
        collapsed_per_tier={t: sum(r.collapsed_per_tier.get(t, 0) for r in reports) for t in tiers},
        dropped_small=[n for r in reports for n in r.dropped_small],
        n_sources=sum(r.n_sources for r in reports),
    )


class MageFlowDataset(Dataset):
    """Serves cached latents plus a freshly-built caption.

    The caption is rebuilt per __getitem__ rather than precomputed, so tag shuffling and dropout
    produce a different view of the same image on every epoch. That is the whole reason text
    embeddings are not cached by default.
    """

    def __init__(self, config: DatasetConfig, require_cache: bool = True, caption_seed: int = 0):
        self.config = config
        self.bucket_managers = config.build_bucket_managers()
        # Back-compat alias: single-tier callers (the cache CLI, older tests) want one manager.
        self.bucket_mgr = self.bucket_managers[config.tiers[0]]
        self.entries: list[ImageEntry] = []
        self.require_cache = require_cache
        self.caption_seed = caption_seed
        self.texture_cfg = config.texture
        self.tier_report: TierReport | None = None
        self.masked = config.masked
        self._scan()

    def _scan(self) -> None:
        subsets = self.config.effective_subsets()
        workers = _scan_workers(len(subsets))
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(workers) as pool:
                modes = list(pool.map(lambda s: self._resolve_source(Path(s.path)), subsets))
            resolved = list(zip(subsets, modes))
        else:
            resolved = [(s, self._resolve_source(Path(s.path))) for s in subsets]

        # `auto` is resolved per directory, so two subsets can legitimately disagree -- but a batch
        # carries either `pixels` or `latents`, never both (`collate` picks one key), so a mixed
        # dataset would fail at the first batch that happened to span them. Refuse up front and say
        # which subset disagreed, rather than at some random step.
        modes = {m for _, m in resolved}
        if len(modes) > 1:
            detail = ", ".join(f"{Path(s.path).name}={m}" for s, m in resolved)
            raise ValueError(
                f"dataset.subsets resolve to different source modes ({detail}). A batch is either "
                f"pixels or latents, so all subsets must agree. Set `dataset.source` explicitly "
                f"instead of leaving it 'auto'."
            )

        # "encode" reads pixels at train time, so a cache is not merely optional -- requiring one
        # would defeat the mode. Recorded on self because __getitem__ has to know which it is.
        self.encode_on_the_fly = modes.pop() == "encode"
        if self.encode_on_the_fly:
            self.require_cache = False

        reports: list[TierReport] = []
        self.subset_report: list[tuple[SubsetConfig, int]] = []
        scanned = None
        if workers > 1:
            # Thousands of small per-artist folders are I/O-bound to scan (directory listings,
            # image headers, cache stats), so threads overlap the waits. Results are merged in
            # subset order below, so the entry list -- and with it every batch -- is identical to
            # a sequential scan.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(workers) as pool:
                scanned = list(pool.map(lambda sm: self._scan_isolated(Path(sm[0].path), sm[1]),
                                        resolved))
        for i, (sub, mode) in enumerate(resolved):
            root = Path(sub.path)
            start = len(self.entries)
            self.tier_report = None   # so a subset's report can never be counted twice
            if scanned is not None:
                self.entries.extend(scanned[i][0])
                self.tier_report = scanned[i][1]
            elif mode == "latents":
                self._scan_latents(root)
            else:
                self._scan_images(root)
            fresh = self.entries[start:]
            if not sub.texture:
                for e in fresh:
                    e.texture_ok = False
            if self.masked and sub.mask_path:
                self._attach_masks(fresh, Path(sub.mask_path))
            # Per-subset repeats, applied to that subset's slice only. This is the knob that
            # balances a small regularization set against a large training set.
            if sub.num_repeats > 1:
                self.entries.extend(fresh * (sub.num_repeats - 1))
            self.subset_report.append((sub, len(self.entries) - start))
            if self.tier_report is not None:
                reports.append(self.tier_report)

        verify_max_resolution({e.bucket for e in self.entries})
        self.tier_report = _merge_tier_reports(reports)

    def _scan_isolated(self, root: Path, mode: str):
        """One subset's (entries, tier report), scanned on a private shallow copy so threads do
        not share `entries`/`tier_report` -- or the bucket managers, whose `select_bucket` records
        new resolutions as it goes."""
        import copy

        worker = copy.copy(self)
        worker.entries = []
        worker.tier_report = None
        worker.bucket_managers = copy.deepcopy(self.bucket_managers)
        if mode == "latents":
            worker._scan_latents(root)
        else:
            worker._scan_images(root)
        return worker.entries, worker.tier_report
    def _attach_masks(self, entries: list[ImageEntry], root: Path) -> None:
        """Give each entry its painted mask, cropped to the entry's own bucket.

        Masks are built once here, not per step: they are tiny at latent resolution, and a missing
        or mismatched mask should stop the run before the first step rather than hundreds in. The
        crop is re-derived from the source size through the entry's tier manager -- the same
        `select_bucket` call that chose the bucket -- so it matches cached latents exactly.
        """
        cfg = self.config
        masks = find_masks(root)
        missing: list[str] = []
        empty: list[str] = []
        for e in entries:
            cached_name = parse_cache_filename(e.path)
            if e.source_size != (0, 0):
                # A latents-only entry can carry its source size too (taken from the cache metadata
                # to attribute tiers); its path is then the cache file, named `<stem>_WxH_trainer`.
                stem = cached_name[0] if cached_name else e.path.stem
                size = e.source_size
            else:
                # Latents-only: the entry is the cache file, whose metadata carries the source size.
                stem = parse_cache_filename(e.latent_path)[0]
                size = parse_original_size(read_cache_metadata(e.latent_path))
                if size is None:
                    raise RuntimeError(
                        f"{e.latent_path.name}: cached latent has no `original_size` metadata, "
                        f"so its mask cannot be cropped to match. Re-cache it."
                    )
            mask_file = masks.get(stem)
            if mask_file is None:
                missing.append(stem)
                continue
            bucket, resized, _ = self.bucket_managers[e.resolution].select_bucket(*size)
            if bucket != e.bucket:
                raise RuntimeError(
                    f"{stem}: re-derived bucket {bucket} differs from the entry's {e.bucket}; the "
                    f"mask would not line up with its latent"
                )
            coverage = load_mask(mask_file, size, bucket, resized,
                                 dilate_px=cfg.mask_dilate_px, feather_px=cfg.mask_feather_px)
            if not coverage.any():
                empty.append(stem)
            e.mask = loss_weights(coverage, cfg.mask_outside_weight)

        if missing:
            raise ValueError(
                f"mask folder {root} has no mask for {len(missing)} image(s) (matched by filename "
                f"stem), e.g. {missing[:10]}. Every image needs one; paint missing ones white to "
                f"train them unmasked."
            )
        if empty and cfg.mask_outside_weight == 0.0:
            import warnings

            warnings.warn(
                f"{len(empty)} mask(s) in {root} are entirely black (e.g. {empty[:5]}); with "
                f"mask_outside_weight = 0 those images train nothing.",
                stacklevel=3,
            )

    def _resolve_source(self, root: Path) -> str:
        if not root.exists():
            raise FileNotFoundError(f"dataset path not found: {root}")
        source = self.config.source
        if source == "auto":
            has_images = any(p.suffix.lower() in IMAGE_EXTENSIONS for p in root.iterdir())
            source = "images" if has_images else "latents"
        if source not in ("images", "latents", "encode"):
            raise ValueError(f"unknown source mode: {source!r}")
        return source

    def _scan_latents(self, root: Path) -> None:
        """Build entries from cached latents; source images need not exist.

        Multiple cached buckets per stem is the *normal* case under multi-resolution, so the old
        "more than one valid bucket -> stale, delete them" rule cannot simply be relaxed or the
        stale-cache guard goes with it. Instead each cache is attributed to the tier that explains
        it: `original_size` from the cache metadata re-derives what `select_bucket` would have
        chosen at every declared tier. A file matching no tier is stale; a tier with no file is
        missing. Exact either way, and it works on caches already on disk.
        """
        cached = find_cached_latents(root)
        if not cached:
            raise ValueError(
                f"no cached latents in {root} (looked for *{CACHE_SUFFIX}). "
                f"Cache with source='images' first."
            )

        tiers = self.config.tiers
        min_px = self._min_source_pixels()
        per_tier = {t: 0 for t in tiers}
        collapsed = {t: 0 for t in tiers}
        dropped: list[str] = []
        skipped: list[str] = []
        stale: list[str] = []
        missing: list[str] = []
        unattributed = 0

        for stem, variants in sorted(cached.items()):
            caption_path = root / f"{stem}.txt"
            if not caption_path.exists():
                skipped.append(stem)
                continue

            by_bucket = {b: p for p, b in variants}
            size = None
            for p, _ in variants:
                size = parse_original_size(read_cache_metadata(p))
                if size:
                    break

            if size is None:
                # Pre-metadata cache: fall back to the area filter. Cannot attribute tiers, so
                # multi-res is refused rather than guessed at.
                unattributed += 1
                if len(tiers) > 1:
                    raise RuntimeError(
                        f"{stem}: cached latents carry no `original_size` metadata, so they cannot "
                        f"be attributed to a tier. Re-cache to use multiple resolutions."
                    )
                wanted = [(p, b) for p, b in variants if self._bucket_allowed(b)]
                if not wanted:
                    skipped.append(stem)
                    continue
                if len(wanted) > 1:
                    raise RuntimeError(
                        f"{stem}: {len(wanted)} cached buckets are all valid for this config "
                        f"({[b for _, b in wanted]}). Delete the stale ones."
                    )
                wanted_pairs = [(tiers[0], wanted[0][1])]
            else:
                w, h = size
                if min_px and w * h < min_px:
                    dropped.append(stem)
                    continue
                for t in self._collapsed_tiers(w, h):
                    collapsed[t] += 1
                wanted_pairs = self._tier_buckets(w, h)

                expected = {b for _, b in wanted_pairs}
                for b in sorted(set(by_bucket) - expected):
                    stale.append(f"{stem} @ {b[0]}x{b[1]}")
                for b in sorted(expected - set(by_bucket)):
                    missing.append(f"{stem} @ {b[0]}x{b[1]}")

            nl_path = root / f"{stem}_nl.txt"
            tags = caption_path.read_text(encoding="utf-8").strip()
            nl = nl_path.read_text(encoding="utf-8").strip() if nl_path.exists() else None
            for tier, bucket in wanted_pairs:
                p = by_bucket.get(bucket)
                if p is None:
                    continue  # reported via `missing` below
                per_tier[tier] += 1
                self.entries.append(
                    ImageEntry(
                        path=p,  # the cache file itself; there may be no image any more
                        latent_path=p, bucket=bucket, tags=tags, nl=nl, resolution=tier,
                        source_size=tuple(size) if size else (0, 0),   # layouts map boxes into the crop
                    )
                )

        if missing:
            raise RuntimeError(
                f"{len(missing)} declared (source, tier) pair(s) have no cached latent (e.g. "
                f"{missing[:4]}). Cache every tier ({' '.join(str(t) for t in tiers)}) first."
            )
        if stale:
            # Warn, do not raise. Exact tier attribution means an unmatched cache is never *chosen*
            # -- it is simply not used -- so ignoring it is safe, and a genuinely stale cache still
            # hard-fails above via `missing`. Erroring here would also make latents mode disagree
            # with images mode, which ignores extra cache files silently, and would forbid the
            # normal workflow of keeping several tiers cached and switching between them in config.
            import warnings

            warnings.warn(
                f"{len(stale)} cached latent(s) match no declared tier and are being ignored "
                f"(e.g. {stale[:3]}). That is expected if you cached other resolutions; delete "
                f"them to reclaim the disk, or add their tier to dataset.resolutions to train on "
                f"them.",
                stacklevel=3,
            )
        if not self.entries:
            raise ValueError(f"{root}: found {len(cached)} cached latents but none had a .txt caption")
        if skipped:
            raise RuntimeError(
                f"{len(skipped)} cached latents have no usable .txt caption or bucket "
                f"(e.g. {skipped[:5]}). They would be silently dropped; fix or remove them."
            )

        self.tier_report = TierReport(
            tiers=tiers, entries_per_tier=per_tier, collapsed_per_tier=collapsed,
            dropped_small=dropped, n_sources=len(cached) - len(dropped),
        )
        if not unattributed:
            self._warn_collapse(self.tier_report)

    def _bucket_allowed(self, bucket: tuple[int, int]) -> bool:
        """Legacy filter, used only for caches predating `original_size` metadata.

        Deliberately does NOT enforce `min_bucket_reso`: `select_bucket`'s no_upscale branch never
        applies a minimum side, so requiring one here rejected caches the image path had happily
        created (a 200x1500 source yields a 192x1472 bucket). The two paths now agree.
        """
        c = self.config
        return (
            max(bucket) <= c.max_bucket_reso
            and bucket[0] % c.bucket_reso_steps == 0
            and bucket[1] % c.bucket_reso_steps == 0
            and bucket[0] * bucket[1] <= c.resolution * c.resolution
        )

    def _tier_buckets(self, w: int, h: int) -> list[tuple[int, tuple[int, int]]]:
        """(tier, bucket) for one source, with `tier_collapse` applied.

        `bucket_no_upscale` caps a bucket by the *source*, so an image already under a tier's area
        budget is left untouched and every tier above it yields the identical bucket. Buckets are
        therefore monotone non-decreasing in tier and duplicates form a contiguous run at the top,
        which is why keeping the first occurrence is the correct dedup.

        This is a per-image *ceiling*, not a floor -- `min_bucket_reso` has no bearing on it, since
        `select_bucket`'s no_upscale branch never reads `min_size`.
        """
        out: list[tuple[int, tuple[int, int]]] = []
        seen: set[tuple[int, int]] = set()
        for tier in self.config.tiers:
            bucket = self.bucket_managers[tier].select_bucket(w, h)[0]
            if self.config.tier_collapse == "dedup" and bucket in seen:
                continue
            seen.add(bucket)
            out.append((tier, bucket))
        return out

    def _collapsed_tiers(self, w: int, h: int) -> list[int]:
        """Tiers whose bucket duplicates a lower tier's -- i.e. that bought no new resolution."""
        collapsed, seen = [], set()
        for tier in self.config.tiers:
            bucket = self.bucket_managers[tier].select_bucket(w, h)[0]
            if bucket in seen:
                collapsed.append(tier)
            seen.add(bucket)
        return collapsed

    def _min_source_pixels(self) -> float:
        cfg = self.config
        return cfg.min_source_area * min(cfg.tiers) ** 2 if cfg.min_source_area else 0.0

    def _warn_collapse(self, report: TierReport) -> None:
        """Multi-res degrading into plain `num_repeats` is invisible otherwise: it costs the same
        VRAM, disk and wall-clock while producing no extra resolution. Say so."""
        if len(report.tiers) < 2 or not report.n_sources:
            return
        import warnings

        for tier in report.tiers:
            frac = report.collapsed_per_tier[tier] / report.n_sources
            if frac >= 0.01:
                warnings.warn(
                    f"tier {tier} exceeds the native area of {frac * 100:.1f}% of your sources "
                    f"({report.collapsed_per_tier[tier]}/{report.n_sources}); those images "
                    f"duplicate a lower tier's bucket rather than gaining resolution"
                    + ("  -- deduplicated, so they simply get fewer repeats."
                       if self.config.tier_collapse == "dedup"
                       else "  -- tier_collapse='repeat', so they are trained twice at the same "
                            "bucket.")
                    + " Lower the top tier, or check `cache_latents.py audit` for a ladder that "
                      "fits this dataset's size distribution.",
                    stacklevel=3,
                )

    def _scan_images(self, root: Path) -> None:
        files = sorted(
            p for p in root.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.endswith(".safetensors")
        )
        if not files:
            raise ValueError(f"no images found in {root}")

        from PIL import Image

        tiers = self.config.tiers
        min_px = self._min_source_pixels()
        per_tier = {t: 0 for t in tiers}
        collapsed = {t: 0 for t in tiers}
        dropped: list[str] = []
        missing_cache: list[tuple[str, tuple[int, int]]] = []

        for p in files:
            with Image.open(p) as im:
                w, h = im.size

            if min_px and w * h < min_px:
                dropped.append(p.name)
                continue

            for t in self._collapsed_tiers(w, h):
                collapsed[t] += 1

            tags = nl = None
            for tier, bucket in self._tier_buckets(w, h):
                if self.require_cache and not cache_path(p, bucket).exists():
                    missing_cache.append((p.name, bucket))
                    continue
                if tags is None:
                    tags, nl = read_caption_files(p)
                per_tier[tier] += 1
                self.entries.append(
                    ImageEntry(
                        path=p, latent_path=cache_path(p, bucket), bucket=bucket,
                        tags=tags, nl=nl, resolution=tier, source_size=(w, h),
                    )
                )

        if missing_cache:
            shown = ", ".join(f"{n} @ {b[0]}x{b[1]}" for n, b in missing_cache[:4])
            raise RuntimeError(
                f"{len(missing_cache)} (image, bucket) pair(s) have no cached latent (e.g. "
                f"{shown}). Run latent caching for every tier "
                f"({' '.join(str(t) for t in tiers)}) first."
            )

        self.tier_report = TierReport(
            tiers=tiers, entries_per_tier=per_tier, collapsed_per_tier=collapsed,
            dropped_small=dropped, n_sources=len(files),
        )
        self._warn_collapse(self.tier_report)

    def bucket_of(self, index: int) -> tuple[int, int]:
        return self.entries[index].bucket

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int | tuple[int, int]) -> dict:
        # The sampler yields (index, epoch) so the caption RNG can be seeded from the sample
        # itself. Deriving it from global `random` would be wrong twice over: forked DataLoader
        # workers inherit the parent's RNG state, and every DDP rank forks from the same seed --
        # so ranks would draw *identical* tag shuffles and dropouts. That silently collapses
        # caption diversity instead of failing, which is why it is seeded explicitly here.
        phase = canvas = None
        if isinstance(index, tuple):
            index, epoch, phase, canvas = (index + (None, None))[:4]
        else:
            index, epoch = index, 0
        e = self.entries[index]
        rng = random.Random(_mix(self.caption_seed, epoch, index))

        if canvas is not None:
            return self._texture_sample(e, canvas, rng)

        caption = (self.caption_variation_cache.caption(index, epoch)
                   if hasattr(self, "caption_variation_cache")
                   else build_caption(e.tags, e.nl, self.config.caption, rng,
                                      annotations_for(e, self.config.caption)))
        sample = {"caption": caption, "bucket": e.bucket, "path": str(e.path)}
        if self.masked:
            # An entry from an unmasked subset still carries a uniform mask, so every sample of a
            # masked dataset has the key `collate` stacks.
            sample["mask"] = (e.mask if e.mask is not None else
                              torch.ones(1, e.bucket[1] // LATENT_SCALE, e.bucket[0] // LATENT_SCALE))

        if self.encode_on_the_fly:
            # Crop through the same helper the cacher uses, and against this entry's *own* tier's
            # bucket manager -- otherwise a multi-tier run would re-derive one bucket for every
            # tier and silently break the sampler's bucket-homogeneity guarantee.
            img, _ = load_and_crop(e.path, self.bucket_managers[e.resolution])
            sample["pixels"] = image_to_tensor(img).squeeze(0)  # (3, 1, H, W) in [-1, 1]
        else:
            sample["latents"] = load_cached_latent(e.latent_path)
        return sample


    def _texture_sample(self, e: ImageEntry, canvas: tuple[int, int], rng: random.Random) -> dict:
        """A canvas-sized crop chosen by detail content, plus its feathered loss mask.

        Always encodes live -- the crop is chosen per step from image content, so no cache can
        express it. That is the entire reason `source = "encode"` exists.
        """
        from PIL import Image as _Image

        from .texture import crop_canvas, feather_mask

        with _Image.open(e.path) as im:
            img = im.convert("RGB")
        crop, valid = crop_canvas(img, canvas, self.texture_cfg, rng)
        if crop is None:
            # `oversize = "skip"` on an image that cannot hold the canvas. Returning a shape the
            # batch cannot stack would be worse, so fall back to covering and say so once.
            from .texture import TextureConfig
            cover = TextureConfig(**{**self.texture_cfg.__dict__, "oversize": "cover"})
            crop, valid = crop_canvas(img, canvas, cover, rng)

        cw, ch = canvas
        mask = feather_mask(ch // 16, cw // 16, self.texture_cfg, rng)
        if valid is not None and self.texture_cfg.mask_padding:
            # Invented pixels stay in the forward pass as context but carry no target. Reduced to
            # latent resolution by AREA (mean over each 8x8 block), not by sampling, so a latent
            # cell that is half padding is weighted 0.5 rather than being all-or-nothing on
            # whichever pixel a nearest-neighbour resize happened to land on.
            v = torch.from_numpy(valid).view(ch // 16, 16, cw // 16, 16).mean(dim=(1, 3))
            mask = mask * v
        return {
            # Trigger only, never the image's tags -- see TextureConfig.trigger for why this is
            # load-bearing rather than a convenience.
            "caption": self.texture_cfg.trigger,
            "bucket": canvas,
            "path": str(e.path),
            "pixels": image_to_tensor(crop).squeeze(0),
            "mask": mask,
        }


def collate(batch: list[dict]) -> dict:
    buckets = {b["bucket"] for b in batch}
    if len(buckets) != 1:
        raise RuntimeError(f"batch mixes buckets {buckets}; the sampler must group by bucket")
    key = "pixels" if "pixels" in batch[0] else "latents"
    out = {
        key: torch.stack([b[key] for b in batch]),
        "captions": [b["caption"] for b in batch],
        "bucket": batch[0]["bucket"],
        "paths": [b["path"] for b in batch],
    }
    if "mask" in batch[0]:
        out["mask"] = torch.stack([b["mask"] for b in batch])
    return out


class BucketBatchSampler(Sampler[list[int]]):
    """Yields batches whose members all share a bucket.

    `batch_size` is an int, or a map keyed by **resolution tier** -- the values in
    `dataset.resolutions` (or the single `dataset.resolution`), matched exactly:

        resolutions = [768, 1024, 1280]
        batch_size  = { 768 = 16, 1024 = 12, 1280 = 8 }

    Keyed on the tier, not on the bucket's longest side, because the tier is what predicts cost and
    the longest side is not. Bucketing targets a constant *area*, so `tier^2 / 256` is an exact
    upper bound on a tier's token count (measured on a mixed set: tiers 768/1024/1280 top out at exactly
    2304/4096/6400 tokens) while the longest side ranges overlap almost completely across tiers --
    704-1280 at tier 768 against 768-1792 at tier 1024. Sorting by longest side therefore sorts by
    something that is not the cost.

    This used to be a threshold lookup on `max(bucket)`, where each key was a floor covering
    everything up to the next one. That reads like a label and behaves like a range: `{512: 32,
    1024: 12}` silently gave a 960x960 bucket 32 images -- 2.4x the load of the 1024x1024 bucket
    next to it, and an OOM several minutes into a run. Exact tier matching cannot express that, and
    `load_config` rejects a key that is not a declared tier.

    A bucket can host more than one tier (two different sources can land on the same bucket from
    different tiers). Cost depends only on the bucket, so any of their sizes would do; the largest
    tier present wins, which is the smaller batch and therefore the safe direction.

    **This sampler is rank-agnostic on purpose.** `accelerator.prepare(dataloader)` wraps the batch
    sampler in Accelerate's `BatchSamplerShard`, which hands whole batches to each rank round-robin
    and pads with `even_batches` so ranks stay in lockstep. Because it distributes *whole* batches
    it preserves bucket homogeneity, so there is nothing for us to add. Sharding here as well would
    shard twice -- each rank would see roughly 1/world_size^2 of the data, which shows up only as a
    quietly reduced step count rather than an error.
    """

    def __init__(
        self,
        dataset: MageFlowDataset,
        batch_size: int | dict[int, int],
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self._by_tier = dict(batch_size) if isinstance(batch_size, dict) else None
        self._flat = None if isinstance(batch_size, dict) else int(batch_size)

        # `bucket_indices` stays keyed by bucket alone: it is what the trainer reads to size
        # `use_quantized_matmul` and to report the bucket table, and both want the whole dataset.
        # `_groups` is the finer split that batching uses.
        self.bucket_indices: dict[tuple[int, int], list[int]] = {}
        self._groups: dict[tuple[tuple[int, int], bool], list[int]] = {}
        self._bucket_tier: dict[tuple[int, int], int] = {}
        for i in range(len(dataset)):
            e = dataset.entries[i]
            bucket = e.bucket
            self.bucket_indices.setdefault(bucket, []).append(i)
            self._groups.setdefault((bucket, e.texture_ok), []).append(i)
            self._bucket_tier[bucket] = max(self._bucket_tier.get(bucket, 0), e.resolution)

    def batch_size_for(self, bucket: tuple[int, int]) -> int:
        if self._flat is not None:
            return self._flat
        tier = self._bucket_tier.get(bucket, 0)
        if tier in self._by_tier:
            return self._by_tier[tier]
        # Unreachable through `load_config`, which requires the keys to be exactly the declared
        # tiers. Falling back to the smallest configured size keeps a hand-built sampler safe
        # rather than silently generous.
        return min(self._by_tier.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        # Grouped by (bucket, texture eligibility), not bucket alone. A batch gets ONE canvas, so a
        # batch that mixed a texture-eligible entry with an exempt one could not honour both -- the
        # exempt entry would either be cropped anyway or force the whole batch to fullres. Splitting
        # here makes the guarantee structural: `__iter__` can read the flag off any member.
        for (bucket, _), indices in self._groups.items():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            bs = self.batch_size_for(bucket)
            for i in range(0, len(indices), bs):
                chunk = indices[i : i + bs]
                if len(chunk) < bs and self.drop_last:
                    continue
                batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)  # decorrelate bucket order across the epoch

        # No rank sharding and no truncation: Accelerate does both. A trailing partial
        # accumulation group is fine -- Accelerate forces a gradient sync at end_of_dataloader
        # (`Accelerator._do_sync`), so the remainder steps and no sample is dropped. This is the
        # same behaviour sd-scripts relies on.
        return batches

    def set_curriculum(self, curriculum, total_epochs: int, texture=None) -> None:
        """Let the sampler resolve the curriculum phase for each batch it emits.

        The phase has to reach `__getitem__`, and `__getitem__` runs in forked workers that
        prefetch ahead -- so setting a flag on the dataset from the training loop would arrive
        late by however deep the prefetch queue is (8 workers x prefetch 2 is ~16 batches, about 4
        optimizer steps at accum 4, against 5%-of-run phases only ~21 steps long). Resolving here
        instead is exact: the sampler runs in the main process and knows its own batch ordering,
        and `batches_consumed / total_batches` is *identically* the trainer's
        `global_step / total_steps` -- both divide out accumulation and world size. So the mode the
        dataset used and the t_range the trainer applied always come from the same phase, with no
        shared state and no cross-rank communication.
        """
        from .texture import TextureConfig
        self.curriculum = curriculum
        self.total_epochs = max(1, total_epochs)
        self.texture = texture or TextureConfig()

    def _phase_for(self, batch_ordinal: int) -> object | None:
        if not getattr(self, "curriculum", None):
            return None
        total = max(1, len(self._batches()) * self.total_epochs)
        consumed = self.epoch * len(self._batches()) + batch_ordinal
        return self.curriculum.resolve(consumed / total)

    def __iter__(self):
        # (index, epoch, phase, canvas): the dataset needs the epoch to seed its caption RNG, the
        # phase to know whether this is a texture crop, and the canvas so every sample in the
        # batch is the same shape. The canvas is drawn *here*, once per batch, for that reason --
        # drawing it per sample would produce a batch that cannot be stacked.
        rng = random.Random(self.seed + self.epoch)
        for ordinal, batch in enumerate(self._batches()):
            phase = self._phase_for(ordinal)
            canvas = None
            # `_batches` guarantees the batch is homogeneous in `texture_ok`, so member 0 speaks for
            # all of them. A `texture = false` subset therefore stays fullres -- captions intact --
            # no matter what the curriculum says.
            if (phase is not None and getattr(phase, "mode", "fullres") == "texture"
                    and self.dataset.entries[batch[0]].texture_ok):
                canvas = self._choose_canvas(batch, rng)
            yield [(i, self.epoch, phase, canvas) for i in batch]

    def _choose_canvas(self, batch: list[int], rng: random.Random) -> tuple[int, int]:
        """Pick the batch's canvas -- from those every image in it can hold, when possible.

        Drawing blind and forcing the fit is what makes padding common (56% of draws on one texture set);
        drawing from what fits makes it rare (1%), without upscaling anything. The choice is per
        BATCH, not per sample, because the batch has to stack -- so a larger batch has fewer
        canvases available to it, and at some size the intersection empties and this degrades to
        the blind draw. That is a real cost of batching in texture mode, and it degrades smoothly
        rather than failing.
        """
        canvases = self.texture.canvases
        if not getattr(self.texture, "fit_aware", True):
            return rng.choice(canvases)
        sizes = [self.dataset.entries[i].source_size for i in batch]
        if any(s == (0, 0) for s in sizes):
            return rng.choice(canvases)          # unknown source size: cannot reason about fit
        w, h = min(s[0] for s in sizes), min(s[1] for s in sizes)
        fits = fitting_canvases((w, h), canvases)
        # No canvas fits every image: fall back to the blind draw and let the crop cascade handle
        # it. Choosing the "least bad" canvas here instead would quietly bias the aspect mix toward
        # whatever the smallest image in the batch happens to be.
        return rng.choice(fits or canvases)

    def __len__(self) -> int:
        return len(self._batches())
