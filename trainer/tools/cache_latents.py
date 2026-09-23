"""Latent caching and cache auditing.

    python -m trainer.tools.cache_latents cache /path/to/dataset --resolution 1024
    python -m trainer.tools.cache_latents audit /path/to/dataset

Caching is a prerequisite for training: the dataset layer reads latents, never images. It is also
where bucket assignment is frozen, so a change to `resolution` / `bucket_reso_steps` /
`bucket_no_upscale` invalidates the cache and needs a re-run.

`audit` exists because of the latents-only workflow. Deleting source images is irreversible, and
the two ways to lose data silently are an image with no cache and a cache with no caption. It
reports both and refuses to bless the directory unless neither exists.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

from ..training.config import DEFAULT_MODEL_PATH
from ..data.bucket import verify_max_resolution
from ..data.cache import CACHE_SUFFIX, LatentCacher, audit_cache, cache_path
from ..data.dataset import IMAGE_EXTENSIONS, DatasetConfig


def _images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.endswith(CACHE_SUFFIX)
    )


def config_cache_commands(args):
    """Use the config's exact subset paths and bucket/VAE settings."""
    from ..training.config import load_config
    cfg = load_config(args.config)
    commands = []
    for subset in cfg.dataset.effective_subsets():
        command = [sys.executable, "-u", "-m", "trainer.tools.cache_latents", "cache", subset.path,
                   "--model-path", cfg.train.model_path, "--model-family", cfg.train.model_family,
                   "--resolution", *map(str, args.resolution or cfg.dataset.tiers),
                   "--min-bucket-reso", str(cfg.dataset.min_bucket_reso),
                   "--max-bucket-reso", str(cfg.dataset.max_bucket_reso),
                   "--bucket-reso-steps", str(cfg.dataset.bucket_reso_steps),
                   "--batch-size", str(args.batch_size)]
        if cfg.train.vae_path:
            command += ["--vae-path", cfg.train.vae_path]
        if cfg.train.flux2_vae:
            command.append("--flux2-vae")
        if not cfg.dataset.bucket_no_upscale:
            command.append("--upscale")
        if cfg.dataset.multires_training:
            command.append("--multires")
        if args.devices:
            command += ["--devices", args.devices]
        for flag in ("overwrite", "dry_run", "allow_missing_captions"):
            if getattr(args, flag):
                command.append("--" + flag.replace("_", "-"))
        commands.append(command)
    return commands


def cmd_cache_config(args):
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    commands = config_cache_commands(args)
    for i, command in enumerate(commands, 1):
        print(f"\nCaching subset {i}/{len(commands)}: {command[5]}", flush=True)
        result = subprocess.run(command)
        if result.returncode:
            return result.returncode
    return 0


def cmd_cache(args) -> int:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    devices = [d.strip() for d in (args.devices or "").split(",") if d.strip()]
    if len(devices) > 1 and args.num_shards == 1:
        # One coordinator spawns one independent encoder per selected physical GPU.
        # Each child gets a disjoint round-robin image shard, so no cache files race.
        base = list(sys.argv[1:])
        at = base.index("--devices")
        del base[at:at + 2]
        children = []
        for rank, gpu in enumerate(devices):
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
            command = [sys.executable, "-u", "-m", "trainer.tools.cache_latents", *base,
                       "--shard-index", str(rank), "--num-shards", str(len(devices))]
            children.append(subprocess.Popen(command, env=env))
        statuses = [child.wait() for child in children]
        return 0 if all(status == 0 for status in statuses) else 1
    root = Path(args.path)
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 1

    files = _images(root)
    if not files:
        print(f"no images in {root}")
        return 1

    tiers = sorted(set(args.resolution))
    cfg = DatasetConfig(
        path=str(root),
        resolutions=tiers,
        min_bucket_reso=args.min_bucket_reso,
        max_bucket_reso=args.max_bucket_reso,
        bucket_reso_steps=args.bucket_reso_steps,
        bucket_no_upscale=not args.upscale,
        multires_training=args.multires,
    )
    managers = cfg.build_bucket_managers()

    from PIL import Image

    # Assign every bucket first: a RoPE-illegal bucket must fail before we write 2000 files.
    # Dedup per (image, bucket) matches what the dataset will actually ask for -- a tier above an
    # image's native area yields the same bucket, hence the same file.
    plan: dict[Path, list[tuple[int, tuple[int, int]]]] = {}
    sizes: dict[Path, tuple[int, int]] = {}
    collapsed = {t: 0 for t in tiers}
    for p in files:
        with Image.open(p) as im:
            sizes[p] = im.size
        seen, keep = set(), []
        for t in tiers:
            b = managers[t].select_bucket(*sizes[p])[0]
            if b in seen:
                collapsed[t] += 1
                continue
            seen.add(b)
            keep.append((t, b))
        plan[p] = keep
    verify_max_resolution({b for v in plan.values() for _, b in v})

    print(f"\n{len(files)} images x {len(tiers)} tier(s) -> "
          f"{sum(len(v) for v in plan.values())} cache files")
    for t in tiers:
        bs = [b for v in plan.values() for tt, b in v if tt == t]
        mb = sum(16 * 4 * (b[0] * b[1] // 64) for b in bs) / 1e6
        note = f", {collapsed[t]} collapsed onto a lower tier" if collapsed[t] else ""
        print(f"  tier {t:>5}: {len(bs):>5} images, {len(set(bs)):>4} buckets, ~{mb:>7.0f} MB{note}")

    counts: dict[tuple[int, int], int] = {}
    for v in plan.values():
        for _, b in v:
            counts[b] = counts.get(b, 0) + 1
    print(f"\n{len(counts)} distinct buckets overall")
    for b, n in sorted(counts.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {b[0]:>5}x{b[1]:<5} {n:>5}")
    if len(counts) > 12:
        print(f"  ... {len(counts) - 12} more")

    missing_caption = [p.name for p in files if not p.with_suffix(".txt").exists()]
    if missing_caption and not args.allow_missing_captions:
        print(f"\n{len(missing_caption)} images have no .txt caption "
              f"(e.g. {missing_caption[:3]}). They would be unusable for training.")
        print("Re-run with --allow-missing-captions to cache them anyway.")
        return 1

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    if args.num_shards > 1:
        files = files[args.shard_index::args.num_shards]
        plan = {path: plan[path] for path in files}
        print(f"\nshard {args.shard_index + 1}/{args.num_shards}: {len(files)} images on {args.device}")

    from ..modeling.loader import load_components

    # A named FLUX.2 checkpoint is unambiguous. Keep the explicit switch for
    # generic filenames, but do not let a GUI/CLI wiring omission feed Flux
    # weights into MageVAE and fail several seconds after planning the cache.
    args.flux2_vae = bool(args.flux2_vae or "flux2" in Path(args.vae_path or "").name.lower())
    # The common ComfyUI `flux2-vae.safetensors` export is the legacy LDM
    # implementation. Its compatible module is shipped with ComfyUI and is
    # compiled for ComfyUI's Python, not this trainer's virtualenv. Re-exec
    # only that cache operation there; training subsequently reads ordinary
    # safetensor latents and remains in this environment.
    if args.flux2_vae and args.vae_path and Path(args.vae_path).is_file():
        from safetensors import safe_open
        with safe_open(str(args.vae_path), framework="pt") as handle:
            legacy_comfy_flux2 = "bn.running_mean" in handle.keys() and "encoder.quant_conv.weight" in handle.keys()
        comfy_python = Path("/home/bluvoll/ComfyUI/venv/bin/python")
        if legacy_comfy_flux2 and Path(sys.executable).resolve() != comfy_python.resolve():
            if not comfy_python.is_file():
                raise RuntimeError("This legacy ComfyUI FLUX.2 VAE needs /home/bluvoll/ComfyUI/venv/bin/python for caching.")
            print("restarting cache under ComfyUI's Python for the legacy FLUX.2 VAE", flush=True)
            os.execv(str(comfy_python), [str(comfy_python), *sys.argv])
    vae_label = "FLUX.2" if args.flux2_vae else "Mage"
    print(f"\nloading {vae_label} VAE from {args.vae_path or args.model_path}")
    components = load_components(
        args.model_path, dtype=torch.bfloat16,
        vae_path=args.vae_path,
        flux2_vae=args.flux2_vae,
        load_text_encoder=False, load_vae=True, load_tokenizers=False, load_transformer=False,
    )
    cacher = LatentCacher(components.vae, device=args.device, flux2_vae=args.flux2_vae)

    work: dict[tuple[int, tuple[int, int]], list[Path]] = {}
    for path in files:
        for tier, bucket in plan[path]:
            work.setdefault((tier, bucket), []).append(path)
    total = sum(len(paths) for paths in work.values())
    done = skipped = seen = 0
    for (tier, bucket), paths in work.items():
        for start in range(0, len(paths), args.batch_size):
            batch = paths[start:start + args.batch_size]
            pending = [path for path in batch if args.overwrite or not cache_path(path, bucket).exists()]
            skipped += len(batch) - len(pending)
            if pending:
                cacher.cache_batch(pending, managers[tier], overwrite=args.overwrite)
                done += len(pending)
            seen += len(batch)
            if seen % 25 < len(batch) or seen == total:
                print(f"  {seen}/{total} cache entries  cached {done}, skipped {skipped}", flush=True)

    print(f"\ncached {done}, skipped {skipped} (already present)")
    return 0


def _source_sizes(root: Path) -> list[tuple[int, int]]:
    """Source dimensions, from the images if present, else from the caches' `original_size`."""
    from PIL import Image

    files = _images(root)
    if files:
        out = []
        for p in files:
            with Image.open(p) as im:
                out.append(im.size)
        return out

    from ..data.cache import find_cached_latents, parse_original_size, read_cache_metadata

    out = []
    for variants in find_cached_latents(root).values():
        size = parse_original_size(read_cache_metadata(variants[0][0]))
        if size:
            out.append(size)
    return out


def _report_size_distribution(root: Path, steps: int) -> None:
    """Print the source-area distribution and propose a ladder.

    This is the number that decides whether multi-resolution does anything at all. With
    `bucket_no_upscale`, a tier above an image's native area yields the *same* bucket as the tier
    below it -- so a ladder sitting above the dataset's size distribution silently degrades into
    plain `num_repeats` at full VRAM, disk and wall-clock cost.
    """
    sizes = _source_sizes(root)
    if not sizes:
        return
    areas = sorted(w * h for w, h in sizes)

    def q(p: float) -> int:
        return areas[min(int(p * len(areas)), len(areas) - 1)]

    def side(area: int) -> int:
        return max(64, int(area ** 0.5) // steps * steps)

    print(f"\nsource-area distribution ({len(areas)} sources)")
    print("   pct     MP    tier at this source's own ceiling")
    for p in (0.01, 0.05, 0.10, 0.25, 0.50, 0.90):
        a = q(p)
        print(f"   p{int(p * 100):<4} {a / 1e6:>6.2f}   {side(a):>5}px")

    # Top rung at the p10 ceiling, so ~10% of sources collapse there. Lower rungs step *down*
    # geometrically rather than clustering under the top: every tier below an image's own ceiling
    # produces a genuinely different bucket, so spreading down is what buys real resolution
    # variation. Spacing is geometric because cost scales with area, not with side length.
    top = side(q(0.10))
    ladder = sorted({max(steps, side(int((top / 1.25 ** k) ** 2))) for k in range(3)})
    over = sum(1 for a in areas if a < ladder[-1] ** 2)
    print(f"\n   suggested ladder: resolutions = {ladder}")
    print(f"   -> {over}/{len(areas)} sources ({100 * over / len(areas):.1f}%) sit below the top "
          f"rung, so they collapse onto a lower tier there and simply get fewer repeats.")
    print("   Lower the top rung to shrink that; raise it to train higher at that cost.")


def cmd_audit(args) -> int:
    root = Path(args.path)
    report = audit_cache(root, IMAGE_EXTENSIONS)

    print(f"\n{root}")
    print(f"  trainable (cache + caption)  {len(report.trainable)}")
    print(f"  cached but no caption        {len(report.missing_caption)}")
    print(f"  images with no cache         {len(report.uncached_images)}")
    print(f"  cached under >1 bucket       {len(report.multi_bucket)}")
    print(f"  cache size                   {report.total_cache_bytes / 1e6:.1f} MB")
    print(f"  image size                   {report.total_image_bytes / 1e6:.1f} MB")
    if report.total_cache_bytes:
        print(f"  ratio                        "
              f"{report.total_image_bytes / report.total_cache_bytes:.1f}x smaller")

    for label, items in (("no caption", report.missing_caption),
                         ("no cache", report.uncached_images)):
        if items:
            print(f"\n  {label}: {items[:5]}{' ...' if len(items) > 5 else ''}")
    if report.multi_bucket:
        n = len(report.multi_bucket)
        widths = sorted({len(v) for v in report.multi_bucket.values()})
        print(f"\n  {n} stem(s) cached under multiple buckets ({widths} each), e.g. "
              f"{dict(list(report.multi_bucket.items())[:2])}")
        print("  Expected under multi-resolution -- each is one tier. Otherwise they are stale")
        print("  caches from an earlier config; the trainer ignores unmatched ones with a warning.")

    _report_size_distribution(root, args.bucket_reso_steps)

    if report.safe_to_delete_images:
        print("\nSAFE to delete source images: every image is cached and every cache has a caption.")
        print("Training then continues with source = \"latents\" (or \"auto\").")
        return 0

    print("\nNOT safe to delete source images -- the items listed above would be lost.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Mage-Flow latent cache tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    cc = sub.add_parser("cache-config", help="cache every dataset subset from a training TOML")
    cc.add_argument("config")
    cc.add_argument("--resolution", type=int, nargs="+", help="Override config resolution tiers")
    cc.add_argument("--batch-size", type=int, default=1)
    cc.add_argument("--devices", default="", help="Physical GPU IDs, e.g. 0,1; batch size is per GPU")
    cc.add_argument("--overwrite", action="store_true")
    cc.add_argument("--dry-run", action="store_true")
    cc.add_argument("--allow-missing-captions", action="store_true")
    cc.set_defaults(func=cmd_cache_config)

    c = sub.add_parser("cache", help="encode images to cached latents")
    c.add_argument("path")
    c.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    c.add_argument("--model-family", choices=["auto", "mage_flow"], default="auto")
    c.add_argument("--vae-path", help="Separate Mage-Flow VAE .safetensors (overrides --model-path)")
    c.add_argument("--flux2-vae", action="store_true", help="Experimental: encode with FLUX.2 VAE, pack 32c/8x to 128c/16x, and apply vae_bn normalization")
    c.add_argument("--resolution", type=int, nargs="+", default=[1024],
                   help="AREA budget(s), not side lengths. Several values cache every image at "
                        "every tier, which is what dataset.resolutions then trains on.")
    c.add_argument("--min-bucket-reso", type=int, default=256)
    c.add_argument("--max-bucket-reso", type=int, default=1920,
                   help="PER-SIDE cap; 1920 = in spec, 2048 = hard RoPE limit")
    c.add_argument("--bucket-reso-steps", type=int, default=64)
    c.add_argument("--upscale", action="store_true", help="allow upscaling small images")
    c.add_argument("--multires", action="store_true", help="area tie-break between buckets")
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--allow-missing-captions", action="store_true")
    c.add_argument("--dry-run", action="store_true",
                   help="report the bucket plan and cache size, write nothing")
    c.add_argument("--device", default="cuda")
    c.add_argument("--batch-size", type=int, default=1, help="Images per VAE encode within one resolved bucket")
    c.add_argument("--devices", default="", help="Comma-separated physical GPU IDs; launches one cache shard per GPU")
    c.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    c.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    c.set_defaults(func=cmd_cache)

    a = sub.add_parser("audit", help="check a directory before deleting its images")
    a.add_argument("path")
    a.add_argument("--bucket-reso-steps", type=int, default=64)
    a.set_defaults(func=cmd_audit)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
