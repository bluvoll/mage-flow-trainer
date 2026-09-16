"""GUI editor definitions and layout for the Mage-Flow configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import fields as F
from .widgets import RAW_GROUP_TITLES, TRANSFORMED_GROUP_TITLES


@dataclass(frozen=True)
class Spec:
    label: str
    tooltip: str
    make: Callable[[], F.Editor]
    # Rendered as a standalone checkbox with no separate label column.
    inline_label: bool = False


COMPONENT_TOOLTIPS = {
    "image_attn": "Image Q/K/V and attention output projections.",
    "text_attn": "Text Q/K/V and attention output projections in joint attention.",
    "mlp": "Both image and text feed-forward networks.",
    "adaln": "Timestep embeddings, modulation and output normalization.",
    "base": "Image/text input projections, text input norm and image output projection.",
}


def _spec(label, tooltip, make, inline_label=False):
    return Spec(label, tooltip, make, inline_label)


SPEC: dict[str, Spec] = {

    "train.text_cache_batch_size": _spec(
        "Text cache batch size",
        "Captions per encoder forward per GPU while building either text cache. Independent of training batch size. "
        "Increase for throughput when VRAM permits; reduce if caching runs out of memory.",
        lambda: F.IntEditor(1, 1024)),
    "train.caption_variations": _spec("Caption variations per image", "0 uses fixed-caption RAM caching. A positive count creates persistent augmented caption slots; exact duplicate embeddings share disk storage. Enable text caching.", lambda: F.IntEditor(0, 100000)),
    "train.caption_cache_path": _spec("Caption cache database", "SQLite path. Empty places caption_variations.sqlite in the dataset folder. Reused across runs; stores embeddings on disk, not all in RAM.", lambda: F.PathEditor("file")),
    "train.cache_text_embeddings": _spec("Cache text embeddings", "Unload Qwen3-VL before training. Set caption variations above zero to persist augmented captions and embeddings in SQLite; zero requires fixed captions.", lambda: F.BoolEditor("Cache text embeddings"), inline_label=True),
    "train.offload_text_encoder": _spec("Offload text encoder", "Move the live encoder to GPU for caption encoding and back to CPU before transformer execution. Trades transfers for lower GPU residency.", lambda: F.BoolEditor("Offload text encoder"), inline_label=True),
    "train.pack_resolutions": _spec("Pack native resolutions", "Mix differently sized images in one batch without image-token padding. Requires varlen attention, integer batch size, no curriculum or OT.", lambda: F.BoolEditor("Pack native resolutions"), inline_label=True),
    "train.checkpoint_blocks": _spec("Checkpoint blocks", "Zero-based block indices; blank checkpoints all blocks.", lambda: F.IntListEditor("e.g. 0, 2, 4")),
    "train.attention_backend": _spec("Attention backend", "torch_varlen uses PyTorch's built-in packed attention. SDPA uses padded attention.", lambda: F.ChoiceEditor(["sdpa", "torch_varlen", "flash_attn_2", "flash_attn_3"])),
    "train.max_text_tokens": _spec("Maximum text tokens", "Caption token budget after removing the Qwen3-VL system prefix.", lambda: F.IntEditor(1, 4096)),
    # ------------------------------------------------------------------ train
    "train.model_path": _spec(
        "Model (diffusers dir)",
        'Local Mage-Flow repository: transformer/, text_encoder/ and vae/.',
        lambda: F.PathEditor("folder")),
    "train.output_dir": _spec(
        "Output directory",
        "Checkpoints land in <output_dir>/<run_name>/<tag>/.",
        lambda: F.PathEditor("folder")),
    "train.transformer_path": _spec("Transformer file (optional)",
        "Native Mage-Flow .safetensors. Overrides the Diffusers transformer; leave blank to use the directory.",
        lambda: F.PathEditor("file")),
    "train.text_encoder_path": _spec("Text encoder file (optional)",
        "Qwen3-VL-4B BF16 .safetensors. Overrides the Diffusers text encoder. Uses the bundled tokenizer by default.",
        lambda: F.PathEditor("file")),
    "train.vae_path": _spec("VAE file (optional)",
        "Mage-Flow VAE .safetensors. Used by training and latent caching. Overrides the Diffusers VAE.",
        lambda: F.PathEditor("file")),
    "train.tokenizer_path": _spec("Tokenizer directory (optional)",
        "Override tokenizer assets. Blank uses the bundled Qwen3-VL tokenizer for a separate encoder file, or the Diffusers text_encoder directory.",
        lambda: F.PathEditor("folder")),
    "train.run_name": _spec(
        "Run name", "Names the output subdirectory and the checkpoint files.",
        lambda: F.TextEditor("mageflow")),
    "train.epochs": _spec(
        "Epochs", "Passes over the entry list. Under multi-resolution one epoch already covers "
        "every image at every tier, so N tiers make an epoch N times longer.",
        lambda: F.IntEditor(1, 10_000)),
    "train.max_steps": _spec(
        "Max steps", "Hard stop on optimizer steps; overrides `epochs` when it is reached first. "
        "Empty = run all epochs.",
        lambda: F.OptIntEditor("all epochs")),
    "train.batch_size": _spec(
        "Batch size",
        "One number, or one entry per resolution tier: `768=16, 1024=12, 1280=8`.\n\nKeys are "
        "matched EXACTLY against the tiers in Resolution -- this is not a threshold ladder, and a "
        "key that is not a declared tier is a config error.\n\nKeyed on the tier because the tier "
        "predicts cost and the bucket's longest side does not: bucketing targets a constant area, "
        "so tier^2/256 is an exact upper bound on tokens (measured 2304/4096/6400 at "
        "768/1024/1280), while longest sides overlap across tiers (704-1280 at tier 768 against "
        "768-1792 at tier 1024).",
        lambda: F.BucketMapEditor()),
    "train.gradient_accumulation_steps": _spec(
        "Gradient accumulation",
        'Accumulate this many microbatches per optimizer update. Increases effective batch size without retaining every microbatch graph.',
        lambda: F.IntEditor(1, 256)),
    "train.gradient_checkpointing": _spec(
        "Gradient checkpointing",
        'Recompute transformer activations during backward to reduce activation memory.',
        lambda: F.BoolEditor("Gradient checkpointing"), inline_label=True),
    "train.dtype": _spec(
        "Precision", "bfloat16 on Ada/Ampere. float32 is for debugging only.",
        lambda: F.ChoiceEditor(["bfloat16", "float16", "float32"])),
    "train.compressed_adaln_dtype": _spec(
        "Compressed AdaLN precision",
        "Only compressed Mage-Flow: precision of the shared projection and per-block heads. "
        "FP32 is the default; BF16 reduces their weight and gradient memory. Ordinary models are unaffected.",
        lambda: F.ChoiceEditor(["float32", "bfloat16"])),
    "train.seed": _spec(
        "Seed", "Seeds sampling, caption RNG and the batch order. Each sample folds the seed with "
        "its index and epoch, so DDP ranks draw different captions rather than identical ones.",
        lambda: F.IntEditor(0, 2**31 - 1)),
    "train.num_workers": _spec(
        "Dataloader workers",
        "Re-forked every epoch on purpose: the sampler's epoch is read at __iter__ time, so a "
        "persistent worker would keep serving batches built from a stale epoch.",
        lambda: F.IntEditor(0, 32)),
    "train.vae_encode_chunk": _spec(
        "VAE encode chunk",
        'Images per frozen Mage-VAE forward when encoding live. Smaller chunks reduce transient memory.',
        lambda: F.IntEditor(1, 32)),
    "train.save_every_steps": _spec(
        "Save every N steps", "Empty = never; use epochs instead.",
        lambda: F.OptIntEditor("off")),
    "train.save_every_epochs": _spec(
        "Save every N epochs", "Empty = only the final save.",
        lambda: F.OptIntEditor("off")),
    "train.save_native": _spec(
        "Native single-file save",
        "Write the ComfyUI-loadable layout as well as the diffusers directory. Verified "
        "bit-identical round trip on 685 tensors.",
        lambda: F.BoolEditor("Also write ComfyUI single-file layout"), inline_label=True),
    "train.keep_last_n": _spec(
        "Keep last N checkpoints",
        "Prunes older periodic checkpoints as new ones land. Only ever considers directories this "
        "trainer wrote with an epochNNN/stepNNNNNN tag AND a state.json -- `final` and anything "
        "else in the output directory are never touched.\n\nWorth setting for long full finetunes, "
        "where optimizer state costs ~3.5GB per checkpoint.",
        lambda: F.OptIntEditor("keep all")),
    "train.resume_from": _spec(
        "Resume from",
        "Path to a checkpoint directory. Refuses to start if it holds no accelerator/ state, "
        "rather than silently restarting the optimizer moments from zero.",
        lambda: F.OptTextEditor("no resume")),
    "train.save_optimizer_state": _spec(
        "Save optimizer state",
        'Save Accelerate model, optimizer, scheduler and RNG state for resume; consumes additional disk space.',
        lambda: F.BoolEditor("Save optimizer state (required to resume)"), inline_label=True),
    "train.allow_multi_gpu_texture": _spec(
        "Allow texture mode on multiple GPUs",
        "OFF by default, and the trainer REFUSES the combination rather than warning about it.\n\n"
        "Two multi-GPU texture runs degraded anatomy progressively against a single-GPU run at "
        "MATCHED optimizer steps -- so it is not the halved step count, and more epochs do not fix "
        "it. The cause is not identified: ranks seeding identically was found and fixed, but was "
        "never shown to be the mechanism.\n\n"
        "This is not an 'I know better' switch. It exists so re-testing that fix is deliberate. If "
        "a run under it comes out clean against a single-GPU control at matched steps, the gate "
        "should be removed -- not the override left on.",
        lambda: F.BoolEditor("Allow texture curricula on >1 GPU (not recommended)"),
        inline_label=True),
    "train.skip_final_save": _spec(
        "Skip final save", "Benchmarking only.",
        lambda: F.BoolEditor("Skip final save (benchmarking)"), inline_label=True),
    "train.log_every": _spec(
        "Log every N steps", "Also the sampling rate of the live graphs.",
        lambda: F.IntEditor(1, 1000)),
    "train.progress": _spec(
        "Progress display",
        "Under the GUI stdout is a pipe, so `auto` resolves to the plain per-step line -- which is "
        "the one the graphs parse. Leave it on auto unless you are debugging.",
        lambda: F.ChoiceEditor(["auto", "plain", "bar", "off"])),
    "train.compile": _spec(
        "torch.compile",
        'Compile the shared block callable. Supports default or max-autotune-no-cudagraphs with SDPA or torch_varlen. First steps include compilation.',
        lambda: F.ChoiceEditor(["off", "default", "max-autotune-no-cudagraphs"],
                               [None, "default", "max-autotune-no-cudagraphs"])),
    "train.compile_dynamic": _spec(
        "Dynamic shapes",
        "Effectively mandatory with bucketing. Without it every bucket compiles its own graph and "
        "dynamo's recompile limit is exhausted within a few buckets, after which it SILENTLY falls "
        "back to eager for the rest of the run.",
        lambda: F.BoolEditor("Dynamic shapes (keep on)"), inline_label=True),
    "train.compile_regional": _spec(
        "Regional compilation",
        "Compile the repeated block once instead of the whole 28-block trunk. Without it, "
        "full-model compile ran 25+ minutes here without completing a single step.",
        lambda: F.BoolEditor("Compile the repeated block, not the whole trunk"), inline_label=True),

    # ------------------------------------------------------------------ dataset
    "dataset.path": _spec(
        "Dataset directory",
        "Images with .txt (tags) and optional _nl.txt (natural language) sidecars, or cached "
        "latents named <stem>_WWWWxHHHH_trainer.safetensors.\n\nIgnored once Extra subsets below has "
        "any rows -- the two forms are mutually exclusive and the subset list wins.",
        lambda: F.PathEditor("folder")),
    "dataset.subsets": _spec(
        "Extra subsets",
        "Train from several folders, each with its own repeat count and texture eligibility. "
        "Leave empty for the ordinary single-directory config above.\n\nThe Texture crops column "
        "is the reason this exists: unchecked keeps a folder in full resolution for the whole run "
        "with its captions intact. That is what a regularization set needs -- texture crops "
        "replace the caption with texture.trigger, so an uncaptioned flat colour would train the "
        "unconditional branch, and CFG subtracts the unconditional.",
        F.SubsetEditor),
    "dataset.source": _spec(
        "Source",
        "auto = use cached latents when present, else encode images. latents = cache only, which "
        "is what lets you delete the images (884MB -> 67MB on a uniform set). Run the audit first: it "
        "refuses to call deletion safe while any image is uncached or any latent has lost its "
        "caption.\n\nencode = run the VAE every step instead of reading a cache. REQUIRED for "
        "texture crops, which are chosen per sample per step and so cannot be cached. Measured "
        "cost against cached latents: +0.3-0.4GB and +16% step time, with no reduction in the "
        "batch size that fits.",
        lambda: F.ChoiceEditor(["auto", "images", "latents", "encode"])),
    "dataset.resolution": _spec(
        "Resolution (single tier)",
        "An AREA budget, not a side length: 1024 means ~1024x1024 worth of pixels, so 1920x1080 "
        "becomes 1344x768.\n\nIgnored when a multi-resolution ladder is set below.",
        lambda: F.IntEditor(128, 1920, 64)),
    "dataset.resolutions": _spec(
        "Multi-resolution ladder",
        "Every image trains at every tier -- the cross product, not a curriculum. N tiers = N x "
        "the epoch and N x the cache.\n\nChoose against your dataset's own size distribution: a "
        "tier above an image's native area yields the same bucket and only buys a repeat. Run the "
        "Audit button first; it prints the percentile table and proposes a ladder.\n\nEmpty = "
        "single tier.",
        lambda: F.IntListEditor()),
    "dataset.tier_collapse": _spec(
        "Tier collapse handling",
        "What to do when two tiers pick the same bucket for one image, which happens above that "
        "image's native area.\n\ndedup: one entry (default). repeat: keep both, so a small image's "
        "repeat factor silently equals the ladder length.",
        lambda: F.ChoiceEditor(["dedup", "repeat"])),
    "dataset.min_source_area": _spec(
        "Min source area",
        "Drop sources below this fraction of the smallest tier's area budget. 0 = keep everything. "
        "Mage-Flow has no size micro-conditioning, so it cannot tell 'this source was small' from "
        "'this concept has little detail' -- a floor is the only lever.",
        lambda: F.FloatEditor(0.0, 1.0, 0.05, 2)),
    "dataset.min_bucket_reso": _spec(
        "Min bucket resolution",
        "INERT while `no upscale` is on -- the no-upscale branch never reads it (verified: 256, "
        "128 and 64 give byte-identical buckets). Every image's floor is its own native area.",
        lambda: F.IntEditor(64, 1024, 64)),
    "dataset.max_bucket_reso": _spec(
        "Max bucket resolution (per side)",
        'Maximum image side in pixels. Resolution controls area separately; memory use grows with image size.',
        lambda: F.IntEditor(256, 2**31 - 1, 64)),
    "dataset.bucket_reso_steps": _spec(
        "Bucket step", "Bucket sides are multiples of this. 64 matches sd-scripts.",
        lambda: F.IntEditor(8, 256, 8)),
    "dataset.bucket_no_upscale": _spec(
        "Never upscale",
        "Cap each image's bucket by its own source area. The per-side limit still shrinks an "
        "ultrawide source uniformly -- only ever down, so the no-upscale promise holds exactly.",
        lambda: F.BoolEditor("Never upscale a source to reach the target"), inline_label=True),
    "dataset.multires_training": _spec(
        "Area tie-break",
        "sd-scripts fork behaviour: break bucket ties by area rather than by aspect error. On a "
        "1920x1080 source this picks an exact 1024x576 instead of 1344x768.\n\nUnrelated to the "
        "multi-resolution ladder above, despite the name -- it is sd-scripts' spelling.",
        lambda: F.BoolEditor("Break bucket ties by area"), inline_label=True),
    "dataset.num_repeats": _spec(
        "Repeats", "Multiplies every entry. Stacks on top of the tier cross product.",
        lambda: F.IntEditor(1, 100)),

    # ------------------------------------------------------------------ texture crops
    "curriculum": _spec(
        "Training curriculum",
        "A step function over training progress: from `Start %` onward, draw timesteps from "
        "[t min, t max] in this mode. The last phase whose start is at or below the current "
        "progress wins, so phases run until the next one begins.\n\n"
        "This is the ONLY way to reach texture mode -- `texture` is a phase mode, not a dataset "
        "switch. With no phases the run is plain full-resolution training, which is the default.\n"
        "\nTexture phases require Dataset source = encode: a crop is chosen per sample per step, "
        "so there is no fixed latent to have cached.",
        lambda: F.CurriculumEditor()),

    # ------------------------------------------------------------------ captions
    "dataset.texture.oversize": _spec(
        "Oversize handling",
        "What to do when a source is smaller than the chosen canvas.\n\ncover: scale up to cover, "
        "then crop -- invents no pixels.\npad: reproduce TrainTrain exactly, including its black "
        "bands (PIL fills out-of-bounds crops with zeros). Measured on one texture set this fires for "
        "36-43% of images at the square canvases and 83% at 1536x640. Use it only to reproduce an "
        "existing checkpoint.\nskip: exclude the image for that batch.",
        lambda: F.ChoiceEditor(["cover", "pad", "skip"])),
    "dataset.texture.canvases": _spec(
        "Texture canvases",
        "Crop sizes, WxH, drawn one per batch so the batch can be stacked. Duplicates are kept "
        "and are meaningful -- the reference lists 1024x1024 twice to give it a 1/3 chance.\n\n"
        "Crop freedom depends entirely on source resolution: median slack for a 1024x1024 canvas "
        "is 1001px on a mixed set (4.4MP sources) but 176px on a texture set (1.2MP). The startup report says "
        "which canvases your dataset can actually fill.",
        lambda: F.CanvasListEditor()),
    "dataset.texture.energy_power": _spec(
        "Crop energy power",
        "Crop positions are sampled with probability proportional to Laplacian detail raised to "
        "this power. 0 makes position uniform, which is the honest control for whether energy "
        "selection is doing anything. Measured on a synthetic image with detail only past x=1100: "
        "mean crop x is 510 at power 0, 720 at power 1, 846 at power 3.",
        lambda: F.FloatEditor(0.0, 8.0, step=0.5)),
    "dataset.texture.energy_downscale": _spec(
        "Energy map downscale",
        "The energy map is built at 1/N scale to keep crop scoring cheap. The Laplacian itself is "
        "taken at FULL resolution and only then averaged down -- downscaling first would low-pass "
        "away the fine texture this is meant to find.",
        lambda: F.IntEditor(1, 32)),
    "dataset.texture.feather_latent_px": _spec(
        "Mask feather (latent px)",
        "Cosine taper on the supervised sub-region's edges, in latent pixels (1 latent px = 8 "
        "image px).",
        lambda: F.IntEditor(0, 32)),
    "dataset.texture.mask_ratio": _spec(
        "Mask size range",
        "The supervised sub-region as a fraction of the canvas, drawn per sample. The unmasked "
        "remainder is real texture that still conditions the forward pass but carries no target, "
        "so it adds context without adding gradient variance. Varying position stops the model "
        "tying learned content to a fixed spot on the canvas.",
        lambda: F.NumListEditor("0.5, 1.0", cast=float, as_tuple=True)),
    "dataset.texture.fit_aware": _spec(
        "Fit-aware canvas choice",
        "Draw each batch's canvas from those its images can actually hold, instead of drawing "
        "blind and forcing the fit. Measured on one texture set: padding falls from 56% of draws to 1%, "
        "and mean black area from 10.7% to 0.1%, with NO upscaling -- 99% of that dataset fits "
        "some canvas.\n\nThe trade is that the canvas mix follows your dataset's shapes rather "
        "than the flat preset weighting, so extreme aspects get rarer. Those were the ones "
        "producing 15-25% black area.\n\nPer BATCH, not per sample, since a batch must stack -- so "
        "a larger batch has fewer canvases available and this degrades toward the blind draw.",
        lambda: F.BoolEditor("Pick a canvas the batch can hold"), inline_label=True),
    "dataset.texture.cover_max_scale": _spec(
        "Gentle upscale ceiling",
        "A small upscale beats an invented pixel; a large one does not -- unrestricted cover lost "
        "its A/B by softening faces at a mean 1.25x. This caps it. At 1.15x about a quarter of "
        "oversize draws resolve here and the rest fall through to padding. 1.0 disables the rung.",
        lambda: F.FloatEditor(1.0, 4.0, step=0.05)),
    "dataset.texture.pad_mode": _spec(
        "Padding fill",
        "What the remainder is padded with.\n\nreflect: mirror the image's own content into the "
        "overflow -- implausible as composition, but every pixel is real texture.\nblack: PIL's "
        "zero fill, the reference's behaviour. A padded run learned this well enough to emit a "
        "black bar in roughly 1 of 50 generations, which is why it is no longer the default.",
        lambda: F.ChoiceEditor(["reflect", "black"])),
    "dataset.texture.mask_padding": _spec(
        "Exclude padding from the loss",
        "Invented pixels stay in the forward pass as context but carry no target, so no gradient "
        "ever tells the model to produce them. This is the direct fix for the black-bar artifact "
        "and it composes with every other rung -- worth leaving on even at reflect padding, since "
        "mirrored content is plausible but still not what was in the image.",
        lambda: F.BoolEditor("Invented pixels carry no gradient"), inline_label=True),
    "dataset.texture.trigger": _spec(
        "Texture trigger word",
        "Texture crops are captioned with this ALONE, never the image's tags. Not a convenience: "
        "Mage-Flow has no size micro-conditioning, so it cannot tell a zoomed crop from a large "
        "subject. Captioning a 1024px crop of a 4000px image with whole-image tags teaches that "
        "confusion into every one of those tags. Empty = unconditional.",
        lambda: F.TextEditor()),
    "dataset.caption.caption_mode": _spec(
        "Caption mode",
        "tags = .txt, nl = _nl.txt, tags_nl / nl_tags concatenate in that order, mixed draws per "
        "sample from the weights below.\n\nText embeddings are never cached, precisely so this can "
        "vary per epoch.",
        lambda: F.ChoiceEditor(["tags", "nl", "tags_nl", "nl_tags", "mixed"])),
    "dataset.caption.mixed_weights": _spec(
        "Mixed weights",
        "Relative weights, not percentages -- they need not sum to 100. Only used when the mode "
        "is `mixed`.",
        lambda: F.WeightsEditor(["tags", "nl", "tags_nl", "nl_tags"])),
    "dataset.caption.shuffle_tags": _spec(
        "Shuffle tags", "Breaks positional memorisation of tag order.",
        lambda: F.BoolEditor("Shuffle tags"), inline_label=True),
    "dataset.caption.tag_delimiter": _spec(
        "Tag delimiter",
        "How tags are split and rejoined. Whitespace is significant -- the default is a comma "
        "AND a space, and trimming it to a bare comma rewrites every caption.",
        lambda: F.TextEditor(", ", strip=False)),
    "dataset.caption.shuffle_keep_first_n": _spec(
        "Pin first N tags",
        "Hold the first N tags in place while shuffling the rest -- the usual way to keep a "
        "trigger word first. Verified 3000/3000.",
        lambda: F.IntEditor(0, 20)),
    "dataset.caption.tag_dropout_percent": _spec(
        "Tag dropout",
        "Fraction of tags dropped per sample. The floor below is respected exactly, even at 100%.",
        lambda: F.FloatEditor(0.0, 1.0, 0.05, 2)),
    "dataset.caption.min_tags_kept": _spec(
        "Min tags kept", "Dropout never takes a caption below this many tags.",
        lambda: F.IntEditor(0, 50)),
    "dataset.caption.protected_tags": _spec(
        "Protected tags",
        "Never dropped, never shuffled out of position. Comma separated. Verified 3000/3000.",
        lambda: F.StrListEditor("trigger, character name, ...", as_set=True, height=60)),
    "dataset.caption.caption_dropout_percent": _spec(
        "Caption dropout",
        "Fraction of samples trained with an empty caption -- classifier-free guidance "
        "conditioning.",
        lambda: F.FloatEditor(0.0, 1.0, 0.05, 2)),
    "dataset.caption.nl_shuffle_sentences": _spec(
        "Shuffle NL sentences", "Applies to the natural-language caption only.",
        lambda: F.BoolEditor("Shuffle NL sentences"), inline_label=True),
    "dataset.caption.nl_keep_first_sentence": _spec(
        "Pin first NL sentence", "Hold sentence one in place while shuffling the rest.",
        lambda: F.BoolEditor("Pin the first NL sentence"), inline_label=True),

    # ------------------------------------------------------------------ flow
    "flow.timestep_sample_method": _spec(
        "Timestep sampling",
        "logit_normal concentrates t near 0.5; uniform is flat.\n\nNo per-timestep loss weighting "
        "exists here on purpose: for rectified flow uniform weighting is already correct, and the "
        "timestep DISTRIBUTION is the lever. Weighting as well would double-count.",
        lambda: F.ChoiceEditor(["logit_normal", "uniform"])),
    "flow.sigmoid_scale": _spec(
        "Sigmoid scale",
        "Widens or narrows the logit-normal draw. Measured p10/p90: 0.5 -> 0.34/0.65, "
        "1.0 -> 0.21/0.78, 2.0 -> 0.07/0.93.\n\nHAS NO EFFECT under `uniform` -- setting both is a "
        "config error rather than a silent no-op.",
        lambda: F.FloatEditor(0.05, 10.0, 0.1, 2)),
    "flow.shift": _spec(
        "Shift",
        "Static timestep shift; pushes t higher (more noise). Inference default is 3.0. Latents "
        "here normalise to std ~0.69 rather than 1.0, so effective noise is already higher than a "
        "unit-variance schedule assumes -- shift < 3 is worth an A/B on illustration data.\n\n"
        "Leave EMPTY for no shift. Do not type 0: the map sends every timestep to exactly 0, so "
        "the model is fed clean latents and asked to predict pure noise. Loss sits at 1.0 for the "
        "whole run and nothing is learned. Values <= 0 are rejected.",
        lambda: F.SciEditor("none", optional=True)),
    "flow.flux_shift": _spec(
        "Resolution-aware shift",
        'Optional resolution-dependent shift. Mage-Flow pretraining uses the static shift instead; default is shift=6.',
        lambda: F.BoolEditor("Scale shift with resolution (flux_shift)"), inline_label=True),
    "flow.phase_mapping": _spec(
        "Curriculum phase mapping",
        "How a curriculum phase's t_range restricts the timestep draw. Inert without a "
        "[[curriculum]].\n\nrescale (TrainTrain's behaviour): squeeze the whole distribution into "
        "the range, so shape is preserved and every quantile is span x the full one -- "
        "logit_normal(1.3) on [0, 0.6] goes median 0.499 -> 0.299.\n\ntruncate: keep the "
        "pretraining density and drop out-of-range draws, which skews mass toward the top of the "
        "range -- same setup gives median 0.344. Raises rather than hangs if the range holds "
        "almost no probability.",
        lambda: F.ChoiceEditor(["rescale", "truncate"])),
    "flow.use_ot": _spec(
        "Optimal transport pairing",
        "Pair noise to latents within the batch by cosine cost instead of at random, which "
        "shortens the average transport path and lowers gradient variance.\n\nA NO-OP at batch "
        "size 1 -- there is nothing to permute. The log prints `ot X.XX`, the fraction of rows "
        "actually reordered (measured 0.00 at batch 1, 1.00 at batch 2, 0.83 at batch 12). "
        "Requires scipy.",
        lambda: F.BoolEditor("Cosine optimal-transport noise pairing"), inline_label=True),
    "flow.hf_scale": _spec(
        "High-frequency loss scale",
        "Weight on the Laplacian-energy-weighted x0 term, added to plain MSE. 0 = off, and off is "
        "bit-identical to the term not existing, so a same-seed A/B is a real A/B.\n\nSuggested "
        "start 0.25. The log decomposes as `loss X (mse Y + hf Z)`.",
        lambda: F.FloatEditor(0.0, 10.0, 0.05, 3)),
    "flow.hf_exponent": _spec(
        "High-frequency exponent",
        'Controls detail weighting in the auxiliary token loss. Mage-Flow uses one latent pixel per token, so patch-local variance is zero and this exponent has no effect.',
        lambda: F.FloatEditor(0.1, 5.0, 0.1, 2)),

    # ------------------------------------------------------------------ optimizer
    "optimizer.kind": _spec(
        "Optimizer",
        "Grouped by implementation. Switching resets betas, epsilon, weight decay and state options "
        "to upstream defaults, while retaining your learning rate. SDNQ supports state quantization/offload.",
        lambda: F.OptimizerEditor()),
    "optimizer.lr": _spec(
        "Learning rate",
        "Scientific notation accepted. Note the schedule changes what a given LR does: mean "
        "multiplier over 1000 steps is 0.50 for linear/cosine, 0.76 for rerex, 0.83 for rex -- so "
        "moving cosine -> rex at the same LR is a ~1.7x increase in total movement.",
        lambda: F.SciEditor("3e-5")),
    "optimizer.betas": _spec(
        "Betas", "Defaults follow the optimizer. SDNQ Adafactor uses a negative decay exponent; CAME and Optimi Adan use three values.",
        lambda: F.FloatListEditor("0.9, 0.99")),
    "optimizer.eps": _spec("Epsilon", "Denominator floor.", lambda: F.SciEditor("1e-8")),
    "optimizer.weight_decay": _spec(
        "Weight decay", "Decoupled (AdamW).", lambda: F.FloatEditor(0.0, 1.0, 0.005, 4)),
    "optimizer.max_grad_norm": _spec(
        "Gradient clipping", "0 disables. Clipping runs only on sync steps, under Accelerate.",
        lambda: F.FloatEditor(0.0, 100.0, 0.1, 2)),
    "optimizer.quantize_state": _spec(
        "Quantize optimizer state",
        'Quantize SDNQ optimizer moments to reduce state storage.',
        lambda: F.BoolEditor("Quantize optimizer state"), inline_label=True),
    "optimizer.offload_state": _spec(
        "Offload optimizer state",
        'Keep SDNQ optimizer state in host memory; trades GPU memory for transfer overhead.',
        lambda: F.BoolEditor("Offload optimizer state to CPU"), inline_label=True),
    "optimizer.use_kahan": _spec(
        "SDNQ Kahan summation",
        "SDNQ use_kahan: retain rounding residuals alongside stochastic rounding. Costs an additional buffer.",
        lambda: F.BoolEditor("SDNQ Kahan summation"), inline_label=True),
    "optimizer.kahan_sum": _spec(
        "Optimi Kahan summation",
        "Optimi kahan_sum: Auto enables compensation for low-precision parameters. Adan defaults to Disabled.",
        lambda: F.ChoiceEditor(["Optimizer default", "Auto (parameter dtype)", "Enabled", "Disabled"], [None, "auto", True, False])),
    "optimizer.gradient_release": _spec(
        "Optimi gradient release",
        "Update parameters during backward and immediately release gradients. Single GPU only. "
        "Requires gradient accumulation=1 and gradient clipping=0. DDP is blocked before model loading.",
        lambda: F.BoolEditor("Optimi gradient release (single GPU)"), inline_label=True),
    "optimizer.momentum": _spec(
        "SGD momentum", "Only used by Optimi SGD.", lambda: F.FloatEditor(0.0, 0.9999, 0.01, 4)),

    # ------------------------------------------------------------------ schedule
    "schedule.kind": _spec(
        "Schedule",
        "rex and rerex hold the LR near peak far longer than cosine. Both were ported from "
        "sd-scripts and verified to 0.0 max|diff| against it -- but there `d` is hardcoded and "
        "here every parameter is tunable.\n\nReRex is MONOTONE, not a warm-restart schedule: "
        "segment i ends exactly where i+1 begins.",
        lambda: F.ChoiceEditor(["constant", "cosine", "linear", "rex", "rerex"])),
    "schedule.warmup_steps": _spec(
        "Warmup steps", "Linear ramp from 0 to the peak LR.", lambda: F.IntEditor(0, 100_000)),
    "schedule.min_lr_ratio": _spec(
        "Min LR ratio",
        "Floor for every decaying schedule, as a fraction of peak. 0.001 reproduces sd-scripts; "
        "leaving it at 0 decays all the way to zero instead.",
        lambda: F.FloatEditor(0.0, 0.999, 0.001, 4)),
    "schedule.d": _spec(
        "REX d",
        "Decay sharpness. 0 is linear; higher holds the peak longer. sd-scripts hardcodes 0.9. "
        "Must be < 1: at exactly 1 the REX denominator collapses onto the numerator and the last "
        "step is 0/0, so it is rejected rather than clamped.",
        lambda: F.FloatEditor(0.0, 0.999, 0.01, 3)),
    "schedule.global_d": _spec(
        "ReREX global d", "`d` of the outer curve the segment endpoints ride.",
        lambda: F.FloatEditor(0.0, 0.999, 0.01, 3)),
    "schedule.local_d": _spec(
        "ReREX local d", "`d` of the decay inside each segment.",
        lambda: F.FloatEditor(0.0, 0.999, 0.01, 3)),
    "schedule.weight_power": _spec(
        "ReREX weight power",
        "Skews the step budget toward early segments; 0 gives equal lengths. At 1.5 with 8 "
        "segments the last segment gets ~1.2% of the budget, so below ~85 steps segments round to "
        "zero length and the curve is no longer the documented one.",
        lambda: F.FloatEditor(0.0, 10.0, 0.1, 2)),
    "schedule.num_segments": _spec(
        "ReREX segments", "Number of recursive segments.", lambda: F.IntEditor(1, 64)),

    "component_lr.adaln": _spec(
        "Train AdaLN",
        "Full finetuning only. Enabled by default; disable to freeze AdaLN modulation and timestep projections. "
        "Uses the global learning rate unless the loaded TOML specifies an AdaLN rate, which is preserved. "
        "Training AdaLN adds gradient and optimizer-state memory. LoRA always freezes AdaLN.",
        lambda: F.TrainComponentEditor("Train AdaLN"), inline_label=True),

    # ------------------------------------------------------------------ adapter
    "adapter.kind": _spec(
        "Training mode",
        'none trains base parameters according to component LRs; lora and lokr use PEFT; lycoris_lora uses LyCORIS with compile-compatible adapter kernels.',
        lambda: F.ChoiceEditor(["none (full finetune)", "LoRA (PEFT)", "LyCORIS", "LoKr (PEFT)"], ["none", "lora", "lycoris_lora", "lokr"])),
    "adapter.lycoris_algo": _spec("LyCORIS algorithm", "LoCon and LoKr default to bypass; DoRA defaults to output-axis weight decomposition. AdaLN is always excluded.", lambda: F.ChoiceEditor(["LoCon (LyCORIS)", "LoKr (LyCORIS)", "DoRA (LyCORIS)"], ["locon", "lokr", "dora"])),
    "adapter.lycoris_bypass": _spec("LyCORIS bypass", "Compute the adapter separately from the frozen base. Disabled for weight decomposition so magnitude normalization is applied.", lambda: F.BoolEditor("Bypass base weight merging"), inline_label=True),
    "adapter.lycoris_wd_on_output": _spec("Decompose on output", "Normalize weight decomposition along the output dimension (wd_on_out).", lambda: F.BoolEditor("Weight decomposition on output"), inline_label=True),
    "adapter.dtype": _spec("Adapter dtype", "FP32 is the default. BF16 reduces adapter weights, gradients and projection activation memory; validate its training quality separately.", lambda: F.ChoiceEditor(["float32", "bfloat16"])),
    "adapter.rank": _spec("Rank", "LoRA rank / LoKr dimension.", lambda: F.IntEditor(1, 512)),
    "adapter.alpha": _spec(
        "Alpha",
        "peft scales by alpha/r and kohya by alpha/dim, so this transfers unchanged between the "
        "two export formats.",
        lambda: F.FloatEditor(0.1, 512.0, 1.0, 2)),
    "adapter.dropout": _spec(
        "Dropout", "Applied to the adapter path only.", lambda: F.FloatEditor(0.0, 0.9, 0.05, 2)),
    "adapter.lokr_factor": _spec(
        "LoKr factor", "Passed as factor to LyCORIS LoKr. -1 selects balanced factorization; positive values select the factorization factor. Independent of rank.", lambda: F.IntEditor(-1, 2**31 - 1)),
    "adapter.lokr_decompose_both": _spec(
        "LoKr decompose both", "Factor both Kronecker operands rather than one.",
        lambda: F.BoolEditor("Decompose both LoKr factors"), inline_label=True),

    # ------------------------------------------------------------------ quant
    "quant.mode": _spec(
        "SDNQ mode",
        'frozen trains adapters on quantized base weights; training updates quantized master weights for full finetuning.',
        lambda: F.ChoiceEditor(["none", "frozen", "training"])),
    "quant.weights_dtype": _spec(
        "Weight dtype",
        'SDNQ weight storage dtype. Benchmark quality and throughput on your data.',
        lambda: F.ChoiceEditor(["int8", "uint8", "fp8", "int7", "int6", "int5", "int4"])),
    "quant.use_quantized_matmul": _spec(
        "Quantized matmul",
        'Auto defaults to off. Explicit on quantizes activations too; compatibility and performance depend on the SDNQ/PyTorch versions.',
        lambda: F.ChoiceEditor(["auto", "on", "off"], ["auto", True, False])),
    "quant.skip_policy": _spec(
        "Skip policy",
        'Input/output and timestep projections stay in high precision. Optional policies also protect modulation or MLP down projections.',
        lambda: F.ChoiceEditor(
            ["default", "first_block_adaln", "all_adaln", "mlp_down"])),
    "quant.extra_skip": _spec(
        "Extra skip keys",
        "SDNQ key matching is NOT substring matching: a key matches only as a whole parameter "
        "name, a whole dot-separated component, or a glob. `norm1.linear_` matches nothing.",
        lambda: F.StrListEditor("comma separated", height=54)),

    "quant.quantize_text_encoder": _spec(
        "Quantize Qwen3", 'Quantize the frozen Qwen3-VL encoder with SDNQ for loaded or cached text encoding. Independent of transformer quantization: enabled also with quant.mode=none and Optimi.',
        lambda: F.BoolEditor("Quantize the text encoder"), inline_label=True),
    "quant.text_encoder_weights_dtype": _spec(
        "Text encoder storage dtype",
        "Independent encoder storage dtype. Inherit uses transformer storage dtype. Keep INT8 here when comparing INT8 and UINT8 transformer storage to reuse the same text cache.",
        lambda: F.ChoiceEditor(["Inherit", "int8", "uint8", "fp8"], [None, "int8", "uint8", "fp8"])),
    "quant.dynamic_loss_threshold": _spec(
        "Dynamic loss threshold", "SDNQ internal. Empty = library default.",
        lambda: F.SciEditor(optional=True)),
    "quant.group_size": _spec(
        "Group size", "0 = per-channel. SDNQ internal.", lambda: F.IntEditor(0, 1024)),
    "quant.use_stochastic_rounding": _spec(
        "Stochastic rounding",
        "What keeps small updates from vanishing into a quantized master weight. Leave on for "
        "`training` mode.",
        lambda: F.BoolEditor("Stochastic rounding"), inline_label=True),
}


# ---------------------------------------------------------------- layout

LAYOUT: list[tuple[str, list[tuple[str, list[str]]]]] = [
    ("Dataset", [
        ("Dataset Source", [
            "dataset.path", "dataset.source", "dataset.num_repeats", "dataset.subsets",
        ]),
        ("Resolution", [
            "dataset.resolution", "dataset.resolutions", "dataset.tier_collapse",
            "dataset.min_source_area",
        ]),
        ("Aspect Ratio Bucketing", [
            "dataset.max_bucket_reso", "dataset.min_bucket_reso", "dataset.bucket_reso_steps",
            "dataset.bucket_no_upscale", "dataset.multires_training",
        ]),
        ("Curriculum", [
            "curriculum",
        ]),
        ("Texture Crops", [
            "dataset.texture.canvases", "dataset.texture.trigger",
            "dataset.texture.mask_ratio", "dataset.texture.feather_latent_px",
            "dataset.texture.energy_power", "dataset.texture.energy_downscale",
        ]),
        # The four rungs plus their terminal fallback, kept together and in the order they fire so
        # the cascade reads as one decision rather than five unrelated switches.
        ("Oversize Cascade", [
            "dataset.texture.fit_aware", "dataset.texture.cover_max_scale",
            "dataset.texture.pad_mode", "dataset.texture.mask_padding",
            "dataset.texture.oversize",
        ]),
        ("Captions", [
            "dataset.caption.caption_mode", "dataset.caption.mixed_weights",
            "dataset.caption.tag_delimiter",
        ]),
        ("Caption Augmentation", [
            "dataset.caption.shuffle_tags", "dataset.caption.shuffle_keep_first_n",
            "dataset.caption.tag_dropout_percent", "dataset.caption.min_tags_kept",
            "dataset.caption.caption_dropout_percent", "dataset.caption.protected_tags",
            "dataset.caption.nl_shuffle_sentences", "dataset.caption.nl_keep_first_sentence",
        ]),
    ]),
    ("Training", [
        ("Model / Output", [
            "train.model_path", "train.transformer_path", "train.text_encoder_path",
            "train.vae_path", "train.tokenizer_path", "train.output_dir", "train.run_name",
        ]),
        ("Schedule Length", [
            "train.epochs", "train.max_steps", "train.seed",
        ]),
        ("Batching", [
            "train.batch_size", "train.gradient_accumulation_steps", "train.num_workers",
            "train.vae_encode_chunk",
        ]),
        ("Memory / Precision", [
            "train.dtype", "train.compressed_adaln_dtype", "train.gradient_checkpointing", "train.pack_resolutions", "train.checkpoint_blocks", "train.attention_backend", "train.max_text_tokens", "train.cache_text_embeddings", "train.text_cache_batch_size", "train.caption_variations", "train.caption_cache_path", "train.offload_text_encoder",
        ]),
        ("torch.compile", [
            "train.compile", "train.compile_dynamic", "train.compile_regional",
        ]),
        ("Checkpoints", [
            "train.save_every_epochs", "train.save_every_steps", "train.keep_last_n",
            "train.save_native", "train.save_optimizer_state", "train.resume_from",
            "train.skip_final_save",
        ]),
        ("Logging", [
            "train.log_every", "train.progress",
        ]),
        ("Guardrails", [
            "train.allow_multi_gpu_texture",
        ]),
    ]),
    ("Optimizer", [
        ("Optimizer", [
            "optimizer.kind", "optimizer.lr", "optimizer.betas", "optimizer.eps", "optimizer.momentum",
            "optimizer.weight_decay", "optimizer.max_grad_norm",
        ]),
        ("Optimizer State", [
            "optimizer.quantize_state", "optimizer.offload_state", "optimizer.use_kahan", "optimizer.kahan_sum", "optimizer.gradient_release",
        ]),
        ("Learning Rate Schedule", [
            "schedule.kind", "schedule.warmup_steps", "schedule.min_lr_ratio",
        ]),
        ("REX / ReREX", [
            "schedule.d", "schedule.global_d", "schedule.local_d", "schedule.weight_power",
            "schedule.num_segments",
        ]),
    ]),
    ("Method", [
        ("Adapter", [
            "adapter.kind", "adapter.lycoris_algo",
            "adapter.lokr_factor", "adapter.lokr_decompose_both",
            "adapter.lycoris_bypass", "adapter.lycoris_wd_on_output", "adapter.rank", "adapter.dtype", "adapter.alpha", "adapter.dropout",
        ]),
        ("Full Finetuning", ["component_lr.adaln"]),
        ("Flow Matching", [
            "flow.timestep_sample_method", "flow.sigmoid_scale", "flow.shift",
            "flow.flux_shift", "flow.use_ot", "flow.phase_mapping",
        ]),
        ("High-Frequency Token Loss", [
            "flow.hf_scale", "flow.hf_exponent",
        ]),
        ("SDNQ Quantization", [
            "quant.mode", "quant.weights_dtype", "quant.use_quantized_matmul",
            "quant.skip_policy", "quant.extra_skip",
            "quant.quantize_text_encoder", "quant.text_encoder_weights_dtype", "quant.use_stochastic_rounding",
            "quant.group_size", "quant.dynamic_loss_threshold",
        ]),
    ]),
]


def layout_keys() -> list[str]:
    return [k for _, groups in LAYOUT for _, keys in groups for k in keys]


# Group tinting, in Aozora's two-tone convention: groups that describe the data as it is on disk
# read "raw"; groups that change what the model sees read "transformed".
RAW_GROUP_TITLES.update({
    "Dataset Source", "Model / Output", "Checkpoints", "Logging", "Batching",
    "Aspect Ratio Bucketing", "Memory / Precision", "torch.compile",
})
TRANSFORMED_GROUP_TITLES.update({
    "Resolution", "Captions", "Caption Augmentation", "Flow Matching",
    "Curriculum", "Texture Crops", "Oversize Cascade",
    "High-Frequency Token Loss", "Adapter", "Text Encoder Adapter", "LoKr",
    "Concept Preservation", "Full Finetuning",
    "SDNQ Quantization", "Learning Rate Schedule", "REX / ReREX",
    "Per-Component Learning Rates", "Optimizer State",
})
