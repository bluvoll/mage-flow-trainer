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
    "rti": "Experimental region read/write interface for full finetuning.",
}


def _spec(label, tooltip, make, inline_label=False):
    return Spec(label, tooltip, make, inline_label)


SPEC: dict[str, Spec] = {
    'flow.hf_high_noise_snr_cut': _spec('HF high-noise SNR cutoff', 'HF-only attenuation below this SNR. 1/9 corresponds to noise timestep 0.75.', lambda: F.FloatEditor(1e-12, 1e6, step=.01, decimals=12)),
    'flow.hf_high_noise_min_weight': _spec('HF minimum gate', 'Floor for the HF high-noise gate. Set 1 to recover the previous ungated HF loss.', lambda: F.FloatEditor(0, 1, step=.05, decimals=6)),
    'flow.hf_high_noise_power': _spec('HF gate power', 'Exponent on the high-noise SNR ratio. Leaves the ordinary flow loss unchanged.', lambda: F.FloatEditor(1e-6, 100, step=.1, decimals=6)),
    'paired_texture.scale': _spec('Paired texture loss scale', 'Experimental latent patch-distribution loss. Zero disables all work; no decoder or extra model forward.', lambda: F.FloatEditor(0, 100, step=.01, decimals=8)),
    'paired_texture.levels': _spec('Detail pyramid levels', 'Gaussian reduction and Laplacian detail bands; omits the coarse residual.', lambda: F.IntEditor(1, 32)),
    'paired_texture.patch': _spec('Local patch size', 'Odd patch width in latent pixels, independent of transformer token size.', lambda: F.IntEditor(1, 255, 2)),
    'paired_texture.tokens': _spec('Sampled patches per level', 'Reproducible sampling without replacement; preserves signed patch values.', lambda: F.IntEditor(1, 1048576)),
    'paired_texture.features': _spec('Fourier frequencies', 'Number of fixed random frequencies at three bandwidths.', lambda: F.IntEditor(1, 65536)),
    'paired_texture.seed': _spec('Texture projection seed', 'Local RNG only; does not advance training noise RNG.', lambda: F.OptIntEditor('1729')),
    'paired_texture.snr_cut': _spec('Texture SNR cutoff', 'Attenuate texture loss at high noise independently of the HF gate.', lambda: F.FloatEditor(1e-12, 1e6, step=.01, decimals=12)),
    'paired_texture.min_gate': _spec('Texture minimum gate', 'Zero removes the texture term at pure noise. No batch normalization of the gate.', lambda: F.FloatEditor(0, 1, step=.05, decimals=6)),
    'adversarial.enabled': _spec('Enable eye discriminator', 'Experimental. Requires dedicated low-noise eye examples and LPIPS weight zero.', lambda: F.BoolEditor('Enable eye adversarial loss'), inline_label=True),
    'adversarial.weight': _spec('Adversarial weight', 'Zero trains a shadow discriminator but adds no adversarial gradient to the adapter.', lambda: F.FloatEditor(0, 100, step=0.0001, decimals=8)),
    'adversarial.discriminator_lr': _spec('Discriminator LR', 'FP32 Adam learning rate for the small eye discriminator.', lambda: F.FloatEditor(0, 1, step=0.00001, decimals=8)),
    'adversarial.warm_start': _spec('Discriminator warm start', 'Optional eye-discriminator safetensors checkpoint.', lambda: F.PathEditor('file')),
    'perceptual.enabled': _spec('Enable decoded supervision', 'Frozen VSF3 decoder path. LPIPS is loaded only when its weight is positive; eye adversarial training uses weight zero here.', lambda: F.BoolEditor('Enable decoded supervision'), inline_label=True),
    'perceptual.weight': _spec('LPIPS weight', 'Added to ordinary flow MSE.', lambda: F.FloatEditor(0, 100, 6)),
    'perceptual.max_t': _spec('Maximum noise timestep', 'Apply only to t at or below this threshold (0 is clean).', lambda: F.FloatEditor(.001, 1, 3)),
    'perceptual.every': _spec('Every N microbatches', 'Frequency per GPU; independent of gradient accumulation.', lambda: F.IntEditor(1, 10000)),
    'perceptual.cache_dir': _spec('Decoded target cache', 'Cache keyed by VAE checksum and latent contents.', lambda: F.PathEditor('folder')),
    'perceptual.eye_manifest': _spec('Eye annotation manifest', 'Optional approved eye manifest JSON. Empty uses whole-image LPIPS; unannotated samples receive flow loss only.', lambda: F.PathEditor('file')),
    'perceptual.dedicated_low_noise': _spec('Dedicated low-noise example', 'Experimental: add one approved eye example per GPU per update, t uniform in [0.05, max_t]. Weight=0 gives the matched flow-only control.', lambda: F.BoolEditor('Add low-noise example'), inline_label=True),

    'self_flow.enabled': _spec('Enable Self-Flow', 'Experimental full finetuning: configurable batch size, packed cached latents/text, dual timestep; student block 4, teacher block 8. Saves inference checkpoints and EMA resume state.', lambda: F.BoolEditor('Enable experimental Self-Flow'), inline_label=True),
    'self_flow.ema_dtype': _spec('EMA storage precision', 'BF16 saves memory; FP32 retains small EMA updates directly.', lambda: F.ChoiceEditor(['bfloat16', 'float32'])),
    'self_flow.ema_device': _spec('EMA storage device', 'CPU streams the teacher and greatly reduces speed in our tests.', lambda: F.ChoiceEditor(['cuda', 'cpu'])),
    'self_flow.adaln_fp32': _spec('Keep AdaLN EMA in FP32', 'Protects block modulation, timestep embeddings and shared modulation projection.', lambda: F.BoolEditor('FP32 AdaLN EMA'), inline_label=True),
    'self_flow.stochastic_rounding': _spec('EMA stochastic rounding', 'Preserves small BF16 EMA updates in expectation. Not optimizer Kahan. FP32 EMA ignores this.', lambda: F.BoolEditor('Stochastic rounding for BF16 EMA'), inline_label=True),
    'self_flow.decay': _spec('EMA decay', 'EMA = decay * teacher + (1-decay) * student after each optimizer update.', lambda: F.FloatEditor(0, .999999, 6)),
    'self_flow.weight': _spec('Alignment loss weight', 'Weight applied to cosine feature alignment loss.', lambda: F.FloatEditor(0, 100, 4)),
    "rti.enabled": _spec("Enable RTI", "Experimental packed native-resolution region-token finetuning. Requires full finetuning, varlen packing, and dynamic compile.", lambda: F.BoolEditor("Enable experimental RTI"), inline_label=True),
    "rti.dense_prefix_blocks": _spec("Dense prefix blocks", "Dense blocks before the region read boundary.", lambda: F.IntEditor(0, 256)),
    "rti.dense_suffix_blocks": _spec("Dense suffix blocks", "Dense blocks after the region write boundary. Must be at least one.", lambda: F.IntEditor(1, 256)),
    "rti.size_buckets": _spec("Region size buckets", "Fixed log2 size-embedding buckets stored in the checkpoint.", lambda: F.IntEditor(1, 128)),
    "rti.start_keep": _spec("Initial keep fraction", "Fraction of image tokens retained at the beginning of RTI compression.", lambda: F.FloatEditor(.001, 1.0, 4)),
    "rti.target_keep": _spec("Target keep fraction", "Fraction retained after annealing.", lambda: F.FloatEditor(.001, 1.0, 4)),
    "rti.identity_steps": _spec("Identity steps", "Optimizer steps with one region per token; trains WRITE and size embeddings before compression.", lambda: F.IntEditor(0, 10_000_000)),
    "rti.warmup_steps": _spec("RTI warmup steps", "Optimizer steps held at initial keep fraction before annealing.", lambda: F.IntEditor(0, 10_000_000)),
    "rti.anneal_steps": _spec("RTI anneal steps", "Cosine anneal duration from initial to target keep fraction.", lambda: F.IntEditor(0, 10_000_000)),
    "rti.budget_steps": _spec("Discrete keep grid", "Optional comma-separated keep fractions. Each scheduled value snaps to its nearest entry.", lambda: F.FloatListEditor("e.g. 1.0, 0.9, 0.75")),

    "train.text_cache_batch_size": _spec(
        "Text cache batch size",
        "Captions per encoder forward per GPU while building either text cache. Independent of training batch size. "
        "Increase for throughput when VRAM permits; reduce if caching runs out of memory.",
        lambda: F.IntEditor(1, 1024)),
    "train.caption_variations": _spec("Caption variations per image", "0 uses fixed-caption RAM caching. A positive count creates persistent augmented caption slots; exact duplicate embeddings share disk storage. Enable text caching.", lambda: F.IntEditor(0, 100000)),
    "train.caption_cache_path": _spec("Caption cache database", "SQLite path. Empty places caption_variations.sqlite in the dataset folder. Reused across runs; stores embeddings on disk, not all in RAM.", lambda: F.PathEditor("file")),
    "train.cache_text_embeddings": _spec("Cache text embeddings", "Unload Qwen3-VL before training. Set caption variations above zero to persist augmented captions and embeddings in SQLite; zero requires fixed captions.", lambda: F.BoolEditor("Cache text embeddings"), inline_label=True),
    "train.offload_text_encoder": _spec("Offload text encoder", "Move the live encoder to GPU for caption encoding and back to CPU before transformer execution. Trades transfers for lower GPU residency.", lambda: F.BoolEditor("Offload text encoder"), inline_label=True),
    "train.compile_text_encoder": _spec("Compile Loaded Text Encoder", "Compile Qwen3 decoder blocks with dynamic shapes. Independent of transformer compilation and SDNQ. First encoding includes compilation; disable text encoder quantization for BF16.", lambda: F.BoolEditor("Compile Loaded Text Encoder"), inline_label=True),
    "train.text_encoder_embedding_only": _spec("Loaded Text Encoder: embedding-only output", "Skip unused logits and intermediate hidden states. Checks equivalence to the original output on the first batch; disabled for Cached Text Encoder.", lambda: F.BoolEditor("Loaded Text Encoder: embedding-only output"), inline_label=True),
    "train.pack_resolutions": _spec("Pack native resolutions", "Mix differently sized images in one batch without image-token padding. Requires varlen attention, integer batch size, no curriculum or OT.", lambda: F.BoolEditor("Pack native resolutions"), inline_label=True),
    "train.checkpoint_blocks": _spec("Checkpoint blocks", "Zero-based block indices; blank checkpoints all blocks.", lambda: F.IntListEditor("e.g. 0, 2, 4")),
    "train.attention_backend": _spec("Attention backend", "torch_varlen uses PyTorch's built-in packed attention. SDPA uses padded attention.", lambda: F.ChoiceEditor(["sdpa", "torch_varlen", "flash_attn_2", "flash_attn_3"])),
    "train.max_text_tokens": _spec("Maximum text tokens", "Caption token budget after removing the Qwen3-VL system prefix.", lambda: F.IntEditor(1, 4096)),
    # ------------------------------------------------------------------ train
    "train.moss_qkv_format": _spec(
        "MOSS Q/K/V saving format", "Default: fused for MOSS full finetunes and plain LoRA/LoCon; other models and LoKr/DoRA retain their native layout. Fused supports fused-base loaders such as Forge. Plain LoRA/LoCon uses three times the joint QKV rank with equivalent scaling and larger files. Training and optimizer resume state stay native.",
        lambda: F.ChoiceEditor(["Default (fused for MOSS LoRA/LoCon and finetunes)", "fused", "separated"], values=["auto", "fused", "separated"])),
    "train.model_family": _spec("Model family", "Flux3/MOSS text-to-image support is experimental; select SDPA or torch_varlen attention.", lambda: F.ChoiceEditor(["auto", "mage_flow", "flux3", "moss_image"])),
    "train.model_path": _spec(
        "Model directory",
        'Local Mage-Flow repository, Flux3 BF16 package, or MOSS-Image package with transformer/, text_encoder/, tokenizer/ and vae/.',
        lambda: F.PathEditor("folder")),
    "train.output_dir": _spec(
        "Output directory",
        "Checkpoints land in <output_dir>/<run_name>/<tag>/.",
        lambda: F.PathEditor("folder")),
    "train.transformer_path": _spec("Transformer file (optional)",
        "Native Mage-Flow, BF16 Flux3, or unquantized MOSS .safetensors. Overrides transformer weights; leave blank to use the model directory.",
        lambda: F.PathEditor("file")),
    "train.text_encoder_path": _spec("Text encoder file (optional)",
        "Qwen3-VL-4B BF16 .safetensors. Overrides the Diffusers text encoder. Uses the bundled tokenizer by default.",
        lambda: F.PathEditor("file")),
    "train.vae_path": _spec("VAE file (optional)",
        "Mage-Flow VAE .safetensors. Used by training and latent caching. Overrides the Diffusers VAE.",
        lambda: F.PathEditor("file")),
    "train.flux2_vae": _spec("Use FLUX.2 VAE (experimental)", "Packs FLUX.2 32-channel /8 latents into Mage-Flow's 128-channel /16 layout and applies vae_bn normalization. Supports finetuning and adapters on a base adapted to the selected VAE, including VSF3. Use latent caches made with that same VAE.", lambda: F.BoolEditor("Use experimental FLUX.2 VAE"), inline_label=True),
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
    "train.nonfinite_patience": _spec(
        "Invalid-update limit", "Skip invalid ordinary optimizer updates on all ranks; stop after this many consecutive failures. Gradient release stops immediately.",
        lambda: F.IntEditor(1, 100)),
    "train.activation_offload": _spec(
        "Offload checkpoint inputs", "Bounded CPU activation storage. Requires gradient checkpointing and torch.compile disabled.",
        lambda: F.BoolEditor("Offload checkpoint inputs"), inline_label=True),
    "train.activation_offload_budget_mb": _spec(
        "Activation CPU budget (MiB/rank)", "Maximum live offloaded checkpoint inputs per GPU process. Two GPUs can use twice this RAM. Excess inputs stay on GPU.",
        lambda: F.IntEditor(1, 262144)),
    "train.activation_offload_pin_memory": _spec(
        "Pin activation memory", "Optional pinned RAM for transfers. Disabled by default to reduce pressure on host memory.",
        lambda: F.BoolEditor("Pin activation memory"), inline_label=True),
    "train.activation_offload_prefetch": _spec(
        "Prefetch checkpoint inputs", "One-block transfer lookahead; requires pinned activation memory. Uses extra transient GPU memory.",
        lambda: F.BoolEditor("Prefetch checkpoint inputs"), inline_label=True),
    "dataset.cache_validation": _spec(
        "Latent cache validation", "warn: permit unverified legacy caches but reject known VAE mismatches. strict: require verified provenance and local VAE. off: bypass validation.",
        lambda: F.ChoiceEditor(["warn", "strict", "off"])),
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
        "latents named <stem>_WWWWxHHHH_trainer.safetensors.\n\nDisabled once Dataset subsets below has "
        "any rows -- the two forms are mutually exclusive and the subset list wins.",
        lambda: F.PathEditor("folder")),
    "dataset.subsets": _spec(
        "Dataset subsets",
        "Train from several folders, each with its own repeat count and texture eligibility. "
        "Leave empty for the ordinary single-directory config above.\n\nThe Texture crops column "
        "is the reason this exists: unchecked keeps a folder in full resolution for the whole run "
        "with its captions intact. That is what a regularization set needs -- texture crops "
        "replace the caption with texture.trigger, so an uncaptioned flat colour would train the "
        "unconditional branch, and CFG subtracts the unconditional.",
        F.SubsetEditor),
    "dataset.subsets_file": _spec(
        "Subsets file",
        "A CSV / JSON / TOML listing folders, for datasets with too many folders to list here -- "
        "one per artist. Illustration Scrapping Studio's training-layout export (folders.csv, "
        "folders.json or dataset.toml) works as written; any CSV with a `path` column and "
        "optional `num_repeats`/`repeats` and `texture` does too. Folders with 0 repeats or 0 "
        "images are skipped. Combines with the rows above; replaces Dataset path.",
        lambda: F.PathEditor("file")),
    "dataset.subsets_root": _spec(
        "Subsets root (optional)",
        "Re-root every folder from the subsets file under this directory, keeping only its "
        "folder name. For a file written on another machine: /home/jovyan/lib/images/emily "
        "becomes <root>/emily.",
        lambda: F.PathEditor("folder")),
    "dataset.latent_dtype": _spec(
        "Latent cache precision",
        "Precision newly cached latents are stored in. float16 halves the cache (a 1024px "
        "latent is 2 MB in float32) at ~5e-4 relative rounding, far below the VAE's own error; "
        "training reads either. Existing caches keep theirs -- `python -m "
        "trainer.tools.cache_latents convert --config <toml>` rewrites them in place.",
        lambda: F.ChoiceEditor(["float32", "float16", "bfloat16"])),
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
    "dataset.mask_path": _spec(
        "Mask folder",
        "Grayscale loss masks for the Dataset directory, one per image, matched by filename stem "
        "(img_001.png -> img_001.png here). White = learn this region, black = do not, grey = in "
        "between. The whole image still conditions the model; only the training target is "
        "weighted, so a character can be learned without its backgrounds.\n\nEvery image needs a "
        "mask once this is set. With Dataset subsets, set the Masks column per folder instead. "
        "Empty = unmasked.",
        lambda: F.PathEditor("folder")),
    "dataset.use_masks": _spec(
        "Use masks",
        "Off trains unmasked without clearing the mask folders, for an A/B against the same config.",
        lambda: F.BoolEditor("Apply the painted loss masks"), inline_label=True),
    "dataset.mask_outside_weight": _spec(
        "Outside weight",
        "Loss weight where a mask is black. 0 ignores those pixels entirely; 0.1-0.3 keeps a "
        "little supervision on the background; 1 is the same as no mask.",
        lambda: F.FloatEditor(0.0, 1.0, 0.05, 2)),
    "dataset.mask_dilate_px": _spec(
        "Grow (px)",
        "Grow every mask by this many pixels at bucket resolution before it is reduced to the "
        "16x latent grid; negative shrinks. A mask painted tight to a silhouette leaves its edge "
        "half-supervised -- a few pixels of slack hands it back.",
        lambda: F.IntEditor(-256, 256)),
    "dataset.mask_feather_px": _spec(
        "Feather (px)",
        "Gaussian-soften every mask edge, in bucket pixels, after growing. A hard edge between "
        "learned and ignored regions can train as a seam; about half the grow value is a sane start.",
        lambda: F.FloatEditor(0.0, 128.0, 0.5, 1)),

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
        lambda: F.ChoiceEditor(["tags", "nl", "tags_nl", "nl_tags", "short", "medium", "long", "layout", "mixed"])),
    "dataset.caption.mixed_weights": _spec(
        "Mixed weights",
        "Relative weights, not percentages -- they need not sum to 100. Only used when the mode "
        "is `mixed`.\n\nshort / medium / long / layout come from MOSS-Image annotation files next to "
        "each image (<stem>_moss.json, <stem>_layout.json, written by the MOSS captioning script); a "
        "sample without them falls back to its NL caption, then its tags.",
        lambda: F.WeightsEditor(["tags", "nl", "tags_nl", "nl_tags", "short", "medium", "long", "layout"])),
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
    "dataset.caption.attribution_patterns": _spec(
        "Attribution patterns",
        "Regexes, one per line, case-insensitive, that mark a tag or NL sentence as an artist / "
        "trigger attribution -- e.g. ^drawn by\\s for `Drawn by emily (pure dream)`. Matched "
        "anywhere in the caption, and a post crediting several artists has several. Empty "
        "turns the feature off.",
        lambda: F.LineListEditor("^drawn by\\s", height=54)),
    "dataset.caption.attribution_position": _spec(
        "Attribution position",
        "fixed pins attribution entries to the front of their field (tags or NL) in their "
        "original order, whatever shuffling does. random lets each land anywhere in its field "
        "(still exactly one copy).",
        lambda: F.ChoiceEditor(["fixed", "random"])),
    "dataset.caption.attribution_dropout_immune": _spec(
        "Attribution immune to dropout",
        "Take attribution entries out before tag dropout and shuffling, so a trigger is never "
        "dropped. Off treats them like any other tag.",
        lambda: F.BoolEditor("Attribution is never dropped"), inline_label=True),
    "dataset.caption.keep_order_through_attribution": _spec(
        "Keep tag order up to the artist",
        "For MOSS-Image's tag order (quality, meta, year, character, series, @artist, general): every tag "
        "up to and including the last attribution tag stays in place and is never dropped; shuffling and "
        "dropout apply only to the general tags after it. Needs attribution patterns such as ^@.",
        lambda: F.BoolEditor("Keep the ordered head, shuffle only general tags"), inline_label=True),
    "dataset.caption.annotation_attribution_template": _spec(
        "Artist in MOSS captions",
        "How the artist (the caption's attribution tags) is added to the short / medium / long captions "
        "and the layout summary, which never name the artist themselves. {attribution} and {caption}; "
        "empty = leave the artist out.",
        lambda: F.TextEditor("{attribution}. {caption}", strip=False)),
    "dataset.caption.layout_item_dropout": _spec(
        "Layout granularity",
        "MOSS layouts: each layout sample leaves out a random share of its items, drawn between 0 and this "
        "value, so the model learns both complete and sparse layouts. At least one item stays. 0 = off; "
        "0.5 = from all items down to about half.",
        lambda: F.FloatEditor(0.0, 0.9, 0.05, 2)),
    "dataset.caption.layout_short_captions": _spec(
        "Short item captions",
        "Share of layout samples whose boxes use their short caption (a few words, e.g. \"Marisa holding a "
        "spoon\") instead of the full sentence, so short and detailed box prompts both work. Needs layouts "
        "with short captions (the captioner's layout or shorten mode). 0 = always the full caption.",
        lambda: F.FloatEditor(0.0, 1.0, 0.05, 2)),
    "dataset.caption.layout_parts": _spec(
        "Character detail boxes",
        "Layout boxes for character details (a ribbon, a hat, a worn weapon), given to the model as plain objects. "
        "Off = only whole characters, standalone objects, text and panels.",
        lambda: F.BoolEditor("Keep character details"), inline_label=True),
    "dataset.caption.layout_panels": _spec(
        "Panel items in layouts",
        "MageTrail layouts box each comic / collage panel with a caption of what happens in it, beyond MOSS's "
        "character / object / text items. Off = train on MOSS's vocabulary only.",
        lambda: F.BoolEditor("Keep panel items"), inline_label=True),
    "dataset.caption.layout_panels_as_objects": _spec(
        "Panels as object boxes",
        "Give panels to the model as object boxes captioned \"panel: ...\", which is what a box node that only "
        "draws objects (ComfyUI's default) sends, instead of the separate panel label.",
        lambda: F.BoolEditor("Panels as objects"), inline_label=True),
    "dataset.caption.attribution_dedupe_on_combine": _spec(
        "Dedupe attribution on tags+NL",
        "tags_nl / nl_tags join both fields, and datasets carry `Drawn by X` in both so it "
        "survives any variant. On: the trailing field's copy of an attribution the leading field "
        "already has is removed, so it appears once. Attributions only in the trailing field "
        "are kept.",
        lambda: F.BoolEditor("Write each attribution once in combined captions"), inline_label=True),
    "dataset.caption.protected_tags": _spec(
        "Protected tags",
        "Never dropped, never shuffled out of position. Comma separated. Verified 3000/3000.",
        lambda: F.StrListEditor("trigger, character name, ...", as_set=True, height=60)),
    "dataset.caption.protected_tags_file": _spec(
        "Protected tags file",
        "One tag per line, added to the list above: for example every character and copyright tag of the "
        "dataset (Illustration Scrapping Studio's training export writes protected_tags.txt). A path on the "
        "machine that trains.",
        lambda: F.PathEditor("file")),
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
    "flow.dual_timestep": _spec(
        "Dual-timestep noising (experimental)",
        "Self-Flow noising only: assign a second sampled timestep to a subset of image tokens. "
        "No EMA teacher or alignment loss. Potential convergence gains require validation.",
        lambda: F.BoolEditor("Enable dual-timestep noising"), inline_label=True),
    "flow.dual_timestep_mask_ratio": _spec(
        "Second-timestep token fraction",
        "Probability of assigning each image token the second timestep; range (0, 0.5]. "
        "Only used when dual-timestep noising is enabled. Default 0.25.",
        lambda: F.FloatEditor(0.001, 0.5, 0.05, 3)),
    "flow.timestep_sample_method": _spec(
        "Timestep sampling",
        "logit_normal samples a sigmoid of normal noise; uniform is flat. "
        "moss uses Kohaku's static MOSS recipe: a truncated/rescaled logit-normal distribution "
        "on a discrete grid including pure noise. It locks shared sigmoid scale and shift at 1 "
        "and disables resolution-aware shift. These change sampling frequency, not loss weights.",
        lambda: F.ChoiceEditor(["logit_normal", "uniform", "moss"])),
    "flow.moss_mu": _spec(
        "MOSS normal mean", "Mean before the inverted sigmoid; default -1.09. More negative biases toward noise.",
        lambda: F.FloatEditor(-20, 20, 0.01, 3)),
    "flow.moss_std": _spec(
        "MOSS normal std", "Distribution width for MOSS sampling, independent of sigmoid_scale. Default 0.95.",
        lambda: F.FloatEditor(0.001, 20, 0.01, 3)),
    "flow.moss_cutoff": _spec(
        "MOSS cutoff", "Condition the base distribution below this cutoff, then rescale onto the full grid "
        "up to t=1. This is NOT the maximum sampled timestep. 1 disables truncation; default 0.95.",
        lambda: F.FloatEditor(0.001, 1, 0.01, 3)),
    "flow.moss_uniform_mix": _spec(
        "MOSS uniform mixture", "Fraction mixed with a uniform grid distribution. Default 0.01 (1%).",
        lambda: F.FloatEditor(0, 1, 0.01, 4)),
    "flow.moss_p_full": _spec(
        "MOSS extra pure-noise mass", "Extra probability assigned to t=1. Default 0; the final grid bin "
        "already has nonzero probability without an extra atom.",
        lambda: F.FloatEditor(0, 1, 0.01, 4)),
    "flow.moss_steps": _spec(
        "MOSS grid size", "Number of discrete timesteps j/N, j=1..N. Default 1000. Not inference steps.",
        lambda: F.IntEditor(2, 100000)),
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
    "optimizer.norm_mode": _spec(
        "SDNQ update normalization",
        "Adafactor defaults to relative; CAME defaults to rms_clip. Relative normalization "
        "can make zero-initialized LoRA updates extremely small. Use rms_clip for LoRA "
        "with an explicitly tuned learning rate. Separate from gradient clipping.",
        lambda: F.ChoiceEditor(["Upstream default", "relative", "rms_clip", "rms", "clip", "none"],
                               [None, "relative", "rms_clip", "rms", "clip", "none"])),
    "optimizer.use_first_moment": _spec(
        "Adafactor first moment",
        "SDNQ Adafactor: smooth normalized updates using the second beta (default 0.999). "
        "Adds a full-size momentum buffer; Quantize optimizer state quantizes eligible buffers. "
        "Off preserves existing behavior. This is separate from Optimi SGD momentum.",
        lambda: F.BoolEditor("Adafactor first moment"), inline_label=True),
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
        "segment i ends exactly where i+1 begins.\n\nstage is StageLR: chained linear / cosine / "
        "constant / REX segments, each a percentage of the run, with absolute LRs (see the "
        "StageLR table).",
        lambda: F.ChoiceEditor(["constant", "cosine", "linear", "rex", "rerex", "stage"])),
    "schedule.stages": _spec(
        "StageLR stages",
        "Segments in order, sized as a percentage of the post-warmup run (must total 100%). The "
        "run's exact optimizer-step count is known before step 1, so lengths are fitted to it "
        "exactly and printed at startup.\n\nLR is absolute: linear/cosine move from where the "
        "previous stage ended to it, constant holds it, rex decays from it to Min LR. The first "
        "stage starts from Optimizer LR. Groups with a component LR follow the same curve scaled "
        "by their ratio to Optimizer LR.\n\nSame table format as the diffusion-pipe fork's "
        "[StageLR] stages, so those lists paste into the TOML unchanged.",
        lambda: F.StageListEditor()),
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
        "Leave AdaLN LR blank to use the main optimizer LR, or enter a separate rate (e.g. 2e-6). "
        "Applies to shared and per-block modulation, timestep projections, and final normalization. Zero freezes AdaLN. "
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
    "adapter.rank": _spec("Rank / dimension", "LoRA rank / LoKr dimension. Large LoKr dimensions such as 10000 are supported; actual factorization depends on layer shapes and factor.", lambda: F.IntEditor(1, 2**31 - 1)),
    "adapter.alpha": _spec(
        "Alpha",
        "peft scales by alpha/r and kohya by alpha/dim, so this transfers unchanged between the "
        "two export formats.",
        lambda: F.FloatEditor(0.1, float(2**31 - 1), 1.0, 2)),
    "adapter.dropout": _spec(
        "Dropout", "Applied to the adapter path only.", lambda: F.FloatEditor(0.0, 0.9, 0.05, 2)),
    "adapter.lokr_factor": _spec(
        "LoKr factor", "Passed as factor to LyCORIS LoKr. -1 selects balanced factorization; positive values select the factorization factor. Independent of rank.", lambda: F.IntEditor(-1, 2**31 - 1)),
    "adapter.lokr_decompose_both": _spec(
        "LoKr decompose both", "Factor both Kronecker operands rather than one.",
        lambda: F.BoolEditor("Decompose both LoKr factors"), inline_label=True),

    # ------------------------------------------------------------------ quant
    "quant.backend": _spec(
        "Transformer quantization backend",
        "ConvRot is experimental MOSS BF16 LyCORIS LoCon training with a frozen INT8 base, "
        "rotation group 256 and INT8 forward matmul. Requires the convrot extra. "
        "Text encoder and optimizer settings remain independent.",
        lambda: F.ChoiceEditor(["SDNQ", "ConvRot INT8 (experimental)"], ["sdnq", "convrot"])),
    "quant.mode": _spec(
        "Quantization mode",
        'frozen trains adapters on quantized base weights; training updates quantized master weights for full finetuning.',
        lambda: F.ChoiceEditor(["none", "frozen", "training"])),
    "quant.weights_dtype": _spec(
        "Weight dtype",
        'SDNQ weight storage dtype (ConvRot always uses INT8). Also the inherited text encoder dtype.',
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

    # ------------------------------------------------------------------ monitoring
    "tracking.backends": _spec(
        "Trackers",
        "Where metrics (loss, LR per component, grad norm, throughput, memory, ETA), eval loss "
        "and validation samples go, besides the console and the Metrics tab. Any combination.\n\n"
        "tensorboard: local files under the run's tracking/ folder.\nwandb: needs WANDB_API_KEY "
        "or `wandb login` on the training machine -- never put a key in the config, which is "
        "copied into every checkpoint.\ntrackio: local dashboard (`trackio show`), a Hugging Face "
        "Space, or your own server; no network needed in local mode.\n\nA tracker that fails is "
        "disabled with a warning; training never stops for it.",
        lambda: F.CheckSetEditor(["tensorboard", "wandb", "trackio"])),
    "tracking.project": _spec("Project", "wandb/Trackio project name.",
                              lambda: F.TextEditor("mage-flow")),
    "tracking.run_name": _spec("Tracker run name", "Empty uses the training run name.",
                               lambda: F.OptTextEditor("= run name")),
    "tracking.log_dir": _spec(
        "Tracking folder", "TensorBoard and local Trackio/wandb files. Empty = "
        "<output_dir>/<run_name>/tracking.", lambda: F.PathEditor("folder")),
    "tracking.resume_run": _spec(
        "Continue tracker run on resume",
        "A resumed training continues the same wandb run (id kept in the run folder) and the "
        "same Trackio run, so its curves stay one line.",
        lambda: F.BoolEditor("Continue the same tracker run when resuming"), inline_label=True),
    "tracking.wandb_entity": _spec("wandb entity", "Team or user. Empty = your default.",
                                   lambda: F.OptTextEditor("default")),
    "tracking.wandb_mode": _spec(
        "wandb mode", "offline writes the run to disk; upload it later with `wandb sync`.",
        lambda: F.ChoiceEditor(["online", "offline"])),
    "tracking.wandb_base_url": _spec("wandb server", "Self-hosted W&B URL. Empty = wandb.ai.",
                                     lambda: F.OptTextEditor("https://api.wandb.ai")),
    "tracking.wandb_tags": _spec("wandb tags", "One per line.",
                                 lambda: F.LineListEditor("one tag per line", height=48)),
    "tracking.wandb_offline_on_failure": _spec(
        "wandb offline fallback",
        "If the online start fails (network block, bad key), log offline instead of losing the "
        "run's data.", lambda: F.BoolEditor("Fall back to offline on failure"), inline_label=True),
    "tracking.trackio_space_id": _spec(
        "Trackio Space", "e.g. user/my-dashboard. Uses your Hugging Face login. Empty = local.",
        lambda: F.OptTextEditor("local")),
    "tracking.trackio_server_url": _spec(
        "Trackio server", "Self-hosted Trackio server URL. Empty = local.",
        lambda: F.OptTextEditor("local")),

    "sampling.prompts": _spec(
        "Sample prompts",
        "One prompt per line. Empty = no validation samples.\n\nA line can be an inline table "
        "overriding settings for that prompt:\n{ prompt = \"a castle\", width = 1216, height = "
        "832, seed = 7, cfg = 4 }\n\nPrompts are encoded once with the training captions, so "
        "sampling never keeps the text encoder loaded. Images go to <run>/samples/ and to the "
        "trackers.",
        lambda: F.PromptListEditor("Drawn by emily (pure dream), 1girl, smile, outdoors")),
    "sampling.every_n_steps": _spec("Sample every N steps", "Empty = off.",
                                    lambda: F.OptIntEditor("off")),
    "sampling.every_n_epochs": _spec("Sample every N epochs", "Empty = off.",
                                     lambda: F.OptIntEditor("off")),
    "sampling.at_start": _spec(
        "Sample before training", "A baseline round at step 0 (skipped on resume).",
        lambda: F.BoolEditor("Sample the untrained model first"), inline_label=True),
    "sampling.steps": _spec("Sampling steps", "Euler steps per image. Reference default 30.",
                            lambda: F.IntEditor(1, 500)),
    "sampling.cfg": _spec("CFG scale", "1 or below skips the negative pass (half the cost).",
                          lambda: F.FloatEditor(0.0, 30.0, 0.5, 2)),
    "sampling.shift": _spec("Sampling shift", "Static flow shift at inference. Mage-Flow uses 6.",
                            lambda: F.FloatEditor(0.01, 100.0, 0.5, 2)),
    "sampling.width": _spec("Width", "Multiple of 16.", lambda: F.IntEditor(64, 4096, 16)),
    "sampling.height": _spec("Height", "Multiple of 16.", lambda: F.IntEditor(64, 4096, 16)),
    "sampling.negative_prompt": _spec(
        "Negative prompt", "Blank (a single space) is the reference default.",
        lambda: F.TextEditor(" ", strip=False)),
    "sampling.seed": _spec("Seed", "Base seed for every prompt unless it sets its own.",
                           lambda: F.IntEditor(0, 2**31 - 1)),
    "sampling.seed_strategy": _spec(
        "Seed strategy",
        "fixed: same noise every round, so changes between rounds come from training. walk: "
        "seed + round number, for variety.", lambda: F.ChoiceEditor(["fixed", "walk"])),
    "sampling.renormalize_cfg": _spec(
        "Renormalize CFG",
        "Rescale the guided velocity to the conditional one's norm; reduces oversaturation at "
        "high CFG.", lambda: F.BoolEditor("CFG renormalization"), inline_label=True),
    "sampling.save_to_disk": _spec(
        "Save samples to disk", "PNG files plus prompts.json under <run>/samples/<step>/.",
        lambda: F.BoolEditor("Save sample PNGs"), inline_label=True),

    "eval.path": _spec(
        "Eval folder",
        "Held-out images (cached like any subset; Cache latents includes it). Each eval round "
        "scores the same images at the same timestep quantiles against the same noise with "
        "un-augmented captions, so the curve moves only when the model does -- unlike training "
        "loss. Empty = off.",
        lambda: F.PathEditor("folder")),
    "eval.every_n_steps": _spec("Eval every N steps", "Empty = off.",
                                lambda: F.OptIntEditor("off")),
    "eval.every_n_epochs": _spec("Eval every N epochs", "Empty = off.",
                                 lambda: F.OptIntEditor("off")),
    "eval.at_start": _spec("Eval before training", "Baseline at step 0 (skipped on resume).",
                           lambda: F.BoolEditor("Evaluate the untrained model first"),
                           inline_label=True),
    "eval.quantiles": _spec(
        "Timestep quantiles",
        "Quantiles of the training timestep distribution (shift included) to score at. Logged "
        "per quantile and as their mean: low quantiles track detail, high ones composition.",
        lambda: F.FloatListEditor("0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9")),
    "eval.batch_size": _spec("Eval batch size", "Images per forward, within one bucket.",
                             lambda: F.IntEditor(1, 256)),
    "eval.max_samples": _spec("Eval max images", "Cap; a fixed spread of the folder. Empty = all.",
                              lambda: F.OptIntEditor("all")),
    "eval.seed": _spec("Eval noise seed", "Fixes the per-image noise.",
                       lambda: F.IntEditor(0, 2**31 - 1)),
}


# ---------------------------------------------------------------- layout

LAYOUT: list[tuple[str, list[tuple[str, list[str]]]]] = [
    ("Dataset", [
        ("Dataset Source", [
            "dataset.path", "dataset.source", "dataset.cache_validation", "dataset.num_repeats", "dataset.subsets",
            "dataset.subsets_file", "dataset.subsets_root", "dataset.latent_dtype",
        ]),
        ("Loss Masks", [
            "dataset.mask_path", "dataset.use_masks", "dataset.mask_outside_weight",
            "dataset.mask_dilate_px", "dataset.mask_feather_px",
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
            "dataset.caption.protected_tags_file",
            "dataset.caption.nl_shuffle_sentences", "dataset.caption.nl_keep_first_sentence",
        ]),
        ("Artist Attribution", [
            "dataset.caption.attribution_patterns", "dataset.caption.attribution_position",
            "dataset.caption.attribution_dropout_immune",
            "dataset.caption.attribution_dedupe_on_combine",
            "dataset.caption.keep_order_through_attribution",
            "dataset.caption.annotation_attribution_template",
            "dataset.caption.layout_item_dropout",
            "dataset.caption.layout_panels",
            "dataset.caption.layout_panels_as_objects",
            "dataset.caption.layout_short_captions",
            "dataset.caption.layout_parts",
        ]),
    ]),
    ("Training", [
        ("Model / Output", [
            "train.model_family", "train.model_path", "train.transformer_path", "train.text_encoder_path",
            "train.vae_path", "train.flux2_vae", "train.tokenizer_path", "train.output_dir", "train.run_name",
        ]),
        ("Schedule Length", [
            "train.epochs", "train.max_steps", "train.seed",
        ]),
        ("Batching", [
            "train.batch_size", "train.gradient_accumulation_steps", "train.num_workers",
            "train.vae_encode_chunk",
        ]),
        ("Memory / Precision", [
            "train.activation_offload", "train.activation_offload_budget_mb", "train.activation_offload_pin_memory", "train.activation_offload_prefetch", "train.nonfinite_patience",
            "train.dtype", "train.compressed_adaln_dtype", "train.gradient_checkpointing", "train.pack_resolutions", "train.checkpoint_blocks", "train.attention_backend", "train.max_text_tokens", "train.cache_text_embeddings", "train.text_cache_batch_size", "train.caption_variations", "train.caption_cache_path", "train.offload_text_encoder",
        ]),
        ("torch.compile", [
            "train.compile", "train.compile_dynamic", "train.compile_regional",
            "train.compile_text_encoder", "train.text_encoder_embedding_only",
        ]),
        ("Checkpoints", [
            "train.save_every_epochs", "train.save_every_steps", "train.keep_last_n",
            "train.save_native", "train.moss_qkv_format", "train.save_optimizer_state", "train.resume_from",
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
            "optimizer.weight_decay", "optimizer.max_grad_norm", "optimizer.norm_mode",
        ]),
        ("Optimizer State", [
            "optimizer.quantize_state", "optimizer.offload_state", "optimizer.use_kahan", "optimizer.kahan_sum", "optimizer.gradient_release", "optimizer.use_first_moment",
        ]),
        ("Learning Rate Schedule", [
            "schedule.kind", "schedule.warmup_steps", "schedule.min_lr_ratio",
        ]),
        ("REX / ReREX", [
            "schedule.d", "schedule.global_d", "schedule.local_d", "schedule.weight_power",
            "schedule.num_segments",
        ]),
        ("StageLR", [
            "schedule.stages",
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
            "flow.moss_mu", "flow.moss_std", "flow.moss_cutoff",
            "flow.moss_uniform_mix", "flow.moss_p_full", "flow.moss_steps",
            "flow.dual_timestep", "flow.dual_timestep_mask_ratio",
        ]),
        ("High-Frequency Token Loss", [
            "flow.hf_scale", "flow.hf_exponent", 'flow.hf_high_noise_snr_cut',
            'flow.hf_high_noise_min_weight', 'flow.hf_high_noise_power',
        ]),
        ("Experimental RTI", [
            "rti.enabled", "rti.dense_prefix_blocks", "rti.dense_suffix_blocks", "rti.size_buckets",
            "rti.start_keep", "rti.target_keep", "rti.identity_steps", "rti.warmup_steps",
            "rti.anneal_steps", "rti.budget_steps",
        ]),
        ('Paired Latent Texture Loss', [
            'paired_texture.scale', 'paired_texture.levels', 'paired_texture.patch',
            'paired_texture.tokens', 'paired_texture.features', 'paired_texture.seed',
            'paired_texture.snr_cut', 'paired_texture.min_gate',
        ]),
        ('Experimental Decoded Perceptual Loss', [
            'perceptual.enabled', 'perceptual.weight', 'perceptual.max_t',
            'perceptual.every', 'perceptual.cache_dir', 'perceptual.eye_manifest', 'perceptual.dedicated_low_noise',
        ]),
        ('Experimental Eye Adversarial Loss', [
            'adversarial.enabled', 'adversarial.weight', 'adversarial.discriminator_lr', 'adversarial.warm_start',
        ]),
        ('Experimental Self-Flow', [
            'self_flow.enabled', 'self_flow.ema_dtype', 'self_flow.ema_device',
            'self_flow.adaln_fp32', 'self_flow.stochastic_rounding', 'self_flow.decay', 'self_flow.weight',
        ]),
        ("Quantization", [
            "quant.backend", "quant.mode", "quant.weights_dtype", "quant.use_quantized_matmul",
            "quant.skip_policy", "quant.extra_skip",
            "quant.quantize_text_encoder", "quant.text_encoder_weights_dtype", "quant.use_stochastic_rounding",
            "quant.group_size", "quant.dynamic_loss_threshold",
        ]),
    ]),
    ("Monitoring", [
        ("Experiment Tracking", [
            "tracking.backends", "tracking.project", "tracking.run_name", "tracking.log_dir",
            "tracking.resume_run",
        ]),
        ("Weights & Biases", [
            "tracking.wandb_entity", "tracking.wandb_mode", "tracking.wandb_base_url",
            "tracking.wandb_tags", "tracking.wandb_offline_on_failure",
        ]),
        ("Trackio", [
            "tracking.trackio_space_id", "tracking.trackio_server_url",
        ]),
        ("Validation Samples", [
            "sampling.prompts", "sampling.every_n_steps", "sampling.every_n_epochs",
            "sampling.at_start", "sampling.steps", "sampling.cfg", "sampling.shift",
            "sampling.width", "sampling.height", "sampling.negative_prompt", "sampling.seed",
            "sampling.seed_strategy", "sampling.renormalize_cfg", "sampling.save_to_disk",
        ]),
        ("Held-out Eval", [
            "eval.path", "eval.every_n_steps", "eval.every_n_epochs", "eval.at_start",
            "eval.quantiles", "eval.batch_size", "eval.max_samples", "eval.seed",
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
    "Experiment Tracking", "Weights & Biases", "Trackio", "Validation Samples", "Held-out Eval",
})
TRANSFORMED_GROUP_TITLES.update({
    "Resolution", "Captions", "Caption Augmentation", "Flow Matching",
    "Curriculum", "Texture Crops", "Oversize Cascade",
    "High-Frequency Token Loss", "Adapter", "Text Encoder Adapter", "LoKr",
    "Concept Preservation", "Full Finetuning",
    "Quantization", "Learning Rate Schedule", "REX / ReREX",
    "Per-Component Learning Rates", "Optimizer State", "Artist Attribution", "StageLR",
})
