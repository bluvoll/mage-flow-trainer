# Mage-Flow training guide

Mage-Flow text-to-image training with native model weights, frozen Qwen3-VL conditioning, Mage-VAE, PEFT LoRA/LoKr, LyCORIS LoCon/LoKr/DoRA, and SDNQ quantized base-weight training. The implementation adapts `diffusion-pipe-mageflow-ft` commit `40bf63a59269b5cfd73508ff25b5da7357cb1db1` without a runtime dependency on that checkout or DeepSpeed.

Use the `trainer` package for Mage-Flow training and caching. Image latent caches must be generated with the Mage-Flow VAE.

## Run

`./start-gui.sh` prefers the existing `.venv`, falling back to `venv`; `MAGE_FLOW_PYTHON` can select an interpreter explicitly. Use the existing `.venv` in this checkout, or install a fresh environment with `./install.sh` / `install.bat` (these create `venv`). PyTorch remains pinned to **2.10.0 + CUDA 12.8**; its built-in varlen attention works on the tested RTX 4090s. Dependencies are locked in `uv.lock` and exported to `requirements.txt`.

```bash
# Existing environment
.venv/bin/python -m trainer.gui

# Edit dataset.path and train.model_path in a copy of an example first.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m trainer.training.train configs/mageflow-lora.toml
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m trainer.training.train configs/mageflow-finetune-cached-int8.toml
```

`train.model_path` is a local model directory (default `mage-flow`, overridden by `MAGE_FLOW_MODEL`) containing:

```text
transformer/config.json
transformer/diffusion_pytorch_model.safetensors
text_encoder/                           # Qwen3-VL model and tokenizer files
vae/diffusion_pytorch_model.safetensors
```

Loading checks transformer weights strictly. Only the text-to-image model with `patch_size=1` is supported; image editing/reference-image training is not implemented.

Separate files are also supported: set `train.transformer_path`, `train.text_encoder_path`, and `train.vae_path` to native Mage-Flow transformer, Qwen3-VL-4B, and Mage-VAE `.safetensors` files. Each nonempty path overrides its component from `train.model_path`; supplying all three removes the need for that directory. The GUI exposes these paths under Model / Output and forwards the VAE selection to latent caching (`--vae-path` on the CLI).

Single-file Qwen3-VL loading uses the bundled 4B architecture configuration and tokenizer, with an optional `train.tokenizer_path` directory override. The tied language-model output head is restored when omitted by ComfyUI. Transformer configuration comes from embedded `model_config`, a same-stem JSON sidecar, or the bundled standard Mage-Flow architecture, in that order; weights must match strictly. Changing the encoder or tokenizer selects a new text-cache namespace. Transformer/VAE overrides do not invalidate text embeddings. VAE changes require regenerating image latents separately.

## Data and latents

Images have matching `.txt` captions. Existing caption shuffling/dropout, natural-language variants, dataset subsets, multi-resolution tiers, texture crops and schedules remain available. By default captions are encoded live after augmentation using the upstream Qwen3-VL template and 34-token prefix removal. The frozen encoder keeps the original wrapper, requests hidden states and projects only one unused logit token.

`dataset.source="encode"` encodes images live. To cache latents first:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m trainer.tools.cache_latents cache /path/to/images \
  --model-path mage-flow --resolution 1024
```

The cacher loads only the VAE. It uses posterior means, discards the unused decoder for training, and applies **no additional latent normalization**. Cached tensors are `[128,1,H/16,W/16]`, named `*_trainer.safetensors`. Wrong channel counts or bucket shapes are rejected. Set `dataset.source="latents"` to train from cache; images may then be absent. In `auto`, directories containing images use their caches, otherwise the loader scans latent files; choose `encode` explicitly for live VAE encoding.

`resolution` is an area budget; `max_bucket_reso` limits each side. Bucketing still resizes/crops images to the selected dimensions.

## Cached text and encoder offloading

Set `train.cache_text_embeddings=true` to unload Qwen3-VL **before** loading the transformer. With `train.caption_variations=0`, fixed captions are cached in RAM each launch and augmentation must be off. With a positive variation count, augmented caption slots and deduplicated embeddings persist in SQLite and are read on demand. See [caption variation caching](caption-variation-cache.md). Texture curricula and preservation probes are rejected in either cache mode.

For changing captions, `train.offload_text_encoder=true` moves Qwen3-VL to GPU for each text forward and back to CPU before transformer execution. This reduces GPU residency but incurs weight transfers each step. It cannot be combined with text caching, where the encoder is already absent during training.

## Variable-length attention and native resolutions

```toml
[train]
attention_backend = "torch_varlen"
# Optional: mix different bucket shapes in one microbatch.
pack_resolutions = true
batch_size = 2
```

`sdpa` uses a padded joint text/image sequence. `torch_varlen` uses [`torch.nn.attention.varlen.varlen_attn`](https://docs.pytorch.org/tutorials/intermediate/variable_length_attention_tutorial.html), with no separate FlashAttention installation. Optional `flash_attn_2` and `flash_attn_3` backends require their own CUDA packages. Unavailable backends fail explicitly.

With ordinary bucket batches, varlen removes caption padding from attention. With `pack_resolutions=true`, image and text streams are packed through the whole transformer: each image keeps its own spatial RoPE origin and timestep modulation, and attention boundaries prevent samples from attending to each other. Loss is averaged per image, so larger images do not receive more weight merely because they have more tokens.

Packed resolution batches use a **fixed image count**, not a token-budget scheduler. Set batch size for the largest possible combination of images. This mode requires an integer batch size, a packed attention backend, no curriculum, and `flow.use_ot=false`. It preserves bucket preprocessing rather than bypassing it to use raw source dimensions.

## Quantization and execution

- `quant.mode="frozen"`: quantized base weights with trainable LoRA/LoKr adapters.
- `quant.mode="training"`: SDNQ quantized master weights, stochastic rounding and base-weight updates. Use an SDNQ optimizer such as `adamw8bit`; its state can be quantized and offloaded to host memory.
- `quant.quantize_text_encoder=true`: quantize Qwen3-VL in **frozen** mode, including during transformer full finetuning.
- Input/output projections, text input norm and timestep embeddings stay in high precision. `all_adaln` additionally protects both streams' modulation; `first_block_adaln` and `mlp_down` are available alternatives.
- `use_quantized_matmul="auto"` currently resolves to **off**. Explicit `true` enables SDNQ quantized matmul; its memory, throughput, and numerical effects depend on the workload.

With SDNQ 0.2.4 and PyTorch 2.10.0, the tested INT8/UINT8 dynamic matmul
full-finetune paths fail during regional model compilation at SDNQ's
`ctx.use_hadamard` branch. Omit `train.compile` to exercise those paths;
SDNQ still compiles its internal kernels. Storage-only quantization continues
to work with model compilation. The installed SDNQ configuration defaults
`use_grad_ckpt` to `true`, including for these matmul tests.
See the [controlled matmul benchmark](quantized-matmul-benchmark.md) for
measured memory, throughput, and loss differences.

`adapter.dtype` explicitly controls trainable adapter precision: `float32` (default) or `bfloat16`. Frozen base weights keep their configured SDNQ storage. BF16 adapters reduce both gradient storage and LoRA projection activations; validate training quality for that setting.

Component names are `image_attn`, `text_attn`, `mlp`, `adaln` and `base`. Image/text attention groups are projections inside joint attention, not separate self/cross-attention layers. Full finetuning trains all components, including `adaln` and `base`, at the global optimizer learning rate by default. Explicit `[component_lr]` values override that rate; zero still freezes a component. `quant.skip_policy="all_adaln"` protects AdaLN from quantization, not from training. LoRA continues to freeze the base model and exclude AdaLN adapters. The text encoder remains frozen. The inherited high-frequency loss has no patch-local detail weighting at Mage-Flow's patch size of one; leave `flow.hf_scale=0` unless you deliberately want its auxiliary clean-latent MSE.

```toml
[train]
gradient_checkpointing = true
checkpoint_blocks = [0, 2, 4, 6, 8, 10]  # optional; omit for all blocks, [] for none
compile = "default"                      # defaults on for LyCORIS
compile_dynamic = true
compile_regional = true
```

Compilation applies to the shared block callable and preserves checkpoint key names. Supported modes are `default` and `max-autotune-no-cudagraphs`, with SDPA or Torch varlen. Metadata construction stays outside compiled/checkpointed blocks. Selective checkpointing trades activation memory against recomputation; smaller sets are not automatically faster overall if they force smaller batches.

## Optimizer families

The GUI groups optimizers into **PyTorch**, **SDNQ**, and **Optimi**. Switching
optimizers resets betas, epsilon, weight decay, and optimizer-state options to
that implementation's defaults. Your learning rate and gradient-clipping setting
are retained. Loading a TOML preserves explicitly configured values; omitted
betas, epsilon, and weight decay now use optimizer-specific defaults.

Optimi choices are `optimi_adam`, `optimi_adamw`, `optimi_adan`, `optimi_lion`,
`optimi_radam`, `optimi_ranger`, `optimi_sgd`, and `optimi_stableadamw`.
For example:

```toml
[optimizer]
kind = "optimi_adamw"
lr = 2e-5
betas = [0.9, 0.99]
eps = 1e-6
kahan_sum = "auto"
```

Optimi's `kahan_sum` accepts `"auto"`, `true`, or `false`. Auto uses compensation
for low-precision parameters. Adan defaults to `false`, following upstream.
SDNQ retains its separate `use_kahan` setting and stochastic rounding. State
quantization and CPU state offload are SDNQ-only controls. Epsilon is disabled for
SDNQ, Optimi Lion, and Optimi SGD because these implementations do not accept it;
legacy SDNQ TOMLs containing epsilon remain readable, but the value has no effect.
Optimi SGD exposes `momentum` instead of betas.

SDNQ Adafactor defaults to `betas = [-0.8, 0.999]`; its first value is a negative
second-moment decay exponent. SDNQ CAME uses `[0.9, 0.999, 0.9999]`, and Optimi Adan
uses `[0.98, 0.92, 0.99]`. Explicit AdamW-style positive decay exponents for Adafactor
are rejected before model loading.

SDNQ Adafactor also defaults to `norm_mode="relative"`, which scales each
update by the parameter tensor's norm. Zero-initialized LoRA up projections
can consequently learn extremely slowly at AdamW-style learning rates such
as `1e-4`. For LoRA, explicitly set `optimizer.norm_mode="rms_clip"` and tune
the learning rate for that mode. The GUI exposes **SDNQ update normalization**
for Adafactor and CAME; **Upstream default** preserves existing behavior
(relative for Adafactor, rms_clip for CAME). This setting is separate from
`max_grad_norm`; disabling gradient clipping does not disable normalization.
Changing normalization requires restarting training or loading adapter weights
without old optimizer state, because a resumed optimizer restores its groups.

A controlled WAI LoCon check (30 steps, batch 1, LR `1e-4`, five warmup
steps, BF16 adapters, frozen INT8 base, compiled varlen, Cached Text Encoder)
reproduced this: relative normalization left the median up-projection RMS at
`2.29e-9`; rms_clip reached `3.64e-4`. Identical initial weights, captions,
batches, and noise seeds produced finite gradients in both runs. On the same
fixed in-sample noise/timestep probe, loss changed from `0.365681` to
`0.365714` with relative normalization and to `0.356144` with rms_clip.
This validates meaningful updates and short-run learning, not final image
quality or the optimal learning rate for a longer run.

Optimi currently supports ordinary trainable tensors: use `quant.mode = "none"`
for full finetuning. Adapters can use a frozen SDNQ INT8 base with
`quant.mode = "frozen"`. SDNQ quantized full-finetune tensors are rejected with
Optimi until that combination is validated. The frozen text encoder has an
independent `quant.quantize_text_encoder` toggle: it can remain INT8 with
`quant.mode = "none"`, allowing reuse of existing INT8 caption embeddings while
training the transformer with Optimi. Changing optimizer or transformer-only
quantization settings does not invalidate that encoder cache. Changing actual
encoder precision, weights, tokenization, or caption content still can.

For storage-only SDNQ comparisons, `weights_dtype = "uint8"` uses asymmetric
8-bit storage with an offset; `int8` uses symmetric storage. Set
`text_encoder_weights_dtype = "int8"` alongside `quantize_text_encoder = true` to
keep conditioning unchanged while switching transformer storage dtype. If the
encoder override is omitted, it inherits `weights_dtype`, preserving existing
config behavior. The GUI exposes both UINT8 and this independent encoder setting.

### Optimi gradient release (single GPU)

Enable **Optimi gradient release** in the GUI, or set:

```toml
[train]
gradient_accumulation_steps = 1

[optimizer]
kind = "optimi_adamw"
gradient_release = true
max_grad_norm = 0
```

This updates each parameter during backward and immediately frees its gradient.
Ordinary global clipping and gradient accumulation are incompatible and rejected.
Learning-rate schedules and curriculum LR multipliers still apply; optimizer
checkpoints preserve per-parameter step counts. Compilation and non-reentrant
activation checkpointing can remain enabled. Gradient release does not shrink
weights or optimizer moments, and no total-VRAM reduction is guaranteed.

DDP training with gradient release is blocked before loading the model. In an
isolated two-4090 BF16/NCCL test, ordinary AdamW kept replicas identical, while
release mode diverged by a maximum absolute parameter difference of 0.0103 after
one step and 0.0300 after three. Hooks updated local parameters before DDP gradient
synchronization. The run completed without an error, so checking only for crashes
would miss this failure. A compiled single-GPU control with checkpointed BF16
blocks and a shared FP32 projection matched ordinary AdamW exactly for three steps.
These are small correctness probes, not full Mage-Flow convergence or VRAM tests.

To reproduce on disposable models (no dataset or model checkpoint is loaded):

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.tools.probe_gradient_release --device cuda --compile
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 -m trainer.tools.probe_gradient_release --device cuda --ddp
```

The DDP diagnostic deliberately bypasses the training guard and reports replica
drift. It never exports a model.


## Save and resume

Adapter exports contain native `diffusion_model.*` keys, per-module alpha tensors, embedded adapter config, and a JSON sidecar. Base-weight exports contain native transformer keys; SDNQ master weights are dequantized for ordinary safetensors serialization. `save_native=false` writes `transformer/config.json` plus `transformer/diffusion_pytorch_model.safetensors` under the checkpoint directory; use the original frozen VAE/text encoder with these weights.

New full-model and adapter exports embed readable JSON in the safetensors header:
`training_config` contains the resolved configuration, including optimizer, LR,
normalization, scheduler, quantization, adapter, captions, and dataset/model paths.
`training_state` records the step, export dtype, actual optimizer implementation
and parameter-group settings at save time (including current LR and normalization),
GPU count, nominal effective batch size, and save tag. The live optimizer settings
are authoritative when resumed state differs from the requested configuration.
`training_versions` records installed library versions. These fields travel with
the checkpoint when it is renamed; optimizer tensors are not embedded.

Read the header without loading model weights:

```bash
python -m trainer.tools.inspect_checkpoint path/to/checkpoint.safetensors
python -m trainer.tools.inspect_checkpoint path/to/checkpoint.safetensors --section config
python -m trainer.tools.inspect_checkpoint path/to/checkpoint.safetensors --section state
```

Older checkpoints remain readable but do not gain missing training settings.
Restart the trainer for new saves to include this metadata; an already-running
process keeps its loaded export code.

Enable `train.save_optimizer_state=true` to save resumable Accelerate state. Resume skips already-consumed microbatches within the current epoch. `train.resume_from` accepts the exported safetensors path or its adjacent `-state` directory. Inference exports alone cannot resume optimizer state. Keep the dataset and batching configuration consistent when resuming.

## Verification

See [validation results](mageflow-validation.md) and [attention measurements](mageflow-attention-benchmark.json). These include real pretrained-model smoke steps and small-model numerical tests; they do not establish long-run image quality. For resolution capacity, see the [single- and dual-GPU VRAM benchmark](readme-vram-benchmark.md).

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 MAGE_GPU_TEST=1 PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m trainer.tools.benchmark_attention
QT_QPA_PLATFORM=offscreen .venv/bin/python trainer/parity/test_gui.py
```

## License

The combined trainer is distributed under GPL-3.0 because the adapted diffusion-pipe code is GPL-3.0. Microsoft Mage modules retain their MIT notice; existing Apache-2.0 code retains its original notices. See `LICENSE`, `LICENSE.original`, `NOTICE`, and the vendored license files.

See the [10-image LoRA VRAM comparison](kuse-lora-vram.md) for Loaded Text Encoder versus Cached Text Encoder at 1024-area resolution, including INT8 and encoder offloading.

The Method tab has a **Train AdaLN** toggle under **Full Finetuning**, enabled by default and available only in full finetune mode. Disable it to freeze AdaLN (`component_lr.adaln = 0.0`); other components keep their configured training behavior. Enabling it uses the global learning rate, preserving any positive AdaLN override loaded from TOML. LoRA always freezes AdaLN.

The GUI omits component targeting, per-component learning-rate inputs, concept-preservation probes, and spectral initialization/cache controls. Their advanced TOML settings are retained when loading existing files and logged when non-default. The maximum bucket side accepts values above 2048; choose it separately from the resolution area budget.

SDNQ optimizer state quantization can be used with LoRA, independently of frozen-model quantization. Kahan compensation is optional and adds a buffer. The trainer supplies a compatibility alias for older SDNQ releases whose quantized Kahan initialization reads `use_svd_quantization` instead of `quantized_buffers_use_svd`; checkpoints retain the canonical keys.

LoRA always leaves AdaLN frozen, including block modulation, output modulation, and timestep projections. Explicit `adaln` adapter targets are rejected before model loading; changing component learning rates cannot unfreeze the base AdaLN weights.

LyCORIS LoRA is available as `adapter.kind = "lycoris_lora"`, including SDNQ frozen bases and block compilation. See [setup, target audit, and validation](lycoris-mageflow.md). Both LoRA backends now omit the final block's unused text-output targets (140 default targets).

[Persistent caption variation caching](caption-variation-cache.md) trades disk space and initial text encoding for low training VRAM while retaining a configurable pool of augmented caption slots. The LyCORIS algorithm selector is limited to LoCon, LoKr, and DoRA.
