# Experimental compressed Mage-Flow modulation

This implementation replaces the large block AdaLN projections with a shared,
trainable low-rank input projection and separate small output projections for
every block and stream. It preserves block-specific modulation. The original
checkpoint is not overwritten, and ordinary Mage-Flow loading stays unchanged.

The rank-256 SVD checkpoint uses **a shared rank-256 input projection with
separate AdaLN output heads**, not one identical AdaLN projection reused at every
block. One 3072-to-256 projection feeds 24 separate heads (image and text for
each of the 12 blocks). SVD initializes this factorization; the factors remain
trainable and are no longer constrained to an orthogonal PCA basis during finetuning.

This is an **experimental architecture**, not a drop-in checkpoint for the
standard ComfyUI Mage-Flow loader. Use this trainer or the dedicated
[Load Compressed Mage-Flow custom node](https://github.com/bluvoll/ComfyUI-MageFlow-Compressed).
The node leaves built-in loaders and model detection unchanged.
Small prediction errors do not establish equal
image quality, and a short finetuning run does not establish long-run stability.

## Architecture and precision

For the original post-SiLU timestep feature `x`, calibration estimates its mean
`m` and a PCA basis `B`. Each original block projection `W x + b` becomes:

```text
z = B.T @ (x - m)                 # computed once, shared across blocks
y_block = (W_block @ B) @ z + W_block @ m + b_block
```

The shared projection and all block heads are trainable. Attention, MLPs, the
timestep embedder, and final output modulation retain their original structure.
Full finetuning trains all of them by default. The existing AdaLN freeze control
also freezes the compressed factors. LoRA leaves these factors frozen.

| Block modulation architecture | Parameters | Factor storage |
| --- | ---: | ---: |
| Original dense projections | 1,359,396,864 | 2.532 GiB, BF16 |
| Rank 128 shared input + separate heads | 57,458,816 | 219.19 MiB, FP32 |
| Rank 64 shared input + separate heads | 28,950,592 | 110.44 MiB, FP32 |
| Rank 256 shared input + separate heads | 114,475,264 | 436.69 MiB, FP32 |

These counts cover the replaced block projections and shared input projection;
the timestep embedder and final modulation are additional. Rank 128 leaves
2,813,807,360 total transformer parameters. By default, the factors stay
**FP32**, including checkpoint export/reload and calls to `model.to(bfloat16)`.
Their outputs return to the trunk dtype. SDNQ skips them regardless of skip
policy. The default variant is therefore not full BF16.

With the default setting, both the shared projection and all separate heads have FP32
parameters and FP32 parameter gradients. Their linear projections receive FP32
inputs; the timestep embedding and post-SiLU feature originate in the trunk
dtype, and the resulting modulation values return to that dtype before use.
This does not make the entire conditioning or normalization path FP32.
Quantized optimizer state, when enabled, remains a separate precision choice:
excluding these weights from SDNQ weight quantization does not exclude their
optimizer moments from state quantization.

TF32 is a reduced-precision execution option for FP32 matrix multiplication,
not a parameter-storage dtype or an upgrade over FP32 accuracy. Keep full FP32
matmul precision as the numerical baseline; enabling TF32 should be evaluated
separately for speed and quality. FP32 factors protect small parameter updates
from BF16 rounding, but the short finetuning tests do not establish immunity
to collapse or long-run training stability.

Set the compressed factors' training precision independently of the trunk:

```toml
[train]
dtype = "bfloat16"
compressed_adaln_dtype = "float32" # Default; alternatively "bfloat16".
```

The GUI exposes **Compressed AdaLN precision** under training precision settings.
This setting affects only the shared projection and separate block heads of a
compressed model; ordinary Mage-Flow, the timestep embedder, and final output
modulation are unaffected. BF16 factors use BF16 parameter gradients and receive
BF16 projection inputs. They remain excluded from SDNQ weight quantization.
Checkpoint exports keep these factors in FP32 for inference compatibility;
upcasting BF16-trained values does not recover discarded precision. Reloading
uses FP32 by default; keep the BF16 setting in your config to resume in BF16.

## Measured full-finetuning memory

Both converted initializations completed **ten optimizer steps** on one RTX 4090
24 GB, with every remaining transformer parameter trainable. Settings: batch 1,
accumulation 1, cached Bisque image latents at 1024-area resolution (1344×768 in
these batches), Cached Text Encoder with five caption variations, packed
`torch_varlen`, gradient checkpointing, regional `torch.compile`, INT8 SDNQ
training, AdamW8bit with quantized states, no Kahan buffer, **no CPU optimizer
offloading**, and learning rate 2e-6 with REX.

| Variant | Trainable parameters | Peak allocated VRAM | Peak reserved VRAM | Median step, excluding first two |
| --- | ---: | ---: | ---: | ---: |
| Rank 128 | 2,813,807,360 | 15.53 GiB | 16.34 GiB | 1.51 s |
| Rank 64 | 2,785,299,136 | 15.25 GiB | 16.05 GiB | 1.50 s |

These are PyTorch allocator measurements, not total `nvidia-smi` usage. Compilation
and initialization are excluded from step timing. The runs shared the host with
a distillation process on the other GPU, so timing is indicative. An available
host-RAM guard stopped only experiment processes if headroom fell below 12 GiB;
it did not trigger. This does not predict larger batches or resolutions.

The rank-128 initialization also passed **ten DDP optimizer steps on both 4090s**,
batch 1 per GPU (effective batch 2), with the same quantization and **no CPU
optimizer offloading**:

| DDP rank / physical GPU | Peak allocated VRAM | Peak reserved VRAM |
| --- | ---: | ---: |
| Rank 0 / GPU 0 | 16.00 GiB | 16.10 GiB |
| Rank 1 / GPU 1 | 16.01 GiB | 16.13 GiB |

Rank 0's median step after the first two was 2.68 s. Host available RAM remained
above 49.23 GiB. Distributed ordering produced different captions from the
single-GPU run, including three empty-caption steps on rank 0 and one on rank 1.
These measurements establish that this short DDP run fits, not that its memory
or throughput will be identical for every caption mix.

The audits verified gradients for all trainable parameters, changes to an INT8
attention-weight probe and the shared FP32 factor, absence of the text encoder,
and exclusively GPU-resident optimizer-state tensors. The rank-128 run also
saved and reloaded its full finetuned checkpoint, preserving every compressed
factor exactly and matching the trained shared-factor hash. Ten steps verify
execution and memory use, not eventual finetuning quality.

## Distillation results

Modulation-only optimization ran 1,000 steps for each rank and did not improve
held-out error; both exports retained their calibrated initialization. Flow
distillation then used cached Bisque, Nicorima, and Kuse data, balancing the three
sources. The path split contained 324 training and 80 held-out images. Validation
sampled five held-out images total, each at four timesteps: **20 prediction
probes**, not all 80 images.

| Candidate | Mean relative flow-prediction error | Worst probe |
| --- | ---: | ---: |
| Rank 64, calibrated initialization | 1.674% | 2.900% |
| Rank 128, calibrated initialization | 1.588% | 2.859% |
| Rank 128, conservative distillation, selected step 200 | 1.552% | 2.739% |

The first flow trial used head LR 1e-5 for 1,000 steps and ended worse (2.173%
mean error); its exported checkpoint therefore retained initialization. A second
300-step trial used head LR 1e-7 and shared-projection LR 1e-8, selecting step 200.
These are small validation gains on a small set. The error is
`||student - teacher||₂ / ||teacher||₂` per probe, averaged across probes; it is
**not a percentage loss of image quality**. No broad perceptual-equivalence or
long-run finetuning claim follows from these measurements.

Raw step measurements, gradient/optimizer audits, and validation histories are
in [the results JSON](compressed-modulation-results.json). The earlier
[feasibility experiment](adaln-sharing-experiment.md) used a different calibration
pool and different probes; its numbers are not directly comparable.

Matched generation used three prompts (still life, an anime street scene, and
a mountain lake), one seed each, 1024×1024, 32 Euler steps, shift 6, and CFG 4.
All three compressed candidates retained the original compositions closely on
visual inspection. Local details changed, including flowers, a hand shape, and
lighting. These twelve images are a preliminary comparison, not a perceptual
benchmark. Final-latent relative errors were 5.13–18.48% for rank 128, 5.09–16.42%
for rank 64, and 6.42–16.94% for distilled rank 128. Numerical differences can
accumulate over a sampling trajectory without an equally large visual change;
neither latent error nor these examples establish a consistent quality ranking.

The local image comparison is
`benchmarks/readme-vram/compressed-samples/index.html`; PNGs, prompt/seed settings,
and sampled latents are alongside it. Generated models and these local artifacts
are excluded from Git. Unit/GPU checks cover conversion, FP32 preservation,
export/reload, adapter freezing, packed mixed-resolution conditioning and
gradients, and existing SDNQ/compiled Mage-Flow paths.

## Convert a checkpoint

### Rank 256 and calibration precision follow-up

The rank-256 transformer has 2,870,823,808 parameters. Both the original FP32
calibration and the improved FP64/QR calibration passed ten full-finetuning
steps with the settings above, with no CPU optimizer offloading: **16.08 GiB
allocated / 17.01 GiB reserved**. The FP64/QR run used GPU 0 and verified updates
to both the INT8 attention-weight probe and the shared FP32 factor. FP64 is used
only during conversion; saved coefficients and modulation arithmetic stay FP32.
Rank 256 has not been tested with DDP here.

The following use the same 20 held-out prediction probes, without distillation:

| Rank / calibration | Mean prediction error |
| --- | ---: |
| 128, FP32 / default CUDA SVD | 1.588% |
| 256, FP32 / default CUDA SVD | 1.576% |
| 256, FP32 / QR SVD | 1.181% |
| 128, FP64 / QR SVD, stored FP32 | 1.299% |
| 256, FP64 / QR SVD, stored FP32 | **1.180%** |

The converter uses nonrandomized `torch.linalg.svd`. `full_matrices=False` selects
the reduced decomposition, not randomized approximation. CUDA's default uses
Jacobi (`gesvdj`), with QR (`gesvd`) as fallback; PyTorch recommends QR for
accuracy-sensitive cases. [PyTorch 2.10 SVD documentation](https://docs.pytorch.org/docs/2.10/generated/torch.linalg.svd.html).

Controls isolated the cause: FP64 factor construction with the original FP32
Jacobi basis still measured 1.576%; switching the basis solver to QR in FP32
already reached 1.181%. Merely evaluating the original coefficients in FP64
measured 1.578%, while FP64 calibration plus FP64 evaluation measured 1.179%.
The main improvement therefore came from obtaining a more accurate basis, not
from using FP64 during inference. None of these percentages is image-quality loss.

The original rank-256 initialization also underwent 300 conservative distillation
steps, selecting step 200 at 1.537% error. The better **1.180% candidate is
calibration-only**, with no gradient descent. A fair high-precision calibration
comparison now favors rank 256 over rank 128 on these probes; long-run finetuning
quality remains untested.

Use these explicit options for the higher-accuracy conversion (defaults remain
FP32 and PyTorch's automatic driver for reproducibility):

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python \
  -m trainer.tools.distill_modulation \
  --model mage-flow --rank 256 --steps 0 \
  --calibration-dtype float64 --svd-driver gesvd \
  --output output/compressed-mageflow/rank256-fp64-qr.safetensors
```

For a different starting checkpoint, supply `--transformer /path/to/model.safetensors`.
Set `train.transformer_path` to the new file in the existing experimental
finetuning preset. No FP64 inference/training mode is required. Precision controls
and the new measurements are in [the follow-up results](rank256-precision-results.json).

The new matched-image comparison is local at
`benchmarks/readme-vram/compressed-samples-r256/index.html`. The original and prior
rank-128 baseline PNGs matched the previous run byte for byte. The high-precision
rank-128 and rank-256 samples both preserved overall compositions, with local
detail differences. Neither consistently had the smaller sampled-latent error
across all three prompts; better held-out flow error is not proof of better
perceptual quality.

### Original rank-128 reproduction

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python \
  -m trainer.tools.distill_modulation \
  --model mage-flow --rank 128 --steps 0 \
  --output output/compressed-mageflow/rank128-modulation.safetensors
```

Use `--transformer /path/to/magetrail.safetensors` to convert a separate native
transformer. Calibration must use the checkpoint being converted. `--steps 0`
exports the calibrated initialization; a positive step count also attempts
modulation-only distillation, retaining the best held-out checkpoint including
step zero. Calibration and validation use distinct representable BF16 timesteps.
The JSON beside the checkpoint records the selected step and error history.

## Optional flow distillation

Prepare ordinary trainer configs referring to **existing cached image latents
and caption-variation SQLite caches**, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m trainer.tools.distill_flow \
  --model mage-flow \
  --student output/compressed-mageflow/rank128-modulation.safetensors \
  --data-config configs/my-cached-dataset.toml \
  --steps 300 --lr 1e-7 \
  --output output/compressed-mageflow/rank128-flow.safetensors
```

This is teacher/student model placement on two GPUs, not DDP. Only compressed
factors train; the teacher and student trunk stay frozen. Repeat `--data-config`
for additional datasets. If you converted a separate transformer, pass that
original file as `--teacher-transformer` here and in the sample tool below.
A deterministic path-based split excludes held-out
images from training. Validation uses up to two images per source at four fixed
timesteps; its small size is a limitation. The objective combines teacher flow
prediction matching with modulation matching. Both models use packed
`torch_varlen`; the student uses gradient checkpointing. The tool retains the
best validation checkpoint, including initialization if training does not help.

## Finetune and inspect samples

Copy [the experimental preset](../configs/mageflow-compressed-finetune.toml), edit
its dataset/cache/model paths, and run it with the normal training CLI. It uses
INT8 SDNQ training, quantized optimizer state, no CPU optimizer offloading,
Cached Text Encoder, batch size 1, and a ten-step smoke-test limit. Increase that
limit for a real run. Saved checkpoints preserve the architecture metadata and
FP32 factors.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.training.train \
  configs/my-compressed-finetune.toml
```

For the tested two-GPU placement, use the same config with:

```bash
CUDA_VISIBLE_DEVICES=0,1 TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no \
  -m trainer.training.train configs/my-compressed-finetune.toml
```

Generate matched-seed reference images with the original and compressed models:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python \
  -m trainer.tools.compare_compressed_samples \
  --model mage-flow \
  --student output/compressed-mageflow/rank128-modulation.safetensors \
  --output output/compressed-samples
```

The sampler uses 32 Euler steps, shift 6, CFG 4, and the native Mage-VAE decoder.
It is a reference comparison, not a claim of ComfyUI sampler parity. It loads
the text encoder for prompt encoding and unloads it before loading transformers.
Each tool writes new output files and rejects an existing final output target.
