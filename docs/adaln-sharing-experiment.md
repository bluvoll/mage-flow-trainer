# Mage-Flow AdaLN sharing feasibility

This is the initial feasibility report. See [compressed modulation](compressed-modulation.md)
for the subsequent trainable architecture, checkpoint conversion, and finetuning tests.

GPU 1 (RTX 4090), original local `mage-flow/transformer/diffusion_pytorch_model.safetensors`, BF16 model, no SDNQ or compilation. This is an isolated approximation experiment, not a trained replacement or a released model. No checkpoint or production model code was changed.

## Findings

Directly sharing image modulation across all blocks, and separately sharing text modulation, performed poorly even with fitted per-block offsets. A shared low-dimensional timestep representation with separate small projections for each block/stream performed much better. This preserves block-specific timestep behavior; it is not identical modulation broadcast to every block.

The original 24 block projections have 1,359,396,864 parameters. A rank-64 representation needs approximately 28,953,600 stored coefficients including its shared basis, mean, and per-block output offsets. Keeping these coefficients in FP32 uses 110.45 MiB, versus 2.532 GiB for the original BF16 projections: **2.424 GiB of estimated weight-storage savings**. This is not a measured end-to-end training VRAM reduction. The timestep embedder and final output modulation remain unchanged.

## Method

1. Select 512 calibration and 256 heldout inputs from distinct representable BF16 timesteps in [0,1], seed 42. These sets do not overlap. Sampling is over representable values from a dense grid, not the training logit-normal distribution.
2. Compute the original post-SiLU timestep features. Fit a shared PCA basis to their centered calibration values.
3. For each original projection `W x + b`, use `W mean + b + (W basis) basis.T (x - mean)`. Coefficients and projection arithmetic are FP32; modulation outputs are cast back to BF16. No gradient descent or end-to-end distillation is performed.
4. Compare with a shared-projection baseline: mean image/text projection weights across blocks, with a per-block offset matching the calibration mean.
5. Probe full-model flow predictions at five heldout timesteps for each of three actual cached Bisque latents, paired with their own cached augmented captions. Inputs are `(1-t)*latent + t*noise`, at 1024-area resolution. These are a small sample from one dataset, not a broad evaluation set.

The timestep basis is shared in storage in this prototype; its projection is still recomputed in each block. A production implementation could compute it once per forward. Neither export compatibility nor backward/training behavior was tested.

## Results

Relative prediction error is `||student - teacher||₂ / ||teacher||₂`, computed separately for each probe and then averaged. It is not a percentage loss of image quality.

| Replacement | Mean prediction error | Range across 15 probes |
| --- | ---: | ---: |
| Shared projection + per-block offsets | 127.30% | 100.99–224.30% |
| Shared rank-16 basis + per-block projections | 9.01% | 2.18–25.51% |
| Shared rank-64 basis + per-block projections | 2.41% | 1.20–3.70% |
| Shared rank-128 basis + per-block projections | 2.25% | 1.11–3.32% |
| Uncompressed projections evaluated in FP32, returning BF16 | 1.45% | 0.73–2.46% |

All predictions were finite. The precision control demonstrates sensitivity to projection arithmetic alone; its error cannot simply be subtracted from compressed-model error. Direct sharing failing this initialization does not establish that a trained shared-AdaLN architecture cannot work.

The initial modulation-only scan also evaluated ranks 4, 8, 32 and 256. Small modulation-output errors did not guarantee small full-model errors: rank 16 illustrates the accumulation through the network. Raw metrics, singular values, and the four preliminary random-latent probes are in [the results JSON](adaln-sharing-results.json).

## Resources and next step

The validation experiment peaked at 13.64 GiB allocated GPU memory while retaining multiple candidates and the FP32 control; that is experiment overhead, not the footprint of a standalone compressed model. Host available RAM remained above 53.79 GiB across both runs. Both runs used GPU 1 only, one CPU thread, and a monitor that terminates only the experiment process group if available host RAM falls below 12 GiB.

Local reproduction scripts/logs are under `benchmarks/readme-vram/adaln-sharing/` and `benchmarks/readme-vram/adaln-sharing-validation/` (ignored experiment artifacts). Each `guard.py` runs its `run.py` with the memory guard. They require the local base checkpoint and existing Bisque latent/text caches.

Rank 64 is a promising starting point for further work: validate on more checkpoints, captions, aspect ratios, and training-distributed timesteps; evaluate full sampling trajectories and decoded images; then consider output/feature distillation. This experiment does not establish perceptual equivalence, adapter compatibility, or stable finetuning of the compressed architecture.
