# Mage-Flow quantized matmul test

Measured on 2026-09-16 using one RTX 4090 (physical GPU 1), PyTorch
2.10.0+cu128, and SDNQ 0.2.4. This tests full finetuning, not frozen-base LoRA.

## Controls

- Ten Bisque images from existing 1344×768 image latents; batch size 1,
  accumulation 1, 100 optimizer steps per completed run.
- Evaluated MageTrail rank-256 compressed-modulation checkpoint, with all
  2.871B parameters trainable. BF16 trunk and FP32 compressed modulation.
- Identical Cached Text Encoder embeddings from INT8 Qwen3-VL; fixed tag
  captions without augmentation. Every run reported zero pending embeddings.
- Native-resolution packing, `torch_varlen`, gradient checkpointing, SDNQ
  AdamW with quantized states and stochastic rounding, no Kahan or CPU offload.
  Learning rate 2e-6, REX, no warmup.
- The SDNQ configuration has `use_grad_ckpt=true`. Matmul forward selection
  is audited after conversion; both matmul variants select their corresponding
  dynamic quantized-matmul implementation for all 144 converted linear layers.
- Noise/timestep randomness resets to the same seed for each step. The report
  verifies image paths, caption hashes, buckets, batch sizes, and seeds match.
- Runs are sequential on the same GPU. Speed is median step time after the
  first ten steps; memory is PyTorch peak allocated memory, not `nvidia-smi`.

## Compilation compatibility

Both INT8 and UINT8 matmul failed before the first optimizer step with regional
`train.compile="default"`. Dynamo reported data-dependent branching at
`if ctx.use_hadamard` in SDNQ's dynamic matmul autograd forward.

Removing `train.compile` allowed both to complete 100 steps. SDNQ still
compiles its internal kernels, including specializations for caption lengths;
the first epoch is excluded from speed measurements. No dependency patches
or upgrades were applied. Storage-only quantization supports regional model
compilation in the same environment.

## Measurements

All four rows below omit model compilation. All 400 steps had matched inputs
and finite losses. Exact summary values and selected SDNQ forward functions
are in [the results JSON](quantized-matmul-results.json).

| Transformer mode | Peak allocated GiB | Peak reserved GiB | Warm median s/step | Mean loss | Mean absolute loss deviation from BF16 |
|---|---:|---:|---:|---:|---:|
| BF16 reference | 18.225 | 19.377 | 1.570 | 0.324066 | 0.000% |
| UINT8 storage only | 15.966 | 16.879 | 1.563 | 0.328621 | 1.545% |
| INT8 storage + matmul | 15.992 | 16.945 | 1.410 | 0.381089 | 13.181% |
| UINT8 storage + matmul | 16.106 | 17.072 | 1.465 | 0.335502 | 3.128% |

Loss deviation is the mean of `abs(loss_variant - loss_BF16) / loss_BF16`
over matched steps. It compares diverging training trajectories, not isolated
forward-pass quantization error. INT8 matmul had several loss spikes (maximum
1.703), despite remaining finite.

UINT8 matmul reduced median step time by 6.3% versus storage-only UINT8
without model compilation, while adding about 143 MiB of allocated VRAM and
roughly doubling loss deviation. The earlier storage-only UINT8 run **with**
regional model compilation measured 16.070 GiB, 1.486 s/step, and 1.533%
deviation from its compiled BF16 reference. That leaves only a 1.4% timing
difference versus UINT8 matmul, too small to establish a reliable advantage
from a single run. Compile modes differ in this last comparison.

Keep `use_quantized_matmul="auto"` resolving to off. These results favor UINT8
storage-only training with model compilation for this workload; they do not
justify enabling quantized matmul by default.

## Reproduction

The benchmark runner accepts storage-only and quantized-matmul configurations:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.tools.benchmark_storage \
  path/to/controlled-config.toml --output result.json
```

Use `quant.mode="training"`, `quant.weights_dtype="int8"` or `"uint8"`,
and `quant.use_quantized_matmul=true` for matmul. Set the last option to
`false` for storage-only training. Keep `quant.text_encoder_weights_dtype="int8"`
fixed to reuse the same text cache, even for the BF16 reference with
`quant.mode="none"`. Omit `train.compile` for a matched comparison in this
environment. The runner requires `train.gradient_accumulation_steps=1` and
`train.log_every=1`.

Local configs, logs, and per-step JSON are under
`benchmarks/quantized-matmul-controlled/`; source manifests and compiled
storage-only baselines are under `benchmarks/uint8-controlled/`. Those local
artifacts depend on the dataset and checkpoint paths on the benchmark machine.

This small, single-seed test compares training trajectories. It does not
establish generated-image quality, long-run convergence, performance at other
resolutions, or compatibility on other SDNQ/PyTorch versions. The BF16
reference uses the same quantized optimizer states, not an FP32 optimizer.
