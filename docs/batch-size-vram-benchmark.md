# Single-GPU batch-size VRAM sweep

Measured on physical GPU 1 (RTX 4090, 24 GB), with no training on GPU 0. Each trial starts a fresh process with `CUDA_VISIBLE_DEVICES=1`; CUDA's logical device 0 therefore refers to physical GPU 1. Batch size increases by one until confirmed CUDA OOM, separately for LyCORIS LoCon and PEFT LoRA.

All passing trials complete **10 optimizer steps with ten full batches**, accumulation 1, at the **1024-area resolution tier**. They use **Cached Text Encoder**, five augmented caption slots per image, cached Mage-VAE image latents, packed `torch_varlen` attention, compilation, full gradient checkpointing, rank/alpha 32, BF16 adapters, and INT8 SDNQ frozen weights with `all_adaln` protected. The AdamW preset uses Kahan compensation; optimizer-state quantization and CPU offloading are disabled.

The same ten Nicorima images and caption cache from the [resolution sweep](readme-vram-benchmark.md) are reused. Set `dataset.num_repeats` equal to batch size: ten images × B repeats gives exactly ten batches of B samples in one epoch. The benchmark checks every actual batch size and the number of optimizer steps. Repeats reuse existing latent and caption caches, and do not create new image content. This is a memory-capacity test, not a training-quality comparison.

## Results

Both backends completed ten steps at every batch size from **1 through 15**. At batch size **16**, both hit CUDA OOM during the fifth step, after four successful steps. Batch size 15 peaked at **21.69 GiB allocated**; this measured limit applies to the stated 1024-area configuration and caption pool.

VRAM is peak PyTorch **allocated / reserved GiB**, measured over all ten training steps including compilation. CUDA context and external allocations require additional memory. The GPU was idle before the sweep. Warm time is the median of steps 3–10; later recompilation can still affect it. OOM trials are failures even if some steps completed; their partial measurements are retained in the [raw JSON](batch-size-vram-results.json).

| Adapter | Batch size | Completed steps | Status | Peak allocated / reserved GiB | Warm seconds / step |
| --- | ---: | ---: | --- | ---: | ---: |
| PEFT LoRA | 1 | 10 | ok | 6.75 / 7.07 | 0.582 |
| PEFT LoRA | 2 | 10 | ok | 7.84 / 8.14 | 0.977 |
| PEFT LoRA | 3 | 10 | ok | 8.88 / 9.25 | 1.395 |
| PEFT LoRA | 4 | 10 | ok | 9.96 / 10.58 | 1.791 |
| PEFT LoRA | 5 | 10 | ok | 11.01 / 11.71 | 2.197 |
| PEFT LoRA | 6 | 10 | ok | 12.08 / 12.90 | 2.607 |
| PEFT LoRA | 7 | 10 | ok | 13.13 / 14.06 | 3.000 |
| PEFT LoRA | 8 | 10 | ok | 14.20 / 15.27 | 3.443 |
| PEFT LoRA | 9 | 10 | ok | 15.24 / 16.46 | 3.853 |
| PEFT LoRA | 10 | 10 | ok | 16.32 / 17.46 | 4.260 |
| PEFT LoRA | 11 | 10 | ok | 17.37 / 18.67 | 4.670 |
| PEFT LoRA | 12 | 10 | ok | 18.45 / 19.80 | 5.076 |
| PEFT LoRA | 13 | 10 | ok | 19.54 / 21.07 | 5.499 |
| PEFT LoRA | 14 | 10 | ok | 20.58 / 21.95 | 5.881 |
| PEFT LoRA | 15 | 10 | ok | 21.69 / 22.99 | 6.444 |
| PEFT LoRA | 16 | 4 | oom | — | — |
| LyCORIS LoCon | 1 | 10 | ok | 6.75 / 7.07 | 0.567 |
| LyCORIS LoCon | 2 | 10 | ok | 7.84 / 8.14 | 0.975 |
| LyCORIS LoCon | 3 | 10 | ok | 8.88 / 9.25 | 1.397 |
| LyCORIS LoCon | 4 | 10 | ok | 9.96 / 10.58 | 1.786 |
| LyCORIS LoCon | 5 | 10 | ok | 11.01 / 11.71 | 2.207 |
| LyCORIS LoCon | 6 | 10 | ok | 12.08 / 12.90 | 2.619 |
| LyCORIS LoCon | 7 | 10 | ok | 13.13 / 14.06 | 3.019 |
| LyCORIS LoCon | 8 | 10 | ok | 14.20 / 15.27 | 3.454 |
| LyCORIS LoCon | 9 | 10 | ok | 15.24 / 16.46 | 3.855 |
| LyCORIS LoCon | 10 | 10 | ok | 16.32 / 17.46 | 4.259 |
| LyCORIS LoCon | 11 | 10 | ok | 17.37 / 18.67 | 4.685 |
| LyCORIS LoCon | 12 | 10 | ok | 18.45 / 19.80 | 5.079 |
| LyCORIS LoCon | 13 | 10 | ok | 19.54 / 21.07 | 5.501 |
| LyCORIS LoCon | 14 | 10 | ok | 20.58 / 21.95 | 5.896 |
| LyCORIS LoCon | 15 | 10 | ok | 21.69 / 22.99 | 6.443 |
| LyCORIS LoCon | 16 | 4 | oom | — | — |

## Reproduce

Copy [the benchmark preset](../configs/mageflow-readme-benchmark.toml), set your model/dataset/cache paths, and use exactly ten cached images. For each batch size B:

```toml
[dataset]
# Keep the preset's other dataset settings.
num_repeats = B

[train]
# Keep the preset's other training settings.
batch_size = B
epochs = 1
max_steps = 10
gradient_accumulation_steps = 1
```

Replace B with a positive integer; it is a placeholder, not a TOML value. Start at 1 and increment by 1. Use `adapter.kind="lycoris_lora"` with `lycoris_algo="locon"`, or `adapter.kind="lora"` for PEFT.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.tools.benchmark_lora_memory \
  configs/my-batch-benchmark.toml --output batch-B.json
```

Local per-trial configs and logs are under the ignored `benchmarks/readme-vram/batch-sweep/` directory. Stop at CUDA OOM; other errors do not establish a memory limit.
