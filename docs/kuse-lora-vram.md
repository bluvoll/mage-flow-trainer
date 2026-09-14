# Mage-Flow LoRA VRAM measurements

Measured on one RTX 4090 using PyTorch 2.10.0+cu128. These are actual forward/backward/optimizer runs on the pretrained Mage-Flow checkpoint, with the seven INT8 variants reported below covering 140 optimizer steps.

## Method

Ten captioned images were selected from the supplied kuse dataset using `random.Random(42).sample(sorted(eligible), 10)` (121 eligible images). Copies and a selection manifest are in the ignored local directory `benchmarks/kuse-lora/`; the source dataset was unchanged.

All runs used batch size 1, accumulation 1, rank/alpha 32, image attention + text attention + MLP adapters (42,467,328 trainable parameters), cached VAE latents, full gradient checkpointing, BF16 base computation, native-resolution `torch_varlen` attention, and no compilation. Images were resized into 1024-area buckets: 832×1152 (4), 768×1280 (3), 768×1152 (2), and 1088×896 (1). These measurements do not cover training at the images' original resolutions.

Each variant ran two epochs (20 optimizer steps). Loaded Text Encoder runs used tag shuffle, preserving the first tag, 10% caption dropout, and 10% tag dropout. The short seeded runs realized five empty captions out of 20 steps. Cached Text Encoder runs used fixed captions without these augmentations. Text length was capped at 512 tokens. Warm timing is the median of the second epoch, excluding initialization and preprocessing. No checkpoints were saved.

SDNQ quantizes frozen transformer and Qwen3-VL text-encoder weights. Trainable LoRA weights remain FP32 or BF16 as indicated. The AdamW optimizer also uses quantized states offloaded to CPU, at learning rate 1e-4. Quantized matrix multiplication is disabled: this measures quantized weight storage with floating-point computation. `all_adaln` preserves block modulation layers in BF16; `default` quantizes those layers too, while protecting input/output/time projections.

## Results

Memory is PyTorch peak **allocated GiB**, with peak **reserved GiB** shown separately. CUDA context, driver, and other processes require additional memory; these are not minimum GPU capacity guarantees. Initialization includes model loading and, for Cached Text Encoder runs, text-cache construction. VAE preprocessing is a separate prerequisite and is excluded.

**Cached Text Encoder** uses precomputed text embeddings and unloads the encoder before training. **Loaded Text Encoder** encodes captions during training. **Loaded Text Encoder (CPU offload)** moves the encoder between CPU and GPU each step. Adapter precision and quantization skip policy are stated explicitly in each row.

| Variant | Training allocated GiB | Reserved GiB | Initialization allocated GiB | Warm seconds/step | Maximum loss |
|---|---:|---:|---:|---:|---:|
| Cached Text Encoder — INT8, BF16 adapters, default policy | 5.403 | 5.605 | 4.617 | 0.839 | 0.849 |
| Cached Text Encoder — INT8, FP32 adapters, default policy | 5.822 | 6.102 | 4.617 | 0.911 | 0.878 |
| Cached Text Encoder — INT8, FP32 adapters, all_adaln policy | 7.083 | 7.326 | 5.311 | 0.923 | 0.878 |
| Loaded Text Encoder — INT8, BF16 adapters, default policy | 9.908 | 10.129 | 8.472 | 0.904 | 0.853 |
| Loaded Text Encoder — INT8, FP32 adapters, default policy | 10.328 | 10.625 | 8.551 | 0.973 | 0.876 |
| Loaded Text Encoder — INT8, FP32 adapters, all_adaln policy | 11.589 | 11.891 | 9.812 | 0.981 | 0.872 |
| Loaded Text Encoder (CPU offload) — INT8, FP32 adapters, all_adaln policy | 9.942 | 10.029 | 6.618 | 3.010 | 0.872 |

## Interpretation

The conservative INT8 comparison drops from 11.59 GiB with Loaded Text Encoder to 7.08 GiB with Cached Text Encoder (39% less). Quantizing block modulation and using BF16 adapters brings INT8 to 9.91 GiB with Loaded Text Encoder or 5.40 GiB with Cached Text Encoder.

The lowest reported Cached Text Encoder configuration uses INT8 + BF16 adapters at 5.40 GiB and 0.84 seconds/step. With Loaded Text Encoder, the same weight and adapter settings use 9.91 GiB at 0.90 seconds/step. These are minima within the reported configurations, not global minima.

All reported runs completed with finite losses below 0.88. These short memory tests do not establish long-run model quality.

These historical Cached Text Encoder runs used fixed embeddings in CPU RAM, rebuilt each launch before loading the transformer. The current trainer also supports [persistent augmented caption caches](caption-variation-cache.md); see the [current benchmark](readme-vram-benchmark.md) for that mode.

## Reproduce

Local TOML configurations, per-step JSON, logs, copied inputs, and latent caches are under `benchmarks/kuse-lora/`. Aggregate results without captions are in [kuse-lora-vram-results.json](kuse-lora-vram-results.json).

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m trainer.tools.benchmark_lora_memory \
  benchmarks/kuse-lora/cached-int8-default-bf16.toml \
  --output benchmarks/kuse-lora/cached-int8-default-bf16.json
```

Use the corresponding `variant_id` from the aggregate JSON as the filename stem to repeat another row. Display labels use Cached Text Encoder and Loaded Text Encoder; existing configuration filenames remain valid. The local checkpoint is `mage-flow/`. To reproduce these historical measurements, use batch size 1, accumulation 1, and logging every step. The benchmark also supports larger full batches for batch-size sweeps.
