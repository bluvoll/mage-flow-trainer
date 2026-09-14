# Mage-Flow VRAM benchmark

Measured on RTX 4090 24 GB GPUs with PyTorch 2.10.0+cu128. Every passing trial completed **10 optimizer steps**, **batch size 1 per GPU**, accumulation 1, rank/alpha 32, BF16 adapters, INT8 SDNQ frozen base weights (`all_adaln` skip policy), gradient checkpointing, `torch.compile`, packed `torch_varlen` attention, and **Cached Text Encoder (five augmented variations per image)** with cached image latents. The `adamw8bit` preset uses Kahan compensation with **optimizer-state quantization and offloading disabled**.

Ten large, file-deduplicated images from the Nicorima dataset were used. Sources are approximately 2K; larger tiers deliberately upscale them to test capacity. Resolutions are target-area tiers with aspect-ratio buckets, not fixed square dimensions. Single-GPU tests use GPU 1; dual-GPU tests use GPUs 0 and 1. GPU 0 retains approximately 1.4–1.6 GiB of desktop usage. CUDA reports about 23.52 GiB usable capacity per card.

Values in the README are **peak PyTorch allocated / reserved GiB per GPU**, including the first training step. They exclude external processes, CUDA context, and NCCL allocations. Model/cache initialization is recorded separately in the detailed results. OOM means a tested tier failed, not an exact resolution limit. These are short memory tests, not training-quality benchmarks.

The public [configuration](../configs/mageflow-readme-benchmark.toml) derives from the local `bisque-mageflow-lora-SDQN-AdamW.toml`. It keeps caption mode `mixed`, tag shuffling, tag dropout 0.1, caption dropout 0.1, and natural-language sentence shuffling. LoCon uses bypass; PEFT changes only `adapter.kind` to `lora`. Both train 41,091,072 BF16 adapter parameters. AdaLN is not trained.

The ten selected source sizes are 2048×2048, 2032×2048, 2006×2048, 2048×1997, 2048×1996, 1978×2048, 1948×2048, 1947×2048, 2048×1937, and 1929×2048. Byte-identical image files were excluded. No source files were modified. Upscaling above source resolution measures memory capacity, not additional image detail.

## Reproduce

Copy ten captioned images to a working dataset and cache each resolution with the real Mage-VAE:

```bash
.venv/bin/python -m trainer.tools.cache_latents cache /path/to/benchmark-data \
  --model-path mage-flow --resolution 1024 2048 3072 4096 5120 \
  --max-bucket-reso 8192 --upscale
```

Copy the public preset, set the local paths, select one `dataset.resolution`, and use the same `train.caption_cache_path` for all trials. For PEFT set `adapter.kind="lora"`; for LoCon use `kind="lycoris_lora"` and `lycoris_algo="locon"`. Each command must start a fresh process. Increase resolution until CUDA OOM.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.tools.benchmark_lora_memory \
  configs/my-benchmark.toml --output single.json

CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module trainer.tools.benchmark_lora_memory \
  configs/my-benchmark.toml --output dual.json
```

DDP writes `dual.rank0.json` and `dual.rank1.json`. CUDA OOM is recorded separately from other failures; a failed rank can cause its peer to terminate without completing a report. No higher tier is attempted after OOM for that configuration.

## Text-cache storage

The measured five-slot cache contains 51 unique embeddings including the shared empty caption: 48,266,240 bytes of tensor payload and a 48,472,064-byte SQLite main file (46.23 MiB). Embeddings range from 5 to 383 tokens, averaging 184.84 tokens including the empty caption, with 2,560 BF16 features. This is one small dataset measurement; the README's larger-dataset table is a projection assuming 200 tokens and no deduplication. WAL/lock sidecars and filesystem allocation may add overhead.

## Detailed measurements

[Machine-readable results](readme-vram-results.json) retain per-step timings, losses, text-token counts, actual bucket dimensions, and per-rank memory. Private paths and caption text are omitted. The first trial prepares the text cache; subsequent trials reuse it. Training peaks exclude that preparation, and the text encoder is absent during optimizer steps. Reported initialization peaks include initialization and, for the first trial, cold text-cache preparation; image-latent preprocessing was a separate VAE run.

Warm timing below is the median of steps 3–10; initial graph compilation is excluded from that timing but included in the training memory peak. Later shape recompilation may still affect timing. DDP timing includes synchronization and is not a controlled throughput comparison.

| Adapter | GPUs | Tier | Rank | Status | Initialization peak allocated GiB | Training allocated / reserved GiB | Warm seconds / step |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| lycoris_lora | 1 | 1024 | 0 | ok | 5.23 | 6.75 / 7.05 | 0.565 |
| lycoris_lora | 1 | 2048 | 0 | ok | 5.22 | 9.88 / 10.55 | 2.676 |
| lycoris_lora | 1 | 3072 | 0 | ok | 5.22 | 15.11 / 16.21 | 9.245 |
| lycoris_lora | 1 | 4096 | 0 | ok | 5.22 | 22.44 / 23.04 | 24.991 |
| lycoris_lora | 1 | 5120 | 0 | oom | — | — | — |
| lycoris_lora | 2 | 1024 | 0 | ok | 5.80 | 6.99 / 7.19 | 0.637 |
| lycoris_lora | 2 | 1024 | 1 | ok | 5.80 | 6.97 / 7.17 | 0.633 |
| lycoris_lora | 2 | 2048 | 0 | ok | 5.80 | 9.88 / 10.41 | 2.982 |
| lycoris_lora | 2 | 2048 | 1 | ok | 5.80 | 9.87 / 10.22 | 2.971 |
| lycoris_lora | 2 | 3072 | 0 | ok | 5.80 | 14.97 / 15.82 | 10.155 |
| lycoris_lora | 2 | 3072 | 1 | ok | 5.80 | 14.97 / 16.02 | 10.162 |
| lycoris_lora | 2 | 4096 | 0 | oom | — | — | — |
| lora | 1 | 1024 | 0 | ok | 5.22 | 6.75 / 7.07 | 0.566 |
| lora | 1 | 2048 | 0 | ok | 5.22 | 9.88 / 10.55 | 2.673 |
| lora | 1 | 3072 | 0 | ok | 5.22 | 15.11 / 16.21 | 9.245 |
| lora | 1 | 4096 | 0 | ok | 5.22 | 22.44 / 23.04 | 24.995 |
| lora | 1 | 5120 | 0 | oom | — | — | — |
| lora | 2 | 1024 | 0 | ok | 5.80 | 6.99 / 7.15 | 0.661 |
| lora | 2 | 1024 | 1 | ok | 5.80 | 6.97 / 7.11 | 0.667 |
| lora | 2 | 2048 | 0 | ok | 5.80 | 9.88 / 10.41 | 2.912 |
| lora | 2 | 2048 | 1 | ok | 5.80 | 9.86 / 10.22 | 2.904 |
| lora | 2 | 3072 | 0 | ok | 5.80 | 14.97 / 16.02 | 10.091 |
| lora | 2 | 3072 | 1 | ok | 5.80 | 14.96 / 16.02 | 10.085 |
| lora | 2 | 4096 | 0 | oom | — | — | — |
