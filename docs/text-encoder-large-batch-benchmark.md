# Loaded Text Encoder: larger batches on RTX 4090

This follow-up measures encoder batches of 8, 16, 24, 30, and 32 on GPU 1.
It does not load the Mage-Flow transformer or measure full training memory.
All batches fit. Training configuration and encoder defaults remain unchanged.

## Controlled setup

- Qwen3-VL from the local `mage-flow` model; frozen encoder, BF16 compute.
- RTX 4090; PyTorch 2.10.0+cu128, Transformers 5.14.1, SDNQ 0.2.4.
- Fixed manifest of real WAI tag captions, using its first N captions per batch.
- Every batch padded to **220 tokens**, the longest caption in the largest
  tested batch including the prompt prefix. The caption cap is 512; actual
  captions are shorter. Fixed padding prevents growing sequence length from
  obscuring the batch-size comparison.
- Three warmups, then median of 20 synchronized forwards per case. Loading,
  tokenization, compilation startup, and transformer training are excluded.
- Each precision/compilation combination runs in its own process. Compiled
  modes compile decoder blocks with `dynamic=True`.
- Peak memory means PyTorch allocated memory, including the resident encoder
  and benchmark reference tensors. It excludes allocator reserve and other
  CUDA overhead, so it is not equivalent to `nvidia-smi` device usage.

The tables use the experimental embedding-only output path. It reproduced
the current wrapper's output exactly within each tested mode, while avoiding
the unused logits and retained intermediate hidden states. Raw JSON also
contains measurements for the current wrapper.

## Batch 24 comparison

| Mode | Latency (ms) | Captions/s | Peak (GiB) | Relative RMSE | Mean token cosine |
|---|---:|---:|---:|---:|---:|
| BF16 eager | 377.9 | 63.5 | 8.77 | 0.00% | 1.000000 |
| BF16 compiled | 301.0 | 79.7 | 8.68 | 1.93% | 0.999816 |
| INT8 storage | 389.9 | 61.5 | 5.00 | 4.21% | 0.999117 |
| INT8 storage compiled | 311.6 | 77.0 | 4.95 | 4.01% | 0.999200 |
| INT8 matmul | 224.4 | 106.9 | 4.99 | 16.66% | 0.986045 |

Relative RMSE uses eager BF16 as a reference and excludes padding and the
system prefix. It is numerical deviation, not a measure of image quality.
Compilation changes rounding; quantization introduces additional differences.
No FP32 reference or image-quality evaluation was performed.

## Throughput versus batch size

Values are captions per second; higher is better.

| Mode | Batch 8 | Batch 16 | Batch 24 | Batch 30 | Batch 32 |
|---|---:|---:|---:|---:|---:|
| BF16 eager | 67.5 | 67.0 | 63.5 | 63.0 | 63.1 |
| BF16 compiled | 79.2 | 80.6 | 79.7 | 81.2 | 81.6 |
| INT8 storage | 61.6 | 63.5 | 61.5 | 61.5 | 61.5 |
| INT8 storage compiled | 71.7 | 76.6 | 77.0 | 78.7 | 79.2 |
| INT8 matmul | 95.4 | 112.7 | 106.9 | 108.4 | 104.8 |

Compiled BF16 reaches about 79–82 captions/s across batches 8–32: increasing
the encoder batch beyond 8–16 provides little throughput benefit on this
workload. Compiled storage-only INT8 is close at batch 24 (312 ms versus
301 ms) and saves about 3.7 GiB. INT8 matmul is faster at batch 24 (224 ms),
but its embedding deviation is considerably larger. For a speed/precision
compromise with enough VRAM, compiled BF16 is the leading candidate from
these measurements. Eager BF16 retains exact reference behavior.

The production wrapper at batch 24 used 9.63 GiB in eager BF16 and took
397 ms. Compilation reduced wrapper time to 319 ms. The experimental
embedding-only compiled path reduced that to 301 ms and 8.68 GiB.

A separately configurable encoder chunk size is worth considering: training
batches of 24–30 need not require encoding all captions at once. However,
chunking performance and equivalence should be measured on the same complete
batch before adopting it; the current trainer encodes all captions together.

These results apply to this card and caption workload. H100 needs its own
measurements; these do not establish its optimal batch size. Longer captions
can substantially change speed and activation memory.

## Reproduction

Run eager BF16 first to generate references, then repeat with other modes.
Use a new output directory when changing captions or padding settings.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.tools.benchmark_text_encoder \
  --mode bf16 --captions /path/to/captions \
  --batches 8 16 24 30 32 --limits 512 --fixed-padding --steps 20 \
  --output benchmarks/text-encoder-large-batch
```

Repeat with `--compile-blocks`, `--mode int8`, `--mode int8 --compile-blocks`,
and `--mode int8_matmul`. Each invocation runs sequentially on GPU 1.
Local manifests, reference tensors, and timing JSON are in
`benchmarks/text-encoder-large-batch/`.
