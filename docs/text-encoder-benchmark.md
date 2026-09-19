# Loaded Text Encoder benchmark

Local experiment on GPU 1 (RTX 4090, 24 GiB), using the Qwen3-VL encoder
loaded from `mage-flow`. Software: PyTorch 2.10.0+cu128, Transformers 5.14.1,
SDNQ 0.2.4. These are encoder-only measurements, not complete training steps
or H100 projections.

## Method

`trainer.tools.benchmark_text_encoder` runs each precision/compilation mode
in a fresh process. It saves a fixed caption manifest and eager BF16 reference
embeddings in the output directory. Run eager BF16 first. For example:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.tools.benchmark_text_encoder \
  --mode bf16 --captions /path/to/captions --output benchmarks/text-encoder
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.tools.benchmark_text_encoder \
  --mode int8 --captions /path/to/captions --output benchmarks/text-encoder
```

Other options are `--mode int8_matmul` and `--compile-blocks`. Compilation
uses `dynamic=True` on each decoder block. Production training defaults are
not changed by this experiment.

The first 1, 4, and 8 captions from the fixed sorted sample of WAI tag captions
were tested with caption caps of 128 and 512. The table below uses the 512 cap,
but actual padded sequence lengths were **136, 174, and 174 tokens**, including
the prompt prefix. This is not a long-caption benchmark.

Timing is the median of 10 synchronized forwards after three warmups. Loading,
tokenization, compilation startup, and transformer training are excluded.
Memory is peak PyTorch allocated VRAM with the full encoder resident, including
its unused vision tower and small reference tensors; it is not total device
usage or the incremental memory cost inside training.

The current wrapper requests all hidden states and computes one token's logits.
The experimental embedding-only path reads the backbone's normalized
`last_hidden_state`. For the installed Transformers version this was exactly
equal to the wrapper's conditioning within each tested execution mode.
Do not replace it with the last decoder layer's pre-normalization output.

Embedding relative RMSE is measured against eager BF16, over valid caption
tokens after removing the system prefix. It measures numerical deviation,
not resulting image quality or convergence.

## Results

| Encoder mode / output path | Batch 1 (ms) | Batch 4 (ms) | Batch 8 (ms) | Batch 8 peak (GiB) | Batch 8 relative RMSE |
|---|---:|---:|---:|---:|---:|
| BF16 / wrapper | 36.1 | 54.9 | 101.7 | 8.67 | 0.00% |
| BF16 / embedding only | 34.2 | 51.8 | 95.4 | 8.45 | 0.00% |
| INT8 storage / wrapper | 61.9 | 67.2 | 113.5 | 4.92 | 4.01% |
| INT8 storage / embedding only | 61.0 | 64.5 | 107.0 | 4.69 | 4.01% |
| INT8 matmul / wrapper | 76.6 | 81.1 | 84.6 | 4.89 | 16.25% |
| INT8 matmul / embedding only | 74.3 | 78.0 | 77.8 | 4.67 | 16.25% |
| BF16 compiled / wrapper | 26.0 | 46.9 | 87.4 | 8.64 | 1.90% |
| BF16 compiled / embedding only | 25.0 | 43.7 | 81.3 | 8.42 | 1.90% |
| INT8 storage compiled / wrapper | 32.0 | 56.4 | 98.0 | 4.92 | 3.93% |
| INT8 storage compiled / embedding only | 31.1 | 53.1 | 92.4 | 4.69 | 3.93% |

## Interpretation

Storage-only INT8 saves roughly 3.7 GiB but can be slower than BF16. INT8
storage does not itself accelerate matrix multiplication. Quantized matmul
helped at batch size 8 here, but hurt small-batch latency and produced larger
embedding deviations. It should not become the default based on speed alone.

The embedding-only path avoids unnecessary outputs without changing the tested
conditioning. Regional compilation is another candidate, but changes numerical
results slightly and has startup/recompilation costs excluded from this table.
Both are now available as opt-in training controls; see the
[Loaded Text Encoder configuration](../README.md#compiled-loaded-text-encoder).
The production path passed batch-1 and batch-24 GPU smoke checks, including
downstream backward. Training defaults remain unchanged.

Repeat on H100 with representative caption lengths and the intended training
batch size. These local tests narrow the candidates; their speed rankings do
not establish H100 rankings or end-to-end training speedups.
