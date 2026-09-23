# Experimental Self-Flow full finetuning

For saved training checkpoints and GUI usage, see
[Self-Flow training](self-flow-training.md). This page describes the isolated
measurement tool and its earlier benchmark results.

The isolated `trainer.tools.probe_self_flow_lokr` tool also supports Mage-Flow
full finetuning with `--use-config-settings` and `adapter.kind = "none"`.
Its historical module name is retained. This is a batch-one-per-GPU
feasibility probe, not a production training option: it writes JSON measurements,
does not save trained checkpoints or support resume. Experimental DDP measurement
requires the explicit `--ddp-probe` flag.

## Measured results, September 21, 2026

All five single-GPU runs completed ten updates on GPU 1 (RTX 4090), batch size 1, with
finite losses and gradients. Images, caption hashes, and sampled timestep pairs
matched across the runs. The eager uniform-forward parity check was exact.

| Method | EMA storage | Peak allocated VRAM | Peak reserved VRAM | Median warm step |
| --- | ---: | ---: | ---: | ---: |
| Dual-timestep finetune baseline | — | 9.28 GiB | 9.52 GiB | 0.583 s |
| Self-Flow, FP32 EMA | 7.11 GiB | 16.46 GiB | 16.73 GiB | 0.655 s |
| Self-Flow, BF16 EMA + stochastic rounding | 3.55 GiB | 12.91 GiB | 13.18 GiB | 0.720 s |
| Self-Flow, BF16 stochastic EMA + FP32 AdaLN | 3.71 GiB | 13.07 GiB | 13.31 GiB | 0.706 s |
| Self-Flow, CPU FP32 EMA | 7.11 GiB **RAM** | 9.35 GiB | 9.62 GiB | 6.665 s |

No CPU optimizer offload was used. The CPU row explicitly offloads EMA storage
and updates, while executing the teacher on GPU one block at a time.
BF16 EMA saved 3.55 GiB of peak
allocated VRAM against FP32 EMA, while its stochastic-rounding update cost more
time. FP32 EMA already fits comfortably for this particular batch-one experiment.

Timings include diagnostic gradient checks, forward/backward, optimizer work,
and EMA updates, excluding cached-data loading/text lookup. The reported median
uses steps 3–10; shape-dependent compilation still affected step 4 in the two
teacher runs (about 22 seconds), so these are feasibility measurements rather
than rigorous throughput benchmarks. Initial compilation affected steps 1–2
as well. The former batch-54 LoCon configuration was not tested as a full finetune.

Raw reports are stored locally under `benchmarks/self-flow-finetune-512/` as
`baseline.json`, `fp32-ema.json`, `bf16-sr-ema.json`, `mixed-ema.json`, and
`cpu-fp32-ema.json`.

The follow-up mixed EMA run keeps block modulation, the shared modulation
down-projection, and timestep embedding EMA weights in FP32. It costs only
0.162 GiB more than all-BF16 EMA on this compressed model. Six unit tests pass,
including a check of the per-parameter storage policy and FP32 modulation update.

The synchronous CPU implementation is about 9.4 times slower than mixed GPU EMA
here. Median teacher streaming/evaluation takes 1.245 s and the EMA update/sync
4.836 s. This includes PCIe transfers, CPU memory traffic, and implementation
overhead; it does not isolate the effect of DDR4. No transfer prefetching or
overlap optimization was used. Each additional CPU EMA replica would require its
own storage and add memory/transfer traffic; CPU DDP was not benchmarked.

## Teacher and precision

The student feature comes from block 4 and the teacher stops at block 8.
For a full finetune, EMA covers all parameters in those first eight blocks,
plus the image/text input projections, text normalization, timestep embedding,
and shared compressed AdaLN down-projection. Sharing the *current* student input
projections would not give a consistent EMA teacher. Output layers and later
blocks need no teacher copy because the teacher never executes them.

SDNQ training tensors are explicitly dequantized before EMA initialization and
updates. Teacher evaluation temporarily substitutes ordinary floating-point EMA
tensors, one module/block at a time, and restores student parameters before
backpropagation. Quantized matmul is disabled in the tested configuration.

FP32 is the default EMA storage. `--ema-dtype bfloat16
--ema-stochastic-rounding` instead computes each update in temporary FP32 and
stochastically rounds it back into BF16 storage, without a persistent FP32 master
copy. This halves EMA storage. Ordinary BF16 rounding can discard small EMA
updates; stochastic rounding preserves them in expectation but adds rounding
noise. It is not equivalent to an FP32 EMA trajectory.

Add `--ema-adaln-fp32` to retain FP32 EMA storage for parameters classified by
the trainer as AdaLN, including timestep embeddings and compressed modulation.
Only the remaining BF16 EMA tensors receive stochastic rounding. This policy
does not change the student's parameter dtype or learning rate, and uses no
Kahan compensation buffer.

The DDP probe executes the student and feature projector through DDP's forward,
with the existing SDNQ-aware reducer configuration and gradient bucket views.
Data/noising RNG differs by rank; optimizer and EMA stochastic-rounding RNG are
synchronized separately so replicated weights receive the same rounding draws.
At the end it compares fingerprints of sampled values in every student storage
tensor and EMA tensor. This detects sampled replica divergence, not an exhaustive
element-by-element equality proof. Output JSON files receive `.rank0`, `.rank1`,
etc. suffixes.

### Two-GPU result

The mixed GPU EMA test completed 10/10 updates on **each** RTX 4090, batch size
1 per GPU (effective batch 2), with finite losses and gradients. Sampled student
and teacher fingerprints matched across ranks after the final update.

| Rank | Peak allocated VRAM | Peak reserved VRAM | Median warm step |
| --- | ---: | ---: | ---: |
| GPU 0 | 13.159 GiB | 13.311 GiB | 1.841 s |
| GPU 1 | 13.162 GiB | 13.313 GiB | 1.835 s |

Observed process-level usage in `nvidia-smi` was about 13.9 GiB per process,
including allocations outside PyTorch. This is an observation, not a separately
sampled peak measurement. GPU 0's roughly 2 GiB desktop allocation remained
running and is not included in the table. This setup did not add a fixed 3 GiB
per GPU: gradient bucket views reuse storage. These results do not establish
memory requirements at larger batches or resolutions.

At this small per-GPU batch, communication overhead makes DDP slower in aggregate
image throughput than the single-GPU run. Its effective batch, rank-dependent
noising, and distributed scheduler stepping also differ; do not compare losses
as a convergence experiment. Reports are `mixed-ema-ddp.rank0.json` and
`mixed-ema-ddp.rank1.json`. An earlier run ending in an out-of-bounds *post-training
diagnostic* was superseded after replacing floating-point sample indices with
integer indices and adding a regression test.

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module trainer.tools.probe_self_flow_lokr \
  --config benchmarks/self-flow-finetune-512/probe.toml --use-config-settings \
  --ddp-probe --mode self_flow --ema-device cuda --ema-dtype bfloat16 \
  --ema-stochastic-rounding --ema-adaln-fp32 --steps 10 \
  --output benchmarks/self-flow-finetune-512/mixed-ema-ddp.json
```

Teacher parameters are cast to the corresponding student's execution dtype.
Consequently, even FP32 EMA normally executes the large linear layers in BF16.
Compressed AdaLN remains FP32 at execution when configured that way, but BF16 EMA
storage has already rounded its saved values before that upcast.

## Reproducing the local experiment

The local config is `benchmarks/self-flow-finetune-512/probe.toml`, derived from
`configs/Overfit-mageflow-LoCON-Adafactor-swap-vsf3-512px.toml` with the adapter
removed and SDNQ changed from frozen to training mode. It uses an isolated
ten-image subset of existing 512-area VSF3 latents and cached captions. Original
dataset files and the original training config are unchanged.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.tools.probe_self_flow_lokr \
  --config benchmarks/self-flow-finetune-512/probe.toml --use-config-settings \
  --mode baseline --steps 10 \
  --output benchmarks/self-flow-finetune-512/baseline.json

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.tools.probe_self_flow_lokr \
  --config benchmarks/self-flow-finetune-512/probe.toml --use-config-settings \
  --mode self_flow --ema-dtype float32 --steps 10 \
  --output benchmarks/self-flow-finetune-512/fp32-ema.json

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.tools.probe_self_flow_lokr \
  --config benchmarks/self-flow-finetune-512/probe.toml --use-config-settings \
  --mode self_flow --ema-dtype bfloat16 --ema-stochastic-rounding --steps 10 \
  --output benchmarks/self-flow-finetune-512/bf16-sr-ema.json

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.tools.probe_self_flow_lokr \
  --config benchmarks/self-flow-finetune-512/probe.toml --use-config-settings \
  --mode self_flow --ema-dtype bfloat16 --ema-stochastic-rounding \
  --ema-adaln-fp32 --steps 10 \
  --output benchmarks/self-flow-finetune-512/mixed-ema.json

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.tools.probe_self_flow_lokr \
  --config benchmarks/self-flow-finetune-512/probe.toml --use-config-settings \
  --mode self_flow --ema-device cpu --ema-dtype float32 --steps 10 \
  --output benchmarks/self-flow-finetune-512/cpu-fp32-ema.json
```

These runs retain SDNQ Adafactor RMS clipping, nominal LR 2e-4, REX with
100 warmup steps, quantized optimizer state, no first moment, no Kahan, and no
CPU optimizer offload. All ten steps are inside warmup. Both the baseline and
Self-Flow use the same dual-timestep student noising. Execution uses checkpointing,
torch_varlen attention, and default regional compilation. EMA decay is 0.99 and
alignment loss weight is 0.8.

Short-run losses and finite gradients establish execution, not faster convergence,
equivalent teacher quality, or a suitable learning rate for a long full finetune.
An image-quality comparison requires saved checkpoints and held-out evaluation.

## Validation

Small-model tests cover uniform-forward parity, adapter and full-finetune teacher
isolation, shared AdaLN/input projection averaging, checkpointed backward, and
sub-BF16-resolution EMA updates with stochastic rounding. The real-model parity
check compares eager paths to avoid conflating compiler-induced BF16 rounding
differences with differences in the experimental forward.
