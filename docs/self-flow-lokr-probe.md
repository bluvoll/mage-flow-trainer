# Experimental Self-Flow with a LoKr EMA teacher

On September 16, 2026, a ten-step feasibility test completed on one RTX 4090
using LoKr dimension 10000, alpha 10000, factor 1, with an EMA adapter teacher
sharing the student's frozen SDNQ base. This is an isolated experimental tool,
not an option in the production training loop or GUI.

## Measured results

| Method | Peak allocated VRAM | Peak reserved VRAM | Median warm step |
| --- | ---: | ---: | ---: |
| Ordinary LoKr | 14.01 GiB | 14.86 GiB | 1.357 s |
| Self-Flow with GPU FP32 adapter EMA | 20.86 GiB | 21.68 GiB | 1.589 s |

Both completed 10/10 updates with finite losses and adapter gradients. Warm
timings exclude the first two steps; they include diagnostic gradient checks,
forward/backward, optimizer work, and EMA updates, but exclude fetching cached
latents and text. Self-Flow added approximately 17% to this measured step time.
No CPU EMA offload or second GPU was necessary. The CPU streaming option exists
in the probe but has not been benchmarked on the full model.

These are corrected runs with validated Adafactor defaults (betas -0.8, 0.999).
An earlier harness retained AdamW betas when switching optimizer kind after
config loading. Its raw reports are preserved locally as `*-legacy-betas.json`;
the corrected runs supersede them. Memory use was unchanged.

The uniform-token experimental forward matched the existing forward exactly
on the real model. A separate small-model test verifies that teacher evaluation
uses EMA adapters without modifying student weights, EMA updates follow the
configured decay, dual-timestep backward works with checkpointing, and frozen
parameters receive no gradients.

## Configuration and interpretation

- Base: compressed MageTrail v0.2, shared rank-256 modulation, frozen UINT8 SDNQ
  weights, quantized matmul disabled. Frozen compressed AdaLN remains FP32.
- Data: the existing ten-image Bisque control dataset, cached 1344×768 image
  latents and Cached Text Encoder; one fixed caption per image.
- Batch size 1, torch_varlen attention, gradient checkpointing, outer compile off.
- LoKr uses full matrices at this dimension/factor. Actual adapters contain
  **2,623,537,292 parameters**, including scalar factors; effective alpha scaling
  is **1**, not 10000. The production target selector excludes unused final
  text-output branches. AdaLN remains frozen.
- SDNQ Adafactor, RMS clipping, LR 1e-4 constant, no warmup, no first moment,
  no Kahan, no global gradient clipping, quantized optimizer buffers enabled,
  optimizer offload disabled.
- Student feature layer 4, teacher feature layer 8; two-layer feature projector
  with hidden width 6144. Both runs initialize the same projector, although the
  baseline does not train it.
- EMA decay 0.99, alignment weight 0.8, token mask probability 0.25. Teacher
  uses the smaller timestep because this trainer defines zero as clean.
- Only adapters in the teacher's first eight blocks need EMA copies:
  **6.75 GiB FP32**. Teacher evaluation stops at block 8 and casts one block's
  EMA adapters at a time, without a full second BF16 adapter copy.
- EMA averages the adapter parameters, not their materialized Kronecker products.
- Image order, caption hashes, timestep pairs and random seeds match between
  runs. Student noising intentionally differs between the two objectives.

This establishes that the requested network and EMA teacher fit and execute.
It does **not** demonstrate improved image quality, stronger style, generalization,
or faster convergence. The mean flow loss was 0.393 for baseline and 0.611 for
Self-Flow, with different student noising; those numbers are not a fair quality
comparison. A longer comparison must evaluate the same held-out prompts and
ordinary denoising objective, including equal wall-clock training budgets.

## Running the isolated probe

Use a config with existing cached latents and text embeddings. The tool overrides
the adapter, optimizer and execution settings described above. It saves JSON
measurements, not checkpoints, and does not support resume or DDP.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.tools.probe_self_flow_lokr \
  --config benchmarks/uint8-controlled/uint8.toml \
  --mode baseline --steps 10 --output benchmarks/self-flow-lokr/baseline.json

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.tools.probe_self_flow_lokr \
  --config benchmarks/uint8-controlled/uint8.toml \
  --mode self_flow --ema-device cuda --steps 10 \
  --output benchmarks/self-flow-lokr/self-flow-gpu.json
```

The local dataset, config and raw reports under `benchmarks/` are not distributed
with the repository. Supply your own existing cached-data config elsewhere.
Peak memory depends on resolution, caption length, model and optimizer.

Research reference: [Self-Supervised Flow Matching for Scalable Multi-Modal
Synthesis](https://arxiv.org/abs/2603.06507).
