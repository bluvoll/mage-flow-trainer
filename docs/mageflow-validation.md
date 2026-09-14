# Mage-Flow migration validation

Environment: Linux, Python 3.11, `.venv`, PyTorch 2.10.0+cu128, RTX 4090. No PyTorch upgrade was needed. The local pretrained model has 12 blocks, hidden dimension 3072, 24 attention heads and head dimension 128.

## Real checkpoint smoke tests

These use synthetic images, not a quality-evaluation dataset. All losses were finite.

| Path | Test | Peak allocated GPU memory |
|---|---|---:|
| SDNQ frozen base + rank-2 LoRA | Two 256×256 steps, adapter export and Accelerate state save | 11.1 GB |
| Saved LoRA state | Resume at step 2 and complete step 3 | 11.0 GB |
| SDNQ base-weight training | One 256×256 optimizer step, quantized/offloaded AdamW state | 16.7 GB |
| Packed native resolutions | One LoRA step containing 256×256 and 192×256 images | 11.1 GB |

The base-weight smoke test trained image/text attention and both MLP streams (2.72B parameters); modulation and input/output groups were frozen. Its first step included kernel compilation and took 23.66 seconds. These figures are not estimates for 1024-pixel training, higher ranks, or all-parameter finetuning.

A separate mid-epoch test saved step 1 and resumed at step 2, skipping the consumed batch.

## Numerical and integration checks

The regression suite checks padded-caption isolation, selective-checkpoint gradient parity, strict native export/reload, adapter gradients and scale metadata, Qwen3-VL wrapper/prefix semantics, component grouping, SDNQ skip policy, Torch-varlen output/gradient agreement, and mixed-resolution packing against independent image forwards. GPU tests also cover compiled varlen backward and quantized weight update/export/optimizer-state reload.

All 9 GPU regression tests passed. A separate LoKr backward/export check passed. The GUI gate passed 150/150 checks; multi-resolution 23/23, encode mode 15/15, and texture 62/62. The lock check, wheel build, and Python undefined-name/import checks passed.

The GUI gate checks real editor construction and config coverage. Existing multi-resolution, encode-mode and texture gates exercise the retained dataset paths with Mage-VAE dimensions.

## Attention-only benchmark

24 heads × 128 dimensions, BF16, forward + backward, 3 warmups and median of 10 measured iterations. Varlen timings include Q/K/V gathering and output scattering, but exclude one-time microbatch metadata construction. SDPA receives a boolean key-padding mask. Token counts include image and text tokens.

| Valid joint sequence lengths | Masked SDPA | Torch varlen | Ratio |
|---|---:|---:|---:|
| 4160, 4608 | 34.82 ms | 16.03 ms | 2.17× |
| 1088, 4608 | 34.99 ms | 9.81 ms | 3.57× |
| 1088, 2112, 4160 | 53.90 ms | 11.15 ms | 4.83× |

This is not whole-training throughput. MLPs, projections, Qwen3-VL, VAE, optimization and checkpoint recomputation are outside these timings. Varlen is not universally lower-memory: pack/scatter buffers can increase transient allocations. Raw measurements and the reproducible benchmark tool are included.

## Remaining validation limits

- No long-run sample-quality comparison, full-model compiled throughput study, or SDNQ QMM crossover sweep.
- No end-to-end multi-GPU or Windows run for the new model integration.
- Mixed-resolution packing currently excludes curricula and optimal-transport noise pairing; batches are capped by sample count, not tokens.
- Optional external FlashAttention backends were ported but not installed/tested here.
