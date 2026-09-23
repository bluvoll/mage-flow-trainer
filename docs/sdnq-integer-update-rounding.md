# INT8/UINT8 training update preservation

The trainer applies a compatibility fix when SDNQ training mode and stochastic
rounding are enabled. It recognizes the upstream implementation that adds
Gaussian noise with standard deviation 0.1 before rounding integer codes, and
replaces that operation for INT8/UINT8 with `floor(x + uniform(0, 1))`.
For a fractional code `n + f`, this chooses `n + 1` with probability `f`, preserving
the requested code in expectation. The old rule has a strong bias toward the
nearest integer, which can repeatedly erase small optimizer updates.

This is a trainer-local runtime compatibility fix, not an edit to the installed
SDNQ package. It is installed before training quantization and compilation. It
also affects stochastic integer optimizer-buffer quantization in that process.
Deterministic quantization, floating-point quantization, and processes using only
frozen model quantization are unchanged. The installation is idempotent and does
not override upstream code that no longer matches the recognized implementation.

## Controlled diagnosis

Tests used 256 rows of the Jaba faces base checkpoint's first image-query weight
on GPU 1. Twenty AdamW steps received identical, fixed unit-norm gradients at
LR 5e-5. We captured AdamW's requested updates and compared them with actual
dequantized weight displacement. Directional preservation below is the mean
displacement along the intended gradient sign divided by the corresponding mean
requested displacement; it is not a per-element guarantee or quality metric.

| Weight storage | Optimizer states | Original preservation | Corrected preservation |
|---|---|---:|---:|
| INT8 | Unquantized | 0.082% | 99.78% |
| INT8 | Quantized UINT8 | 0.081% | 100.05% |
| UINT8 | Unquantized | 0.258% | 99.70% |
| UINT8 | Quantized UINT8 | 0.262% | 100.08% |

The effect persists without optimizer-state quantization and with identical
gradients, isolating loss of small updates in integer weight writeback in this
probe. It is not explained by different model gradients or global clipping.
SDNQ 0.2.4 and upstream 0.2.7 at commit
`f3a7ea5cdd3ec4ee07b6e39a3c136a3ac503da0c` produced identical original probe
results. The venv was restored to 0.2.4 after that update experiment.

Tests cover signed sub-bin expectations, unchanged deterministic behavior, and
zero-scale groups for both INT8 and UINT8. A compiled INT8 forward/backward,
quantized-AdamW and export smoke test completed eight steps with finite gradients
and exports; its fixed-target loss decreased from 1.0474 to 1.0088. Compiled
real-weight optimizer probes cover both integer storage formats. Evidence and
reproducers are under `benchmarks/sdnq-update-check/`.

## Using the fix

Start a new training process with `quant.mode = "training"` and
`quant.use_stochastic_rounding = true` (the default). The startup log confirms
that unbiased INT8/UINT8 rounding is enabled. No new GUI option is needed.
Already-running training processes do not acquire the fix.

This preserves small updates statistically; it does not make an aggressive LR
safe, eliminate quantization noise, or establish model-quality equivalence to
BF16. The user's BF16 stress run produced deformed generations and is not a
quality target. Reassess learning rates on a short validation run rather than
assuming settings tuned with suppressed updates remain appropriate. Previously
lost updates cannot be recovered from saved checkpoints. DDP determinism and
full-model convergence have not been revalidated for this fix.
