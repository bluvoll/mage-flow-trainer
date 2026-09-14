# Mage-Flow LoRA targets and LyCORIS

LyCORIS 4.0.0 was cloned and installed into `.venv` from latest upstream commit `4a6a333819356795d22170fe661a84f16b299b6b`. The same revision is pinned in `pyproject.toml`, `uv.lock`, and `requirements.txt`. Existing PyTorch and SDNQ versions were retained.

Choose `lycoris_lora` in the GUI's Adapter method, or use [the example config](../configs/mageflow-lycoris-lora.toml):

```toml
[adapter]
kind = "lycoris_lora"
rank = 32
alpha = 32

[train]
compile = "default"
attention_backend = "torch_varlen"
gradient_checkpointing = true
```

Selecting a LyCORIS method/algorithm in the GUI enables block compilation by default. TOML configs also default to `train.compile = "default"` when omitted for LyCORIS; explicit `compile = false` disables it. LoKr factor controls apply to the LyCORIS LoKr algorithm too.

Adapter dtype defaults to FP32; `dtype = "bfloat16"` under `[adapter]` is also supported. The GUI exposes only LoCon, LoKr, and DoRA. Set `adapter.lycoris_algo` to `locon`, `lokr`, or `dora`; the legacy `lora` spelling maps to `locon`. LoCon and LoKr default to bypass mode so SDNQ retains the frozen-base computation. DoRA disables bypass and defaults to `lycoris_wd_on_output = true`; SDNQ weights are dequantized as needed for magnitude normalization without retaining a second complete base model. Adapters are registered under each Linear module so device transfers, optimizer grouping, DDP parameter discovery, and Accelerate checkpoints include their tensors.

LyCORIS automatically dispatches its low-rank operations among fused and compiled backends. To explicitly select its compile backend, launch with `LYCORIS_KERNEL_BACKEND=compile`. This is separate from `train.compile`, which compiles Mage-Flow blocks. The [4.0 release notes](https://github.com/KohakuBlueleaf/LyCORIS/releases/tag/v4.0.0) describe this dispatch and the available backend choices. Spectral initialization and the retired concept-preservation implementation are not supported for this adapter kind; incompatible configurations are rejected.

## Target audit

Default PEFT and LyCORIS adapters use the same exact target names. For the local 12-block checkpoint:

| Component | Targets |
|---|---:|
| Image Q/K/V and output projection | 48 |
| Text Q/K/V and output projection | 46 |
| Image and text MLP up/down projections | 46 |
| Total | 140 |

The final block's text Q projection, text output projection, and both text MLP projections are excluded: they feed only the discarded final text output. Text K/V remain because image attention consumes them. The audit reduced the previous 144 targets to 140, or 41,091,072 adapter parameters at rank 32. The earlier VRAM report used the old target set.

AdaLN, timestep projections, and input/output projections are excluded from the default targets. Explicit AdaLN requests are rejected for both LoRA backends. Full finetuning is unaffected. The complete target names and shapes are in [mageflow-adapter-targets.json](mageflow-adapter-targets.json).

## Checkpoints and validation

Exports use native dotted Mage-Flow target names prefixed with `diffusion_model.`, and LyCORIS `lora_down.weight`, `lora_up.weight`, and `alpha` keys. A JSON sidecar and embedded metadata record the adapter configuration. `trainer.training.lycoris.load_lycoris_state_dict` strictly reloads these exports into a matching adapter configuration. This format was tested within the trainer; external inference-loader compatibility has not been tested.

Accelerate full-state save/load was verified separately. Switching PEFT to LyCORIS, or resuming a previous 144-target optimizer state with the new 140-target configuration, changes parameter names/counts and is not a compatible optimizer-state resume. Start a new run for the new backend/target set.

Validation includes CPU forward/backward, frozen-base/AdaLN checks, export/reload numerical agreement, and Accelerate state restoration. GPU tests compare eager and compiled outputs/gradients with SDNQ INT8 bases, checkpointing, and torch varlen attention, using BF16 adapters with automatic LyCORIS dispatch and FP32 adapters with its explicit compile backend. The explicit compile test also performs a quantized AdamW optimizer update. Dynamic block compilation also passed backward and an optimizer update across two image shapes with automatic LyCORIS dispatch. All 168 GUI checks passed. These are small-model correctness checks; full-checkpoint throughput, VRAM, and long-run image quality have not been measured for this backend.

```bash
CUDA_VISIBLE_DEVICES=1 MAGE_LYCORIS_GPU=1 LYCORIS_KERNEL_BACKEND=auto \
  .venv/bin/python -m unittest discover -s tests -p test_lycoris.py -v
```

The three exposed algorithms passed GPU SDNQ + compiled varlen forward/backward comparisons. Adapter-path dropout is supported only for LoCon; caption augmentation/dropout is independent and works with all three.

The **LoKr factor** control is in the main Adapter section and passes `adapter.lokr_factor` as LyCORIS's `factor` argument. It defaults to -1 (balanced factorization), accepts integers from -1 through the GUI integer limit (2,147,483,647), and is independent of rank. Previously this path incorrectly passed PEFT's `decompose_factor` keyword, which LyCORIS ignored; non-default factors now take effect. Changing the factor can change adapter tensor shapes, so match it when reloading weights or resuming an optimizer state.
