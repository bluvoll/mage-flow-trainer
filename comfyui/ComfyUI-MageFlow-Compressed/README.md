# ComfyUI Mage-Flow Compressed

A dedicated **Load Compressed Mage-Flow** node for Mage-Flow transformers with
shared low-rank AdaLN conditioning and separate per-block modulation heads.
**ComfyUI's built-in loaders, model detection, and registry stay unchanged.**

Requires a ComfyUI installation with native Mage-Flow support
(`comfy.ldm.mage_flow`). No extra Python dependencies or automatic downloads.

## Install

Clone this repository into `ComfyUI/custom_nodes/`, then restart ComfyUI:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/bluvoll/ComfyUI-MageFlow-Compressed.git
```

Download the transformer from
[Bluvoll on Hugging Face](https://huggingface.co/Bluvoll)
and place it in `ComfyUI/models/diffusion_models/`.

## Use

Replace your Mage-Flow diffusion model loader with **Load Compressed Mage-Flow**
under `model/loaders`. Select the checkpoint and connect `MODEL` to your sampler.
Keep the rest of your Mage-Flow workflow:

- **Load CLIP:** `qwen3vl_4b_bf16.safetensors`, type **mage**.
- **Load VAE:** `mage-flow-vae.safetensors`.
- **Empty Flux 2 Latent**, positive/negative text encoding, KSampler, VAE Decode, and Save Image.

Start with BF16, Euler, 32 steps, CFG 4, and 1024×1024. The model uses shift 6.
The node keeps calibrated modulation factors in FP32 through loading and
offloading; the trunk uses BF16 by default on supported hardware. FP32 inference
is also selectable.

## Weighted prompts (experimental)

Use **Mage-Flow Weighted Text Encode** under `conditioning/mage-flow` in place
of ordinary text encoding. Connect Load CLIP (type **mage**) to its `clip` input
and its CONDITIONING output to the sampler's positive or negative conditioning.
This works with both full and compressed Mage-Flow transformers.

Example: `a landscape, (night sky:1.5), (clouds:0.8)`. Parentheses without a
number use ComfyUI's default emphasis; escape literal parentheses as `\(` and
`\)`. Start with modest weights around 0.8–1.3.

The node removes weighting syntax before encoding and preserves the clean
prompt's tokenization. Weight 1.0 matches ordinary encoding. Tokens crossing a
weight boundary receive the average weight over their UTF-8 bytes. ComfyUI's
existing encoder applies weights relative to an empty reference embedding;
this is an experimental conditioning adjustment, not a literal attention
multiplier or a guarantee of stronger visual adherence. Non-unit weights need
an additional reference sequence during text encoding. Text-to-image only;
chat templates and reference-image prompts are not supported by this node.

Run `verify_weighted_text.py COMFY_DIRECTORY` with ComfyUI's Python for tokenizer
checks; add `--encoder /path/to/qwen3vl_4b_bf16.safetensors` to test real embeddings.

## Supported checkpoints

Loads `mageflow-lowrank-modulation-v1` checkpoints exported by
[mage-flow-trainer](https://github.com/bluvoll/mage-flow-trainer), using strict
weight loading. Factors must be stored in FP32; the trainer preserves this
export format even when finetuning those factors in BF16.

Dense Mage-Flow models, LoRA adapters, and FP8/INT8 checkpoint formats are not
handled by this node. Compressed models remain experimental and are not
bit-identical to the trainer's BF16 inference.

## Compressed RTI checkpoints

**Load Compressed RTI Mage-Flow** loads checkpoints that combine shared AdaLN
compression with RTI full finetuning (`architecture = mageflow-rti-v1`). Set
**keep fraction** to the value used at the end of RTI training, normally the
checkpoint's `rti.target_keep` value. The first implementation supports one
target image per batch and no reference-image latents.

## Verification and license

Run `verify.py COMFY_DIRECTORY COMPRESSED_CHECKPOINT` with ComfyUI's Python to
check loading, sampling, factor precision, and unchanged built-in loaders.
The rank-256 checkpoint also passed partial-offload testing and a complete
1024px, 32-step image generation.

GPL-3.0-or-later. The forward implementation is adapted from ComfyUI; see
[LICENSE](LICENSE). Model weights are distributed separately under their own license.
