"""Load native Mage-Flow transformer, Mage-VAE and frozen Qwen3-VL."""

from dataclasses import dataclass, fields
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from .mage_flow import MageFlow, MageFlowParams
from .batched import PROMPT_TEMPLATE_ENCODE, PROMPT_TEMPLATE_ENCODE_START_IDX
from .mageflow_text import encode_text_hidden


ASSETS = Path(__file__).with_name("assets")


def model_load_kwargs(train):
    return {name: getattr(train, name, "float32" if name == "compressed_adaln_dtype" else None) for name in
            ("transformer_path", "text_encoder_path", "vae_path", "tokenizer_path", "compressed_adaln_dtype", "flux2_vae")}


def text_sources(path, text_encoder_path=None, tokenizer_path=None):
    encoder = Path(text_encoder_path) if text_encoder_path else Path(path) / "text_encoder"
    tokenizer = Path(tokenizer_path) if tokenizer_path else (
        ASSETS / "qwen3vl4b" if encoder.is_file() else encoder)
    return encoder, tokenizer


def _load_single_text_encoder(path, dtype):
    from accelerate import init_empty_weights
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig.from_pretrained(ASSETS / "qwen3vl4b", local_files_only=True)
    config._attn_implementation = "sdpa"
    with init_empty_weights():
        encoder = Qwen3VLForConditionalGeneration(config)
    state = load_file(str(path))
    # ComfyUI omits the tied output head. All other weights must be present.
    if "lm_head.weight" not in state and config.tie_word_embeddings:
        state["lm_head.weight"] = state["model.language_model.embed_tokens.weight"]
    encoder.load_state_dict(state, strict=True, assign=True)
    encoder.tie_weights()
    return encoder.to(dtype=dtype)


@dataclass
class MageFlowComponents:
    transformer: MageFlow | None
    text_encoder: torch.nn.Module | None
    vae: torch.nn.Module | None
    tokenizer: object | None


def load_components(
    path,
    dtype=torch.bfloat16,
    load_text_encoder=True,
    load_vae=True,
    load_tokenizers=True,
    load_transformer=True,
    transformer_path=None,
    text_encoder_path=None,
    vae_path=None,
    tokenizer_path=None,
    compressed_adaln_dtype="float32",
    flux2_vae=False,
):
    path = Path(path)
    transformer = None
    if load_transformer:
        source = Path(transformer_path) if transformer_path else path / "transformer"
        if source.is_file():
            weights = source
            with safe_open(str(weights), framework="pt") as f:
                embedded = (f.metadata() or {}).get("model_config")
            sidecar = source.with_suffix(".json")
            config = json.loads(embedded) if embedded else json.loads(
                (sidecar if sidecar.is_file() else ASSETS / "mageflow.json").read_text())
        else:
            config = json.loads((source / "config.json").read_text())
            weights = source / "diffusion_pytorch_model.safetensors"
        params = MageFlowParams(
            **{
                f.name: config[f.name]
                for f in fields(MageFlowParams)
                if f.name in config and f.name != "checkpoint"
            },
            checkpoint=False,
        )
        if params.patch_size != 1:
            raise ValueError("Only Mage-Flow patch_size=1 is supported")
        with torch.device("meta"):
            transformer = MageFlow(params)
        state = load_file(str(weights))
        for prefix in ("model.diffusion_model.", "diffusion_model."):
            if state and all(k.startswith(prefix) for k in state):
                state = {k[len(prefix):]: v for k, v in state.items()}
                break
        has_rti = any(k.startswith("region_interface.") for k in state)
        if has_rti != bool(params.rti_size_buckets):
            raise ValueError(
                "RTI checkpoint metadata and state tensors disagree. Restore the checkpoint's "
                "model_config (rti_size_buckets, rti_core_start, rti_core_end) before loading."
            )
        transformer.load_state_dict(state, strict=True, assign=True)
        # RoPE tables are ordinary attributes, so load_state_dict(assign=True) cannot
        # materialize those created inside the meta context.
        from .modules.mage_layers import MageFlowEmbedRope

        transformer.pos_embed = MageFlowEmbedRope(
            theta=10000, axes_dim=params.axes_dim, scale_rope=True
        )
        from .compressed_modulation import set_modulation_dtype

        set_modulation_dtype(transformer, getattr(torch, compressed_adaln_dtype))
        transformer.to(dtype=dtype)
        transformer.configure_execution()
    encoder = tokenizer = vae = None
    encoder_source, tokenizer_source = text_sources(path, text_encoder_path, tokenizer_path)
    if load_text_encoder:
        from transformers import Qwen3VLForConditionalGeneration

        encoder = (_load_single_text_encoder(encoder_source, dtype) if encoder_source.is_file()
                   else Qwen3VLForConditionalGeneration.from_pretrained(
                       encoder_source, torch_dtype=dtype, attn_implementation="sdpa"))
        encoder.requires_grad_(False).eval()
    if load_tokenizers:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, padding_side="right", local_files_only=True
        )
    if load_vae:
        if flux2_vae:
            from diffusers import AutoencoderKLFlux2
            source = Path(vae_path) if vae_path else path / "vae"
            if source.is_dir():
                vae = AutoencoderKLFlux2.from_pretrained(source, torch_dtype=dtype)
            else:
                state = load_file(str(source))
                state = {k.removeprefix("first_stage_model.").removeprefix("module."): v for k, v in state.items()}
                vae = AutoencoderKLFlux2()
                result = vae.load_state_dict(state, strict=False)
                if result.missing_keys or result.unexpected_keys:
                    # ComfyUI's Flux2 VAE export uses the original LDM key layout.
                    # Its Legacy engine already pixel-unshuffles and applies `bn` in
                    # encode(), producing exactly Mage-Flow's 128c /16 latent layout.
                    if "bn.running_mean" not in state or "encoder.quant_conv.weight" not in state:
                        raise RuntimeError(f"FLUX.2 VAE load failed: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
                    import sys
                    comfy_root = Path("/home/bluvoll/ComfyUI")
                    if str(comfy_root) not in sys.path:
                        sys.path.insert(0, str(comfy_root))
                    from comfy.ldm.models.autoencoder import AutoencodingEngineLegacy
                    ddconfig = {"double_z": True, "z_channels": 32, "resolution": 256,
                                "in_channels": 3, "out_ch": 3, "ch": 128,
                                "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2,
                                "attn_resolutions": [], "dropout": 0.0,
                                "batch_norm_latent": True}
                    vae = AutoencodingEngineLegacy(embed_dim=32, ddconfig=ddconfig,
                        regularizer_config={"target": "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"})
                    result = vae.load_state_dict(state, strict=True)
                    vae.outputs_packed_flux2 = True
        else:
            from .modules.mage_vae import MageVAE
            vae = MageVAE(str(vae_path or path / "vae/diffusion_pytorch_model.safetensors"), sample_posterior=False)
            del vae.decoder_model
        vae.requires_grad_(False).eval().to(dtype=dtype)
    return MageFlowComponents(transformer, encoder, vae, tokenizer)


@torch.no_grad()
def encode_prompts(components, prompts, device, max_length=512):
    tokens = components.tokenizer(
        [PROMPT_TEMPLATE_ENCODE.format(p) for p in prompts],
        padding=True,
        truncation=True,
        max_length=max_length + PROMPT_TEMPLATE_ENCODE_START_IDX,
        return_tensors="pt",
    )
    ids, mask = tokens.input_ids.to(device), tokens.attention_mask.to(device)
    hidden = encode_text_hidden(components.text_encoder, ids, mask)
    offset = PROMPT_TEMPLATE_ENCODE_START_IDX
    return hidden[:, offset:].contiguous(), mask[:, offset:].bool().contiguous()
