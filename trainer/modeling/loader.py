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
    return {name: getattr(train, name, None) for name in
            ("transformer_path", "text_encoder_path", "vae_path", "tokenizer_path")}


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
        transformer.load_state_dict(state, strict=True, assign=True)
        # RoPE tables are ordinary attributes, so load_state_dict(assign=True) cannot
        # materialize those created inside the meta context.
        from .modules.mage_layers import MageFlowEmbedRope

        transformer.pos_embed = MageFlowEmbedRope(
            theta=10000, axes_dim=params.axes_dim, scale_rope=True
        )
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
        from .modules.mage_vae import MageVAE

        vae = MageVAE(
            str(vae_path or path / "vae/diffusion_pytorch_model.safetensors"),
            sample_posterior=False,
        )
        # Training only encodes; discard the decoder before moving the VAE to GPU.
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
