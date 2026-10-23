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
            ("transformer_path", "text_encoder_path", "vae_path", "tokenizer_path", "compressed_adaln_dtype", "flux2_vae", "model_family")}


def resolve_model_family(path, model_family="auto"):
    if model_family not in (None, "auto", "mage_flow", "flux3", "moss_image"):
        raise ValueError(f"Unsupported model family: {model_family}")
    if model_family not in (None, "auto"):
        return model_family
    config = Path(path) / 'config.json'
    if config.is_file() and json.loads(config.read_text()).get('model') == 'flux-3-image-klein':
        return 'flux3'
    index = Path(path) / 'model_index.json'
    if index.is_file() and json.loads(index.read_text()).get('_class_name') == 'MOSSImagePipeline':
        return 'moss_image'
    return "mage_flow"


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
    model_family: str = "mage_flow"


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
    vae_decoder=False,
    model_family="auto",
):
    """`vae_decoder=True` loads the VAE for decoding (validation samples): Mage-VAE keeps its
    decoder and drops the encoder instead of the other way round."""
    path = Path(path)
    family = resolve_model_family(path, model_family)
    transformer = None
    if family == 'flux3' and load_transformer:
        from .flux3_loader import load_transformer as load_flux3
        transformer = load_flux3(path, transformer_path, dtype)
    if family == 'moss_image' and load_transformer:
        from .moss_loader import load_transformer as load_moss
        transformer = load_moss(path, transformer_path, dtype)
    if load_transformer and family not in ('flux3', 'moss_image'):
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
    if family == 'moss_image':
        from .moss_loader import component_config, text_sources as moss_text_sources
        encoder_source, tokenizer_source = moss_text_sources(path, text_encoder_path, tokenizer_path)
    if load_text_encoder and family == 'moss_image':
        from transformers import AutoModel
        if encoder_source.is_file():
            from .moss_text_loader import load_single_text_encoder
            encoder = load_single_text_encoder(encoder_source, component_config(path, 'text_encoder'), dtype)
        else:
            encoder = AutoModel.from_pretrained(encoder_source, torch_dtype=dtype,
                                               attn_implementation='sdpa', local_files_only=True)
        encoder.requires_grad_(False).eval()
    if load_text_encoder and family != 'moss_image':
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
        if flux2_vae or family in ("flux3", "moss_image"):
            from .flux2_vae_loader import load_flux2_vae
            source = Path(vae_path) if vae_path else path / "vae"
            vae = load_flux2_vae(source, torch.float32 if family in ("flux3", "moss_image") else dtype)
        else:
            from .modules.mage_vae import MageVAE
            vae = MageVAE(str(vae_path or path / "vae/diffusion_pytorch_model.safetensors"), sample_posterior=False)
            if vae_decoder:
                del vae.dconv_encoder
            else:
                del vae.decoder_model
        vae.requires_grad_(False).eval().to(dtype=torch.float32 if family in ("flux3", "moss_image") else dtype)
    return MageFlowComponents(transformer, encoder, vae, tokenizer, family)


@torch.no_grad()
def encode_prompts(components, prompts, device, max_length=512, *, embedding_only=False):
    family = getattr(components, 'model_family', 'mage_flow')   # component bundles built by hand may omit it
    if family == 'moss_image':
        from .moss_loader import encode_prompts as encode_moss
        return encode_moss(components, prompts, device, max_length)
    if family == 'flux3':
        from .flux3_loader import encode_prompts as encode_flux3
        return encode_flux3(components, prompts, device, max_length)
    tokens = components.tokenizer(
        [PROMPT_TEMPLATE_ENCODE.format(p) for p in prompts],
        padding=True,
        truncation=True,
        max_length=max_length + PROMPT_TEMPLATE_ENCODE_START_IDX,
        return_tensors="pt",
    )
    ids, mask = tokens.input_ids.to(device), tokens.attention_mask.to(device)
    hidden = encode_text_hidden(components.text_encoder, ids, mask, embedding_only=embedding_only)
    offset = PROMPT_TEMPLATE_ENCODE_START_IDX
    return hidden[:, offset:].contiguous(), mask[:, offset:].bool().contiguous()
