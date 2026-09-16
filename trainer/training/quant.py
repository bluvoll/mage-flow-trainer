"""SDNQ quantization for Mage-Flow. Automatic QMM stays off until benchmarked."""

from __future__ import annotations


from dataclasses import dataclass, field, replace

import torch
from torch import nn

_ADALN_ALL = ["img_mod", "txt_mod"]
_ADALN_FIRST_BLOCK = [
    "transformer_blocks.0.img_mod.1.weight",
    "transformer_blocks.0.txt_mod.1.weight",
]
_MLP_DOWN_ALL = ["*.img_mlp.net.2.weight", "*.txt_mlp.net.2.weight"]
_HIGH_PRECISION = [
    "time_text_embed",
    "img_in",
    "txt_in",
    "txt_norm",
    "norm_out",
    "proj_out",
    "pos_embed",
]
SKIP_POLICIES = ("default", "first_block_adaln", "all_adaln", "mlp_down")


@dataclass
class QuantConfig:
    """Quantization settings. `mode='none'` leaves the model in bf16."""

    mode: str = "none"  # "none" | "frozen" | "training"
    weights_dtype: str = (
        "int8"  # int8 default: faster than fp8 here, and 2x lower error
    )
    use_quantized_matmul: bool | str = "auto"
    quantize_text_encoder: bool = False

    skip_policy: str = "default"
    extra_skip: list[str] = field(default_factory=list)

    dynamic_loss_threshold: float | None = None

    group_size: int = 0
    use_stochastic_rounding: bool = True

    def __post_init__(self):
        if self.mode not in ("none", "frozen", "training"):
            raise ValueError(f"unknown quant mode: {self.mode!r}")
        if self.skip_policy not in SKIP_POLICIES:
            raise ValueError(
                f"unknown skip_policy: {self.skip_policy!r} (expected {SKIP_POLICIES})"
            )
        if self.use_quantized_matmul not in (True, False, "auto"):
            raise ValueError(
                f'use_quantized_matmul must be true, false, or "auto", '
                f"got {self.use_quantized_matmul!r}"
            )

    def skip_keys(self) -> list[str]:
        keys = _HIGH_PRECISION + list(self.extra_skip)
        if self.skip_policy == "all_adaln":
            keys += _ADALN_ALL
        elif self.skip_policy == "first_block_adaln":
            keys += _ADALN_FIRST_BLOCK
        elif self.skip_policy == "mlp_down":
            keys += _MLP_DOWN_ALL
        return keys


def text_encoder_quant_config(cfg: QuantConfig) -> QuantConfig | None:
    """Effective frozen encoder settings, independent of transformer training mode."""
    if not cfg.quantize_text_encoder or cfg.mode == "none":
        return None
    return replace(
        cfg,
        mode="frozen",
        skip_policy="default",
        extra_skip=["lm_head"],
        use_quantized_matmul=False,
    )


def resolve_quantized_matmul(cfg, tokens=None):
    """Require an explicit opt-in: there is no measured Mage-Flow crossover."""
    return (
        False if cfg.use_quantized_matmul == "auto" else bool(cfg.use_quantized_matmul)
    )


def tokens_for_bucket(bucket: tuple[int, int]) -> int:
    """Pixels -> DiT tokens: 16x Mage-VAE downsampling and patch_size=1."""
    w, h = bucket
    return (w // 16) * (h // 16)


def build_sdnq_config(
    cfg: QuantConfig,
    device: torch.device,
    is_training: bool,
    use_qmm: bool | None = None,
):
    from sdnq import SDNQConfig

    return SDNQConfig(
        weights_dtype=cfg.weights_dtype,
        use_quantized_matmul=resolve_quantized_matmul(cfg, None)
        if use_qmm is None
        else use_qmm,
        group_size=cfg.group_size,
        dynamic_loss_threshold=cfg.dynamic_loss_threshold,
        use_stochastic_rounding=cfg.use_stochastic_rounding,
        modules_to_not_convert=cfg.skip_keys(),
        add_skip_keys=True,  # also apply SDNQ's generic list
        quantization_device=device,
        return_device=device,
        is_training=is_training,
    )


def quantize_module(
    module: nn.Module,
    cfg: QuantConfig,
    device: torch.device,
    dtype: torch.dtype,
    use_qmm: bool | None = None,
) -> nn.Module:
    """Quantize one module in place-ish (returns the converted module)."""
    from sdnq import apply_sdnq_to_module
    from sdnq.training import add_module_skip_keys, apply_sdnq_training_to_module

    is_training = cfg.mode == "training"
    if getattr(getattr(module, "params", None), "modulation_rank", 0):
        cfg = replace(
            cfg,
            extra_skip=cfg.extra_skip + ["modulation_down", "img_mod", "txt_mod"],
        )
    sdnq_cfg = build_sdnq_config(cfg, device, is_training, use_qmm)

    module, sdnq_cfg = add_module_skip_keys(module, sdnq_cfg)

    apply = apply_sdnq_training_to_module if is_training else apply_sdnq_to_module
    module, _ = apply(module, sdnq_cfg, torch_dtype=dtype)
    return module


def dequantize_state_dict(
    state_dict: dict, dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    """Turn `SDNQTensor` master weights back into plain tensors.

    Required before export: safetensors cannot serialize an SDNQTensor (it raises
    "Attempted to access the data pointer on an invalid python storage"), so a quantized full
    finetune would train fine and then fail at the first checkpoint without this.

    The exported weights are the dequantized values, so the checkpoint is an ordinary bf16 model --
    it carries the quantization error the training accumulated, but needs no special loader.
    """
    from sdnq.training import SDNQTensor

    out = {}
    for key, value in state_dict.items():
        if isinstance(value, SDNQTensor):
            value = value.dequantize(dtype)
        out[key] = value.detach().to(dtype)
    return out


def quantized_layer_report(module: nn.Module) -> tuple[int, int, list[str]]:
    """-> (quantized Linear count, total Linear count, unquantized layer names).

    Worth printing: a skip list that matches nothing is indistinguishable from a correct one until
    quality drops, and a typo'd key fails silently.
    """
    from sdnq.layers import SDNQLayer, SDNQLinear

    quantized, skipped = 0, []
    for name, m in module.named_modules():
        if isinstance(m, (SDNQLinear, SDNQLayer)):
            quantized += 1
        elif isinstance(m, nn.Linear):
            skipped.append(name)
    return quantized, quantized + len(skipped), skipped
