"""LyCORIS LoRA registered under each target, so DDP and checkpointing see its tensors."""

from dataclasses import asdict

import torch


ALGORITHMS = ("locon", "lokr", "dora")
DECOMPOSED = {"dora"}


def dense_weight(module):
    if hasattr(module, "sdnq_dequantizer"):
        return module.sdnq_dequantizer(
            module.weight,
            module.scale,
            zero_point=module.zero_point,
            svd_up=module.svd_up,
            svd_down=module.svd_down,
            skip_quantized_matmul=True,
        ).detach()
    return module.weight.detach()


class SDNQWeightAccess:
    def _current_weight(self):
        return dense_weight(self.org_module[0])

    @property
    def org_weight(self):
        return self._current_weight()


_ADAPTER_CLASSES = None


def algorithm_classes():
    global _ADAPTER_CLASSES
    if _ADAPTER_CLASSES is None:
        from lycoris.modules import LoConModule, LokrModule

        base = dict(locon=LoConModule, lokr=LokrModule, dora=LoConModule)
        _ADAPTER_CLASSES = {
            k: type("Mage" + k.title(), (SDNQWeightAccess, cls), {})
            for k, cls in base.items()
        }
    return _ADAPTER_CLASSES


def apply_lycoris_lora(model, cfg):
    from .params import adapter_target_names, classify

    cfg._validate_lora_targets()
    targets = [
        (name, model.get_submodule(name))
        for name in adapter_target_names(model, cfg.components)
    ]
    model.requires_grad_(False)
    for name, module in targets:
        if classify(name + ".weight") == "adaln":
            raise ValueError("AdaLN must remain frozen for LoRA")
        # Bypass keeps SDNQ's frozen-base forward intact; only the low-rank
        # activation path is dispatched through LyCORIS kernels.
        wd = cfg.lycoris_algo in DECOMPOSED
        source = module
        if wd:
            # Materialize one target only to initialize the magnitude; release it immediately.
            source = torch.nn.Linear(
                module.in_features, module.out_features, bias=False, device="meta"
            )
            source.weight = torch.nn.Parameter(
                dense_weight(module), requires_grad=False
            )
        adapter = algorithm_classes()[cfg.lycoris_algo](
            name,
            source,
            lora_dim=cfg.rank,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            bypass_mode=cfg.lycoris_bypass and not wd,
            weight_decompose=wd,
            wd_on_out=cfg.lycoris_wd_on_output,
            factor=cfg.lokr_factor,
            decompose_both=cfg.lokr_decompose_both,
        )
        adapter.org_module = [module]
        adapter.org_forward = module.forward
        adapter.bypass_mode = cfg.lycoris_bypass and not wd
        del source
        alpha = (
            adapter.alpha.detach().clone().float()
            if hasattr(adapter, "alpha")
            else None
        )
        adapter.to(device=module.weight.device, dtype=getattr(torch, cfg.dtype))
        # Preserve the algorithm's effective alpha (LoKr can normalize its scale).
        if alpha is not None:
            adapter.alpha = alpha.to(module.weight.device)
        module.add_module("lycoris_adapter", adapter)
        adapter.apply_to()
    model._lycoris_config = asdict(cfg)
    return model


def lycoris_state_dict(model):
    """Native dotted Mage-Flow targets with LyCORIS down/up/alpha tensor names."""
    return {
        "diffusion_model." + name.removesuffix(".lycoris_adapter") + "." + key: value
        for name, module in model.named_modules()
        if name.endswith(".lycoris_adapter")
        for key, value in module.state_dict().items()
    }


def load_lycoris_state_dict(model, state):
    expected = lycoris_state_dict(model)
    if set(state) != set(expected):
        raise ValueError(
            "LyCORIS checkpoint targets do not match this adapter configuration"
        )
    for name, module in model.named_modules():
        if name.endswith(".lycoris_adapter"):
            prefix = "diffusion_model." + name.removesuffix(".lycoris_adapter") + "."
            module.load_state_dict(
                {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)},
                strict=True,
            )
