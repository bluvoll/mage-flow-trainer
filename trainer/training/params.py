"""Mage-Flow component groups and PEFT adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from torch import nn

_COMPONENT_PATTERNS = [
    ("adaln", re.compile(r"^transformer_blocks\.\d+\.(img_mod|txt_mod)\.")),
    (
        "text_attn",
        re.compile(r"^transformer_blocks\.\d+\.attn\.(add_|to_add_|norm_added)"),
    ),
    ("image_attn", re.compile(r"^transformer_blocks\.\d+\.attn\.")),
    ("mlp", re.compile(r"^transformer_blocks\.\d+\.(img_mlp|txt_mlp)\.")),
    ("adaln", re.compile(r"^(norm_out|time_text_embed|modulation_down)\.")),
    ("base", re.compile(r"^(img_in|txt_in|txt_norm|proj_out)\.")),
]
COMPONENTS = ("image_attn", "text_attn", "mlp", "adaln", "base")
_LORA_TARGETS = {
    "image_attn": ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"),
    "text_attn": (
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
    ),
    "mlp": (
        "img_mlp.net.0.proj",
        "img_mlp.net.2",
        "txt_mlp.net.0.proj",
        "txt_mlp.net.2",
    ),
    "adaln": (
        "img_mod.1",
        "txt_mod.1",
        "norm_out.linear",
        "time_text_embed.timestep_embedder.linear_1",
        "time_text_embed.timestep_embedder.linear_2",
    ),
    "base": ("img_in", "txt_in", "proj_out"),
}


def classify(name: str) -> str:
    """Map a transformer parameter name to its component. Unmatched names are an error, not a
    silent drop into a default bucket -- a diffusers version bump that renames a submodule would
    otherwise quietly stop training it."""
    for component, pattern in _COMPONENT_PATTERNS:
        if pattern.match(name):
            return component
    raise KeyError(f"unclassified transformer parameter: {name!r}")


@dataclass
class ComponentLRs:
    """Per-component learning rates. None means "use the global lr"; 0.0 means freeze.

    Applies to **both** full finetuning and LoRA. Under LoRA the split is over the adapter tensors,
    grouped by the component of the base module each one wraps -- `classify()` matches LoRA
    parameter names unchanged, because the peft suffix (`.lora_A.default.weight`) sits after the
    part the patterns key on.

    """

    image_attn: float | None = None
    text_attn: float | None = None
    mlp: float | None = None
    adaln: float | None = None
    base: float | None = None

    def resolve(self, component: str, default_lr: float) -> float:
        lr = getattr(self, component)
        return default_lr if lr is None else float(lr)

    def explicit(self) -> dict[str, float]:
        """Components the user actually set, for reporting and for validation against
        `adapter.components` -- a LR on a component with no adapter injected is a no-op."""
        default = ComponentLRs()
        return {
            c: getattr(self, c)
            for c in COMPONENTS
            if getattr(self, c) != getattr(default, c)
        }


@dataclass
class ParamGroupReport:
    groups: list[dict]
    counts: dict[str, int]  # component -> trainable parameter count
    frozen: dict[str, int]  # component -> frozen parameter count

    def summary(self) -> str:
        lines = []
        for c in COMPONENTS:
            n_train, n_frozen = self.counts.get(c, 0), self.frozen.get(c, 0)
            if not (n_train or n_frozen):
                continue
            lr = next((g["lr"] for g in self.groups if g["component"] == c), 0.0)
            state = f"lr={lr:.2e}" if n_train else "FROZEN"
            lines.append(
                f"  {c:<12} {n_train / 1e6:8.2f}M trainable  {n_frozen / 1e6:8.2f}M frozen  {state}"
            )
        total = sum(self.counts.values())
        lines.append(
            f"  {'TOTAL':<12} {total / 1e6:8.2f}M trainable "
            f"({total / max(total + sum(self.frozen.values()), 1):.1%})"
        )
        return "\n".join(lines)


def build_param_groups(
    transformer: nn.Module,
    lrs: ComponentLRs,
    default_lr: float,
    weight_decay: float = 0.0,
) -> ParamGroupReport:
    """Split parameters by component, freeze the zero-LR ones, and return optimizer groups.

    Freezing happens here rather than in the caller so that `requires_grad` and the optimizer
    groups can never disagree -- a parameter is in a group if and only if it is trainable.
    """
    buckets: dict[str, list[torch.nn.Parameter]] = {c: [] for c in COMPONENTS}
    counts: dict[str, int] = {}
    frozen: dict[str, int] = {}

    named = [(classify(n), p) for n, p in transformer.named_parameters()]

    for component, param in named:
        if lrs.resolve(component, default_lr) == 0.0:
            param.requires_grad_(False)
            frozen[component] = frozen.get(component, 0) + param.numel()
            continue
        param.requires_grad_(True)
        buckets[component].append(param)
        counts[component] = counts.get(component, 0) + param.numel()

    groups = [
        {
            "params": params,
            "lr": lrs.resolve(component, default_lr),
            "weight_decay": weight_decay,
            "component": component,
        }
        for component, params in buckets.items()
        if params
    ]
    if not groups:
        raise ValueError("every component is frozen; nothing to train")

    return ParamGroupReport(groups=groups, counts=counts, frozen=frozen)


def build_adapter_param_groups(
    transformer: nn.Module,
    lrs: ComponentLRs,
    default_lr: float,
    weight_decay: float = 0.0,
) -> ParamGroupReport:
    """The LoRA counterpart of `build_param_groups`: split the *adapter* tensors by the component
    of the base module they wrap.

    Only parameters that are already trainable are considered -- `apply_adapter` has frozen the
    bases by then, so this sees adapter tensors and nothing else. A component set to 0.0 is frozen,
    though dropping it from `adapter.components` is better: that skips injecting the adapter at all
    rather than carrying dead zero tensors into the export.
    """
    buckets: dict[str, list[torch.nn.Parameter]] = {c: [] for c in COMPONENTS}
    counts: dict[str, int] = {}
    frozen: dict[str, int] = {}

    named: list[tuple[str, torch.nn.Parameter]] = [
        (classify(n), p) for n, p in transformer.named_parameters() if p.requires_grad
    ]
    for component, param in named:
        if lrs.resolve(component, default_lr) == 0.0:
            param.requires_grad_(False)
            frozen[component] = frozen.get(component, 0) + param.numel()
            continue
        buckets[component].append(param)
        counts[component] = counts.get(component, 0) + param.numel()

    groups = [
        {
            "params": params,
            "lr": lrs.resolve(component, default_lr),
            "weight_decay": weight_decay,
            "component": component,
        }
        for component, params in buckets.items()
        if params
    ]
    if not groups:
        raise ValueError(
            "every adapter component is frozen by component_lr; nothing to train"
        )
    return ParamGroupReport(groups=groups, counts=counts, frozen=frozen)


def lora_target_modules(components: list[str]) -> list[str]:
    """Suffix patterns PEFT matches against module names, for the listed components."""
    unknown = [c for c in components if c not in _LORA_TARGETS]
    if unknown:
        raise ValueError(
            f"no LoRA targets defined for {unknown}; valid: {sorted(_LORA_TARGETS)}"
        )
    targets = [t for c in components for t in _LORA_TARGETS[c]]
    if not targets:
        raise ValueError(
            f"components {components} contain no LoRA-injectable Linear layers"
        )
    return targets


def adapter_target_names(transformer, components):
    """Exact targets, excluding the final block's discarded text output branch."""
    suffixes = lora_target_modules(components)
    last = f"transformer_blocks.{len(transformer.transformer_blocks) - 1}."
    unused = {last + suffix for suffix in (
        "attn.add_q_proj", "attn.to_add_out", "txt_mlp.net.0.proj", "txt_mlp.net.2"
    )}
    names = [name for name, _ in transformer.named_modules()
             if any(name == s or name.endswith('.' + s) for s in suffixes)
             and name not in unused]
    if not names:
        raise ValueError("No trainable Mage-Flow adapter targets matched")
    return names


@dataclass
class AdapterConfig:
    """LoRA / LoKr settings. LoKr factorises the update as a Kronecker product, so it reaches a
    given expressivity with far fewer parameters than LoRA at the same rank -- worth it when
    optimizer state is the binding constraint, which on 24GB it usually is."""

    kind: str = "lora"  # "lora" | "lycoris_lora" | "lokr" | "none"
    rank: int = 32
    dtype: str = "float32"
    alpha: float = 32.0
    dropout: float = 0.0
    components: list[str] = field(
        default_factory=lambda: ["image_attn", "text_attn", "mlp"]
    )
    lycoris_algo: str = "locon"
    lycoris_bypass: bool = True
    lycoris_wd_on_output: bool = True
    # LoKr only.
    lokr_factor: int = -1  # -1 = pick the most balanced factorisation
    lokr_decompose_both: bool = False
    # Which subspace `lora_A` starts in: "random" (PEFT's own), or a band of the base weight's
    # singular directions -- "top" | "mid" | "bottom". See `spectrum_init`. lora_B stays zero in
    # every case, so the starting model is numerically identical and only the subspace differs.
    init: str = "random"
    # Where to cache the singular directions. One SVD pass serves all three bands.
    init_cache: str | None = None

    def __post_init__(self):
        if self.kind not in ("lora", "lycoris_lora", "lokr", "none"):
            raise ValueError(f"unknown adapter kind: {self.kind}")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("adapter.dtype must be float32 or bfloat16")
        if type(self.lokr_factor) is not int or self.lokr_factor < -1:
            raise ValueError("adapter.lokr_factor must be an integer >= -1")
        self._validate_lora_targets()
        if self.lycoris_algo == "lora":
            self.lycoris_algo = "locon"
        from .lycoris import ALGORITHMS
        if self.lycoris_algo not in ALGORITHMS:
            raise ValueError(f"Unknown LyCORIS algorithm: {self.lycoris_algo}")
        if self.kind == "lycoris_lora" and self.lycoris_algo != "locon" and self.dropout:
            raise ValueError("Adapter dropout is only supported for plain LyCORIS LoRA; caption dropout is independent")
        from .spectrum_init import MODES

        if self.init not in MODES:
            raise ValueError(f"adapter.init must be one of {MODES}, got {self.init!r}")
        if self.init != "random" and self.kind != "lora":
            raise ValueError(f"adapter.init={self.init!r} applies to kind='lora' only")

    def _validate_lora_targets(self):
        if self.kind in ("lora", "lycoris_lora") and "adaln" in self.components:
            raise ValueError("AdaLN must remain frozen for LoRA; remove 'adaln' from adapter.components")

    def build(self):
        """-> a peft config. Imported lazily so full-finetune runs need no peft import."""
        # Recheck mutable configs before injection, including programmatic callers.
        self._validate_lora_targets()
        from peft import LoKrConfig, LoraConfig

        targets = lora_target_modules(self.components)
        if self.kind == "lora":
            return LoraConfig(
                r=self.rank,
                lora_alpha=self.alpha,
                lora_dropout=self.dropout,
                target_modules=targets,
                init_lora_weights="gaussian",
            )
        return LoKrConfig(
            r=self.rank,
            alpha=self.alpha,
            rank_dropout=self.dropout,
            target_modules=targets,
            decompose_factor=self.lokr_factor,
            decompose_both=self.lokr_decompose_both,
        )


def apply_adapter(
    transformer: nn.Module,
    cfg: AdapterConfig,
) -> nn.Module:
    """Wrap the modules with PEFT adapters and freeze everything else.

    The base weights are frozen *before* injection so that the only trainable tensors are the
    adapter's own -- which is the whole point, and also what lets the base be quantized later
    without touching the optimizer.
    """
    if cfg.kind == "none":
        return transformer
    if cfg.kind == "lycoris_lora":
        from .lycoris import apply_lycoris_lora

        return apply_lycoris_lora(transformer, cfg)

    peft_config = cfg.build()
    peft_config.target_modules = set(adapter_target_names(transformer, cfg.components))
    transformer.requires_grad_(False)
    transformer.add_adapter(peft_config)

    if cfg.init != "random":
        from .spectrum_init import spectrum_init

        n = spectrum_init(transformer, cfg.init, cfg.init_cache)
        print(
            f"init     lora_A from base-weight SVD, band={cfg.init}, {n} layer(s)",
            flush=True,
        )

    # PEFT defaults to FP32 after injecting into quantized bases. Make adapter
    # storage/compute precision explicit without changing frozen base weights.
    adapter_dtype = getattr(torch, cfg.dtype)
    for param in transformer.parameters():
        if param.requires_grad:
            param.data = param.data.to(adapter_dtype)

    return transformer


def trainable_parameters(*modules: nn.Module | None) -> list[torch.nn.Parameter]:
    return [
        p for m in modules if m is not None for p in m.parameters() if p.requires_grad
    ]


def count_parameters(*modules: nn.Module | None) -> tuple[int, int]:
    """-> (trainable, total)."""
    trainable = total = 0
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            total += p.numel()
            trainable += p.numel() if p.requires_grad else 0
    return trainable, total
