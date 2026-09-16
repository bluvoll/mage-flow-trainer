"""Optimizer capabilities and upstream defaults, shared by CLI and GUI.

SDNQ 0.2.4 and torch-optimi 0.3.3. Keep this module free of GPU imports.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerSpec:
    family: str
    name: str
    betas: tuple[float, ...] = ()
    eps: float | None = None
    weight_decay: float = 0.01


OPTIMIZERS = {
    "adamw": OptimizerSpec("PyTorch", "AdamW", (0.9, 0.999), 1e-8),
    "adamw8bit": OptimizerSpec("SDNQ", "AdamW", (0.9, 0.999)),
    "adafactor": OptimizerSpec("SDNQ", "Adafactor", (-0.8, 0.999)),
    "came": OptimizerSpec("SDNQ", "CAME", (0.9, 0.999, 0.9999)),
    "lion": OptimizerSpec("SDNQ", "Lion", (0.9, 0.999)),
    "muon": OptimizerSpec("SDNQ", "Muon", (0.9, 0.999)),
    "optimi_adam": OptimizerSpec("Optimi", "Adam", (0.9, 0.99), 1e-6, 0),
    "optimi_adamw": OptimizerSpec("Optimi", "AdamW", (0.9, 0.99), 1e-6),
    "optimi_adan": OptimizerSpec("Optimi", "Adan", (0.98, 0.92, 0.99), 1e-6, 0.02),
    "optimi_lion": OptimizerSpec("Optimi", "Lion", (0.9, 0.99), None, 0),
    "optimi_radam": OptimizerSpec("Optimi", "RAdam", (0.9, 0.99), 1e-6, 0),
    "optimi_ranger": OptimizerSpec("Optimi", "Ranger", (0.9, 0.99), 1e-6, 0),
    "optimi_sgd": OptimizerSpec("Optimi", "SGD", (), None, 0),
    "optimi_stableadamw": OptimizerSpec("Optimi", "StableAdamW", (0.9, 0.99), 1e-6),
}


def optimizer_defaults(kind):
    spec = OPTIMIZERS[kind]
    return dict(betas=spec.betas, eps=spec.eps, weight_decay=spec.weight_decay,
                use_kahan=False, kahan_sum=(False if kind == "optimi_adan" else "auto") if spec.family == "Optimi" else None,
                quantize_state=False, offload_state=False, momentum=0.0, gradient_release=False)


def supports(kind, option):
    spec = OPTIMIZERS.get(kind)
    if spec is None:
        return False
    if option == "betas":
        return bool(spec.betas)
    if option == "eps":
        return spec.eps is not None
    if option in ("use_kahan", "quantize_state", "offload_state"):
        return spec.family == "SDNQ"
    if option in ("kahan_sum", "gradient_release"):
        return spec.family == "Optimi"
    if option == "momentum":
        return kind == "optimi_sgd"
    return True
