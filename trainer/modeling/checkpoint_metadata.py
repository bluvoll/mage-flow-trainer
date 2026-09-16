"""Readable settings and actual optimizer state at checkpoint export time."""

from dataclasses import asdict, is_dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, SimpleNamespace):
        return vars(value)
    if isinstance(value, (Path, torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return value.detach().item()
    raise TypeError(f"Cannot encode {type(value).__name__} as checkpoint settings")


def encode_settings(value):
    return json.dumps(value, default=_json_default, indent=2, ensure_ascii=False)


def optimizer_snapshot(optimizer):
    # Accelerate wraps the optimizer; record the actual implementation.
    while hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    groups = []
    for group in optimizer.param_groups:
        settings = {}
        omitted = []
        for key, value in group.items():
            if key == "params":
                continue
            try:
                settings[key] = json.loads(encode_settings(value))
            except (TypeError, ValueError):
                omitted.append(key)
        groups.append({"settings": settings,
                       "parameter_count": sum(p.numel() for p in group["params"]),
                       "omitted_fields": omitted})
    cls = type(optimizer)
    return {"optimizer_class": f"{cls.__module__}.{cls.__qualname__}",
            "optimizer_groups": groups}


def training_metadata(cfg, step, dtype, runtime=None):
    versions = {"torch": str(torch.__version__)}
    for package in ("sdnq", "torch-optimi", "lycoris-lora", "peft", "accelerate"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    state = dict(runtime or {})
    state.update(global_step=step, export_dtype=str(dtype))
    return {
        "training_metadata_version": "1",
        "training_config": encode_settings(cfg),
        "training_state": encode_settings(state),
        "training_versions": encode_settings(versions),
    }
