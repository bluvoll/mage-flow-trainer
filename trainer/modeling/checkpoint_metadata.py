"""Readable settings and actual optimizer state at checkpoint export time."""

from dataclasses import asdict, is_dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open


def checkpoint_identity(path):
    """Identify the actual source weights once, before training modifies them.

    MD5 is a file identity checksum, not an authenticity guarantee. Directory
    sources use the same single weight filename as the model loader.
    """
    source = Path(path).expanduser()
    weights = source / "diffusion_pytorch_model.safetensors" if source.is_dir() else source
    before = weights.stat()
    digest = hashlib.md5(usedforsecurity=False)
    with weights.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    with safe_open(str(weights), framework='pt', device='cpu') as checkpoint:
        keys = list(checkpoint.keys())
        metadata = checkpoint.metadata() or {}
    for prefix in ('model.diffusion_model.', 'diffusion_model.'):
        if keys and all(k.startswith(prefix) for k in keys):
            keys = [k[len(prefix):] for k in keys]
            break
    compressed = 'modulation_down.weight' in keys
    rti = any(k.startswith('region_interface.') for k in keys)
    kind = ('compressed' if compressed else 'full_mage_flow') + ('_rti' if rti else '')
    after = weights.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError(f'Source checkpoint changed while hashing: {weights}')
    return dict(name=weights.name, path=str(weights.resolve()), md5=digest.hexdigest(),
                size_bytes=after.st_size, model_type=kind, compressed=compressed, rti=rti,
                declared_architecture=metadata.get('architecture'))


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
    source = state.pop('source_checkpoint', None)
    state.update(global_step=step, export_dtype=str(dtype))
    result = {
        "training_metadata_version": "2",
        "training_config": encode_settings(cfg),
        "training_state": encode_settings(state),
        "training_versions": encode_settings(versions),
    }
    if source is not None:
        result['source_checkpoint'] = encode_settings(source)
    return result
