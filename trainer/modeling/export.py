"""Native transformer weights and PEFT adapters, without model-key conversion."""

from dataclasses import asdict
import json
from pathlib import Path
import torch
from safetensors.torch import save_file


def export_checkpoint(model, dest, stem, cfg, dtype, step, *, runtime=None):
    dest = Path(dest)
    metadata = {"step": str(step), "run": cfg.train.run_name, "model": "mage_flow"}
    from .checkpoint_metadata import training_metadata

    metadata.update(training_metadata(cfg, step, dtype, runtime))
    if model.params.modulation_rank:
        metadata.update(
            architecture="mageflow-lowrank-modulation-v1",
            experimental="true",
            modulation_rank=str(model.params.modulation_rank),
        )
    if cfg.is_lora and hasattr(model, "_lycoris_config"):
        from ..training.lycoris import lycoris_state_dict

        state = {k: v.detach().to("cpu", torch.float32 if k.endswith(".alpha") else dtype).contiguous()
                 for k, v in lycoris_state_dict(model).items()}
        metadata["adapter_config"] = json.dumps(model._lycoris_config)
        metadata["adapter_format"] = "lycoris_lora_native"
        (dest / f"{stem}.json").write_text(metadata["adapter_config"])
        out = dest / f"{stem}.safetensors"
    elif cfg.is_lora:
        from peft import get_peft_model_state_dict

        state = get_peft_model_state_dict(model)
        state = {
            "diffusion_model." + k: v.detach().to("cpu", dtype).contiguous()
            for k, v in state.items()
        }
        adapter_cfg = model.peft_config["default"]
        # Store a sidecar for PEFT and embed the same config so a renamed single file retains scale.
        config = adapter_cfg.to_dict()

        def serializable(x):
            if isinstance(x, set):
                return sorted(x)
            if hasattr(x, "value"):
                return x.value
            raise TypeError(type(x).__name__)

        alpha = float(config.get("lora_alpha", config.get("alpha", 1)))
        targets = {k.split(".lora_")[0] for k in state if ".lora_A." in k}
        targets |= {k.split(".lokr_")[0] for k in state if ".lokr_" in k}
        for target in targets:
            state[target + ".alpha"] = torch.tensor(alpha, dtype=torch.float32)
        metadata["adapter_config"] = json.dumps(config, default=serializable)
        (dest / f"{stem}.json").write_text(metadata["adapter_config"])
        out = dest / f"{stem}.safetensors"
    else:
        state = model.state_dict()
        from .compressed_modulation import compressed_parameter

        keep_fp32 = {
            k: v.detach().to("cpu", torch.float32).contiguous()
            for k, v in state.items()
            if model.params.modulation_rank and compressed_parameter(k)
        }
        if cfg.quant.mode == "training":
            from ..training.quant import dequantize_state_dict

            state = dequantize_state_dict(state, dtype)
        state = {k: v.detach().to("cpu", dtype).contiguous() for k, v in state.items()}
        state.update(keep_fp32)
        config = asdict(model.params)
        metadata["model_config"] = json.dumps(config)
        if cfg.train.save_native:
            out = dest / f"{stem}.safetensors"
            (dest / f"{stem}.json").write_text(json.dumps(config, indent=2))
        else:
            folder = dest / "transformer"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "config.json").write_text(json.dumps(config, indent=2))
            out = folder / "diffusion_pytorch_model.safetensors"
    save_file(state, str(out), metadata=metadata)
    return len(state)
