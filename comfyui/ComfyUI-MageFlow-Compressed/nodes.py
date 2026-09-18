# SPDX-License-Identifier: GPL-3.0-or-later
import json

import torch
import folder_paths
import comfy.model_management as mm
import comfy.model_patcher
import comfy.supported_models
import comfy.utils

from .model import CompressedMageFlowBase
from .rti import CompressedRTIMageFlowBase


def load_compressed_model(path, precision="default"):
    state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    if (metadata or {}).get("architecture") != "mageflow-lowrank-modulation-v1":
        raise ValueError("This node requires a compressed Mage-Flow checkpoint exported by mage-flow-trainer.")
    params = json.loads(metadata["model_config"])
    rank = params["modulation_rank"]
    if rank <= 0 or state["modulation_down.weight"].shape[0] != rank:
        raise ValueError("Compressed Mage-Flow modulation rank does not match its checkpoint metadata.")
    factor_keys = ["modulation_down.weight", "modulation_down.bias"]
    for i in range(params["depth"]):
        factor_keys.extend(f"transformer_blocks.{i}.{stream}_mod.1.{part}"
                           for stream in ("img", "txt") for part in ("weight", "bias"))
    if any(state[key].dtype != torch.float32 for key in factor_keys):
        raise ValueError("Compressed modulation factors must be stored in FP32 to preserve calibration accuracy.")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}.get(precision)
    if precision == "default":
        dtype = mm.unet_dtype(model_params=comfy.utils.calculate_parameters(state),
                              supported_dtypes=[torch.bfloat16, torch.float32], weight_dtype=torch.bfloat16)
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Compressed Mage-Flow supports BF16 or FP32 inference; select either explicitly.")
    config = comfy.supported_models.MageFlow({
        "image_model": "mage_flow", "in_channels": params["in_channels"],
        "out_channels": params["out_channels"], "num_layers": params["depth"],
        "attention_head_dim": params["hidden_size"] // params["num_heads"],
        "num_attention_heads": params["num_heads"], "joint_attention_dim": params["context_in_dim"],
        "axes_dims_rope": params["axes_dim"], "modulation_rank": rank,
    })
    load_device, offload_device = mm.get_torch_device(), mm.unet_offload_device()
    config.set_inference_dtype(dtype, dtype, device=load_device)
    model = CompressedMageFlowBase(config, device=offload_device)
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=offload_device)
    model.diffusion_model.load_state_dict(state, strict=True, assign=patcher.is_dynamic())
    patcher.cached_patcher_init = (load_compressed_model, (path, precision))
    return patcher


def load_compressed_rti_model(path, precision="default", keep_fraction=.75):
    state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    if (metadata or {}).get("architecture") != "mageflow-rti-v1":
        raise ValueError("This node requires a combined compressed RTI Mage-Flow checkpoint.")
    params = json.loads(metadata["model_config"])
    required = ("modulation_rank", "rti_size_buckets", "rti_core_start", "rti_core_end")
    if any(key not in params for key in required) or params["modulation_rank"] < 1 or params["rti_size_buckets"] < 1:
        raise ValueError("Checkpoint does not contain both shared-AdaLN and RTI architecture metadata.")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}.get(precision)
    if precision == "default":
        dtype = mm.unet_dtype(model_params=comfy.utils.calculate_parameters(state), supported_dtypes=[torch.bfloat16, torch.float32], weight_dtype=torch.bfloat16)
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Compressed RTI Mage-Flow supports BF16 or FP32 inference.")
    config = comfy.supported_models.MageFlow({"image_model":"mage_flow", "in_channels":params["in_channels"], "out_channels":params["out_channels"], "num_layers":params["depth"], "attention_head_dim":params["hidden_size"]//params["num_heads"], "num_attention_heads":params["num_heads"], "joint_attention_dim":params["context_in_dim"], "axes_dims_rope":params["axes_dim"], "modulation_rank":params["modulation_rank"], "rti_size_buckets":params["rti_size_buckets"], "rti_core_start":params["rti_core_start"], "rti_core_end":params["rti_core_end"]})
    load_device, offload_device = mm.get_torch_device(), mm.unet_offload_device(); config.set_inference_dtype(dtype, dtype, device=load_device)
    model = CompressedRTIMageFlowBase(config, device=offload_device)
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=offload_device)
    model.diffusion_model.load_state_dict(state, strict=True, assign=patcher.is_dynamic())
    patcher.model_options.setdefault("transformer_options", {})["rti_keep_fraction"] = float(keep_fraction)
    patcher.cached_patcher_init = (load_compressed_rti_model, (path, precision, keep_fraction))
    return patcher


class LoadCompressedMageFlow:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_name": (folder_paths.get_filename_list("diffusion_models"),),
            "precision": (["default", "bfloat16", "float32"],),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "model/loaders"
    DESCRIPTION = "Load a Mage-Flow checkpoint with shared low-rank modulation. Keeps calibrated factors in FP32."

    def load_model(self, model_name, precision):
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        return (load_compressed_model(path, precision),)


class LoadCompressedRTIMageFlow:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (folder_paths.get_filename_list("diffusion_models"),), "precision": (["default", "bfloat16", "float32"],), "keep_fraction": ("FLOAT", {"default": .75, "min": .01, "max": 1.0, "step": .01})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "model/loaders"
    DESCRIPTION = "Load a shared-AdaLN Mage-Flow checkpoint fine-tuned with RTI. Batch size 1 and no reference image latents."
    def load_model(self, model_name, precision, keep_fraction):
        return (load_compressed_rti_model(folder_paths.get_full_path_or_raise("diffusion_models", model_name), precision, keep_fraction),)
