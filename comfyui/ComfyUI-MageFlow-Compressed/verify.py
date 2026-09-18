"""Run with ComfyUI's Python: verify.py COMFY_DIRECTORY COMPRESSED_CHECKPOINT."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])

import torch
import comfy.model_detection
import comfy.model_management as mm
import comfy.sample
import comfy.sd
import comfy.supported_models

before = (comfy.sd.load_diffusion_model, comfy.model_detection.model_config_from_unet,
          tuple(comfy.supported_models.models))
directory = Path(__file__).parent
spec = importlib.util.spec_from_file_location("compressed_plugin", directory / "__init__.py",
                                              submodule_search_locations=[str(directory)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
assert before == (comfy.sd.load_diffusion_model, comfy.model_detection.model_config_from_unet,
                  tuple(comfy.supported_models.models))
from compressed_plugin.nodes import load_compressed_model

with torch.inference_mode():
    patcher = load_compressed_model(sys.argv[2], "bfloat16")
    model = patcher.model.diffusion_model
    factors = {name: tensor.detach().cpu().clone() for name, tensor in model.named_parameters()
               if name in ("modulation_down.weight", "transformer_blocks.0.img_mod.1.weight")}
    model.to(dtype=torch.bfloat16)
    mm.load_models_gpu([patcher])
    torch.manual_seed(2026)
    latent = torch.zeros(1, 128, 16, 16)
    context = torch.randn(1, 16, 2560)
    result = comfy.sample.sample(patcher, comfy.sample.prepare_noise(latent, 2026), 4, 4.,
                                 "euler", "simple", [[context, {}]],
                                 [[torch.zeros_like(context), {}]], latent, seed=2026)
    assert result.shape == latent.shape and torch.isfinite(result).all()
    mm.unload_all_models()
    for name, reference in factors.items():
        actual = model.get_parameter(name)
        assert actual.dtype == torch.float32 and torch.equal(actual.cpu(), reference), name
print("PASS: strict load, sampler, FP32 preservation, unchanged built-in loaders/registry")
