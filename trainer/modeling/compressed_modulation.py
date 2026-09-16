"""Experimental trainable, shared-input low-rank Mage-Flow modulation."""

import torch
from torch import nn


class ModulationLinear(nn.Linear):
    """Preserve the selected modulation precision across trunk dtype changes."""

    parameter_dtype = torch.float32

    def _apply(self, fn, recurse=True):
        def preserve(tensor):
            if not tensor.is_floating_point():
                return fn(tensor)
            # Probe only the destination device. Casting to BF16 and back would
            # irreversibly round the saved factors before restoring their dtype.
            destination = fn(tensor.new_empty(0)).device
            return tensor.to(device=destination, dtype=self.parameter_dtype)

        return super()._apply(preserve, recurse=recurse)


def set_modulation_dtype(model, dtype):
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Compressed AdaLN dtype must be float32 or bfloat16")
    if not model.params.modulation_rank:
        return
    for module in model.modules():
        if isinstance(module, ModulationLinear):
            module.parameter_dtype = dtype
            module.to(dtype=dtype)


def compressed_parameter(name):
    return name.startswith("modulation_down.") or (
        name.startswith("transformer_blocks.")
        and (".img_mod." in name or ".txt_mod." in name)
    )


@torch.no_grad()
def initialize_compression(model, basis, mean, calculation_dtype=torch.float32):
    """Convert a dense model in place using calibrated timestep-feature PCA.

    basis: [hidden_size, rank], mean: [hidden_size], both before block projection.
    calculation_dtype controls offline factor construction; stored factors are FP32.
    Attention, MLPs, timestep embedder and final modulation remain unchanged.
    """
    if model.params.modulation_rank:
        raise ValueError("Model already has compressed modulation")
    if calculation_dtype not in (torch.float32, torch.float64):
        raise ValueError("calculation_dtype must be float32 or float64")
    rank = basis.shape[1]
    if basis.shape[0] != model.inner_dim or not 0 < rank <= model.inner_dim:
        raise ValueError("Invalid modulation basis")
    device = next(model.parameters()).device
    basis, mean = (
        basis.to(device, calculation_dtype),
        mean.to(device, calculation_dtype),
    )
    down = ModulationLinear(model.inner_dim, rank, device=device, dtype=torch.float32)
    down.weight.copy_(basis.T)
    down.bias.copy_(-mean @ basis)
    for block in model.transformer_blocks:
        for name in ("img_mod", "txt_mod"):
            old = getattr(block, name)[1]
            head = ModulationLinear(
                rank, 6 * model.inner_dim, device=device, dtype=torch.float32
            )
            weight = old.weight.to(calculation_dtype)
            head.weight.copy_(weight @ basis)
            head.bias.copy_(old.bias.to(calculation_dtype) + weight @ mean)
            setattr(block, name, nn.Sequential(nn.Identity(), head))
    model.modulation_down = down
    model.params.modulation_rank = rank
    return model
