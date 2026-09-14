"""DDP initialization and update randomness for SDNQ training tensors."""

from contextlib import contextmanager

import torch
import torch.distributed as dist
from accelerate.utils import DistributedDataParallelKwargs


class QuantizedDDPKwargs(DistributedDataParallelKwargs):
    def to_kwargs(self):
        # Accelerate's kwargs dataclass does not yet expose PyTorch's init_sync.
        # sync_quantized_model replaces that initialization, not gradient reduction.
        return {**super().to_kwargs(), "init_sync": False}


def configure_quantized_ddp():
    # DDPOptimizer partitions compiled graphs at gradient-bucket boundaries. Its
    # subgraphs can return aliases of SDNQ tensor subclasses, which AOTAutograd
    # rejects. Compile complete transformer blocks; DDP still reduces gradients.
    torch._dynamo.config.optimize_ddp = False
    return QuantizedDDPKwargs(find_unused_parameters=True, gradient_as_bucket_view=True)


def model_storage(module):
    """Yield ordinary tensors, never SDNQ wrappers, including quantization metadata."""
    from sdnq.training import SDNQTensor

    for name, value in list(module.named_parameters()) + list(module.named_buffers()):
        if isinstance(value, SDNQTensor):
            names, _ = value.__tensor_flatten__()
            for field in names:
                yield f"{name}.{field}", getattr(value, field)
        else:
            yield name, value


@torch.no_grad()
def sync_quantized_model(module):
    """Verify layouts and broadcast exact stored values without flatten/requantize."""
    from sdnq.training import SDNQTensor

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    values = list(module.named_parameters()) + list(module.named_buffers())
    layout = [
        (
            name,
            tuple(p.shape),
            str(p.dtype),
            p.requires_grad,
            vars(p.sdnq_dequantizer) if isinstance(p, SDNQTensor) else None,
        )
        for name, p in values
    ]
    storage = list(model_storage(module))
    layout += [(name, tuple(p.shape), str(p.dtype)) for name, p in storage]
    layouts = [None] * dist.get_world_size()
    dist.all_gather_object(layouts, layout)
    if any(other != layout for other in layouts):
        raise RuntimeError(
            "SDNQ model layouts differ across DDP ranks; cannot synchronize safely"
        )
    for _, value in storage:
        data = value.detach().contiguous()
        if dist.get_backend() == "nccl" and data.device.type != "cuda":
            data = data.to(torch.cuda.current_device())
        dist.broadcast(data, src=0)
        if data.data_ptr() != value.data_ptr():
            value.copy_(data)


@contextmanager
def quantized_optimizer_rng(device, seed, step):
    """Share stochastic rounding across replicas, preserving per-rank data/noise RNG."""
    device = torch.device(device)
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        update_seed = (seed + (step + 1) * 0x9E3779B97F4A7C15) % (2**63)
        torch.random.default_generator.manual_seed(update_seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(update_seed)
        yield
