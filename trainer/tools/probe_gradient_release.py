"""Isolated Optimi backward-release diagnostic; never loads a training checkpoint.

Single GPU: CUDA_VISIBLE_DEVICES=1 python -m trainer.tools.probe_gradient_release --device cuda --compile
DDP diagnostic (deliberately bypasses the trainer's guard):
    torchrun --standalone --nproc_per_node=2 -m trainer.tools.probe_gradient_release --ddp
"""
import argparse
import copy
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from trainer.training.config import OptimizerConfig
from trainer.training.optim import build_optimizer, enable_gradient_release


class ProbeModel(torch.nn.Module):
    """Shared FP32 projection feeding three checkpointed BF16 blocks on CUDA."""
    def __init__(self, device, dtype):
        super().__init__()
        self.shared = torch.nn.Linear(32, 32, device=device, dtype=torch.float32)
        self.blocks = torch.nn.ModuleList([
            torch.nn.Linear(32, 32, device=device, dtype=dtype) for _ in range(3)])

    def forward(self, x):
        condition = self.shared(x.float()).to(x.dtype)
        for block in self.blocks:
            x = checkpoint(block, x + condition, use_reentrant=False).tanh()
        return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--optimizer', default='optimi_adamw')
    args = parser.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', 0)) if args.ddp else 0
    device = torch.device('cuda', rank) if args.device == 'cuda' else torch.device('cpu')
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if args.ddp:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(123)
        reference = ProbeModel(device, dtype)
        released = copy.deepcopy(reference)
        if args.compile:
            for model in (reference, released):
                for block in model.blocks:
                    block.forward = torch.compile(block.forward, fullgraph=True)
        if args.ddp:
            ddp_args = dict(find_unused_parameters=True)
            if device.type == 'cuda':
                ddp_args['device_ids'] = [rank]
            reference = torch.nn.parallel.DistributedDataParallel(reference, **ddp_args)
            released = torch.nn.parallel.DistributedDataParallel(released, **ddp_args)
        opts = []
        for model, release in ((reference, False), (released, True)):
            cfg = OptimizerConfig(kind=args.optimizer, lr=5e-3, max_grad_norm=0,
                                  gradient_release=release)
            opt = build_optimizer([{'params': list(model.parameters())}], cfg)
            if device.type == 'cpu':
                for group in opt.param_groups:
                    group['triton'] = False
            opts.append(opt)
        if args.ddp:
            # Intentionally unsafe, only on this disposable model: measure actual rank drift.
            from optimi import prepare_for_gradient_release
            prepare_for_gradient_release(released, opts[1])
        else:
            enable_gradient_release(opts[1])
        rows = []
        for step in range(args.steps):
            torch.manual_seed(1000 + step + rank * 100)
            x = torch.randn(4, 32, device=device, dtype=dtype)
            losses = []
            for i, model in enumerate((reference, released)):
                loss = model(x).float().square().mean()
                loss.backward()
                if i == 0:
                    opts[i].step()
                    opts[i].zero_grad(set_to_none=True)
                losses.append(loss.item())
            vectors = [torch.cat([p.detach().float().flatten() for p in model.parameters()])
                       for model in (reference, released)]
            spreads = []
            for vector in vectors:
                if args.ddp:
                    root = vector.clone()
                    dist.broadcast(root, 0)
                    error = (root - vector).abs().max()
                    dist.all_reduce(error, op=dist.ReduceOp.MAX)
                    spreads.append(error.item())
                else:
                    spreads.append(0.)
            error = (vectors[0] - vectors[1]).abs().max()
            if args.ddp:
                dist.all_reduce(error, op=dist.ReduceOp.MAX)
            rows.append(dict(step=step + 1, reference_loss=losses[0], release_loss=losses[1],
                             reference_rank_drift=spreads[0], release_rank_drift=spreads[1],
                             max_parameter_difference=error.item(),
                             remaining_gradients=sum(p.grad is not None for p in released.parameters())))
        if rank == 0:
            print(json.dumps(dict(device=str(device), ddp=args.ddp, compiled=args.compile,
                                  optimizer=args.optimizer, steps=rows), indent=2), flush=True)
    finally:
        if args.ddp:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
