"""Backward update parity, scheduler timing, and resume without model checkpoints."""
import copy
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from trainer.training.config import OptimizerConfig, ScheduleConfig, load_config
from trainer.training.optim import build_optimizer, build_scheduler, enable_gradient_release
from trainer.training.optimizer_specs import OPTIMIZERS
from trainer.tools.probe_gradient_release import ProbeModel


@pytest.mark.parametrize('kind', [k for k in OPTIMIZERS if k.startswith('optimi_')])
def test_release_matches_normal_and_resumes(kind):
    torch.manual_seed(12)
    normal = ProbeModel('cpu', torch.float32)
    released = copy.deepcopy(normal)
    cfg = OptimizerConfig(kind=kind, lr=1e-3, max_grad_norm=0)
    opts = [build_optimizer([{'params': list(m.parameters())}], replace(cfg, gradient_release=flag))
            for m, flag in ((normal, False), (released, True))]
    for opt in opts:
        for group in opt.param_groups:
            group['triton'] = False
    scheds = [build_scheduler(opt, ScheduleConfig(kind='linear', warmup_steps=2), 5) for opt in opts]
    holder = enable_gradient_release(opts[1])
    for step in range(4):
        x = torch.randn(3, 32)
        for i, model in enumerate((normal, released)):
            model(x).square().mean().backward()
            if i == 0:
                opts[i].step()
            else:
                assert all(p.grad is None for p in model.parameters())
            scheds[i].step()
            opts[i].zero_grad(set_to_none=True)
        for p, q in zip(normal.parameters(), released.parameters()):
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
        if step == 1:
            saved = copy.deepcopy(opts[1].state_dict())
            assert all('group' not in state for state in saved['state'].values())
            scheduler_state = copy.deepcopy(scheds[1].state_dict())
            from optimi import remove_gradient_release
            remove_gradient_release(holder)
            opts[1] = build_optimizer([{'params': list(released.parameters())}], replace(cfg, gradient_release=True))
            scheds[1] = build_scheduler(opts[1], ScheduleConfig(kind='linear', warmup_steps=2), 5)
            opts[1].load_state_dict(saved)
            scheds[1].load_state_dict(scheduler_state)
            for group in opts[1].param_groups:
                for p in group['params']:
                    assert opts[1].state[p]['group'] is group
            holder = enable_gradient_release(opts[1])


def test_release_guards():
    with pytest.raises(ValueError, match='Optimi'):
        OptimizerConfig(kind='adamw', gradient_release=True, max_grad_norm=0)
    with pytest.raises(ValueError, match='clipping'):
        OptimizerConfig(kind='optimi_adamw', gradient_release=True)
    with pytest.raises(ValueError, match='single-GPU'):
        enable_gradient_release(None, num_processes=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'bad.toml'
        path.write_text('[dataset]\npath="/images"\n[train]\ngradient_accumulation_steps=2\n'
                        '[optimizer]\nkind="optimi_adamw"\ngradient_release=true\nmax_grad_norm=0\n')
        with pytest.raises(ValueError, match='gradient_accumulation_steps=1'):
            load_config(path)


def test_accelerate_scheduler_and_parameter_steps():
    from accelerate import Accelerator
    from torch.utils.data import DataLoader, TensorDataset
    acc = Accelerator(cpu=True, mixed_precision='no', gradient_accumulation_steps=1)
    model = ProbeModel('cpu', torch.float32)
    raw = build_optimizer([{'params': list(model.parameters())}], OptimizerConfig(
        kind='optimi_adamw', lr=1e-3, gradient_release=True, max_grad_norm=0))
    for group in raw.param_groups:
        group['triton'] = False
    scheduler = build_scheduler(raw, ScheduleConfig(kind='linear'), 3)
    loader = DataLoader(TensorDataset(torch.randn(3, 32)), batch_size=1)
    model, opt, scheduler, loader = acc.prepare(model, raw, scheduler, loader)
    enable_gradient_release(raw)
    for index, (x,) in enumerate(loader):
        with acc.accumulate(model):
            # A curriculum multiplier must be applied before the hooks run.
            for group, lr in zip(opt.param_groups, scheduler.get_last_lr()):
                group['lr'] = lr * 0.5
            acc.backward(model(x).square().mean())
            scheduler.step()
            opt.zero_grad(set_to_none=True)
        for param in model.parameters():
            assert param.grad is None
            assert raw.state[param]['step'].item() == index + 1
            assert raw.state[param]['group'] is raw.param_groups[0]
    assert scheduler.get_last_lr() == [0.0]
