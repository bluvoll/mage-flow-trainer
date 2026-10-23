"""Real loop checks plus distributed equivalence against one concatenated batch."""
import os
from types import SimpleNamespace as NS
import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from trainer.training.step_safety import (
    global_sample_scale, scale_gradients, gradient_norm, globally_finite, guard_released_gradient,
)


def test_real_loop_weights_images_and_skips_invalid_updates(tmp_path, monkeypatch):
    from trainer.training import train as train_module
    from trainer.training.train import Trainer
    acc = Accelerator(cpu=True, gradient_accumulation_steps=2)
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    opt = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
    # First group: 3 + 1 images. Second: nonfinite + finite. Last: one-image tail.
    data = [torch.tensor([[1.], [2.], [3.]]), torch.tensor([[10.]]),
            torch.tensor([[float('nan')]]), torch.tensor([[2.]]), torch.tensor([[5.]])]
    loader = DataLoader(data, batch_size=None, collate_fn=lambda x: {'x': x, 'captions': [''] * len(x)})
    model, opt, sched, loader = acc.prepare(model, opt, sched, loader)
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = NS(train=NS(epochs=1, gradient_accumulation_steps=2, nonfinite_patience=3,
        log_every=100, save_every_steps=None, save_every_epochs=None, skip_final_save=True, seed=42),
        optimizer=NS(gradient_release=False, max_grad_norm=0))
    trainer.accelerator = acc
    trainer.transformer, trainer.optimizer, trainer.scheduler, trainer.loader = model, opt, sched, loader
    trainer.sampler = NS(set_epoch=lambda _: None)
    trainer.self_flow_ema = NS(update=lambda: ema_calls.append(1))
    trainer.perceptual = trainer.adversarial = trainer.bar = trainer._eta_anchor = None
    trainer.start_epoch = trainer.resume_skip_batches = trainer.global_step = 0
    trainer.successful_updates = trainer.skipped_updates = trainer.consecutive_nonfinite = 0
    trainer.total_steps = 3
    trainer.progress_mode = 'off'
    trainer._report = trainer._set_phase = trainer._set_rti_budget = trainer._apply_lr_mul = lambda: None
    trainer._all_modules_params = lambda: model.parameters()
    trainer._step = lambda batch: (model(batch['x']).mean(), None)
    # The fork's loop extras: tracker, save/quit signal files, periodic eval/samples.
    trainer.cfg.train.resume_from = None
    trainer.out_dir = tmp_path
    trainer.evaluator = trainer.validation_sampler = None
    trainer._samples_since_log = 0
    trainer._periodic = lambda *_: False
    monkeypatch.setattr(train_module, 'build_tracker', lambda *a, **k: NS(log=lambda *a, **k: None, finish=lambda: None))
    gradients, ema_calls = [], []
    opt.optimizer.register_step_pre_hook(lambda optim, args, kwargs: gradients.append(model.weight.grad.item()))
    trainer.train()
    assert gradients == [4., 5.]
    assert trainer.successful_updates == 2
    assert trainer.skipped_updates == 1
    assert trainer.global_step == 3
    assert len(ema_calls) == 2
    assert opt.optimizer.state[model.weight]['step'].item() == 2
    assert sched.scheduler.last_epoch == 2
    assert torch.isfinite(model.weight).all()
    acc.free_memory()


def test_release_guard_runs_before_update():
    p = torch.nn.Parameter(torch.ones(2))
    updates = []
    p.register_post_accumulate_grad_hook(guard_released_gradient)
    p.register_post_accumulate_grad_hook(lambda _: updates.append(1))
    with pytest.raises(FloatingPointError, match='partial step'):
        (p * float('nan')).sum().backward()
    assert updates == []
    assert torch.equal(p, torch.ones(2))


@pytest.mark.skipif(os.environ.get('MAGEFLOW_GPU_TESTS') != '1' or os.environ.get('WORLD_SIZE') != '2',
                    reason='requires two-rank GPU opt-in')
def test_ddp_uneven_images_accumulation_and_one_rank_nonfinite():
    import torch.distributed as dist
    from contextlib import nullcontext
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl')
    try:
        torch.manual_seed(8)
        layer = torch.nn.Linear(2, 1).to(device)
        reference = torch.nn.Linear(2, 1).to(device)
        reference.load_state_dict(layer.state_dict())
        model = torch.nn.parallel.DistributedDataParallel(layer, device_ids=[rank])
        generator = torch.Generator().manual_seed(90)
        batches = [[torch.randn(n, 2, generator=generator).to(device) for n in sizes]
                   for sizes in ((3, 1), (1, 2))]
        accumulation = 3  # partial tail: only two microbatches actually exist
        count = 0
        for i, x in enumerate(batches[rank]):
            count += len(x)
            with model.no_sync() if i == 0 else nullcontext():
                (model(x).square().mean() * len(x) / accumulation).backward()
        scale_gradients(model.parameters(), global_sample_scale(count, accumulation, device))
        reference(torch.cat([x for rank_batches in batches for x in rank_batches])).square().mean().backward()
        for p, q in zip(layer.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6)
        assert globally_finite(torch.tensor(True, device=device), gradient_norm(model.parameters(), device))
        if rank == 1:
            layer.weight.grad[0, 0] = float('inf')
        assert not globally_finite(torch.tensor(True, device=device), gradient_norm(model.parameters(), device))
        # Also detect a bad loss on just one rank, even if all gradients are finite.
        layer.weight.grad.zero_()
        assert not globally_finite(torch.tensor(rank == 0, device=device), gradient_norm(model.parameters(), device))
    finally:
        dist.destroy_process_group()
