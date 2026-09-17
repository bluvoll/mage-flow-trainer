"""Experimental single-GPU, batch-one Self-Flow/LoKr feasibility probe.

No production training behavior is changed. Teacher and student share frozen
weights; only the teacher prefix's adapter parameters have FP32 EMA copies.
CPU EMA streams one block at a time. This is not a resumable training command.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from trainer.modeling.batched import _double_stream_block_forward
from trainer.modeling.mageflow_attention import packed_attention_metadata
from trainer.training.config import load_config, OptimizerConfig
from trainer.training.flow import flow_loss, sample_timesteps
from trainer.training.train import Trainer


class BlockCall(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args):
        return _double_stream_block_forward(self.block, *args)


class AdapterEMA:
    def __init__(self, model, depth, device, decay):
        self.decay = decay
        self.shadow = []
        self.params = []
        for block in model.transformer_blocks[:depth]:
            params = {n: p for n, p in block.named_parameters() if p.requires_grad}
            if not params or any('lycoris_adapter.' not in n for n in params):
                raise ValueError('Teacher EMA must contain only adapter parameters')
            self.params.append(params)
            self.shadow.append({n: p.detach().to(device=device, dtype=torch.float32, copy=True)
                                for n, p in params.items()})

    @property
    def nbytes(self):
        return sum(v.numel() * v.element_size() for group in self.shadow for v in group.values())

    @torch.no_grad()
    def forward_block(self, index, block, *args):
        # Cast only this block, rather than allocating a BF16 copy of all EMA weights.
        state = {'block.' + n: v.to(device=self.params[index][n].device,
                                  dtype=self.params[index][n].dtype)
                 for n, v in self.shadow[index].items()}
        return torch.func.functional_call(BlockCall(block), state, args, strict=False)

    @torch.no_grad()
    def update(self):
        for params, shadow in zip(self.params, self.shadow):
            for name, value in shadow.items():
                current = params[name].detach().to(device=value.device, dtype=value.dtype)
                value.lerp_(current, 1 - self.decay)


def forward_probe(model, image, times, context, *, token_ids=None,
                  capture_layer=4, stop_at=None, ema=None):
    """Batch-one native token path; image IDs select from a two-timestep table.

    Text retains the primary timestep. With uniform IDs this reduces to the
    ordinary model forward. Teacher evaluation stops at its feature layer.
    """
    if image.shape[0] != 1 or image.shape[2] != 1:
        raise ValueError('Probe supports one image per batch')
    _, _, _, h, w = image.shape
    text, text_mask = context
    img = model.img_in(image.squeeze(2).flatten(2).transpose(1, 2))
    txt = model.txt_in(model.txt_norm(text))
    temb = model.time_text_embed(times.to(img.dtype), img)
    block_temb = model.block_condition(temb)
    freqs = model.pos_embed([(1, h, w)], device=img.device)
    if model.compiled_blocks:
        freqs = torch.view_as_real(freqs)
    mask = torch.cat((text_mask.bool(), torch.ones(1, h*w, device=img.device,
                                                 dtype=torch.bool)), dim=1)[:, None, None]
    metadata = () if model.attention_backend == 'sdpa' else packed_attention_metadata(mask)
    image_ids = (torch.zeros(h*w, device=img.device, dtype=torch.long)
                 if token_ids is None else token_ids.flatten().long())
    text_ids = torch.zeros(text.shape[1], device=img.device, dtype=torch.long)
    captured = None
    for i, block in enumerate(model.transformer_blocks):
        args = (img, txt, block_temb, freqs, mask, model.num_attention_heads,
                model.attention_backend, metadata, False, (image_ids, text_ids))
        if ema is not None:
            txt, img = ema.forward_block(i, block, *args)
        elif model.training and i in model.checkpoint_blocks:
            txt, img = checkpoint(model.block_forward, block, *args, use_reentrant=False)
        else:
            txt, img = model.block_forward(block, *args)
        if i + 1 == capture_layer:
            captured = img
        if stop_at == i + 1:
            return None, img
    scale, shift = model.norm_out.linear(model.norm_out.silu(temb).to(img.dtype)).chunk(2, -1)
    img = model.norm_out.norm(img) * (1 + scale[image_ids][None]) + shift[image_ids][None]
    output = model.proj_out(img).transpose(1, 2).reshape(1, model.out_channels, 1, h, w)
    return output, captured


class ProbeTrainer(Trainer):
    def _build_model(self):
        super()._build_model()
        d = self.transformer.inner_dim
        # Identical initialization/RNG consumption for baseline and Self-Flow.
        self.transformer.self_flow_projector = nn.Sequential(
            nn.Linear(d, 2*d), nn.SiLU(), nn.Linear(2*d, d)
        ).to(device=self.accelerator.device, dtype=self.dtype)
        self.groups.append(dict(params=list(self.transformer.self_flow_projector.parameters()),
                                lr=self.cfg.optimizer.lr, weight_decay=0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='benchmarks/uint8-controlled/uint8.toml',
                    help='Existing config with cached latents and cached text; experiment settings override training knobs')
    ap.add_argument('--mode', choices=['baseline', 'self_flow'], required=True)
    ap.add_argument('--ema-device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--ema-decay', type=float, default=0.99)
    ap.add_argument('--weight', type=float, default=0.8)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if args.steps < 1 or not 0 <= args.ema_decay < 1 or args.weight < 0:
        ap.error('Invalid steps, EMA decay or alignment weight')
    torch.set_num_threads(4)
    cfg = load_config(args.config)
    if cfg.dataset.source != 'latents' or not cfg.train.cache_text_embeddings:
        ap.error('Use cached image latents and Cached Text Encoder for this probe')
    if cfg.curriculum.phases or cfg.optimizer.gradient_release or cfg.preserve.enabled:
        ap.error('Curriculum, gradient release and preservation are not supported by this probe')
    cfg.adapter.kind = 'lycoris_lora'
    cfg.adapter.lycoris_algo = 'lokr'
    cfg.adapter.rank = cfg.adapter.alpha = 10000
    cfg.adapter.lokr_factor = 1
    cfg.adapter.lycoris_bypass = True
    cfg.adapter.dtype = 'bfloat16'
    cfg.quant.mode = 'frozen'
    cfg.optimizer = OptimizerConfig(kind='adafactor', lr=1e-4, norm_mode='rms_clip',
                                    max_grad_norm=0.0, quantize_state=True,
                                    use_first_moment=False, use_kahan=False)
    cfg.schedule.kind = 'constant'
    cfg.schedule.warmup_steps = 0
    cfg.train.compile = None
    cfg.train.gradient_checkpointing = True
    cfg.train.batch_size = cfg.train.gradient_accumulation_steps = 1
    cfg.train.pack_resolutions = True
    cfg.train.attention_backend = 'torch_varlen'
    cfg.train.max_steps = args.steps
    cfg.train.output_dir = 'benchmarks/self-flow-lokr/output'
    cfg.train.run_name = args.mode + '-' + args.ema_device
    resolved_config = json.loads(json.dumps(asdict(cfg), default=lambda v: sorted(v) if isinstance(v, set) else str(v)))
    result = dict(status='running', experiment=vars(args), config=resolved_config, steps=[])
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = 'initialization'
    try:
        tr = ProbeTrainer(cfg)
        if tr.accelerator.num_processes != 1:
            raise ValueError('This experimental probe is single-process only')
        model = tr.transformer
        adapters = [(n, p) for n, p in model.named_parameters() if 'lycoris_adapter.' in n and p.requires_grad]
        result['adapter_parameters'] = sum(p.numel() for _, p in adapters)
        result['teacher_prefix_layers'] = 8
        result['student_feature_layer'] = 4
        assert all(not p.requires_grad for n, p in model.named_parameters()
                   if 'lycoris_adapter.' not in n and 'self_flow_projector.' not in n)
        stage = 'EMA initialization'
        ema = AdapterEMA(model, 8, args.ema_device, args.ema_decay) if args.mode == 'self_flow' else None
        result['ema_gib'] = ema.nbytes / 2**30 if ema else 0
        tr.sampler.set_epoch(0)
        iterator = iter(tr.loader)
        for step in range(args.steps):
            stage = f'step {step+1}'
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(tr.loader)
                batch = next(iterator)
            torch.manual_seed(cfg.train.seed + 10000 + step)
            torch.cuda.manual_seed_all(cfg.train.seed + 10000 + step)
            latent = batch['latents'][0].to(tr.accelerator.device, torch.float32)
            context = tr._encode(batch['captions'])
            noise = torch.randn_like(latent)
            t = sample_timesteps(cfg.flow, 2, *latent.shape[-2:], latent.device)
            mask = torch.rand(latent.shape[-2:], device=latent.device) < .25
            target = noise - latent
            ids = mask.long() if ema else torch.zeros_like(mask, dtype=torch.long)
            per_token_t = t[ids][None, None, None]
            noisy = ((1-per_token_t)*latent + per_token_t*noise).to(tr.dtype)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            if step == 0:
                # Compare the experimental uniform-token path to production inference.
                model.eval()
                with torch.no_grad():
                    uniform = ((1-t[0])*latent + t[0]*noise).to(tr.dtype)
                    expected = model(uniform, t[:1], context)[0]
                    actual, _ = forward_probe(model, uniform, t[:1], context)
                    result['uniform_forward_max_error'] = (actual.float()-expected.float()).abs().max().item()
                    torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
                    del expected, actual, uniform
            teacher_seconds = 0.0
            if ema:
                model.eval()
                teacher_start = time.perf_counter()
                with torch.no_grad():
                    clean_t = t.min().reshape(1)  # Mage-Flow: t=0 clean, t=1 noise.
                    teacher_image = ((1-clean_t)*latent + clean_t*noise).to(tr.dtype)
                    _, teacher = forward_probe(model, teacher_image, clean_t, context,
                                               stop_at=8, ema=ema)
                    del teacher_image
                torch.cuda.synchronize()
                teacher_seconds = time.perf_counter()-teacher_start
            model.train()
            prediction, features = forward_probe(model, noisy, t, context, token_ids=ids)
            fm = flow_loss(prediction, target)
            alignment = fm.new_zeros(())
            if ema:
                projected = model.self_flow_projector(features)
                alignment = 1-torch.nn.functional.cosine_similarity(
                    projected.float(), teacher.float(), dim=-1, eps=1e-12).mean()
            loss = fm + args.weight*alignment
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            tr.accelerator.backward(loss)
            if not all(torch.isfinite(p.grad).all().item() for _, p in adapters if p.grad is not None):
                raise RuntimeError('Nonfinite adapter gradients')
            tr.optimizer.step()
            tr.scheduler.step()
            tr.optimizer.zero_grad(set_to_none=True)
            update_start = time.perf_counter()
            if ema:
                ema.update()
            torch.cuda.synchronize()
            row = dict(step=step+1, loss=loss.item(), flow_loss=fm.item(), alignment_loss=alignment.item(),
                       seconds=time.perf_counter()-start, teacher_seconds=teacher_seconds,
                       ema_update_and_optimizer_sync_seconds=time.perf_counter()-update_start,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                       paths=[str(p) for p in batch['paths']], timesteps=t.tolist(),
                       caption_sha256=[hashlib.sha256(c.encode()).hexdigest() for c in batch['captions']])
            result['steps'].append(row)
            destination.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(row), flush=True)
            del prediction, features, fm, alignment, loss
            if ema:
                del teacher, projected
        result['status'] = 'ok'
        result['warm_median_seconds'] = statistics.median(r['seconds'] for r in result['steps'][2:] or result['steps'])
        result['peak_allocated_gib'] = max(r['peak_allocated_gib'] for r in result['steps'])
    except Exception as exc:
        result.update(status='oom' if isinstance(exc, torch.OutOfMemoryError) else 'error',
                      stage=stage, error=str(exc))
        traceback.print_exc()
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('steps','config')}, indent=2), flush=True)
    if result['status'] != 'ok':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
