"""Experimental Self-Flow feasibility probe, batch one per GPU.

No production training behavior is changed. Teacher and student share frozen
weights in adapter mode. Full finetuning averages the entire teacher prefix,
including input and timestep projections. EMA storage may be FP32 or BF16.
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
from types import MethodType
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from trainer.modeling.batched import _double_stream_block_forward
from trainer.modeling.mageflow_attention import packed_attention_metadata
from trainer.training.config import load_config, OptimizerConfig
from trainer.training.flow import flow_loss, sample_timesteps
from trainer.training.train import Trainer


from trainer.training.self_flow import AdapterEMA, forward_probe, distributed_probe_forward, fingerprint_indices


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
        if self.accelerator.num_processes > 1:
            # The projector and student forward must both execute inside DDP's
            # forward so its reducer can discover the actual autograd graph.
            self.transformer._probe_original_forward = self.transformer.forward
            self.transformer.forward = MethodType(distributed_probe_forward, self.transformer)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='benchmarks/uint8-controlled/uint8.toml',
                    help='Existing config with cached latents and cached text; experiment settings override training knobs')
    ap.add_argument('--mode', choices=['baseline', 'self_flow'], required=True)
    ap.add_argument('--ema-device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--ema-dtype', choices=['float32', 'bfloat16'], default='float32')
    ap.add_argument('--ema-stochastic-rounding', action='store_true')
    ap.add_argument('--ema-adaln-fp32', action='store_true',
                    help='Keep AdaLN/timestep EMA in FP32 when other EMA weights use BF16')
    ap.add_argument('--ddp-probe', action='store_true', help='Allow experimental multi-process measurement')
    ap.add_argument('--use-config-settings', action='store_true',
                    help='Keep configured adapter/full finetune, optimizer, compile and flow settings; batch remains one')
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--ema-decay', type=float, default=0.99)
    ap.add_argument('--weight', type=float, default=0.8)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if args.steps < 1 or not 0 <= args.ema_decay < 1 or args.weight < 0:
        ap.error('Invalid steps, EMA decay or alignment weight')
    if args.ema_stochastic_rounding and args.ema_dtype != 'bfloat16':
        ap.error('--ema-stochastic-rounding requires --ema-dtype bfloat16')
    torch.set_num_threads(4)
    cfg = load_config(args.config)
    if cfg.self_flow.enabled:
        ap.error('This config enables production Self-Flow; use trainer.training.train or the GUI instead of the probe')
    if cfg.dataset.source != 'latents' or not cfg.train.cache_text_embeddings:
        ap.error('Use cached image latents and Cached Text Encoder for this probe')
    if cfg.curriculum.phases or cfg.optimizer.gradient_release or cfg.preserve.enabled:
        ap.error('Curriculum, gradient release and preservation are not supported by this probe')
    if args.use_config_settings:
        if cfg.adapter.kind not in ('none', 'lycoris_lora') or cfg.rti.enabled or cfg.train.model_family != 'mage_flow':
            ap.error('--use-config-settings requires Mage-Flow full finetuning or LyCORIS without RTI')
        if cfg.adapter.kind == 'none' and cfg.quant.mode == 'frozen':
            ap.error('Full finetuning requires quant.mode=training or none')
        if cfg.adapter.kind == 'none' and cfg.quant.use_quantized_matmul is True:
            ap.error('The full-finetune teacher probe requires quantized matmul disabled')
    else:
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
        distributed = tr.accelerator.num_processes > 1
        if distributed and not args.ddp_probe:
            raise ValueError('This experimental probe is single-process only')
        if distributed:
            destination = destination.with_name(destination.stem + f'.rank{tr.accelerator.process_index}' + destination.suffix)
        result['world_size'] = tr.accelerator.num_processes
        result['rank'] = tr.accelerator.process_index
        model = tr.accelerator.unwrap_model(tr.transformer)
        adapters = [(n, p) for n, p in model.named_parameters() if 'lycoris_adapter.' in n and p.requires_grad]
        result['adapter_parameters'] = sum(p.numel() for _, p in adapters)
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        result['trainable_parameters'] = sum(p.numel() for _, p in trainable)
        result['teacher_prefix_layers'] = 8
        result['student_feature_layer'] = 4
        if cfg.is_lora:
            assert all(not p.requires_grad for n, p in model.named_parameters()
                       if 'lycoris_adapter.' not in n and 'self_flow_projector.' not in n)
        stage = 'EMA initialization'
        ema = AdapterEMA(model, 8, args.ema_device, args.ema_decay,
                         getattr(torch, args.ema_dtype), args.ema_stochastic_rounding,
                         full_finetune=not cfg.is_lora,
                         adaln_fp32=args.ema_adaln_fp32) if args.mode == 'self_flow' else None
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
            data_seed = cfg.train.seed + 10000 + step + tr.accelerator.process_index * 100000
            torch.manual_seed(data_seed)
            torch.cuda.manual_seed_all(data_seed)
            latent = batch['latents'][0].to(tr.accelerator.device, torch.float32)
            context = tr._encode(batch['captions'])
            noise = torch.randn_like(latent)
            t = sample_timesteps(cfg.flow, 2, *latent.shape[-2:], latent.device)
            mask = torch.rand(latent.shape[-2:], device=latent.device) < cfg.flow.dual_timestep_mask_ratio
            target = noise - latent
            dual = cfg.flow.dual_timestep if args.use_config_settings else ema is not None
            ids = mask.long() if dual else torch.zeros_like(mask, dtype=torch.long)
            per_token_t = t[ids][None, None, None]
            noisy = ((1-per_token_t)*latent + per_token_t*noise).to(tr.dtype)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            if step == 0:
                # Compare the experimental uniform-token path to production inference.
                model.eval()
                # Compare eager paths: compiler fusion can change BF16 rounding.
                original_forward, original_compiled = model.block_forward, model.compiled_blocks
                model.block_forward, model.compiled_blocks = _double_stream_block_forward, False
                with torch.no_grad():
                    uniform = ((1-t[0])*latent + t[0]*noise).to(tr.dtype)
                    expected = model(uniform, t[:1], context)[0]
                    actual, _ = forward_probe(model, uniform, t[:1], context)
                    result['uniform_forward_max_error'] = (actual.float()-expected.float()).abs().max().item()
                    torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
                    del expected, actual, uniform
                model.block_forward, model.compiled_blocks = original_forward, original_compiled
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
            if distributed:
                prediction, features = tr.transformer(noisy, t, context, token_ids=ids,
                                                      probe=True, project_features=ema is not None)
            else:
                prediction, features = forward_probe(model, noisy, t, context, token_ids=ids)
            fm = flow_loss(prediction, target)
            alignment = fm.new_zeros(())
            if ema:
                projected = features if distributed else model.self_flow_projector(features)
                alignment = 1-torch.nn.functional.cosine_similarity(
                    projected.float(), teacher.float(), dim=-1, eps=1e-12).mean()
            loss = fm + args.weight*alignment
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            tr.accelerator.backward(loss)
            if not all(torch.isfinite(p.grad).all().item() for _, p in trainable if p.grad is not None):
                raise RuntimeError('Nonfinite training gradients')
            if cfg.optimizer.max_grad_norm > 0:
                tr.accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)
            from trainer.training.distributed import quantized_optimizer_rng
            with quantized_optimizer_rng(tr.accelerator.device, cfg.train.seed, step * 2) if distributed else nullcontext():
                tr.optimizer.step()
            tr.scheduler.step()
            tr.optimizer.zero_grad(set_to_none=True)
            update_start = time.perf_counter()
            if ema:
                with quantized_optimizer_rng(tr.accelerator.device, cfg.train.seed, step * 2 + 1) if distributed else nullcontext():
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
        if distributed:
            stage = 'replica verification'
            from trainer.training.distributed import model_storage
            import torch.distributed as dist
            def fingerprint(values):
                digest = hashlib.sha256()
                for name, value in values:
                    flat = value.detach().reshape(-1)
                    indices = fingerprint_indices(flat.numel(), flat.device)
                    digest.update(name.encode())
                    digest.update(flat[indices].float().cpu().numpy().tobytes())
                return digest.hexdigest()
            signatures = {'student': fingerprint(model_storage(model))}
            if ema:
                signatures['teacher'] = fingerprint((f'{i}.{n}', v) for i,g in enumerate(ema.shadow) for n,v in g.items())
            gathered = [None] * tr.accelerator.num_processes
            dist.all_gather_object(gathered, signatures)
            result['sampled_replica_fingerprints'] = gathered
            if any(s != gathered[0] for s in gathered):
                raise RuntimeError('Sampled student/EMA replica values diverged')
    except Exception as exc:
        result.update(status='oom' if isinstance(exc, torch.OutOfMemoryError) else 'error',
                      stage=stage, error=str(exc))
        traceback.print_exc()
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('steps','config')}, indent=2), flush=True)
    if result['status'] != 'ok':
        raise SystemExit(2)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
