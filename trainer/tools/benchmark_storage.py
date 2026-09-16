"""Controlled SDNQ storage comparison, using fixed per-step randomness and cached text."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from trainer.tools.benchmark_lora_memory import MemoryTrainer
from trainer.training.config import load_config


class StorageTrainer(MemoryTrainer):
    def _quantize(self):
        # Audit representative rows across all blocks without retaining a second model.
        candidates = [(n, p) for n, p in self.transformer.named_parameters()
                      if p.ndim == 2 and 'transformer_blocks.' in n
                      and ('attn' in n or 'mlp' in n)]
        selected = candidates
        references = {n: p[:16].detach().cpu().float().clone() for n, p in selected}
        super()._quantize()
        params = dict(self.transformer.named_parameters())
        errors = []
        for name, reference in references.items():
            param = params[name]
            if not hasattr(param, 'sdnq_dequantizer'):
                continue
            restored = param.dequantize(dtype=torch.float32)[:16].detach().cpu()
            error = restored - reference
            errors.append(dict(name=name, sampled_elements=reference.numel(),
                               mse=error.square().mean().item(),
                               relative_l2=(error.norm()/reference.norm()).item(),
                               max_abs=error.abs().max().item()))
        self.weight_audit = errors
        self.storage_audit = dict(weight_bytes=0, scale_bytes=0, offset_bytes=0, quantized_parameters=0)
        for param in params.values():
            if hasattr(param, 'sdnq_dequantizer'):
                self.storage_audit['quantized_parameters'] += param.numel()
                for attr, key in [('weight','weight_bytes'),('scale','scale_bytes'),('zero_point','offset_bytes')]:
                    tensor = getattr(param, attr)
                    if tensor is not None:
                        self.storage_audit[key] += tensor.numel() * tensor.element_size()
        torch.cuda.empty_cache()

    def _step(self, batch):
        # Quantizer initialization can consume different RNG streams. Explicitly reset
        # before noise/timestep sampling so each A/B training example remains matched.
        seed = self.cfg.train.seed + 10000 + self.global_step
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        loss = super()._step(batch)
        self.current_batch['input_seed'] = seed
        self.current_batch['paths'] = [str(p) for p in batch['paths']]
        self.current_batch['caption_hashes'] = [hashlib.sha256(c.encode()).hexdigest() for c in batch['captions']]
        return loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.train.gradient_accumulation_steps != 1 or cfg.train.log_every != 1:
        raise ValueError('Requires accumulation=1 and log_every=1')
    if cfg.quant.use_quantized_matmul is not False:
        raise ValueError('This benchmark is for storage-only quantization')
    trainer = StorageTrainer(cfg, None if args.audit_only else args.config)
    if args.audit_only:
        Path(args.output).write_text(json.dumps(dict(
            weight_audit=trainer.weight_audit, storage_audit=trainer.storage_audit), indent=2)+'\n')
        return
    trainer.train()
    rows = trainer.measurements
    assert len(rows) == cfg.train.max_steps
    result = dict(config=args.config, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  steps=rows, weight_audit=trainer.weight_audit, storage_audit=trainer.storage_audit,
                  sample_count=len(trainer.dataset), parameter_dtypes=trainer.parameter_dtypes,
                  peak_allocated_gib=max(r['peak_allocated_bytes'] for r in rows)/2**30,
                  peak_reserved_gib=max(r['peak_reserved_bytes'] for r in rows)/2**30,
                  warm_median_seconds=statistics.median(r['seconds'] for r in rows[10:]),
                  mean_loss=statistics.mean(r['loss'] for r in rows),
                  first10_loss=statistics.mean(r['loss'] for r in rows[:10]),
                  last10_loss=statistics.mean(r['loss'] for r in rows[-10:]))
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['steps','weight_audit']},indent=2))


if __name__ == '__main__':
    main()
