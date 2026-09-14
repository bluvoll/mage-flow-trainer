"""Measure real training-step VRAM, separately from model/text-cache initialization."""

import argparse
import collections
import json
import os
import traceback
from pathlib import Path
import statistics
import time

import torch

from ..training.config import load_config
from ..training.train import Trainer


class MemoryTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.measurements = []
        super().__init__(*args, **kwargs)
        torch.cuda.synchronize()
        self.initialization = self.memory()
        self.parameter_dtypes = dict(
            collections.Counter(
                {
                    str(dtype): sum(
                        p.numel()
                        for p in self.transformer.parameters()
                        if p.requires_grad and p.dtype == dtype
                    )
                    for dtype in {
                        p.dtype
                        for p in self.transformer.parameters()
                        if p.requires_grad
                    }
                }
            )
        )

    @staticmethod
    def memory():
        return dict(
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )

    def _step(self, batch):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.step_start = time.perf_counter()
        self.current_batch = dict(
            batch_size=len(batch["captions"]),
            bucket=list(batch["bucket"]),
            empty_captions=sum(not c for c in batch["captions"]),
            caption_characters=[len(c) for c in batch["captions"]],
        )
        return super()._step(batch)

    def _encode(self, captions):
        start = time.perf_counter()
        out = super()._encode(captions)
        torch.cuda.synchronize()
        self.current_batch["text_seconds"] = time.perf_counter() - start
        self.current_batch["text_tokens"] = out[1].sum(1).tolist()
        self.current_batch["after_text_allocated_bytes"] = torch.cuda.memory_allocated()
        return out

    def _log(self, loss, hf, epoch, t0):
        torch.cuda.synchronize()
        self.measurements.append(
            dict(
                step=self.global_step,
                epoch=epoch,
                seconds=time.perf_counter() - self.step_start,
                loss=float(loss.detach()),
                **self.current_batch,
                **self.memory(),
            )
        )
        super()._log(loss, hf, epoch, t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.train.gradient_accumulation_steps != 1:
        raise ValueError("This per-step benchmark requires accumulation=1")
    if cfg.train.log_every != 1:
        raise ValueError("log_every must be 1")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = dict(
        config=str(Path(args.config).resolve()),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        rank_process=rank,
        world_size=world_size,
        adapter=cfg.adapter.kind,
        algorithm=(
            cfg.adapter.lycoris_algo if cfg.adapter.kind == "lycoris_lora" else None
        ),
        resolution=cfg.dataset.resolution,
        batch_size=cfg.train.batch_size,
        text_encoder_mode=(
            "Cached Text Encoder"
            if cfg.train.cache_text_embeddings
            else "Loaded Text Encoder"
        ),
        status="running",
    )
    trainer = None
    try:
        trainer = MemoryTrainer(cfg, args.config)
        initialization_seconds = time.perf_counter() - start
        trainer.train()
        rows = trainer.measurements
        if not rows:
            raise RuntimeError("No optimizer steps measured")
        if cfg.train.max_steps and len(rows) != cfg.train.max_steps:
            raise RuntimeError(
                f"Expected {cfg.train.max_steps} steps, measured {len(rows)}"
            )
        if any(r["batch_size"] != cfg.train.batch_size for r in rows):
            raise RuntimeError(
                "Benchmark encountered a partial batch; increase dataset repeats"
            )
        result.update(
            status="ok",
            sample_count=len(trainer.dataset),
            cache_text_embeddings=cfg.train.cache_text_embeddings,
            offload_text_encoder=cfg.train.offload_text_encoder,
            quant_dtype=cfg.quant.weights_dtype,
            skip_policy=cfg.quant.skip_policy,
            rank=cfg.adapter.rank,
            trainable_parameter_dtypes=trainer.parameter_dtypes,
            initialization_seconds=initialization_seconds,
            initialization=trainer.initialization,
            training_peak_allocated_gib=max(r["peak_allocated_bytes"] for r in rows)
            / 2**30,
            training_peak_reserved_gib=max(r["peak_reserved_bytes"] for r in rows)
            / 2**30,
            warm_median_seconds=statistics.median(
                [r["seconds"] for r in (rows[2:] or rows)]
            ),
            empty_caption_count=sum(r["empty_captions"] for r in rows),
            empty_caption_steps=sum(r["empty_captions"] > 0 for r in rows),
            steps=rows,
        )
    except Exception as exc:
        result.update(
            status="oom" if isinstance(exc, torch.OutOfMemoryError) else "error",
            error=str(exc),
            elapsed_seconds=time.perf_counter() - start,
            memory=MemoryTrainer.memory(),
            steps=trainer.measurements if trainer is not None else [],
        )
        traceback.print_exc()
    out = Path(args.output)
    if world_size > 1:
        out = out.with_name(f"{out.stem}.rank{rank}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("steps", "initialization")},
            indent=2,
        )
    )
    if result["status"] != "ok":
        raise SystemExit(2)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
