"""Experimental two-device teacher/student flow distillation of compressed AdaLN.

Reads existing latent and caption-variation caches; only compressed factors train.
This is model parallel distillation (not DDP). Heldout images never enter training.
"""

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from trainer.data.cache import load_cached_latent
from trainer.data.caption_variations import CaptionVariationCache, encoder_fingerprint
from trainer.data.dataset import MageFlowDataset
from trainer.modeling.compressed_modulation import compressed_parameter
from trainer.modeling.loader import load_components
from trainer.training.config import load_config


def sources(paths):
    groups = []
    for path in paths:
        cfg = load_config(path)
        if not cfg.train.caption_variations or not cfg.train.caption_cache_path:
            raise ValueError(
                "Each data config must have an existing variation cache path"
            )
        ds = MageFlowDataset(cfg.dataset)
        cache = CaptionVariationCache(
            cfg.train.caption_cache_path,
            encoder_fingerprint(cfg, cfg.train.caption_cache_path),
        )
        cache.prepare(
            ds.entries,
            cfg.dataset.caption,
            cfg.train.seed,
            cfg.train.caption_variations,
            write=False,
        )
        train = []
        valid = []
        seen = set()
        for i, e in enumerate(ds.entries):
            identity = str(e.path.resolve())
            if identity in seen:
                continue
            seen.add(identity)
            caption = cache.caption(i, 0)
            # Fail before model loading if any source lacks cached embeddings.
            cache.get([caption], "cpu", torch.bfloat16)
            item = (e.path, i, cache)
            (
                valid
                if int(hashlib.sha256(identity.encode()).hexdigest(), 16) % 5 == 0
                else train
            ).append(item)
        if not train or not valid:
            raise ValueError(f"{path}: need both training and validation images")
        groups.append((train, valid))
        print(
            "dataset", path, "train", len(train), "validation", len(valid), flush=True
        )
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mage-flow")
    ap.add_argument("--teacher-transformer")
    ap.add_argument("--student", required=True)
    ap.add_argument("--data-config", action="append", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-7)
    ap.add_argument("--teacher-device", default="cuda:0")
    ap.add_argument("--student-device", default="cuda:1")
    ap.add_argument("--validate-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--index-only", action="store_true")
    args = ap.parse_args()
    if args.steps < 0 or args.lr <= 0 or args.validate_every < 1:
        ap.error("steps must be nonnegative; lr and validate-every must be positive")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    groups = sources(args.data_config)
    if args.index_only:
        return
    start = time.time()
    td = torch.device(args.teacher_device)
    sd = torch.device(args.student_device)
    dtype = torch.bfloat16
    teacher = (
        load_components(
            args.model,
            transformer_path=args.teacher_transformer,
            load_vae=False,
            load_text_encoder=False,
            load_tokenizers=False,
        )
        .transformer.to(td)
        .eval()
        .requires_grad_(False)
    )
    teacher.configure_execution(False, attention_backend="torch_varlen")
    student = (
        load_components(
            args.model,
            transformer_path=args.student,
            load_vae=False,
            load_text_encoder=False,
            load_tokenizers=False,
        )
        .transformer.to(sd)
        .requires_grad_(False)
    )
    if not student.params.modulation_rank:
        raise ValueError("Student must have compressed modulation")
    student.configure_execution(True, attention_backend="torch_varlen")
    factors = {n: p for n, p in student.named_parameters() if compressed_parameter(n)}
    for p in factors.values():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in factors.items() if n.startswith("modulation_down.")
                ],
                "lr": args.lr * 0.1,
            },
            {
                "params": [
                    p
                    for n, p in factors.items()
                    if not n.startswith("modulation_down.")
                ],
                "lr": args.lr,
            },
        ],
        weight_decay=0,
    )
    teacher_heads = [
        getattr(b, s)
        for b in teacher.transformer_blocks
        for s in ("img_mod", "txt_mod")
    ]
    student_heads = [
        getattr(b, s)
        for b in student.transformer_blocks
        for s in ("img_mod", "txt_mod")
    ]

    def inputs(item, t, seed, epoch=0):
        path, index, cache = item
        clean = load_cached_latent(path).unsqueeze(0)
        noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(seed))
        x = ((1 - t) * clean + t * noise).to(dtype)
        text, mask = cache.get([cache.caption(index, epoch)], "cpu", dtype)
        return {
            "hidden_states": [x],
            "timestep": torch.tensor([t], dtype=dtype),
            "encoder_hidden_states": (text, mask),
        }

    def move(batch, device):
        return {
            "hidden_states": [batch["hidden_states"][0].to(device)],
            "timestep": batch["timestep"].to(device),
            "encoder_hidden_states": tuple(
                v.to(device) for v in batch["encoder_hidden_states"]
            ),
        }

    # Up to two heldout images per source, at four fixed timesteps.
    validation = []
    with torch.no_grad():
        for gi, (_, valid) in enumerate(groups):
            for vi, item in enumerate(valid[:2]):
                for ti, t in enumerate([0.1, 0.4, 0.75, 0.95]):
                    batch = inputs(item, t, 10000 + gi * 100 + vi * 10 + ti)
                    target = teacher(**move(batch, td))[0][0].float().cpu()
                    validation.append((batch, target))

    @torch.no_grad()
    def evaluate():
        student.eval()
        errors = []
        for batch, target in validation:
            pred = student(**move(batch, sd))[0][0].float()
            target = target.to(sd)
            errors.append(
                float(((pred - target).square().sum() / target.square().sum()).sqrt())
            )
        student.train()
        return {
            "mean_relative_rmse": sum(errors) / len(errors),
            "max_relative_rmse": max(errors),
            "errors": errors,
        }

    best = evaluate()
    best_step = 0

    def snapshot():
        return {n: p.detach().cpu().clone() for n, p in factors.items()}

    best_state = snapshot()
    history = [{"step": 0, **best}]

    def report():
        result = {
            "args": vars(args),
            "best_step": best_step,
            "best": best,
            "history": history,
            "elapsed_seconds": time.time() - start,
            "train_images": [len(g[0]) for g in groups],
            "validation_images": [len(g[1]) for g in groups],
            "peak_allocated_gib": {
                str(d): torch.cuda.max_memory_allocated(d) / 2**30 for d in (td, sd)
            },
        }
        out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")

    report()
    print("initial", json.dumps(best), flush=True)
    for step in range(1, args.steps + 1):
        train = groups[(step - 1) % len(groups)][0]
        item = rng.choice(train)
        # Alternate uniform coverage and the usual shifted logit-normal distribution.
        if step % 2:
            t = rng.random()
        else:
            u = 1 / (1 + math.exp(-rng.gauss(0, 1)))
            t = 6 * u / (1 + 5 * u)
        batch = inputs(item, t, rng.randrange(2**31), epoch=step // max(len(train), 1))
        tb = move(batch, td)
        sb = move(batch, sd)
        with torch.no_grad():
            target = teacher(**tb)[0][0].to(sd).float()
            temb = teacher.time_text_embed(tb["timestep"], tb["hidden_states"][0])
            mod_targets = [h(temb).to(sd).float() for h in teacher_heads]
        pred = student(**sb)[0][0].float()
        loss_flow = (pred - target).square().mean() / target.square().mean().clamp_min(
            1e-6
        )
        z = student.block_condition(temb.to(sd))
        auxiliary = []
        for head, y in zip(student_heads, mod_targets):
            y = y.reshape(1, 6, -1)
            p = head(z).reshape_as(y)
            auxiliary.append(
                ((p - y).square().mean(-1) / y.square().mean(-1).clamp_min(1e-6)).mean()
            )
        loss = loss_flow + 0.1 * torch.stack(auxiliary).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite flow distillation loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(factors.values()), 1.0)
        scale = min(step / 25, 1.0) * (
            0.1 + 0.9 * (1 + math.cos(math.pi * step / args.steps)) / 2
        )
        opt.param_groups[0]["lr"] = args.lr * 0.1 * scale
        opt.param_groups[1]["lr"] = args.lr * scale
        opt.step()
        if step % 10 == 0:
            print(
                "train",
                step,
                "flow_mse",
                float(loss_flow.detach()),
                "loss",
                float(loss.detach()),
                flush=True,
            )
        if step % args.validate_every == 0 or step == args.steps:
            val = evaluate()
            history.append({"step": step, "training_loss": float(loss.detach()), **val})
            if val["mean_relative_rmse"] < best["mean_relative_rmse"]:
                best, best_step, best_state = val, step, snapshot()
                save_file(best_state, str(out.with_suffix(".factors.safetensors")))
            report()
            print("validation", step, json.dumps(val), flush=True)
    with torch.no_grad():
        for n, p in factors.items():
            p.copy_(best_state[n])
    state = {n: p.detach().cpu().contiguous() for n, p in student.state_dict().items()}
    tmp = out.with_suffix(".incomplete")
    save_file(
        state,
        str(tmp),
        metadata={
            "model_config": json.dumps(asdict(student.params)),
            "architecture": "mageflow-lowrank-modulation-v1",
            "experimental": "true",
            "source": args.student,
        },
    )
    tmp.replace(out)
    report()
    print("DONE", out, "best", best_step, flush=True)


if __name__ == "__main__":
    main()
