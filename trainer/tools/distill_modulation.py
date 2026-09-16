"""Experimental compressed-modulation conversion and timestep-only distillation.

No image/text encoders are needed. Exports a NEW architecture, not ComfyUI-native Mage.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from trainer.modeling.compressed_modulation import (
    compressed_parameter,
    initialize_compression,
)
from trainer.modeling.loader import load_components


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mage-flow")
    ap.add_argument("--transformer")
    ap.add_argument("--output", required=True)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument(
        "--calibration-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Offline SVD/factor-construction precision; stored factors remain FP32",
    )
    ap.add_argument(
        "--svd-driver",
        choices=("auto", "gesvd", "gesvdj"),
        default="auto",
        help="CUDA SVD solver: auto uses PyTorch's default; gesvd selects QR",
    )
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.rank < 1 or args.steps < 0 or args.batch_size < 1:
        ap.error("rank and batch-size must be positive; steps must be nonnegative")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    calculation_dtype = getattr(torch, args.calibration_dtype)
    if device.type != "cuda" and args.svd_driver != "auto":
        ap.error("explicit SVD drivers require CUDA")
    dtype = torch.bfloat16
    start = time.time()
    model = (
        load_components(
            args.model,
            transformer_path=args.transformer,
            load_vae=False,
            load_text_encoder=False,
            load_tokenizers=False,
        )
        .transformer.eval()
        .to(device)
    )
    if model.params.modulation_rank:
        raise ValueError("Teacher must have dense modulation")
    model.requires_grad_(False)
    # Stratified representable timesteps: calibration and validation have no BF16 overlap.
    pool = torch.linspace(0, 1, 16385, device=device).to(dtype).unique()
    select = torch.arange(len(pool), device=device) % 5 == 2
    train_t, valid_t = pool[~select], pool[select]
    with torch.no_grad():

        def features(t):
            return torch.nn.functional.silu(
                model.time_text_embed(t, torch.empty((), device=device, dtype=dtype))
            ).float()

        x, xt = features(train_t), features(valid_t)
        calibration = x.to(calculation_dtype)
        mean = calibration.mean(0)
        _, singular, vh = torch.linalg.svd(
            calibration - mean,
            full_matrices=False,
            driver=None if args.svd_driver == "auto" else args.svd_driver,
        )
        if args.rank > len(vh):
            raise ValueError("Rank exceeds calibration sample count")
        dense = [
            getattr(b, s)[1]
            for b in model.transformer_blocks
            for s in ("img_mod", "txt_mod")
        ]
        # BF16 targets on CPU: bounded batch transfers, no teacher model during optimization.
        targets = torch.stack([m(x.to(dtype)).cpu() for m in dense], dim=1)
        heldout = torch.stack([m(xt.to(dtype)).cpu() for m in dense], dim=1)
        initialize_compression(model, vh[: args.rank].T, mean, calculation_dtype)
        del dense
    heads = [
        getattr(b, s)[1]
        for b in model.transformer_blocks
        for s in ("img_mod", "txt_mod")
    ]
    factors = [p for n, p in model.named_parameters() if compressed_parameter(n)]
    for p in factors:
        p.requires_grad_(True)
    # Optimize only factors; retain original BF16 timestep embedder and trunk.
    opt = torch.optim.AdamW(
        [
            {"params": model.modulation_down.parameters(), "lr": 1e-5},
            {"params": [p for h in heads for p in h.parameters()], "lr": 1e-4},
        ],
        weight_decay=0,
    )
    normalizer = (
        targets.float()
        .reshape(len(x), len(heads), 6, -1)
        .square()
        .mean((0, 3))
        .clamp_min(1e-6)
        .to(device)
    )

    def predict(features):
        z = model.modulation_down(features)
        return torch.stack([h(z) for h in heads], dim=1)

    @torch.no_grad()
    def evaluate():
        err = den = 0.0
        groups = torch.zeros(len(heads), 6, device=device)
        for a in range(0, len(xt), 32):
            y = heldout[a : a + 32].to(device).float()
            p = predict(xt[a : a + 32]).to(dtype).float()
            e = (p - y).square()
            err += float(e.sum())
            den += float(y.square().sum())
            groups += e.reshape(len(y), len(heads), 6, -1).mean(-1).sum(0) / normalizer
        return {
            "relative_rmse": (err / den) ** 0.5,
            "normalized_loss": float((groups / len(xt)).mean()),
        }

    best = evaluate()
    best_step = 0

    def snapshot():
        return {
            n: p.detach().cpu().clone()
            for n, p in model.named_parameters()
            if compressed_parameter(n)
        }

    best_state = snapshot()
    history = [{"step": 0, **best}]
    print("initial", json.dumps(best), flush=True)
    for step in range(1, args.steps + 1):
        ids = torch.randint(len(x), (args.batch_size,), device="cpu")
        y = targets[ids].to(device).float()
        p = predict(x[ids.to(device)])
        error = (p - y).square().reshape(len(ids), len(heads), 6, -1).mean((0, 3))
        loss = (error / normalizer).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite modulation loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(factors, 1.0)
        opt.step()
        if step % 100 == 0 or step == args.steps:
            val = evaluate()
            history.append({"step": step, "train_loss": float(loss.detach()), **val})
            if val["normalized_loss"] < best["normalized_loss"]:
                best, best_step, best_state = val, step, snapshot()
            print("step", step, json.dumps(history[-1]), flush=True)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best_state:
                p.copy_(best_state[n])
    state = {n: p.detach().cpu().contiguous() for n, p in model.state_dict().items()}
    config = asdict(model.params)
    metadata = {
        "model_config": json.dumps(config),
        "architecture": "mageflow-lowrank-modulation-v1",
        "experimental": "true",
        "source": str(args.transformer or args.model),
    }
    tmp = out.with_suffix(".incomplete")
    save_file(state, str(tmp), metadata=metadata)
    tmp.replace(out)
    report = {
        "args": vars(args),
        "model_config": config,
        "best_step": best_step,
        "best": best,
        "history": history,
        "training_timesteps": len(x),
        "validation_timesteps": len(xt),
        "factor_parameters": sum(p.numel() for p in factors),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": time.time() - start,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "singular_values": singular.cpu().tolist(),
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "DONE",
        out,
        "best step",
        best_step,
        "seconds",
        report["elapsed_seconds"],
        flush=True,
    )


if __name__ == "__main__":
    main()
