"""Generate matched-seed teacher/compressed-model samples for human evaluation.

Uses native Mage-VAE decode and explicit Euler flow sampling; not a benchmark of
ComfyUI samplers. Output checkpoints are not modified. No image encoder is loaded.
"""

import argparse
import gc
import itertools
import json
import time
from pathlib import Path

import torch
from PIL import Image

from trainer.modeling.loader import encode_prompts, load_components
from trainer.modeling.modules.mage_vae import MageVAE

PROMPTS = [
    "A ceramic teapot and two cups on a wooden table by a window, morning sunlight, a small vase of yellow flowers, detailed still life.",
    "An adult man wearing a blue jacket and carrying a closed umbrella, standing on a rainy city street at night, neon reflections, anime illustration.",
    "A mountain lake surrounded by pine trees, snow-covered peaks in the distance, sunset clouds reflected in still water, wide landscape illustration.",
]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mage-flow")
    ap.add_argument("--teacher-transformer")
    ap.add_argument("--student", action="append", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()
    if args.steps < 1 or args.size < 16 or args.size % 16:
        ap.error("steps must be positive; size must be a positive multiple of 16")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    start = time.time()
    c = load_components(args.model, load_transformer=False, load_vae=False)
    c.text_encoder.to(device)
    contexts = []
    for prompt in ["", *PROMPTS]:
        h, m = encode_prompts(c, [prompt], device)
        contexts.append((h.cpu(), m.cpu()))
    del c
    gc.collect()
    torch.cuda.empty_cache()
    results = {}
    latent_paths = []
    for mi, path in enumerate([None, *args.student]):
        label = "teacher" if path is None else Path(path).stem
        model = (
            load_components(
                args.model,
                transformer_path=path or args.teacher_transformer,
                load_vae=False,
                load_text_encoder=False,
                load_tokenizers=False,
            )
            .transformer.to(device)
            .eval()
        )
        model.configure_execution(False, attention_backend="sdpa")
        schedule = torch.linspace(1, 0, args.steps + 1, device=device)
        schedule = 6 * schedule / (1 + 5 * schedule)
        values = []
        for pi, prompt in enumerate(PROMPTS):
            g = torch.Generator(device=device).manual_seed(args.seed + pi)
            latent = torch.randn(
                1, 128, 1, args.size // 16, args.size // 16, device=device, generator=g
            )
            cond = tuple(t.to(device) for t in contexts[pi + 1])
            uncond = tuple(t.to(device) for t in contexts[0])
            for t, tnext in itertools.pairwise(schedule):
                inp = latent.to(dtype)
                ts = t.reshape(1).to(dtype)
                vc = model(inp, ts, cond)[0].float()
                vu = model(inp, ts, uncond)[0].float()
                latent = latent + (tnext - t) * (vu + args.cfg * (vc - vu))
            if not latent.isfinite().all():
                raise RuntimeError("Nonfinite sampled latents")
            value = latent.cpu()
            values.append(value)
            target = out / f"{label}-{pi}.pt"
            torch.save(value, target)
            latent_paths.append((label, pi, target))
            print("sampled", label, pi, flush=True)
        results[label] = values
        del model
        gc.collect()
        torch.cuda.empty_cache()
    vae = MageVAE(
        str(Path(args.model) / "vae/diffusion_pytorch_model.safetensors"),
        sample_posterior=False,
    )
    del vae.dconv_encoder
    vae.to(device, dtype).eval()
    for label, pi, path in latent_paths:
        z = (
            torch.load(path, map_location="cpu", weights_only=True)
            .squeeze(2)
            .to(device, dtype)
        )
        image = (
            vae.decode(z)
            .float()[0]
            .clamp(-1, 1)
            .add(1)
            .mul(127.5)
            .round()
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        Image.fromarray(image).save(out / f"{label}-{pi}.png")
    metrics = {
        label: [
            float(((x - y).square().sum() / y.square().sum()).sqrt())
            for x, y in zip(vals, results["teacher"])
        ]
        for label, vals in results.items()
        if label != "teacher"
    }
    (out / "report.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "prompts": PROMPTS,
                "latent_relative_rmse": metrics,
                "elapsed_seconds": time.time() - start,
            },
            indent=2,
        )
        + "\n"
    )
    print("DONE", metrics, flush=True)


if __name__ == "__main__":
    main()
