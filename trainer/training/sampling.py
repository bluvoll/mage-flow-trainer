"""Validation sampling: images from fixed prompts during training.

Ported from the diffusion-pipe-mageflow-ft fork (`utils/validation_sampling.py`) onto this
trainer's model call. What makes it cheap here:

* **No resident text encoder.** Prompt and negative embeddings are encoded once at startup,
  alongside the training captions -- into the in-RAM text cache, the persistent caption-variation
  cache, or by the loaded encoder -- and `Trainer._encode` serves them like any other caption.
  On a warm variation cache the encoder is never loaded at all.
* **Decoder only.** Training loads Mage-VAE without its decoder. Sampling loads a decoder-only VAE
  once, parks it in host RAM, and moves it to the GPU only for the decode at the end of each image.
* **Multi-GPU.** Each rank samples its own share of the prompts, then rank 0 gathers and logs them,
  instead of N-1 GPUs idling at a barrier while one renders everything.

The sampler matches Microsoft's reference pipeline (`mage_flow/pipeline.py`): static-shift sigmas
`linspace(1, 1/steps, steps)` mapped through `shift*s/(1+(shift-1)*s)` with a terminal 0, the model
fed sigma itself as the timestep, Euler steps `x += (sigma_next - sigma) * v`, and dual-forward CFG
`uncond + cfg * (cond - uncond)`, skipped entirely at cfg <= 1. Noise is drawn on the CPU from a
seeded generator, so a seed gives the same starting noise on any GPU.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import torch


def build_sigmas(steps: int, shift: float) -> torch.Tensor:
    """`steps + 1` descending sigmas from 1 to a terminal 0, in float64."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    base = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
    if shift != 1.0:
        base = shift * base / (1.0 + (shift - 1.0) * base)
    return torch.cat([base, torch.zeros(1, dtype=torch.float64)])


def combine_cfg(cond: torch.Tensor, uncond: torch.Tensor, scale: float,
                renormalize: bool = False) -> torch.Tensor:
    guided = uncond + scale * (cond - uncond)
    if not renormalize:
        return guided
    # Per token, i.e. over the channel axis of the (B, C, T, H, W) velocity.
    return guided * (cond.norm(dim=1, keepdim=True) / (guided.norm(dim=1, keepdim=True) + 1e-6))


def resolve_seed(settings: dict, strategy: str, round_index: int) -> int:
    return settings["seed"] + (round_index if strategy == "walk" else 0)


def _blank(text: str) -> str:
    # A caption that tokenizes to nothing after the template is stripped has no valid text token;
    # the reference pipeline uses a single space for "no negative" for the same reason.
    return text if text.strip() else " "


def sample_texts(sampling_cfg) -> list[str]:
    """Every text a sampling round will encode: prompts, plus negatives where CFG is on."""
    out = []
    for s in sampling_cfg.resolved_prompts():
        out.append(_blank(s["prompt"]))
        if s["cfg"] > 1.0:
            out.append(_blank(s["negative_prompt"]))
    return list(dict.fromkeys(out))


def latent_channels(model) -> int:
    """The model's latent channel count: Mage-Flow keeps it as `in_channels`, MOSS-Image and Flux3 in their
    params (`latent_channels` / `channels`)."""
    if hasattr(model, "in_channels"):
        return int(model.in_channels)
    params = getattr(model, "params", None)
    for name in ("latent_channels", "channels", "in_channels"):
        if hasattr(params, name):
            return int(getattr(params, name))
    raise AttributeError(f"{type(model).__name__} does not say how many latent channels it takes")


@torch.no_grad()
def denoise(model, encode, settings: dict, seed: int, device, dtype,
            keep_fraction: float | None = None) -> torch.Tensor:
    """Run the sampler for one prompt and return clean latents, (1, 128, 1, H/16, W/16) float32.

    `encode(texts)` returns `(hidden, mask)` on `device`, as `Trainer._encode` does.
    """
    h, w = settings["height"] // 16, settings["width"] // 16
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, latent_channels(model), 1, h, w, generator=generator,
                    dtype=torch.float32).to(device)
    sigmas = build_sigmas(settings["steps"], float(settings["shift"])).tolist()
    use_cfg = settings["cfg"] > 1.0
    texts = [_blank(settings["prompt"])] + ([_blank(settings["negative_prompt"])] if use_cfg else [])
    hidden, mask = encode(texts)
    # RTI's uniform path is batch-1 only, so its CFG runs as two forwards; everything else fuses
    # cond and uncond into one batch-2 forward.
    fused = use_cfg and getattr(model, "region_interface", None) is None
    extra = {} if keep_fraction is None else {"keep_fraction": keep_fraction}

    def velocity(latents, context, sigma):
        t = torch.full((latents.shape[0],), sigma, device=device, dtype=dtype)
        return model(hidden_states=latents.to(dtype), timestep=t, encoder_hidden_states=context,
                     return_dict=False, **extra)[0].float()

    for i in range(settings["steps"]):
        sigma, nxt = sigmas[i], sigmas[i + 1]
        if fused:
            v = velocity(x.expand(2, -1, -1, -1, -1), (hidden, mask), sigma)
            v = combine_cfg(v[:1], v[1:], settings["cfg"], settings["renormalize_cfg"])
        elif use_cfg:
            cond = velocity(x, (hidden[:1], mask[:1]), sigma)
            uncond = velocity(x, (hidden[1:], mask[1:]), sigma)
            v = combine_cfg(cond, uncond, settings["cfg"], settings["renormalize_cfg"])
        else:
            v = velocity(x, (hidden, mask), sigma)
        x = x + (nxt - sigma) * v
    return x


def to_image(pixels: torch.Tensor):
    """(1, 3, H, W) in [-1, 1] -> PIL."""
    from PIL import Image

    arr = ((pixels[0].float().clamp(-1, 1) + 1.0) * 127.5).round().byte()
    return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())


class Decoder:
    """Decoder-only VAE, parked on the CPU between rounds."""

    def __init__(self, train_cfg, dtype):
        from ..modeling.loader import load_components, model_load_kwargs

        kwargs = model_load_kwargs(train_cfg)
        self.vae = load_components(
            train_cfg.model_path, dtype=dtype, load_transformer=False, load_text_encoder=False,
            load_tokenizers=False, load_vae=True, vae_decoder=True,
            **{k: kwargs[k] for k in ("vae_path", "flux2_vae")},
        ).vae.to("cpu")
        # MOSS-Image and Flux3 always use the FLUX.2 VAE (packed 128c/16x training latents) and run it in
        # float32, whatever train.flux2_vae says.
        from ..modeling.loader import resolve_model_family
        packed_family = resolve_model_family(train_cfg.model_path, train_cfg.model_family) in ("flux3", "moss_image")
        self.flux2 = bool(train_cfg.flux2_vae) or packed_family
        self.dtype = torch.float32 if packed_family else dtype

    @torch.no_grad()
    def __call__(self, latents: torch.Tensor, device):
        z = latents.squeeze(2)
        self.vae.to(device)
        try:
            if self.flux2 and not getattr(self.vae, "outputs_packed_flux2", False):
                # Inverse of LatentCacher.encode_tensor: un-normalise with the VAE's running
                # batch-norm statistics, then fold 128c/16x back to FLUX.2's 32c/8x.
                bn = self.vae.bn
                z = z.to(device, torch.float32)
                z = z * (bn.running_var[None, :, None, None].float() + bn.eps).sqrt() \
                    + bn.running_mean[None, :, None, None].float()
                z = torch.nn.functional.pixel_shuffle(z, 2)
                out = self.vae.decode(z.to(self.dtype)).sample
            else:
                out = self.vae.decode(z.to(device, self.dtype))
                out = getattr(out, "sample", out)
        finally:
            self.vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return to_image(out)


class ValidationSampler:
    """Owns cadence, seeding, distribution across ranks and logging for one training run."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.cfg = trainer.cfg.sampling
        self.prompts = self.cfg.resolved_prompts()
        self.decoder = None
        self.rounds = 0
        self.last_step = None

    def due(self, step: int, epoch_end: int | None = None) -> bool:
        """`epoch_end` is the 1-based number of an epoch that just finished, else None."""
        if step == self.last_step:
            return False
        if self.cfg.every_n_steps and step % self.cfg.every_n_steps == 0:
            return True
        return bool(epoch_end and self.cfg.every_n_epochs
                    and epoch_end % self.cfg.every_n_epochs == 0)

    def run(self, step: int, tag: str) -> None:
        """Generate this rank's prompts, gather to rank 0, log. Collective: every rank calls it."""
        trainer = self.trainer
        acc = trainer.accelerator
        self.last_step = step
        start = time.time()
        if self.decoder is None:
            self.decoder = Decoder(trainer.cfg.train, trainer.dtype)
        model = trainer._unwrap(trainer.transformer)
        was_training = model.training
        model.eval()
        # RTI samples at the keep fraction currently being trained, so samples show the model as
        # it is actually used at this point of the schedule.
        keep = trainer.keep_fraction if getattr(trainer, "rti_interface", None) is not None else None
        mine = []
        try:
            # Sampling must not consume the training RNG stream, or a run with samples would
            # train on different noise than the same run without them.
            with torch.random.fork_rng(devices=[acc.device] if acc.device.type == "cuda" else []):
                for s in self.prompts[acc.process_index::acc.num_processes]:
                    seed = resolve_seed(s, self.cfg.seed_strategy, self.rounds)
                    try:
                        latents = denoise(model, trainer._encode, s, seed, acc.device,
                                          trainer.dtype, keep)
                        image = self.decoder(latents, acc.device)
                        buf = io.BytesIO()
                        image.save(buf, format="PNG")
                        mine.append((s["index"], seed, buf.getvalue()))
                    except Exception as exc:
                        # One failed prompt must not end a run that has been going for hours.
                        print(f"WARNING: sample {s['index']} ({s['label']!r}) failed: "
                              f"{type(exc).__name__}: {exc}", flush=True)
        finally:
            model.train(was_training)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if acc.num_processes > 1:
            from accelerate.utils import gather_object
            mine = gather_object(mine)
        self.rounds += 1
        if acc.is_main_process:
            self._publish(sorted(mine), step, tag, time.time() - start)

    def _publish(self, results, step, tag, seconds):
        from PIL import Image

        trainer = self.trainer
        by_index = {s["index"]: s for s in self.prompts}
        entries, manifest = [], []
        out = trainer.out_dir / "samples" / tag
        if self.cfg.save_to_disk and results:
            out.mkdir(parents=True, exist_ok=True)
        for index, seed, png in results:
            s = by_index[index]
            image = Image.open(io.BytesIO(png))
            image.load()
            name = f"{index:02d}_{s['label']}"
            caption = (f"{s['prompt']}\nsteps={s['steps']} cfg={s['cfg']} shift={s['shift']} "
                       f"{s['width']}x{s['height']} seed={seed}")
            entries.append((name, image, caption))
            manifest.append({**{k: v for k, v in s.items() if k != "index"}, "seed": seed,
                             "file": f"{name}_seed{seed}.png"})
            if self.cfg.save_to_disk:
                image.save(out / f"{name}_seed{seed}.png")
        if self.cfg.save_to_disk and manifest:
            (out / "prompts.json").write_text(json.dumps(
                {"step": step, "samples": manifest}, indent=2, ensure_ascii=False), encoding="utf-8")
        trainer.tracker.log_images("samples", entries, step)
        trainer.tracker.log({"samples/seconds": seconds}, step)
        trainer._print(f"samples  {len(results)}/{len(self.prompts)} at step {step} in "
                       f"{seconds:.1f}s" + (f" -> {out}" if self.cfg.save_to_disk and results else ""))
