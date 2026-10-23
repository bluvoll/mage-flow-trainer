"""Held-out validation loss at fixed timestep quantiles.

The diffusion-pipe-mageflow-ft fork's eval, rebuilt on this trainer: every evaluation round scores
the same images, at the same timesteps (quantiles of the training timestep distribution, shift
included), against the same noise, with un-augmented captions. Only the weights change between
rounds, so the curve is directly comparable across the whole run -- unlike the training loss, which
re-draws timestep and noise every step and mostly reports which timesteps a step happened to draw.

Reported per quantile (`eval/loss_q0.30`) and as their mean (`eval/loss`). Low quantiles track fine
detail, high ones composition, so a regression shows up in the band that moved.
"""

from __future__ import annotations

import random
from dataclasses import replace

import torch


def eval_caption_config(caption):
    """The training caption config with every stochastic step off. Mixed mode evaluates on tags,
    the variant every sample can produce."""
    return replace(
        caption, caption_dropout_percent=0.0, tag_dropout_percent=0.0, shuffle_tags=False,
        nl_shuffle_sentences=False, attribution_position="fixed",
        caption_mode="tags" if caption.caption_mode == "mixed" else caption.caption_mode,
    )


class Evaluator:
    def __init__(self, cfg):
        from ..data.caption import build_caption
        from ..data.moss_captions import annotations_for
        from ..data.dataset import MageFlowDataset

        self.cfg = cfg.eval
        self.flow = cfg.flow
        dataset_cfg = replace(cfg.dataset, path=self.cfg.path, subsets=[], subsets_file=None,
                              subsets_root=None, num_repeats=1,
                              caption=eval_caption_config(cfg.dataset.caption))
        self.dataset = MageFlowDataset(dataset_cfg, caption_seed=0)
        order = list(range(len(self.dataset.entries)))
        if self.cfg.max_samples and len(order) > self.cfg.max_samples:
            # A fixed spread over the folder rather than its first N files, which tend to be one
            # upload batch of one artist.
            order = sorted(random.Random(self.cfg.seed).sample(order, self.cfg.max_samples))
        self.indices = order
        self.captions = {
            i: build_caption(self.dataset.entries[i].tags, self.dataset.entries[i].nl,
                             dataset_cfg.caption, random.Random(0),
                             annotations_for(self.dataset.entries[i], dataset_cfg.caption))
            for i in order
        }
        self.last_step = None

    def __len__(self) -> int:
        return len(self.indices)

    def texts(self) -> list[str]:
        return list(dict.fromkeys(c if c.strip() else " " for c in self.captions.values()))

    def due(self, step: int, epoch_end: int | None = None) -> bool:
        if step == self.last_step:
            return False
        if self.cfg.every_n_steps and step % self.cfg.every_n_steps == 0:
            return True
        return bool(epoch_end and self.cfg.every_n_epochs
                    and epoch_end % self.cfg.every_n_epochs == 0)

    def _batches(self, rank: int, world: int, batch_size: int):
        mine = self.indices[rank::world]
        by_bucket: dict = {}
        for i in mine:
            by_bucket.setdefault(self.dataset.entries[i].bucket, []).append(i)
        for bucket in sorted(by_bucket):
            group = by_bucket[bucket]
            for k in range(0, len(group), batch_size):
                yield group[k:k + batch_size]

    def _latents(self, trainer, indices):
        from ..data.cache import image_to_tensor, load_and_crop, load_cached_latent

        device = trainer.accelerator.device
        entries = [self.dataset.entries[i] for i in indices]
        if self.dataset.encode_on_the_fly:
            pixels = torch.cat([
                image_to_tensor(load_and_crop(e.path, self.dataset.bucket_managers[e.resolution])[0])
                for e in entries])
            return trainer._encode_pixels(pixels.to(device, trainer.dtype))
        return torch.stack([load_cached_latent(e.latent_path) for e in entries]).to(
            device, torch.float32)

    @torch.no_grad()
    def run(self, trainer, step: int) -> dict | None:
        """Collective: every rank calls it. Returns the metrics on rank 0, None elsewhere."""
        from .flow import sample_timesteps

        acc = trainer.accelerator
        self.last_step = step
        quantiles = self.cfg.quantiles
        sums = torch.zeros(len(quantiles), device=acc.device, dtype=torch.float64)
        count = torch.zeros(1, device=acc.device, dtype=torch.float64)
        model = trainer._unwrap(trainer.transformer)
        was_training = model.training
        model.eval()
        rti = getattr(model, "region_interface", None) is not None
        extra = {"keep_fraction": trainer.keep_fraction} if rti else {}
        batch_size = 1 if rti else self.cfg.batch_size
        try:
            with torch.random.fork_rng(devices=[acc.device] if acc.device.type == "cuda" else []):
                for indices in self._batches(acc.process_index, acc.num_processes, batch_size):
                    latents = self._latents(trainer, indices)
                    b, _, _, h, w = latents.shape
                    context = trainer._encode([self.captions[i] if self.captions[i].strip() else " "
                                               for i in indices])
                    for qi, q in enumerate(quantiles):
                        t = sample_timesteps(self.flow, b, h, w, acc.device, quantile=q)
                        # Noise keyed by (seed, image, quantile): identical every round, and
                        # independent of batch composition or how many ranks share the set.
                        noise = torch.stack([
                            torch.randn(latents.shape[1:], dtype=torch.float32,
                                        generator=torch.Generator().manual_seed(
                                            self.cfg.seed * 1_000_003 + i * 97 + qi))
                            for i in indices]).to(acc.device)
                        tt = t.view(-1, 1, 1, 1, 1)
                        noisy = (1 - tt) * latents + tt * noise
                        pred = model(hidden_states=noisy.to(trainer.dtype),
                                     timestep=t.to(trainer.dtype), encoder_hidden_states=context,
                                     return_dict=False, **extra)[0]
                        err = (pred.float() - (noise - latents)) ** 2
                        sums[qi] += err.flatten(1).mean(dim=1).sum().double()
                    count += b
        finally:
            model.train(was_training)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if acc.num_processes > 1:
            sums = acc.reduce(sums, reduction="sum")
            count = acc.reduce(count, reduction="sum")
        if not acc.is_main_process or count.item() == 0:
            return None
        per_q = (sums / count).tolist()
        metrics = {f"eval/loss_q{q:.2f}": v for q, v in zip(quantiles, per_q)}
        metrics["eval/loss"] = sum(per_q) / len(per_q)
        return metrics
