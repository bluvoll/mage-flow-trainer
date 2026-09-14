"""Concept-preservation regularizer.

Mage-Flow loses concepts that are not in the LoRA. The measured mechanism is *learned interference*,
not passive damage: a random weight perturbation at a trained LoRA's own norm leaves concept
directions at cosine 0.97, while the trained LoRA at the same norm drops them to 0.68. Magnitude
is not the lever -- direction is. So shrinking the update (lower LR, lower rank, weight decay)
buys almost nothing, and the only thing that can help is telling the optimizer *which* directions
are expensive.

That is what this does. A concept is a prompt pair differing by one element ("...,horns" vs
"..."); its direction is `d = v(x, with) - v(x, without)`, the part of the velocity field that the
concept is responsible for. `d_base` is captured once with the adapter disabled, and the loss is

    L_preserve = mean over pairs of (1 - cos(d_base, d_lora))

Notes on what this is and is not:

* It constrains the **trunk only**. Contexts are built under `no_grad`, so no gradient reaches the
  text encoder or the LLM adapter even when those are being trained. The measured damage is
  downstream of the text read (a real LoRA moves every token's attention share by <1.5%), so
  that is the right place to spend the compute.
* It needs **no generated images and no regularization dataset**. The reference is the same model
  with the adapter switched off. This is what makes it cheap relative to classic prior
  preservation, which has to sample the base model first.
* **Precision.** A concept direction is ~0.45% of the conditional velocity, so subtracting two
  bf16 velocities loses most of it: measured `cos(d_bf16, d_fp32) = 0.336`. The gradient survives
  at roughly a third of its strength but is not systematically biased, and with stochastic
  rounding in the optimizer the per-step noise averages out rather than accumulating in the
  weights. The practical consequences are that `weight` needs to be larger than it looks like it
  should be, and that raising `n_latents` is worth more here than in a normal loss -- the noise is
  independent across latents, so the mean improves as sqrt(n). Running the probe in fp32 is not
  an option while the trunk is bf16: an fp32 activation cannot be matmul'd against a bf16 weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class PreserveConfig:
    # Path to the pair file. One pair per line, `with ||| without`. Blank lines and `#` comments
    # are skipped. Unset disables the regularizer entirely.
    prompts: str | None = None
    weight: float = 1.0
    # Evaluate every N optimizer-visible steps. The probe costs `pairs_per_step * n_latents *
    # len(timesteps) * 2` transformer forwards, so this is the main cost dial.
    every: int = 4
    # Pairs sampled per evaluation, cycled deterministically so every pair is covered in turn
    # rather than resampled at random. This is the count ACROSS ALL RANKS -- under DDP the pairs
    # are split between processes, so raising the process count makes the probe cheaper per rank
    # rather than duplicating it. Rounded up to at least one pair per rank.
    pairs_per_step: int = 2
    n_latents: int = 2
    resolution: int = 512
    # Where in the flow to probe. Damage was characterised at these two; the ends of the schedule
    # are less informative (t->0 is nearly pure structure, t->1 nearly pure noise).
    timesteps: tuple[float, ...] = (0.3, 0.6)
    seed: int = 1234
    # Relative weight movement used to calibrate the precision floor at startup. Small enough that
    # the true damage is ~0 (measured: fp32 reads 0.0000 at eps 1e-4), so whatever the probe
    # reports at this perturbation is pure numerical noise. Set to 0 to skip calibration.
    calibrate_eps: float = 1e-4
    # Warm up the regularizer over this many steps. At step 0 the adapter is zero-initialised and
    # `d_lora == d_base` exactly, so the loss is 0 and the ramp costs nothing real; it exists to
    # keep the term from fighting the first few hundred steps of style acquisition.
    warmup_steps: int = 0

    def __post_init__(self):
        self.timesteps = tuple(float(t) for t in self.timesteps)
        if self.prompts is None:
            return
        if not self.timesteps:
            raise ValueError("[preserve] timesteps is empty; give at least one, e.g. [0.3, 0.6]")
        if not 0.0 < min(self.timesteps) and max(self.timesteps) < 1.0:
            raise ValueError(f"[preserve] timesteps must lie in (0, 1), got {list(self.timesteps)}")
        if self.every < 1:
            raise ValueError(f"[preserve] every must be >= 1, got {self.every}")

    @property
    def enabled(self) -> bool:
        return bool(self.prompts)


def load_pairs(path: str | Path) -> list[tuple[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[preserve] prompts file not found: {p}")
    pairs = []
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|||" not in line:
            raise ValueError(
                f"{p}:{lineno}: expected `with ||| without`, got {line!r}. The two sides must "
                f"differ by exactly the concept being preserved."
            )
        a, b = line.split("|||", 1)
        a, b = a.strip(), b.strip()
        if a == b:
            raise ValueError(f"{p}:{lineno}: both sides are identical, so the direction is zero.")
        pairs.append((a, b))
    if not pairs:
        raise ValueError(f"{p}: no pairs found.")
    return pairs


class ConceptPreserver:
    """Holds the frozen reference directions and produces the preservation loss.

    Construction is deliberately cheap; the expensive part (the base pass) happens on the first
    `loss()` call, by which point the model is on-device, prepared and compiled. Doing it earlier
    would either run on CPU or force a second compile of the disabled-adapter graph at a moment
    when the training graph has not been warmed yet.
    """

    def __init__(self, cfg: PreserveConfig, trainer):
        self.cfg = cfg
        self.trainer = trainer
        self.pairs = load_pairs(cfg.prompts)
        self.d_base: torch.Tensor | None = None   # (P, n_latents * n_t, D)
        self.floor = 0.0
        self._ctx: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._cursor = 0
        self._latents: torch.Tensor | None = None

    # ------------------------------------------------------------------ setup

    def _adapted(self):
        """The modules carrying adapters, unwrapped from DDP/compile."""
        unwrap = self.trainer._unwrap
        mods = [unwrap(self.trainer.transformer)]
        return [m for m in mods if hasattr(m, "disable_adapters")]

    def _build(self) -> None:
        t = self.trainer
        device, dtype = t.accelerator.device, t.dtype
        c = self.cfg

        g = torch.Generator(device="cpu").manual_seed(c.seed)
        n = c.resolution // 16
        self._latents = torch.randn(
            c.n_latents, 128, 1, n, n, generator=g, dtype=torch.float32
        ).to(device, dtype)

        mods = self._adapted()
        for m in mods:
            m.disable_adapters()
        try:
            with torch.no_grad():
                # Encoded with the adapter off and then held fixed for the rest of the run. Both
                # sides of every later comparison therefore see the *base* conditioning, which is
                # what makes the measured difference attributable to the trunk alone. Encoding
                # them with the adapter live would let a trained text path move the reference and
                # the probe together, hiding exactly the drift this is meant to catch.
                self._ctx = [(t._encode([a]), t._encode([b])) for a, b in self.pairs]
                self.d_base = torch.stack([self._direction(i) for i in range(len(self.pairs))])
        finally:
            for m in mods:
                m.enable_adapters()

        if c.calibrate_eps > 0:
            self.floor = self._calibrate(c.calibrate_eps)

        if t.accelerator.is_main_process:
            mib = self.d_base.numel() * self.d_base.element_size() / 2**20
            print(
                f"preserve  {len(self.pairs)} pair(s), {c.n_latents} latent(s) x "
                f"{len(c.timesteps)} timestep(s) @ {c.resolution}px, reference {mib:.0f} MiB, "
                f"precision floor {self.floor:.4f}",
                flush=True,
            )

    # ------------------------------------------------------------- calibration

    def _lora_pairs(self):
        """(module, lora_A weight, lora_B weight, scaling) for every adapted Linear."""
        for m in self.trainer._unwrap(self.trainer.transformer).modules():
            A, B = getattr(m, "lora_A", None), getattr(m, "lora_B", None)
            if A is None or B is None or "default" not in A:
                continue
            yield m, A["default"].weight, B["default"].weight, m.scaling.get("default", 1.0)

    def _calibrate(self, target: float) -> float:
        """Damage the probe reports at a perturbation too small to cause any.

        In bf16 a concept direction is ~0.5% of the velocity field, so subtracting two velocities
        destroys most of it: measured, the probe reads 1-cos = 0.53 at a perturbation where fp32
        reads 0.0000. That pedestal is constant with respect to the weights, so it contributes no
        gradient -- but it makes the logged number unreadable and hides the real signal under a
        number three times its size. Measuring it once at startup makes `pres` mean what it says.

        The perturbation is injected through the adapter's own B matrices, which are zero at init,
        so backing them up costs a few MiB rather than a copy of the trunk. Each is scaled to give
        exactly `target` relative movement in its own layer, so the calibration point is the same
        regardless of rank, alpha or layer shape.
        """
        g = torch.Generator(device=self.trainer.accelerator.device).manual_seed(self.cfg.seed + 1)
        backup = []
        # `no_grad` throughout: these are leaf parameters that require grad, so an in-place write
        # outside it is a hard error, and the probe itself must not build a graph.
        with torch.no_grad():
            for m, A, B, scale in self._lora_pairs():
                backup.append((B, B.detach().clone()))
                N = torch.randn(B.shape, generator=g, device=B.device, dtype=torch.float32)
                delta = scale * (N @ A.float())
                ref = m.base_layer.weight if hasattr(m, "base_layer") else m.weight
                ratio = delta.norm() / ref.float().norm()
                if ratio > 0:
                    B.add_((N * (target / ratio)).to(B.dtype))
            try:
                d = torch.stack([self._direction(i) for i in range(len(self.pairs))])
                cos = F.cosine_similarity(self.d_base.float(), d.float(), dim=2).mean()
                return float(1.0 - cos)
            finally:
                for B, saved in backup:
                    B.copy_(saved)

    # ------------------------------------------------------------------ probe

    def _velocity(self, ctx: torch.Tensor, ts: float) -> torch.Tensor:
        t = self.trainer
        lat = self._latents
        b = lat.shape[0]
        v = t.transformer(
            hidden_states=lat,
            timestep=torch.full((b,), ts, device=lat.device, dtype=t.dtype),
            encoder_hidden_states=(ctx[0].expand(b, -1, -1), ctx[1].expand(b, -1)),
            return_dict=False,
        )[0]
        return v.flatten(1)

    def _direction(self, i: int) -> torch.Tensor:
        """(n_latents * n_timesteps, D) for pair `i`, in whatever grad mode the caller set."""
        a, b = self._ctx[i]
        return torch.cat([self._velocity(a, ts) - self._velocity(b, ts) for ts in self.cfg.timesteps])

    # ------------------------------------------------------------------- loss

    def loss(self, step: int) -> torch.Tensor | None:
        """Preservation loss for this step, or None when this step is not an evaluation step."""
        c = self.cfg
        if step % c.every:
            return None
        if self.d_base is None:
            self._build()

        # Split the batch of pairs across ranks. DDP already averages gradients, so a rank
        # evaluating its own slice and every rank averaging over that slice is the same objective
        # as one rank evaluating all of them -- but it costs 1/world_size the forwards instead of
        # duplicating the identical probe on every GPU, which is what made the regularizer the
        # dominant cost of a multi-GPU step.
        acc = self.trainer.accelerator
        world, rank = acc.num_processes, acc.process_index
        n = len(self.pairs)
        # At least one pair per rank: a rank with an empty slice would contribute no preservation
        # gradient at all, quietly turning the averaged objective into a fraction of its weight.
        per_rank = max(1, min(c.pairs_per_step, n) // world)
        window = min(per_rank * world, n)
        idx = [(self._cursor + j) % n for j in range(window)]
        # Advance identically on every rank, so the cursors never drift apart.
        self._cursor = (self._cursor + window) % n
        idx = idx[rank * per_rank:(rank + 1) * per_rank] or idx[-1:]

        terms = [
            1.0 - F.cosine_similarity(self.d_base[i].float(), self._direction(i).float(), dim=1).mean()
            for i in idx
        ]
        # Subtracting the floor leaves the gradient untouched -- it is a constant -- but clamping
        # at zero adds a dead zone below it, which is the useful part: the measurement carries no
        # information there, and in bf16 the response is compressive, over-reporting small damage
        # by ~10x. Without the dead zone the term fights style acquisition hardest at exactly the
        # point where the LoRA has done the least harm.
        out = (torch.stack(terms).mean() - self.floor).clamp_min(0.0) * c.weight
        if c.warmup_steps:
            out = out * min(1.0, step / c.warmup_steps)
        return out
