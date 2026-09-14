"""Initialise `lora_A` from the base weight's own singular directions instead of at random.

Why this exists. Standard LoRA sets `lora_B = 0` and `lora_A` to noise, so `dW = B @ A` is exactly
zero at init -- the model is untouched. The randomness is not in the output, it is in the
*projection*: `lora_A` is a random rank-r slice of a 2048-dimensional input space, and that slice
is the only window the adapter can ever look through. A random r-dimensional subspace of a
2048-dimensional one is nearly orthogonal to any particular structure the model holds, so its
overlap with concept-carrying directions is arbitrary, and the adapter has no way to avoid moving
through them.

Measured context (2026-08-17): concept damage in Mage-Flow is written within the first epoch and is
almost independent of what is being learned -- a LoRA trained on deliberately mismatched captions
forgets *more* than one trained correctly (0.603 vs 0.667). That points at something
content-independent operating at the very start of training, and the random projection is the main
candidate.

This is deliberately NOT PiSSA. PiSSA also subtracts the principal component from the frozen base
(`W_res = W - BA`), which changes the starting model; here `lora_B` stays zero, so every arm starts
from a numerically identical model and the *only* variable is which subspace the adapter reads.

`mode` selects where in the spectrum to sit:

  top     the highest-energy directions -- what PiSSA uses. Note these are the directions shared by
          the most behaviours, so on the evidence above they are a plausible *worst* choice: the
          most damaging component in the model (adaln) is also its most concentrated one.
  mid     the middle of the spectrum.
  bottom  the low-energy tail. If Mage-Flow has slack anywhere -- a subspace an adapter can work in
          without disturbing concept structure -- this is where it should be.
  random  PEFT's own init, untouched. The control.

`lora_A` is rescaled to the Frobenius norm of the random init it replaces. Without that the arms
would differ in effective learning rate as well as in subspace, and the comparison would be
confounded.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

MODES = ("random", "top", "mid", "bottom")


def _lora_linears(module: nn.Module):
    """(name, peft LoraLayer) for every adapted Linear carrying a `default` adapter."""
    for name, m in module.named_modules():
        A = getattr(m, "lora_A", None)
        if A is not None and "default" in A and hasattr(m, "base_layer"):
            yield name, m


def _slice_rows(Vh: torch.Tensor, mode: str, r: int) -> torch.Tensor:
    """r rows of Vh (right singular vectors, most significant first) from the chosen band."""
    n = Vh.shape[0]
    if mode == "top":
        return Vh[:r]
    if mode == "bottom":
        return Vh[-r:]
    start = max(0, (n - r) // 2)
    return Vh[start:start + r]


@torch.no_grad()
def spectrum_init(module: nn.Module, mode: str, cache: str | Path | None = None) -> int:
    """Overwrite every `lora_A` with base-weight singular directions. Returns layers touched.

    The SVD is the expensive part (a few seconds per layer, ~280 layers), so the three bands are
    computed in one pass and cached together -- only r rows per band survive, a few hundred KB per
    layer rather than the full factor.
    """
    if mode not in MODES:
        raise ValueError(f"adapter.init must be one of {MODES}, got {mode!r}")
    if mode == "random":
        return 0

    cache_path = Path(cache) if cache else None
    store: dict[str, torch.Tensor] = {}
    if cache_path is not None and cache_path.exists():
        store = torch.load(cache_path, map_location="cpu")

    # Adapters are applied before the model reaches its device, so the weights are still on the
    # host here. A CPU SVD of a 2048x8192 mlp weight takes the better part of a minute, and there
    # are ~280 of them -- hours. Push each factorisation to the GPU and bring back only the rows.
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    touched, computed = 0, 0
    for name, m in _lora_linears(module):
        A = m.lora_A["default"].weight
        r = A.shape[0]
        key = f"{name}|{mode}|{r}"
        rows = store.get(key)
        if rows is None:
            W = m.base_layer.weight.detach().to(dev, torch.float32)
            # `full_matrices=False` gives Vh with min(out, in) rows -- every direction the weight
            # can actually distinguish, which is all the bands need.
            Vh = torch.linalg.svd(W, full_matrices=False)[2]
            for band in ("top", "mid", "bottom"):
                store[f"{name}|{band}|{r}"] = _slice_rows(Vh, band, r).cpu()
            rows = store[key]
            computed += 1
            del Vh, W
            if computed % 40 == 0:
                print(f"         svd {computed} layer(s)...", flush=True)
        # Match the norm of the init being replaced, so the arms differ only in subspace.
        rows = rows.to(A.device, torch.float32)
        A.copy_((rows * (A.detach().float().norm() / rows.norm())).to(A.dtype))
        touched += 1

    if cache_path is not None and computed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(store, cache_path)
    return touched
