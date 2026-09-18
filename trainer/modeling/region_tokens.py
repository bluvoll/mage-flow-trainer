"""Packed native-resolution region-token interface for experimental Mage-Flow RTI.

The Gilbert traversal below is adapted from Jakub Cerveny's ``gilbert`` project
(BSD-2-Clause); see ``LICENSE.gilbert`` for its retained notice.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

import torch
from torch import nn


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _generate_gilbert_2d(x: int, y: int, ax: int, ay: int, bx: int, by: int) -> Iterator[tuple[int, int]]:
    width, height = abs(ax + ay), abs(bx + by)
    dax, day, dbx, dby = _sign(ax), _sign(ay), _sign(bx), _sign(by)
    if height == 1:
        for _ in range(width):
            yield x, y; x += dax; y += day
        return
    if width == 1:
        for _ in range(height):
            yield x, y; x += dbx; y += dby
        return
    ax2, ay2, bx2, by2 = ax // 2, ay // 2, bx // 2, by // 2
    width2, height2 = abs(ax2 + ay2), abs(bx2 + by2)
    if 2 * width > 3 * height:
        if width2 % 2 and width > 2:
            ax2 += dax; ay2 += day
        yield from _generate_gilbert_2d(x, y, ax2, ay2, bx, by)
        yield from _generate_gilbert_2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
        return
    if height2 % 2 and height > 2:
        bx2 += dbx; by2 += dby
    yield from _generate_gilbert_2d(x, y, bx2, by2, ax2, ay2)
    yield from _generate_gilbert_2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
    yield from _generate_gilbert_2d(x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby), -bx2, -by2, -(ax - ax2), -(ay - ay2))


@lru_cache(maxsize=512)
def gilbert_order(height: int, width: int) -> tuple[tuple[int, ...], tuple[bool, ...], int]:
    """Row-major generalized-Gilbert order, discontinuity mask, and cut floor."""
    height, width = int(height), int(width)
    if height < 1 or width < 1:
        raise ValueError(f"height and width must be positive, got {(height, width)}")
    points = tuple(_generate_gilbert_2d(0, 0, width, 0, 0, height) if width >= height
                   else _generate_gilbert_2d(0, 0, 0, height, width, 0))
    if len(points) != height * width or len(set(points)) != len(points):
        raise RuntimeError(f"Gilbert traversal did not cover {height}x{width} exactly once")
    if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
        raise RuntimeError(f"Gilbert traversal escaped the {height}x{width} grid")
    jumps = tuple(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in zip(points, points[1:]))
    return tuple(y * width + x for x, y in points), jumps, sum(jumps) + 1


_DEVICE_ORDER_CACHE: OrderedDict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor, int]] = OrderedDict()
_CACHE_PER_DEVICE = 256


def gilbert_permutation(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    key = (int(height), int(width), str(device))
    value = _DEVICE_ORDER_CACHE.pop(key, None)
    if value is None:
        order, jumps, minimum = gilbert_order(height, width)
        value = (torch.tensor(order, dtype=torch.long, device=device), torch.tensor(jumps, dtype=torch.bool, device=device), minimum)
    _DEVICE_ORDER_CACHE[key] = value
    device_keys = [k for k in _DEVICE_ORDER_CACHE if k[2] == key[2]]
    while len(device_keys) > _CACHE_PER_DEVICE:
        stale = device_keys.pop(0)
        del _DEVICE_ORDER_CACHE[stale]
    return value


@dataclass
class RegionPlan:
    order: torch.Tensor
    labels: torch.Tensor
    counts: torch.Tensor
    image_lengths: list[int]
    region_lengths: list[int]
    requested: list[int]
    minimum: list[int]
    atom_to_token: torch.Tensor | None = None


@torch.no_grad()
def build_region_plan(features: torch.Tensor, shapes: list[tuple[int, int]], image_lengths: list[int], keep_fraction: float, device: torch.device) -> RegionPlan:
    if not 0 < float(keep_fraction) <= 1:
        raise ValueError(f"RTI keep_fraction must be in (0, 1], got {keep_fraction}")
    if len(shapes) != len(image_lengths) or sum(image_lengths) != features.shape[0]:
        raise ValueError("RTI shapes, image lengths, and packed feature length disagree")
    orders, labels, requested, minimum, region_lengths = [], [], [], [], []
    token_offset = region_offset = 0
    for (height, width), n in zip(shapes, image_lengths):
        if height * width != n:
            raise ValueError("RTI image length does not match its token grid")
        perm, jumps, floor = gilbert_permutation(height, width, device)
        want = max(1, min(n, int(math.floor(n * float(keep_fraction) + 0.5))))
        regions = max(want, floor)
        if regions == n:
            local = torch.arange(n, device=device, dtype=torch.long)
        else:
            ordered = features[token_offset:token_offset + n].index_select(0, perm).float()
            scores = (ordered[1:] - ordered[:-1]).square().sum(-1).masked_fill(jumps, torch.inf)
            cuts = scores.argsort(descending=True, stable=True)[: regions - 1]
            starts = torch.zeros(n, dtype=torch.long, device=device)
            starts[cuts + 1] = 1
            local = starts.cumsum(0)
        orders.append(perm + token_offset)
        labels.append(local + region_offset)
        requested.append(want); minimum.append(floor); region_lengths.append(regions)
        token_offset += n; region_offset += regions
    order, labels = torch.cat(orders), torch.cat(labels)
    counts = torch.zeros(region_offset, dtype=torch.long, device=device)
    counts.scatter_add_(0, labels, torch.ones_like(labels))
    return RegionPlan(order, labels, counts, list(image_lengths), region_lengths, requested, minimum)


class RegionInterface(nn.Module):
    """READ/WRITE boundary around a dense Mage-Flow block span."""
    def __init__(self, width: int, size_buckets: int = 17, *, core_start: int, core_end: int):
        super().__init__()
        if size_buckets < 1:
            raise ValueError("rti size_buckets must be positive")
        self.core_start, self.core_end = int(core_start), int(core_end)
        self.read_score = nn.Linear(width, 1)
        self.size_embedding = nn.Embedding(size_buckets, width)
        self.write_map = nn.Linear(2 * width, width)
        nn.init.zeros_(self.read_score.weight); nn.init.zeros_(self.read_score.bias)
        nn.init.zeros_(self.size_embedding.weight)
        with torch.no_grad():
            self.write_map.weight.zero_()
            self.write_map.weight[:, width:].copy_(torch.eye(width, dtype=self.write_map.weight.dtype, device=self.write_map.weight.device))
            self.write_map.bias.zero_()

    def read(self, img: torch.Tensor, dense_freqs: torch.Tensor, plan: RegionPlan) -> tuple[torch.Tensor, torch.Tensor]:
        ordered = img[0].index_select(0, plan.order)
        regions, width = plan.counts.numel(), ordered.shape[-1]
        scores = self.read_score(ordered).squeeze(-1).float()
        maxima = scores.detach().new_full((regions,), -torch.inf)
        maxima.scatter_reduce_(0, plan.labels, scores.detach(), reduce="amax", include_self=True)
        weights = (scores - maxima.index_select(0, plan.labels)).exp()
        sums = scores.new_zeros(regions).index_add(0, plan.labels, weights).clamp_min(1e-12)
        weights = weights / sums.index_select(0, plan.labels)
        pooled = ordered.new_zeros((regions, width), dtype=torch.float32).index_add_(0, plan.labels, ordered.float() * weights[:, None])
        size_ids = plan.counts.clamp_min(1).float().log2().floor().long().clamp_max(self.size_embedding.num_embeddings - 1)
        pooled = pooled + self.size_embedding(size_ids).float()
        phases = torch.view_as_real(dense_freqs.index_select(0, plan.order)).float()
        reduced = phases.new_zeros((regions, *phases.shape[1:])).index_add_(0, plan.labels, phases)
        reduced = reduced / plan.counts.clamp_min(1)[:, None, None]
        return pooled.to(img.dtype)[None], torch.view_as_complex(reduced.contiguous())

    def write(self, dense_img: torch.Tensor, region_out: torch.Tensor, region_in: torch.Tensor, plan: RegionPlan) -> torch.Tensor:
        delta = region_out[0] - region_in[0]
        member = delta.index_select(0, plan.labels)
        ordered = dense_img[0].index_select(0, plan.order)
        update = self.write_map(torch.cat((ordered, member), dim=-1))
        return dense_img[0].index_copy(0, plan.order, ordered + update)[None]


@dataclass(frozen=True)
class BudgetSchedule:
    start_keep: float = .98
    target_keep: float = .75
    identity_steps: int = 0
    warmup_steps: int = 0
    anneal_steps: int = 1000
    budget_steps: tuple[float, ...] = ()

    def __post_init__(self):
        if not 0 < self.target_keep <= self.start_keep <= 1:
            raise ValueError("RTI keep fractions must satisfy 0 < target_keep <= start_keep <= 1")
        if any(type(v) is not int or v < 0 for v in (self.identity_steps, self.warmup_steps, self.anneal_steps)):
            raise ValueError("RTI step counts must be non-negative integers")
        if any(not 0 < float(v) <= 1 for v in self.budget_steps):
            raise ValueError("rti.budget_steps values must be in (0, 1]")

    def resolve(self, optimizer_step: int) -> tuple[float, str]:
        step = max(0, int(optimizer_step))
        if step < self.identity_steps:
            keep, phase = 1.0, "identity"
        elif step < self.identity_steps + self.warmup_steps:
            keep, phase = self.start_keep, "warmup"
        elif self.anneal_steps and step < self.identity_steps + self.warmup_steps + self.anneal_steps:
            p = (step - self.identity_steps - self.warmup_steps) / self.anneal_steps
            keep, phase = self.start_keep + (self.target_keep - self.start_keep) * .5 * (1 - math.cos(math.pi * p)), "anneal"
        else:
            keep, phase = self.target_keep, "target"
        if self.budget_steps:
            keep = min(self.budget_steps, key=lambda candidate: (abs(candidate - keep), candidate))
        return float(keep), phase
