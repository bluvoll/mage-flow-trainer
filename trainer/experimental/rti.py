"""Experimental Region Token Interface for Mage-Flow.

The production packed implementation is in ``trainer.modeling.region_tokens``.

Implements the RTI paper's region read/core-delta/write design, with rectangular
grid cuts and a proposed budget curriculum. This is not a production training,
checkpoint export, or ComfyUI interface. The prototype supports one image only.

Read/Write design adapted from Eduard Zamfir's RTI (MIT, 2026);
see LICENSE.RTI and https://github.com/eduardzamfir/RTI.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from trainer.modeling.mageflow_attention import packed_attention_metadata


@lru_cache(maxsize=64)
def rectangular_hilbert_order(height: int, width: int) -> tuple[int, ...]:
    """Filter an enclosing-square Hilbert curve; partitioning must cut its jumps."""
    if height < 1 or width < 1:
        raise ValueError('Token grid dimensions must be positive')
    side = 1 << (max(height, width) - 1).bit_length()
    order = []
    for distance in range(side * side):
        x = y = 0
        remaining = distance
        extent = 1
        while extent < side:
            right = (remaining // 2) & 1
            up = (remaining ^ right) & 1
            if up == 0:
                if right:
                    x, y = extent - 1 - x, extent - 1 - y
                x, y = y, x
            x += extent * right
            y += extent * up
            remaining //= 4
            extent *= 2
        if x < width and y < height:
            order.append(y * width + x)
    return tuple(order)


@dataclass
class RegionPartition:
    order: torch.Tensor
    labels: torch.Tensor  # labels in curve order
    counts: torch.Tensor
    requested_regions: int
    minimum_regions: int

    @property
    def regions(self):
        return self.counts.numel()


@torch.no_grad()
def partition_regions(features, height, width, requested_regions):
    """One image's [N,D] features; keep every connected component of the path.

    Spatial jumps after filtering rectangular grids are mandatory boundaries.
    If a budget is infeasible, increase it to the reported connectivity floor.
    """
    n = height * width
    if features.ndim != 2 or features.shape[0] != n:
        raise ValueError('Features must contain one row per image token')
    if not 1 <= requested_regions <= n:
        raise ValueError('Region count must be between 1 and the token count')
    order = torch.tensor(rectangular_hilbert_order(height, width), device=features.device)
    rows, columns = order // width, order % width
    jumps = (rows.diff().abs() + columns.diff().abs()) != 1
    minimum = int(jumps.sum().item()) + 1
    regions = max(requested_regions, minimum)
    if regions == n:
        labels = torch.arange(n, device=features.device)
    else:
        ordered = features[order].float()
        scores = (ordered[1:] - ordered[:-1]).square().sum(-1)
        scores = scores.masked_fill(jumps, torch.inf)
        cuts = scores.topk(regions - 1).indices + 1
        starts = torch.zeros(n, device=features.device, dtype=torch.long)
        starts[cuts] = 1
        labels = starts.cumsum(0)
    counts = torch.bincount(labels, minlength=regions)
    return RegionPartition(order, labels, counts, requested_regions, minimum)


@dataclass(frozen=True)
class BudgetSchedule:
    """Cosine annealing by completed optimizer updates, independent of diffusion t."""
    start_keep: float = 0.98
    target_keep: float = 0.75
    anneal_steps: int = 1000
    warmup_steps: int = 0

    def __post_init__(self):
        if not 0 < self.target_keep <= self.start_keep <= 1:
            raise ValueError('Require 0 < target_keep <= start_keep <= 1')
        if self.anneal_steps < 1 or self.warmup_steps < 0:
            raise ValueError('Require positive anneal_steps and nonnegative warmup_steps')

    def keep_fraction(self, optimizer_step):
        if optimizer_step < 0:
            raise ValueError('Optimizer step must be nonnegative')
        progress = min(1., max(0., (optimizer_step-self.warmup_steps)/self.anneal_steps))
        return self.target_keep + (self.start_keep-self.target_keep) * (1+math.cos(math.pi*progress))/2

    def regions(self, tokens, optimizer_step):
        if tokens < 1:
            raise ValueError('Token count must be positive')
        return max(1, min(tokens, math.ceil(tokens*self.keep_fraction(optimizer_step))))


class RegionInterface(nn.Module):
    """Trainable Read/Write; float32 reductions and averaged complex RoPE phasors."""
    def __init__(self, width, size_buckets=17):
        super().__init__()
        self.read_score = nn.Linear(width, 1)
        self.size_embedding = nn.Embedding(size_buckets, width)
        self.write = nn.Linear(2*width, width)
        nn.init.zeros_(self.read_score.weight)
        nn.init.zeros_(self.read_score.bias)
        nn.init.zeros_(self.size_embedding.weight)
        with torch.no_grad():
            self.write.weight.zero_()
            self.write.weight[:, width:].copy_(torch.eye(width))
            self.write.bias.zero_()

    def read(self, features, frequencies, partition):
        ordered = features[partition.order]
        labels, regions = partition.labels, partition.regions
        scores = self.read_score(ordered).squeeze(-1).float()
        maxima = scores.new_full((regions,), -torch.inf)
        maxima.scatter_reduce_(0, labels, scores, reduce='amax', include_self=True)
        weights = (scores-maxima[labels]).exp()
        sums = scores.new_zeros(regions).index_add(0, labels, weights)
        weights = weights / sums[labels]
        pooled = ordered.new_zeros((regions, ordered.shape[-1]), dtype=torch.float32)
        pooled = pooled.index_add(0, labels, ordered.float()*weights[:, None])
        size_ids = partition.counts.float().log2().floor().long().clamp_max(self.size_embedding.num_embeddings-1)
        pooled = pooled + self.size_embedding(size_ids).float()
        # Average cos/sin independently; do not normalize their attenuated magnitude.
        phases = torch.view_as_real(frequencies[partition.order]).float()
        reduced = phases.new_zeros((regions, *phases.shape[1:])).index_add(0, labels, phases)
        reduced = reduced / partition.counts[:, None, None]
        return pooled.to(features.dtype), torch.view_as_complex(reduced.contiguous())

    def restore(self, features, region_delta, partition):
        ordered = features[partition.order]
        member_delta = region_delta[partition.labels]
        update = self.write(torch.cat((ordered, member_delta), dim=-1))
        # index_copy restores the original row-major order, including its gradient.
        return features.index_copy(0, partition.order, ordered + update)


class RTIMageFlow(nn.Module):
    """Single-image prototype. Wrap after selecting/freezing backbone adapters.

    Core span is zero-based, end-exclusive. No production save/load contract yet.
    The caller owns optimizer setup and budget scheduling. No base parameter's
    trainability is changed by this wrapper.
    """
    def __init__(self, backbone, core_start, core_end, *, dual_timestep=False):
        super().__init__()
        depth = len(backbone.transformer_blocks)
        if not 0 < core_start < core_end < depth:
            raise ValueError('RTI needs a nonempty core and dense blocks at both ends')
        if dual_timestep:
            raise ValueError('RTI and dual-timestep noising are mutually exclusive')
        self.backbone = backbone
        self.core_start, self.core_end = core_start, core_end
        reference = backbone.img_in.weight
        if not reference.is_floating_point():
            raise ValueError('Prototype requires a floating-point input projection; SDNQ integration is pending')
        self.interface = RegionInterface(backbone.inner_dim).to(device=reference.device, dtype=reference.dtype)
        self.last_budget = None

    def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=False,
                *, keep_fraction=1., second_timestep=None, timestep_mask=None):
        if second_timestep is not None or timestep_mask is not None:
            raise ValueError('RTI and dual-timestep noising are mutually exclusive')
        if not 0 < keep_fraction <= 1:
            raise ValueError('keep_fraction must be in (0, 1]')
        if keep_fraction == 1:
            self.last_budget = None
            return self.backbone(hidden_states, timestep, encoder_hidden_states, return_dict)
        if isinstance(hidden_states, list) or hidden_states.shape[:1] != (1,):
            raise ValueError('Experimental compressed RTI currently supports a single unpacked image')
        if hidden_states.ndim != 5 or hidden_states.shape[2] != 1:
            raise ValueError('RTI expects [1,C,1,H,W] image latents')
        m = self.backbone
        h, w = hidden_states.shape[-2:]
        text, text_mask = encoder_hidden_states
        img = m.img_in(hidden_states.squeeze(2).flatten(2).transpose(1, 2))
        txt = m.txt_in(m.txt_norm(text))
        temb = m.time_text_embed(timestep.to(img.dtype), img)
        condition = m.block_condition(temb)
        dense_freqs = m.pos_embed([(1, h, w)], device=img.device)
        freqs = dense_freqs
        before = region_input = partition = None
        for index, block in enumerate(m.transformer_blocks):
            if index == self.core_start:
                before = img[0]
                partition = partition_regions(before, h, w, max(1, math.ceil(h*w*keep_fraction)))
                region_input, freqs = self.interface.read(before, dense_freqs, partition)
                img = region_input[None]
                self.last_budget = dict(tokens=h*w, requested_regions=partition.requested_regions,
                                        actual_regions=partition.regions, minimum_regions=partition.minimum_regions)
            if index == self.core_end:
                img = self.interface.restore(before, img[0]-region_input, partition)[None]
                freqs = dense_freqs
            mask = torch.cat((text_mask.bool(), torch.ones(1, img.shape[1], device=img.device,
                                                          dtype=torch.bool)), dim=1)[:, None, None]
            metadata = () if m.attention_backend == 'sdpa' else packed_attention_metadata(mask)
            rotary = torch.view_as_real(freqs) if m.compiled_blocks else freqs
            args = (block, img, txt, condition, rotary, mask, m.num_attention_heads,
                    m.attention_backend, metadata, False)
            if self.training and index in m.checkpoint_blocks:
                txt, img = checkpoint(m.block_forward, *args, use_reentrant=False)
            else:
                txt, img = m.block_forward(*args)
        scale, shift = m.norm_out.linear(m.norm_out.silu(temb).to(img.dtype)).chunk(2,-1)
        img = m.norm_out.norm(img)*(1+scale[:,None])+shift[:,None]
        return (m.proj_out(img).transpose(1,2).reshape(1,m.out_channels,1,h,w),)
