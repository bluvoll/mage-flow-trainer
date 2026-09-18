# SPDX-License-Identifier: GPL-3.0-or-later
"""Combined shared-AdaLN + RTI Mage-Flow inference for ComfyUI."""
import math
from functools import lru_cache

import torch
from torch import nn

from .model import CompressedMageFlowBase, CompressedMageFlowTransformer


def _sign(value): return (value > 0) - (value < 0)

def _gilbert(x, y, ax, ay, bx, by):
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
    if 2 * width > 3 * height:
        if abs(ax2 + ay2) % 2 and width > 2: ax2 += dax; ay2 += day
        yield from _gilbert(x, y, ax2, ay2, bx, by)
        yield from _gilbert(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
        return
    if abs(bx2 + by2) % 2 and height > 2: bx2 += dbx; by2 += dby
    yield from _gilbert(x, y, bx2, by2, ax2, ay2)
    yield from _gilbert(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
    yield from _gilbert(x + (ax-dax)+(bx2-dbx), y + (ay-day)+(by2-dby), -bx2, -by2, -(ax-ax2), -(ay-ay2))

@lru_cache(maxsize=128)
def _order(h, w):
    points = tuple(_gilbert(0, 0, w, 0, 0, h) if w >= h else _gilbert(0, 0, 0, h, w, 0))
    return tuple(y * w + x for x, y in points), tuple(abs(a[0]-b[0])+abs(a[1]-b[1]) != 1 for a,b in zip(points, points[1:]))


class RTIInterface(nn.Module):
    def __init__(self, width, buckets, start, end, operations, device, dtype):
        super().__init__()
        self.core_start, self.core_end = int(start), int(end)
        self.read_score = operations.Linear(width, 1, device=device, dtype=dtype)
        self.size_embedding = operations.Embedding(buckets, width, device=device, dtype=dtype)
        self.write_map = operations.Linear(2 * width, width, device=device, dtype=dtype)

    def read(self, image, rope, h, w, keep):
        n = h * w; order0, jumps0 = _order(h, w)
        order = torch.tensor(order0, device=image.device)
        jumps = torch.tensor(jumps0, device=image.device)
        requested = max(1, min(n, int(math.floor(n * float(keep) + .5))))
        regions = max(requested, sum(jumps0) + 1)
        ordered = image[0].index_select(0, order)
        if regions == n:
            labels = torch.arange(n, device=image.device)
        else:
            cuts = (ordered.float()[1:] - ordered.float()[:-1]).square().sum(-1).masked_fill(jumps, torch.inf).argsort(descending=True, stable=True)[:regions-1]
            starts = torch.zeros(n, dtype=torch.long, device=image.device); starts[cuts + 1] = 1; labels = starts.cumsum(0)
        counts = torch.zeros(regions, dtype=torch.long, device=image.device).scatter_add_(0, labels, torch.ones_like(labels))
        scores = self.read_score(ordered).squeeze(-1).float()
        maxima = scores.new_full((regions,), -torch.inf).scatter_reduce_(0, labels, scores.detach(), reduce="amax", include_self=True)
        weights = (scores - maxima[labels]).exp(); weights = weights / weights.new_zeros(regions).index_add(0, labels, weights).clamp_min(1e-12)[labels]
        pooled = ordered.new_zeros((regions, ordered.shape[-1]), dtype=torch.float32).index_add_(0, labels, ordered.float() * weights[:, None])
        sizes = counts.clamp_min(1).float().log2().floor().long().clamp_max(self.size_embedding.num_embeddings - 1)
        pooled = pooled + self.size_embedding(sizes).float()
        ordered_rope = rope[0, :, order].float()
        reduced = ordered_rope.new_zeros((rope.shape[1], regions, *rope.shape[3:])).index_add_(1, labels, ordered_rope)
        reduced = reduced / counts[None, :, None, None, None].clamp_min(1)
        return pooled.to(image.dtype)[None], reduced.to(rope.dtype)[None], order, labels

    def write(self, dense, region, region_in, order, labels):
        member = (region[0] - region_in[0]).index_select(0, labels)
        ordered = dense[0].index_select(0, order)
        update = self.write_map(torch.cat((ordered, member), -1))
        return dense[0].index_copy(0, order, ordered + update)[None]


class CompressedRTIMageFlowTransformer(CompressedMageFlowTransformer):
    def __init__(self, rti_size_buckets, rti_core_start, rti_core_end, operations, **kwargs):
        super().__init__(operations=operations, **kwargs)
        self.region_interface = RTIInterface(self.inner_dim, rti_size_buckets, rti_core_start, rti_core_end,
                                             operations, kwargs.get("device"), kwargs.get("dtype"))

    def _forward(self, x, timestep, context, attention_mask=None, ref_latents=None, transformer_options={}, control=None, rti_keep_fraction=.75, **kwargs):
        if x.shape[0] != 1 or ref_latents is not None:
            raise ValueError("Compressed RTI Mage-Flow currently supports batch size 1 and no reference latents.")
        if attention_mask is not None and not torch.is_floating_point(attention_mask): attention_mask = (attention_mask - 1).to(x.dtype) * torch.finfo(x.dtype).max
        keep = float(transformer_options.get("rti_keep_fraction", rti_keep_fraction))
        if not 0 < keep <= 1: raise ValueError("RTI keep fraction must be in (0, 1].")
        image, image_ids, shape = self.process_img(x); num = image.shape[1]; h,w = shape
        txt_ids = torch.zeros((1, context.shape[1], 3), device=x.device)
        image, context = self.img_in(image), self.txt_in(self.txt_norm(context))
        temb = self.time_text_embed(timestep, image); block_temb = self.modulation_down(torch.nn.functional.silu(temb).float())
        rope = self.pe_embedder(torch.cat((txt_ids, image_ids), 1)).contiguous(); text_rope, image_rope = rope[:, :, :context.shape[1]], rope[:, :, context.shape[1]:]
        dense = region_in = order = labels = None
        for i, block in enumerate(self.transformer_blocks):
            if i == self.region_interface.core_start:
                dense = image; region_in, image_rope, order, labels = self.region_interface.read(image, image_rope, h, w, keep); image = region_in
            context, image = block(hidden_states=image, encoder_hidden_states=context, encoder_hidden_states_mask=attention_mask, temb=block_temb, image_rotary_emb=torch.cat((text_rope, image_rope), 2), transformer_options=transformer_options)
            if i + 1 == self.region_interface.core_end:
                image = self.region_interface.write(dense, image, region_in, order, labels); image_rope = rope[:, :, context.shape[1]:]
        image = self.proj_out(self.norm_out(image, temb))[:, :num]
        return image.reshape(1, h, w, self.out_channels).movedim(-1, 1)


class CompressedRTIMageFlowBase(CompressedMageFlowBase):
    def __init__(self, config, device=None):
        import comfy.model_base
        comfy.model_base.QwenImage.__init__(self, config, comfy.model_base.ModelType.FLOW, device=device, unet_model=CompressedRTIMageFlowTransformer)
