"""Mage-Flow weight layout (Microsoft, MIT); batched training execution."""

from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .modules.mage_layers import (
    AdaLayerNormContinuous,
    MageFlowEmbedRope,
    MageFlowTimestepProjEmbeddings,
    MageFlowTransformerBlock,
    RMSNorm,
)
from .batched import _double_stream_block_forward
from .mageflow_attention import packed_attention_metadata, validate_attention_backend
from .region_tokens import RegionInterface, build_region_plan


def _packed_varlen_metadata(text_lengths, image_lengths, device):
    """Attention gather/scatter metadata for packed text/image sample pairs."""
    indices, boundaries = [], [0]
    text_offset, image_offset = 0, sum(text_lengths)
    for text_length, image_length in zip(text_lengths, image_lengths):
        indices.extend(range(text_offset, text_offset + text_length))
        indices.extend(range(image_offset, image_offset + image_length))
        boundaries.append(boundaries[-1] + text_length + image_length)
        text_offset += text_length
        image_offset += image_length
    return (torch.tensor(indices, device=device), torch.tensor(boundaries, device=device, dtype=torch.int32), max(t + i for t, i in zip(text_lengths, image_lengths)))


@dataclass
class MageFlowParams:
    in_channels: int
    out_channels: int
    context_in_dim: int
    hidden_size: int
    num_heads: int
    depth: int
    axes_dim: list[int]
    checkpoint: bool
    patch_size: int = 1
    modulation_rank: int = 0  # 0 preserves the original checkpoint architecture.
    rti_size_buckets: int = 0
    rti_core_start: int = 0
    rti_core_end: int = 0


class MageFlow(nn.Module):
    def __init__(self, params: MageFlowParams):
        super().__init__()
        self.params = params
        self.checkpoint = params.checkpoint
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.inner_dim = params.hidden_size  # num_attention_heads * attention_head_dim
        self.axes_dim = params.axes_dim
        self.num_attention_heads = params.num_heads
        self.attention_head_dim = self.inner_dim // self.num_attention_heads
        self.patch_size = params.patch_size
        assert sum(self.axes_dim) == self.attention_head_dim

        self.pos_embed = MageFlowEmbedRope(
            theta=10000, axes_dim=self.axes_dim, scale_rope=True
        )
        self.img_in = nn.Linear(self.in_channels, self.inner_dim)
        self.txt_norm = RMSNorm(params.context_in_dim, eps=1e-6)
        self.txt_in = nn.Linear(params.context_in_dim, self.inner_dim)

        self.time_text_embed = MageFlowTimestepProjEmbeddings(
            embedding_dim=self.inner_dim
        )

        self.transformer_blocks = nn.ModuleList(
            [
                MageFlowTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                )
                for _ in range(params.depth)
            ]
        )

        if params.modulation_rank:
            from .compressed_modulation import ModulationLinear

            if not 0 < params.modulation_rank <= self.inner_dim:
                raise ValueError("modulation_rank must be in [1, hidden_size]")
            self.modulation_down = ModulationLinear(self.inner_dim, params.modulation_rank)
            for block in self.transformer_blocks:
                for name in ("img_mod", "txt_mod"):
                    setattr(
                        block,
                        name,
                        nn.Sequential(
                            nn.Identity(),
                            ModulationLinear(params.modulation_rank, 6 * self.inner_dim),
                        ),
                    )

        self.region_interface = None
        if params.rti_size_buckets:
            self._validate_rti_span(params.rti_core_start, params.rti_core_end)
            self.region_interface = RegionInterface(
                self.inner_dim, params.rti_size_buckets,
                core_start=params.rti_core_start, core_end=params.rti_core_end,
            )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            self.inner_dim,
            self.patch_size * self.patch_size * self.out_channels,
            bias=True,
        )
        self.configure_execution(params.checkpoint)

    def _validate_rti_span(self, start: int, end: int):
        if not (0 <= start < end < self.params.depth):
            raise ValueError(
                f"RTI core must satisfy 0 <= start < end < depth ({self.params.depth}), got {start}:{end}"
            )

    def configure_rti(self, dense_prefix_blocks: int = 2, dense_suffix_blocks: int = 2, size_buckets: int = 17):
        start, end = int(dense_prefix_blocks), self.params.depth - int(dense_suffix_blocks)
        self._validate_rti_span(start, end)
        if self.region_interface is not None:
            if (self.params.rti_size_buckets, self.params.rti_core_start, self.params.rti_core_end) != (size_buckets, start, end):
                raise ValueError("RTI checkpoint architecture does not match requested RTI settings")
            return self.region_interface
        self.params.rti_size_buckets = int(size_buckets)
        self.params.rti_core_start, self.params.rti_core_end = start, end
        self.region_interface = RegionInterface(self.inner_dim, size_buckets, core_start=start, core_end=end)
        self.region_interface.to(device=self.img_in.weight.device, dtype=self.img_in.weight.dtype)
        return self.region_interface

    def configure_execution(
        self,
        gradient_checkpointing=True,
        checkpoint_blocks=None,
        compile_mode=None,
        compile_dynamic=True,
        attention_backend="sdpa",
    ):
        validate_attention_backend(attention_backend)
        depth = len(self.transformer_blocks)
        if checkpoint_blocks is not None:
            if not gradient_checkpointing:
                raise ValueError("checkpoint_blocks requires gradient_checkpointing")
            if any(
                type(i) is not int or i < 0 or i >= depth for i in checkpoint_blocks
            ) or len(set(checkpoint_blocks)) != len(checkpoint_blocks):
                raise ValueError(
                    f"checkpoint_blocks must be unique indices in [0, {depth - 1}]"
                )
        if compile_mode and attention_backend not in ("sdpa", "torch_varlen"):
            raise ValueError("Block compilation requires SDPA or torch_varlen")
        self.checkpoint_blocks = (
            set(range(depth) if checkpoint_blocks is None else checkpoint_blocks)
            if gradient_checkpointing
            else set()
        )
        self.attention_backend = attention_backend
        self.block_forward = (
            _double_stream_block_forward
            if compile_mode is None
            else torch.compile(
                _double_stream_block_forward,
                mode=compile_mode,
                dynamic=compile_dynamic,
                fullgraph=True,
            )
        )
        self.compiled_blocks = compile_mode is not None

    def add_adapter(self, config):
        from peft import inject_adapter_in_model

        inject_adapter_in_model(config, self)

    def disable_adapters(self):
        from peft.tuners.tuners_utils import BaseTunerLayer

        for layer in self.modules():
            if isinstance(layer, BaseTunerLayer):
                layer.enable_adapters(False)

    def enable_adapters(self):
        from peft.tuners.tuners_utils import BaseTunerLayer

        for layer in self.modules():
            if isinstance(layer, BaseTunerLayer):
                layer.enable_adapters(True)

    def block_condition(self, temb):
        if self.params.modulation_rank:
            return self.modulation_down(nn.functional.silu(temb).to(self.modulation_down.weight.dtype))
        return temb

    def forward(
        self, hidden_states, timestep, encoder_hidden_states, return_dict=False,
        second_timestep=None, timestep_mask=None, keep_fraction=1.0,
    ):
        if isinstance(hidden_states, list):
            return (
                self.forward_packed(hidden_states, timestep, encoder_hidden_states,
                                    second_timestep, timestep_mask, keep_fraction),
            )
        # The data/loss layer retains a singleton frame axis for image tensors.
        if hidden_states.ndim != 5 or hidden_states.shape[2] != 1:
            raise ValueError("Mage-Flow expects [B,128,1,H/16,W/16] image latents")
        image = hidden_states.squeeze(2)
        b, c, h, w = image.shape
        if self.region_interface is not None and b != 1:
            raise ValueError("RTI supports packed native-resolution training; uniform RTI is limited to batch_size=1")
        if c != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} Mage-VAE channels, got {c}; rebuild latent caches"
            )
        text, text_mask = encoder_hidden_states
        img = self.img_in(image.flatten(2).transpose(1, 2))
        txt = self.txt_in(self.txt_norm(text))
        token_ids = None
        if (second_timestep is None) != (timestep_mask is None):
            raise ValueError("Dual conditioning requires both second_timestep and timestep_mask")
        if second_timestep is not None:
            if self.region_interface is not None:
                raise ValueError("RTI and dual timestep conditioning cannot be combined")
            if second_timestep.shape != timestep.shape or timestep_mask.shape != (b, h, w):
                raise ValueError("Dual timestep/mask shape does not match image batch")
            sample_ids = torch.arange(b, device=img.device)[:, None]
            image_ids = sample_ids + timestep_mask.flatten(1).long() * b
            token_ids = (image_ids, sample_ids.expand(b, txt.shape[1]))
            timestep = torch.cat((timestep, second_timestep))
        temb = self.time_text_embed(timestep.to(img.dtype), img)
        block_temb = self.block_condition(temb)
        freqs = self.pos_embed([(1, h, w)], device=img.device)
        if self.compiled_blocks:
            freqs = torch.view_as_real(freqs)
        mask = torch.cat(
            (
                text_mask.bool(),
                torch.ones(b, h * w, device=img.device, dtype=torch.bool),
            ),
            dim=1,
        )[:, None, None, :]
        metadata = (
            () if self.attention_backend == "sdpa" else packed_attention_metadata(mask)
        )
        dense_img = dense_freqs = region_in = plan = None
        for i, block in enumerate(self.transformer_blocks):
            if self.region_interface is not None and i == self.region_interface.core_start:
                dense_img, dense_freqs = img, freqs
                plan = build_region_plan(img[0].detach(), [(h, w)], [h * w], keep_fraction, img.device)
                img, freqs = self.region_interface.read(img, dense_freqs if not self.compiled_blocks else torch.view_as_complex(dense_freqs), plan)
                region_in = img
                if self.compiled_blocks:
                    freqs = torch.view_as_real(freqs)
                token_ids = (torch.zeros(plan.counts.numel(), device=img.device, dtype=torch.long), torch.zeros(txt.shape[1], device=img.device, dtype=torch.long))
                mask = None
                metadata = (torch.arange(txt.shape[1] + plan.counts.numel(), device=img.device), torch.tensor([0, txt.shape[1] + plan.counts.numel()], device=img.device, dtype=torch.int32), txt.shape[1] + plan.counts.numel())
            args = (
                block,
                img,
                txt,
                block_temb,
                freqs,
                mask,
                self.num_attention_heads,
                self.attention_backend,
                metadata,
                False,
                token_ids,
            )
            if self.training and i in self.checkpoint_blocks:
                txt, img = checkpoint(self.block_forward, *args, use_reentrant=False)
            else:
                txt, img = self.block_forward(*args)
            if self.region_interface is not None and i + 1 == self.region_interface.core_end:
                img = self.region_interface.write(dense_img, img, region_in, plan)
                freqs = dense_freqs
                token_ids = None
                mask = torch.cat((text_mask.bool(), torch.ones(b, h * w, device=img.device, dtype=torch.bool)), dim=1)[:, None, None, :]
                metadata = () if self.attention_backend == "sdpa" else packed_attention_metadata(mask)
        scale, shift = self.norm_out.linear(
            self.norm_out.silu(temb).to(img.dtype)
        ).chunk(2, dim=-1)
        if token_ids is None:
            img = self.norm_out.norm(img) * (1 + scale[:, None]) + shift[:, None]
        else:
            img = self.norm_out.norm(img) * (1 + scale[image_ids]) + shift[image_ids]
        result = (
            self.proj_out(img).transpose(1, 2).reshape(b, self.out_channels, 1, h, w)
        )
        return (result,)

    def forward_packed(self, images, timestep, context, second_timestep=None, timestep_mask=None, keep_fraction=1.0):
        """Pack heterogeneous native resolutions through the entire MMDiT.

        Each image retains its own RoPE origin and timestep modulation. Joint
        attention boundaries isolate samples; loss reduction is done per image.
        """
        if self.attention_backend == "sdpa":
            raise ValueError(
                "Packed resolutions require torch_varlen or FlashAttention"
            )
        text, text_mask = context
        if len(images) != text.shape[0] or len(images) != timestep.shape[0]:
            raise ValueError("Packed image, caption and timestep counts must match")
        shapes = []
        for im in images:
            if im.ndim != 5 or im.shape[:3] != (1, self.in_channels, 1):
                raise ValueError("Each packed latent must have shape [1,128,1,h,w]")
            shapes.append(im.shape[-2:])
        device = images[0].device
        image_lengths = [h * w for h, w in shapes]
        # One synchronization per microbatch, outside checkpointed/compiled blocks.
        text_lengths = text_mask.sum(1).tolist()
        if any(n == 0 for n in text_lengths):
            raise ValueError("Packed captions must contain at least one valid token")
        txt = text[text_mask.bool()].unsqueeze(0)
        img = torch.cat([im.flatten(2).transpose(1, 2) for im in images], dim=1)
        img = self.img_in(img)
        txt = self.txt_in(self.txt_norm(txt))
        if (second_timestep is None) != (timestep_mask is None):
            raise ValueError("Dual conditioning requires both second_timestep and timestep_mask")
        if second_timestep is not None:
            if self.region_interface is not None:
                raise ValueError("RTI and dual timestep conditioning cannot be combined")
            if second_timestep.shape != timestep.shape or len(timestep_mask) != len(images):
                raise ValueError("Dual timestep/mask shape does not match packed batch")
            if any(m.shape != (1, *shape) for m, shape in zip(timestep_mask, shapes)):
                raise ValueError("Each packed timestep mask must match its image token grid")
            timestep = torch.cat((timestep, second_timestep))
        temb = self.time_text_embed(timestep.to(img.dtype), img)
        block_temb = self.block_condition(temb)
        img_ids = torch.tensor(
            [i for i, n in enumerate(image_lengths) for _ in range(n)], device=device
        )
        txt_ids = torch.tensor(
            [i for i, n in enumerate(text_lengths) for _ in range(n)], device=device
        )
        if second_timestep is not None:
            img_ids = img_ids + torch.cat([m.flatten().long() for m in timestep_mask]) * len(images)
        # Call separately: multiple reference images in upstream RoPE use different
        # frame offsets, while independent training images must each start at zero.
        freqs = torch.cat(
            [self.pos_embed([(1, h, w)], device=device) for h, w in shapes]
        )
        if self.compiled_blocks:
            freqs = torch.view_as_real(freqs)
        # Block tensors keep [all text, all images]. Gather attention into
        # [text_0,image_0,text_1,image_1,...] and scatter back after the kernel.
        indices, boundaries = [], [0]
        toff, ioff = 0, sum(text_lengths)
        for nt, ni in zip(text_lengths, image_lengths):
            indices.extend(range(toff, toff + nt))
            indices.extend(range(ioff, ioff + ni))
            boundaries.append(boundaries[-1] + nt + ni)
            toff += nt
            ioff += ni
        metadata = (
            torch.tensor(indices, device=device),
            torch.tensor(boundaries, device=device, dtype=torch.int32),
            max(a + b for a, b in zip(image_lengths, text_lengths)),
        )
        dense_img = dense_freqs = region_in = plan = None
        for i, block in enumerate(self.transformer_blocks):
            if self.region_interface is not None and i == self.region_interface.core_start:
                dense_img, dense_freqs = img, freqs
                plan = build_region_plan(img[0].detach(), shapes, image_lengths, keep_fraction, device)
                raw_freqs = torch.view_as_complex(freqs) if self.compiled_blocks else freqs
                img, region_freqs = self.region_interface.read(img, raw_freqs, plan)
                region_in = img
                freqs = torch.view_as_real(region_freqs) if self.compiled_blocks else region_freqs
                img_ids = torch.repeat_interleave(torch.arange(len(images), device=device), torch.tensor(plan.region_lengths, device=device))
                metadata = _packed_varlen_metadata(text_lengths, plan.region_lengths, device)
            args = (
                block,
                img,
                txt,
                block_temb,
                freqs,
                None,
                self.num_attention_heads,
                self.attention_backend,
                metadata,
                False,
                (img_ids, txt_ids),
            )
            if self.training and i in self.checkpoint_blocks:
                txt, img = checkpoint(self.block_forward, *args, use_reentrant=False)
            else:
                txt, img = self.block_forward(*args)
            if self.region_interface is not None and i + 1 == self.region_interface.core_end:
                img = self.region_interface.write(dense_img, img, region_in, plan)
                freqs = dense_freqs
                img_ids = torch.tensor([j for j, n in enumerate(image_lengths) for _ in range(n)], device=device)
                metadata = _packed_varlen_metadata(text_lengths, image_lengths, device)
        scale, shift = self.norm_out.linear(
            self.norm_out.silu(temb).to(img.dtype)
        ).chunk(2, -1)
        img = (
            self.norm_out.norm(img) * (1 + scale.index_select(0, img_ids)[None])
            + shift.index_select(0, img_ids)[None]
        )
        output = self.proj_out(img)
        chunks = output.split(image_lengths, dim=1)
        return [
            v.transpose(1, 2).reshape(1, self.out_channels, 1, h, w)
            for v, (h, w) in zip(chunks, shapes)
        ]
