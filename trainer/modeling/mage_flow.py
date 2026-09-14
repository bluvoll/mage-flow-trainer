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

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            self.inner_dim,
            self.patch_size * self.patch_size * self.out_channels,
            bias=True,
        )
        self.configure_execution(params.checkpoint)

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

    def forward(
        self, hidden_states, timestep, encoder_hidden_states, return_dict=False
    ):
        if isinstance(hidden_states, list):
            return (
                self.forward_packed(hidden_states, timestep, encoder_hidden_states),
            )
        # The data/loss layer retains a singleton frame axis for image tensors.
        if hidden_states.ndim != 5 or hidden_states.shape[2] != 1:
            raise ValueError("Mage-Flow expects [B,128,1,H/16,W/16] image latents")
        image = hidden_states.squeeze(2)
        b, c, h, w = image.shape
        if c != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} Mage-VAE channels, got {c}; rebuild latent caches"
            )
        text, text_mask = encoder_hidden_states
        img = self.img_in(image.flatten(2).transpose(1, 2))
        txt = self.txt_in(self.txt_norm(text))
        temb = self.time_text_embed(timestep.to(img.dtype), img)
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
        for i, block in enumerate(self.transformer_blocks):
            args = (
                block,
                img,
                txt,
                temb,
                freqs,
                mask,
                self.num_attention_heads,
                self.attention_backend,
                metadata,
                False,
            )
            if self.training and i in self.checkpoint_blocks:
                txt, img = checkpoint(self.block_forward, *args, use_reentrant=False)
            else:
                txt, img = self.block_forward(*args)
        scale, shift = self.norm_out.linear(
            self.norm_out.silu(temb).to(img.dtype)
        ).chunk(2, dim=-1)
        img = self.norm_out.norm(img) * (1 + scale[:, None]) + shift[:, None]
        result = (
            self.proj_out(img).transpose(1, 2).reshape(b, self.out_channels, 1, h, w)
        )
        return (result,)

    def forward_packed(self, images, timestep, context):
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
        temb = self.time_text_embed(timestep.to(img.dtype), img)
        img_ids = torch.tensor(
            [i for i, n in enumerate(image_lengths) for _ in range(n)], device=device
        )
        txt_ids = torch.tensor(
            [i for i, n in enumerate(text_lengths) for _ in range(n)], device=device
        )
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
        for i, block in enumerate(self.transformer_blocks):
            args = (
                block,
                img,
                txt,
                temb,
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
