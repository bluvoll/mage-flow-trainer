"""Optional packed attention for MageFlow; importing this needs only PyTorch."""

from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=None)
def resolve_flash_attention(backend):
    """Resolve explicitly requested kernels. Never silently fall back."""
    if backend == "torch_varlen":
        from torch.nn.attention.varlen import varlen_attn

        return varlen_attn
    if backend == "flash_attn_2":
        from flash_attn import flash_attn_varlen_func
    elif backend == "flash_attn_3":
        from flash_attn_3.flash_attn_interface import flash_attn_varlen_func
    else:
        raise ValueError(f"Unsupported MageFlow attention_backend: {backend!r}")
    return flash_attn_varlen_func


def validate_attention_backend(backend, pipeline_stages=1):
    if backend not in ("sdpa", "torch_varlen", "flash_attn_2", "flash_attn_3"):
        raise ValueError(
            "MageFlow attention_backend must be sdpa, torch_varlen, flash_attn_2, or flash_attn_3"
        )
    if backend != "sdpa":
        # Packed token counts can vary between microbatches of the same step.
        # DeepSpeed's current pipeline transport allocates receive buffers once
        # per step. Until that path is validated, keep packing on a single stage.
        if pipeline_stages != 1:
            raise ValueError(
                "MageFlow packed attention currently requires pipeline_stages=1"
            )
        try:
            resolve_flash_attention(backend)
        except ImportError as exc:
            raise ImportError(
                f"MageFlow attention_backend={backend!r} is unavailable. "
                'Install its matching CUDA package or use attention_backend="sdpa".'
            ) from exc


def packed_attention_metadata(key_mask):
    """Build once per microbatch, outside checkpointed transformer blocks.

    The mask is [B, 1, 1, L], True for valid text/image tokens. Returned
    integer tensors carry no autograd graph. nonzero synchronizes CUDA here,
    but is not repeated for each block or checkpoint recomputation.
    """
    keep = key_mask.reshape(key_mask.shape[0], -1)
    indices = keep.flatten().nonzero(as_tuple=False).flatten()
    lengths = keep.sum(dim=1, dtype=torch.int32)
    cu_seqlens = F.pad(lengths.cumsum(dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens


def packed_attention(
    q, k, v, indices, cu_seqlens, max_seqlen=None, *, backend, deterministic=False
):
    """[B,H,L,D] attention with padding removed from both queries and keys.

    Padded query outputs are zero. They never enter the image loss or serve
    as valid keys in subsequent blocks. Valid tokens retain SDPA semantics.
    """
    if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("MageFlow FlashAttention requires CUDA FP16/BF16 Q/K/V")
    fn = resolve_flash_attention(backend)
    return _packed_attention(
        q, k, v, indices, cu_seqlens, fn, deterministic, backend, max_seqlen
    )


def _packed_attention(
    q, k, v, indices, cu_seqlens, fn, deterministic=False, backend=None, max_seqlen=None
):
    """Layout conversion shared by real kernels and numerical reference tests."""
    batch, heads, length, dim = q.shape

    def pack(x):
        return (
            x.transpose(1, 2)
            .reshape(batch * length, heads, dim)
            .index_select(0, indices)
        )

    # Padded length is a safe upper bound: avoids a GPU->CPU max().item().
    args = (
        pack(q),
        pack(k),
        pack(v),
        cu_seqlens,
        cu_seqlens,
        max_seqlen or length,
        max_seqlen or length,
    )
    if backend == "torch_varlen":
        if deterministic:
            raise ValueError(
                "torch_varlen does not expose deterministic backward in the pinned Torch"
            )
        out = fn(*args)
    else:
        out = fn(*args, causal=False, deterministic=deterministic)
    padded = out.new_zeros(batch * length, heads, dim).index_copy(0, indices, out)
    return padded.view(batch, length, heads, dim).transpose(1, 2)
