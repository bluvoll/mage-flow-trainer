"""Batched Mage-Flow kernels adapted from diffusion-pipe 40bf63a (GPL-3.0)."""

import torch
import torch.nn.functional as F
from .mageflow_attention import packed_attention

PROMPT_TEMPLATE_ENCODE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_ENCODE_START_IDX = 34


def _apply_rope_batched(x, freqs_complex):
    """Apply MageFlow 2D multi-scale RoPE to a batched tensor.

    Args:
        x: [B, H, L, Dh]
        freqs_complex: [L, Dh//2] complex
    """
    if not freqs_complex.is_complex():
        # Compiled blocks receive real/imaginary pairs prepared outside the
        # graph. Real arithmetic avoids Inductor's complex-number fallback.
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        real, imag = pairs.unbind(-1)
        cos, sin = freqs_complex.unbind(-1)
        return (
            torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1)
            .flatten(-2)
            .type_as(x)
        )
    x_c = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )  # [B, H, L, Dh/2]
    freqs = freqs_complex.view(1, 1, freqs_complex.shape[0], freqs_complex.shape[1])
    x_out = torch.view_as_real(x_c * freqs).flatten(-2)  # [B, H, L, Dh]
    return x_out.type_as(x)


def _modulate(x, mod):
    """adaLN modulation for a [B, L, D] tensor from [B, 3*D] params.

    Returns (modulated_x, gate) with gate shaped [B, 1, D] for the residual.
    """
    shift, scale, gate = mod.chunk(3, dim=-1)  # each [B, D]
    if mod.ndim == 2:
        shift, scale, gate = shift.unsqueeze(1), scale.unsqueeze(1), gate.unsqueeze(1)
    return x * (1 + scale) + shift, gate


def _double_stream_block_forward(
    block,
    hidden_states,
    encoder_hidden_states,
    temb,
    img_freqs,
    attn_mask,
    num_heads,
    attention_backend="sdpa",
    attention_metadata=(),
    deterministic=False,
    token_sample_ids=None,
):
    """Batched (SDPA) reimplementation of MageFlowTransformerBlock.forward.

    Reuses the block's pretrained submodules but drives them with real batched
    [B, L, D] tensors and a joint [text, image] SDPA (padded text + key mask)
    instead of the upstream varlen path. Returns (encoder_hidden_states,
    hidden_states) to match the upstream (txt, img) return order.
    """
    attn = block.attn

    img_mod1, img_mod2 = block.img_mod(temb).chunk(2, dim=-1)  # each [B, 3*dim]
    txt_mod1, txt_mod2 = block.txt_mod(temb).chunk(2, dim=-1)

    if token_sample_ids is not None:
        img_ids, txt_ids = token_sample_ids
        img_mod1, img_mod2 = (
            m.index_select(0, img_ids).unsqueeze(0) for m in (img_mod1, img_mod2)
        )
        txt_mod1, txt_mod2 = (
            m.index_select(0, txt_ids).unsqueeze(0) for m in (txt_mod1, txt_mod2)
        )

    # --- norm1 + modulation ---
    img_modulated, img_gate1 = _modulate(block.img_norm1(hidden_states), img_mod1)
    txt_modulated, txt_gate1 = _modulate(
        block.txt_norm1(encoder_hidden_states), txt_mod1
    )

    # --- joint attention (order: [text, image]) ---
    B, Li, _ = img_modulated.shape
    Lt = txt_modulated.shape[1]

    def _proj(x, q_proj, k_proj, v_proj, nq, nk):
        q = q_proj(x).unflatten(-1, (num_heads, -1))
        k = k_proj(x).unflatten(-1, (num_heads, -1))
        v = v_proj(x).unflatten(-1, (num_heads, -1))
        if nq is not None:
            q = nq(q)
        if nk is not None:
            k = nk(k)
        # [B, L, H, Dh] -> [B, H, L, Dh]
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    iq, ik, iv = _proj(
        img_modulated, attn.to_q, attn.to_k, attn.to_v, attn.norm_q, attn.norm_k
    )
    tq, tk, tv = _proj(
        txt_modulated,
        attn.add_q_proj,
        attn.add_k_proj,
        attn.add_v_proj,
        attn.norm_added_q,
        attn.norm_added_k,
    )

    # RoPE on image tokens only (text is not rotated in MageFlow).
    iq = _apply_rope_batched(iq, img_freqs)
    ik = _apply_rope_batched(ik, img_freqs)

    q = torch.cat([tq, iq], dim=2)
    k = torch.cat([tk, ik], dim=2)
    v = torch.cat([tv, iv], dim=2)

    # softmax_scale=None upstream -> flash default 1/sqrt(head_dim); SDPA default matches.
    if attention_backend == "sdpa":
        joint = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    else:
        joint = packed_attention(
            q,
            k,
            v,
            *attention_metadata,
            backend=attention_backend,
            deterministic=deterministic,
        )
    joint = joint.transpose(1, 2).flatten(2)  # [B, Lt+Li, dim]

    txt_attn_output = attn.to_add_out(joint[:, :Lt])
    img_attn_output = attn.to_out[0](joint[:, Lt:])
    img_attn_output = attn.to_out[1](
        img_attn_output
    )  # dropout (no-op in eval/train p=0)

    hidden_states = hidden_states + img_gate1 * img_attn_output
    encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

    # --- norm2 + MLP ---
    img_modulated2, img_gate2 = _modulate(block.img_norm2(hidden_states), img_mod2)
    hidden_states = hidden_states + img_gate2 * block.img_mlp(img_modulated2)

    txt_modulated2, txt_gate2 = _modulate(
        block.txt_norm2(encoder_hidden_states), txt_mod2
    )
    encoder_hidden_states = encoder_hidden_states + txt_gate2 * block.txt_mlp(
        txt_modulated2
    )

    return encoder_hidden_states, hidden_states
