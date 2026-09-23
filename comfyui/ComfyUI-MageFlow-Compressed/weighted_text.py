# SPDX-License-Identifier: GPL-3.0-or-later
"""Weighted text-to-image conditioning without changing the Mage-Flow encoder."""

import math

from comfy import sd1_clip
from comfy.text_encoders.mage_flow import MageFlowTokenizer, MAGE_T2I_TEMPLATE


def weighted_tokens(tokenizer, text):
    if not isinstance(tokenizer, MageFlowTokenizer):
        raise ValueError("Use Load CLIP with type mage for Mage-Flow weighted text.")

    segments = sd1_clip.token_weights(sd1_clip.escape_important(text), 1.0)
    segments = [(sd1_clip.unescape_important(s), w) for s, w in segments]
    if any(not math.isfinite(w) for _, w in segments):
        raise ValueError("Prompt weights must be finite numbers.")
    clean = "".join(s for s, _ in segments)
    if "<|" in clean:
        raise ValueError("Use plain prompt text, without chat or image special tokens.")

    prefix, suffix = MAGE_T2I_TEMPLATE.split("{}")
    segments = [(prefix, 1.0), *segments, (suffix, 1.0)]
    encoded = "".join(s for s, _ in segments).encode("utf-8")
    byte_weights = [w for s, w in segments for _ in s.encode("utf-8")]
    qwen = tokenizer.qwen3vl_4b.tokenizer
    ids = qwen(MAGE_T2I_TEMPLATE.format(clean))["input_ids"]
    special_ids = set(qwen.all_special_ids)
    # Qwen's byte-level BPE alphabet (also works with Transformers' fast tokenizer).
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    hidden = [b for b in range(256) if b not in visible]
    byte_decoder = {chr(b): b for b in visible}
    byte_decoder.update({chr(256 + i): b for i, b in enumerate(hidden)})
    pairs = []
    offset = 0
    for token_id in ids:
        token = qwen.convert_ids_to_tokens(token_id)
        raw = token.encode("utf-8") if token_id in special_ids else bytes(byte_decoder[c] for c in token)
        end = offset + len(raw)
        if encoded[offset:end] != raw:
            raise ValueError("Cannot align this Qwen tokenizer with the weighted prompt.")
        # BPE can merge across a weighting boundary: average over the token's bytes.
        weight = sum(byte_weights[offset:end]) / len(raw)
        pairs.append((token_id, weight))
        offset = end
    if offset != len(encoded):
        raise ValueError("Qwen tokenization did not cover the entire prompt.")
    return {"qwen3vl_4b": [pairs]}


class MageFlowWeightedTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/mage-flow"
    DESCRIPTION = "Experimental Mage-Flow prompt weighting: (night sky:1.5). Text-to-image only."

    def encode(self, clip, text):
        tokens = weighted_tokens(clip.tokenizer, text)
        return (clip.encode_from_tokens_scheduled(tokens),)
