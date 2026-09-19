"""Embedding-only Qwen3-VL execution, shared by caching and live captions."""

import torch


@torch.no_grad()
def encode_text_hidden(text_encoder, input_ids, attention_mask, *, embedding_only=False):
    if embedding_only:
        hidden = text_encoder.model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, output_hidden_states=False, return_dict=True,
        ).last_hidden_state
        # Verify once per encoder instance: Transformers versions can differ
        # in which normalization boundary hidden_states[-1] captures.
        if not getattr(text_encoder, '_mage_embedding_only_verified', False):
            reference = encode_text_hidden(text_encoder, input_ids, attention_mask)
            if not torch.equal(hidden, reference):
                raise RuntimeError(
                    'Embedding-only text encoding differs from the original wrapper. '
                    'Set train.text_encoder_embedding_only=false for this Transformers version.'
                )
            text_encoder._mage_embedding_only_verified = True
        return hidden
    # Keep the original wrapper: Transformers output-capture decorators can
    # return pre-norm hidden_states[-1] here but post-norm states on .model.
    # logits_to_keep=1 reduces the unused [B,L,vocab] projection to [B,1,vocab].
    # A single token also avoids empty-matmul edge cases in quantized backends.
    return text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=1,
        output_hidden_states=True,
        return_dict=True,
    ).hidden_states[-1]
