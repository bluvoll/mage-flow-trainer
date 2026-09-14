"""CPU-resident text embeddings for runs with fixed captions."""

import torch
from torch.nn.utils.rnn import pad_sequence


def validate_static_captions(cfg):
    caption = cfg.dataset.caption
    changing = [
        name
        for name in (
            "shuffle_tags",
            "tag_dropout_percent",
            "caption_dropout_percent",
            "nl_shuffle_sentences",
        )
        if getattr(caption, name)
    ]
    if caption.caption_mode == "mixed":
        changing.append("caption_mode=mixed")
    if changing and not getattr(getattr(cfg, "train", None), "caption_variations", 0):
        raise ValueError(
            "cache_text_embeddings requires fixed captions; disable "
            + ", ".join(changing)
        )
    if any(p.mode == "texture" for p in cfg.curriculum.phases):
        raise ValueError(
            "cache_text_embeddings currently does not support texture curricula"
        )
    if cfg.preserve.enabled:
        raise ValueError(
            "cache_text_embeddings currently does not support preservation probes"
        )


class TextEmbeddingCache:
    def __init__(self):
        self.embeddings = {}

    def add(self, caption, hidden, mask):
        # Clone after moving to CPU so neither CUDA tensors nor padded backing storage survive.
        self.embeddings[caption] = (
            hidden[0, mask[0].bool()].detach().cpu().contiguous().clone()
        )

    def get(self, captions, device, dtype):
        values = [self.embeddings[c] for c in captions]
        hidden = pad_sequence(values, batch_first=True).to(device=device, dtype=dtype)
        lengths = torch.tensor([v.shape[0] for v in values], device=device)
        mask = torch.arange(hidden.shape[1], device=device)[None] < lengths[:, None]
        return hidden, mask
