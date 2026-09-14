"""Fixed sample-count batches spanning native image resolutions."""

import random
from .dataset import BucketBatchSampler


class NativeResolutionBatchSampler(BucketBatchSampler):
    def _batches(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        n = self._flat
        return [
            indices[i : i + n]
            for i in range(0, len(indices), n)
            if not self.drop_last or len(indices[i : i + n]) == n
        ]

    def __len__(self):
        n = len(self.dataset)
        return n // self._flat if self.drop_last else (n + self._flat - 1) // self._flat


def collate_native(batch):
    key = "pixels" if "pixels" in batch[0] else "latents"
    return {
        key: [item[key].unsqueeze(0) for item in batch],
        "captions": [item["caption"] for item in batch],
        "bucket": [item["bucket"] for item in batch],
        "paths": [item["path"] for item in batch],
    }
