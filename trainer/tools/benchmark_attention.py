"""Compare native Torch varlen and SDPA joint attention, including backward.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m trainer.tools.benchmark_attention
This measures the attention operation and pack/scatter overhead, not full training.
"""

import argparse
import json
import statistics
import time
import torch
from ..modeling.mageflow_attention import packed_attention, packed_attention_metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--output")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = []
    for lengths in ([4160, 4608], [1088, 4608], [1088, 2112, 4160]):
        b, length = len(lengths), max(lengths)
        torch.manual_seed(42)
        tensors = [
            torch.randn(
                b,
                24,
                length,
                128,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for _ in range(3)
        ]
        mask = (
            torch.arange(length, device="cuda")[None, :]
            < torch.tensor(lengths, device="cuda")[:, None]
        )[:, None, None, :]
        metadata = packed_attention_metadata(mask)
        grad = torch.randn_like(tensors[0]) * mask.transpose(-1, -2)
        for backend in ("sdpa", "torch_varlen"):
            durations = []
            torch.cuda.reset_peak_memory_stats()
            for i in range(args.iterations + 3):
                for t in tensors:
                    t.grad = None
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = (
                    torch.nn.functional.scaled_dot_product_attention(
                        *tensors, attn_mask=mask
                    )
                    if backend == "sdpa"
                    else packed_attention(*tensors, *metadata, backend=backend)
                )
                out.backward(grad)
                torch.cuda.synchronize()
                if i >= 3:
                    durations.append((time.perf_counter() - start) * 1000)
            rows.append(
                dict(
                    backend=backend,
                    lengths=lengths,
                    median_forward_backward_ms=statistics.median(durations),
                    peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                )
            )
        del out, grad, tensors
    result = dict(
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), results=rows
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
