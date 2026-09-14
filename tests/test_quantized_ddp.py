"""Two-rank GPU regression for SDNQ initialization and synchronized updates."""

import os
import unittest

import torch
import torch.distributed as dist

from trainer.training.distributed import (
    configure_quantized_ddp,
    model_storage,
    quantized_optimizer_rng,
    sync_quantized_model,
)


class UpdateRandomnessTests(unittest.TestCase):
    def test_update_rng_is_shared_without_changing_data_rng(self):
        draws = []
        for rank_seed in (41, 42):
            torch.manual_seed(rank_seed)
            before = torch.get_rng_state().clone()
            with quantized_optimizer_rng("cpu", 123, 4):
                draws.append(torch.rand(32))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(*draws))
        with quantized_optimizer_rng("cpu", 123, 5):
            self.assertFalse(torch.equal(draws[0], torch.rand(32)))


@unittest.skipUnless(
    os.environ.get("MAGEFLOW_GPU_TESTS") == "1" and os.environ.get("WORLD_SIZE") == "2",
    "requires two-rank GPU opt-in",
)
class QuantizedDistributedTests(unittest.TestCase):
    def test_compiled_quantized_training_keeps_replicas_identical(self):
        from trainer.training.quant import QuantConfig, quantize_module
        from trainer.training.config import OptimizerConfig
        from trainer.training.optim import build_optimizer

        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group("nccl")
        try:
            torch.manual_seed(100 + rank)
            model = torch.nn.Sequential(
                torch.nn.Linear(128, 384),
                torch.nn.GELU(),
                torch.nn.Linear(384, 128),
            ).to(device=device, dtype=torch.bfloat16)
            model = quantize_module(
                model, QuantConfig(mode="training"), device, torch.bfloat16, False
            )
            sync_quantized_model(model)

            def check_equal():
                for name, tensor in model_storage(model):
                    reference = tensor.detach().clone().contiguous()
                    dist.broadcast(reference, src=0)
                    self.assertTrue(torch.equal(reference, tensor), name)

            check_equal()
            model.forward = torch.compile(model.forward, fullgraph=True)
            ddp = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[rank],
                **configure_quantized_ddp().to_kwargs(),
            )
            optimizer = build_optimizer(
                [{"params": list(model.parameters()), "lr": 1e-3}],
                OptimizerConfig(
                    kind="adamw8bit",
                    quantize_state=True,
                    offload_state=True,
                    use_kahan=True,
                ),
            )
            first = next(model.parameters()).dequantize().clone()
            for step in range(3):
                loss = (
                    ddp(torch.randn(2, 128, device=device, dtype=torch.bfloat16))
                    .float()
                    .square()
                    .mean()
                )
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                rng_before = torch.cuda.get_rng_state(device).clone()
                with quantized_optimizer_rng(device, 456, step):
                    optimizer.step()
                self.assertTrue(
                    torch.equal(rng_before, torch.cuda.get_rng_state(device))
                )
                optimizer.zero_grad(set_to_none=True)
                check_equal()
            self.assertFalse(torch.equal(first, next(model.parameters()).dequantize()))
        finally:
            dist.destroy_process_group()
