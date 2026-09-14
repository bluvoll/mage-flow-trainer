import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from test_mageflow import tiny, inputs
from trainer.training.params import AdapterConfig, apply_adapter, classify
from trainer.training.lycoris import lycoris_state_dict, load_lycoris_state_dict
from trainer.modeling.export import export_checkpoint


class LycorisTests(unittest.TestCase):
    def test_bf16_export_preserves_alpha_precision(self):
        model = tiny()
        apply_adapter(
            model,
            AdapterConfig(kind="lycoris_lora", rank=2, alpha=1.7, dtype="bfloat16"),
        )
        with tempfile.TemporaryDirectory() as folder:
            export_checkpoint(
                model,
                folder,
                "test",
                SimpleNamespace(is_lora=True, train=SimpleNamespace(run_name="test")),
                torch.bfloat16,
                1,
            )
            state = load_file(str(Path(folder) / "test.safetensors"))
            for key, value in state.items():
                if key.endswith(".alpha"):
                    self.assertEqual(value.dtype, torch.float32)
                    self.assertAlmostEqual(value.item(), 1.7, places=6)

    def test_targets_gradients_export_reload(self):
        model = tiny()
        cfg = AdapterConfig(kind="lycoris_lora", rank=2, alpha=2)
        apply_adapter(model, cfg)
        names = [n for n, _ in model.named_modules() if n.endswith(".lycoris_adapter")]
        self.assertEqual(len(names), 12 * len(model.transformer_blocks) - 4)
        self.assertEqual(
            {classify(n + ".weight") for n in names}, {"image_attn", "text_attn", "mlp"}
        )
        x = inputs()
        model(**x)[0].square().mean().backward()
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for n, p in model.named_parameters()
                if "lora_up" in n
            )
        )
        for n, p in model.named_parameters():
            if ".lycoris_adapter." not in n:
                self.assertFalse(p.requires_grad, n)
                self.assertIsNone(p.grad, n)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_up" in n:
                    p.add_(0.01)
        with tempfile.TemporaryDirectory() as folder:
            export_checkpoint(
                model,
                folder,
                "test",
                SimpleNamespace(is_lora=True, train=SimpleNamespace(run_name="test")),
                torch.float32,
                1,
            )
            restored = tiny()
            apply_adapter(restored, cfg)
            state = load_file(str(Path(folder) / "test.safetensors"))
            self.assertTrue(all("lycoris_adapter" not in k for k in state))
            load_lycoris_state_dict(restored, state)
            torch.testing.assert_close(model(**x)[0], restored(**x)[0])
            # The ordinary model state dict also includes adapters for Accelerate resume.
            restored.load_state_dict(model.state_dict(), strict=True)
            self.assertEqual(set(lycoris_state_dict(restored)), set(state))
        from accelerate import Accelerator

        acc = Accelerator(cpu=True)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad]
        )
        wrapped, optimizer = acc.prepare(model, optimizer)
        optimizer.step()
        expected = wrapped(**x)[0].detach().clone()
        with tempfile.TemporaryDirectory() as folder:
            acc.save_state(folder)
            with torch.no_grad():
                for p in wrapped.parameters():
                    if p.requires_grad:
                        p.add_(1)
            acc.load_state(folder)
            torch.testing.assert_close(wrapped(**x)[0], expected)
        acc.free_memory()
        with self.assertRaisesRegex(ValueError, "AdaLN"):
            AdapterConfig(kind="lycoris_lora", components=["adaln"])

    @unittest.skipUnless(os.environ.get("MAGE_LYCORIS_GPU") == "1", "GPU opt-in")
    def test_sdnq_compiled_checkpointed_backward(self):
        from trainer.training.quant import QuantConfig, quantize_module

        model = tiny().to("cuda", torch.bfloat16)
        model = quantize_module(
            model,
            QuantConfig(
                mode="frozen", weights_dtype="int8", use_quantized_matmul=False
            ),
            torch.device("cuda"),
            torch.bfloat16,
            False,
        )
        apply_adapter(
            model,
            AdapterConfig(
                kind="lycoris_lora",
                rank=2,
                alpha=2,
                dtype=os.environ.get("MAGE_LYCORIS_DTYPE", "bfloat16"),
            ),
        )
        x = inputs("cuda", torch.bfloat16)
        eager = model(**x)[0]
        eager.float().square().mean().backward()
        gradients = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad
        }
        model.zero_grad(set_to_none=True)
        model.configure_execution(
            True,
            compile_mode="default",
            compile_dynamic=os.environ.get("MAGE_LYCORIS_DYNAMIC") == "1",
            attention_backend="torch_varlen",
        )
        result = model(**x)[0]
        result.float().square().mean().backward()
        torch.testing.assert_close(result, eager, atol=0.02, rtol=0.03)
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.assertTrue(torch.isfinite(p.grad).all(), n)
                torch.testing.assert_close(
                    p.grad, gradients[n], atol=3e-4, rtol=0.1, msg=n
                )
        self.assertTrue(model.compiled_blocks)
        if os.environ.get("MAGE_LYCORIS_DYNAMIC") == "1":
            changed = dict(x)
            changed["hidden_states"] = torch.randn(
                2, 128, 1, 4, 5, device="cuda", dtype=torch.bfloat16
            )
            model.zero_grad(set_to_none=True)
            model(**changed)[0].float().square().mean().backward()
            self.assertTrue(
                all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.requires_grad
                )
            )
        from trainer.training.optim import build_optimizer
        from trainer.training.config import OptimizerConfig

        trainable = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in trainable]
        opt = build_optimizer(
            [{"params": trainable, "lr": 1e-3}],
            OptimizerConfig(kind="adamw8bit", quantize_state=True),
        )
        opt.step()
        self.assertTrue(
            any(not torch.equal(p, old) for p, old in zip(trainable, before))
        )
