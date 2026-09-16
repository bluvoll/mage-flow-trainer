import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from trainer.modeling.compressed_modulation import (
    compressed_parameter,
    initialize_compression,
    set_modulation_dtype,
)
from trainer.modeling.export import export_checkpoint
from trainer.modeling.loader import load_components
from trainer.modeling.mage_flow import MageFlow, MageFlowParams
from trainer.training.params import (
    AdapterConfig,
    ComponentLRs,
    apply_adapter,
    build_param_groups,
)


class CompressedModulationTests(unittest.TestCase):
    def test_selectable_precision_gradients_and_export(self):
        model = MageFlow(MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], False, modulation_rank=8))
        set_modulation_dtype(model, torch.bfloat16)
        model.float()  # Trunk conversions must preserve the selected factor dtype.
        z = model.block_condition(torch.randn(2, 32))
        loss = sum(getattr(block, stream)(z).float().square().mean()
                   for block in model.transformer_blocks for stream in ("img_mod", "txt_mod"))
        loss.backward()
        for name, param in model.named_parameters():
            if compressed_parameter(name):
                self.assertEqual(param.dtype, torch.bfloat16)
                self.assertEqual(param.grad.dtype, torch.bfloat16)
        cfg = SimpleNamespace(is_lora=False, quant=SimpleNamespace(mode="none"),
                              train=SimpleNamespace(run_name="tiny", save_native=True))
        with tempfile.TemporaryDirectory() as folder:
            export_checkpoint(model, folder, "bf16", cfg, torch.bfloat16, 0)
            for dtype in ("float32", "bfloat16"):
                loaded = load_components("/missing", transformer_path=Path(folder) / "bf16.safetensors",
                                         load_vae=False, load_text_encoder=False, load_tokenizers=False,
                                         compressed_adaln_dtype=dtype).transformer
                self.assertEqual(loaded.modulation_down.weight.dtype, getattr(torch, dtype))
                self.assertTrue(torch.equal(loaded.modulation_down.weight.float(), model.modulation_down.weight.float()))
        dense = MageFlow(MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], False))
        set_modulation_dtype(dense, torch.bfloat16)
        self.assertTrue(all(p.dtype == torch.float32 for p in dense.parameters()))

    def test_fp64_construction_stores_fp32_coefficients(self):
        torch.manual_seed(42)
        model = MageFlow(MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], False))
        basis = torch.linalg.qr(torch.randn(32, 8, dtype=torch.float64)).Q
        mean = torch.randn(32, dtype=torch.float64)
        old = model.transformer_blocks[0].img_mod[1]
        expected_weight = (old.weight.double() @ basis).float()
        expected_bias = (old.bias.double() + old.weight.double() @ mean).float()
        initialize_compression(model, basis, mean, calculation_dtype=torch.float64)
        head = model.transformer_blocks[0].img_mod[1]
        self.assertEqual(head.weight.dtype, torch.float32)
        self.assertTrue(torch.equal(head.weight, expected_weight))
        self.assertTrue(torch.equal(head.bias, expected_bias))
        self.assertTrue(
            torch.equal(model.modulation_down.bias, (-mean @ basis).float())
        )

    @unittest.skipUnless(os.environ.get("MAGE_GPU_TEST") == "1", "GPU opt-in")
    def test_packed_compressed_conditions_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._check_packed_gradients(dtype)

    def _check_packed_gradients(self, dtype):
        torch.manual_seed(42)
        plain = MageFlow(
            MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], False, modulation_rank=8)
        ).to("cuda", torch.bfloat16)
        set_modulation_dtype(plain, dtype)
        plain.configure_execution(False)
        packed = copy.deepcopy(plain)
        packed.configure_execution(True, [0], attention_backend="torch_varlen")
        images = [
            torch.randn(1, 128, 1, h, w, device="cuda", dtype=torch.bfloat16)
            for h, w in ((3, 4), (2, 3))
        ]
        times = torch.tensor([0.2, 0.7], device="cuda", dtype=torch.bfloat16)
        context = torch.randn(2, 5, 24, device="cuda", dtype=torch.bfloat16)
        mask = torch.tensor(
            [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], device="cuda", dtype=torch.bool
        )
        expected = [
            plain(image, times[i : i + 1], (context[i : i + 1], mask[i : i + 1]))[0]
            for i, image in enumerate(images)
        ]
        actual = packed(images, times, (context, mask))[0]
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y, atol=0.02, rtol=0.03)
        sum(x.float().square().mean() for x in expected).backward()
        sum(x.float().square().mean() for x in actual).backward()
        for (name, p), (_, q) in zip(
            plain.named_parameters(), packed.named_parameters()
        ):
            # The final block's text-only output is unused by the image loss.
            # Its gradients may be absent in both paths; modulation must connect.
            if compressed_parameter(name):
                self.assertIsNotNone(q.grad, name)
            torch.testing.assert_close(p.grad, q.grad, atol=4e-4, rtol=0.08, msg=name)

    def test_full_rank_conversion_roundtrip_and_gradients(self):
        torch.manual_seed(7)
        model = MageFlow(
            MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], False)
        ).eval()
        model.configure_execution(False)
        inputs = {
            "hidden_states": torch.randn(1, 128, 1, 2, 2),
            "timestep": torch.tensor([0.4]),
            "encoder_hidden_states": (
                torch.randn(1, 3, 24),
                torch.ones(1, 3, dtype=torch.bool),
            ),
        }
        before = model(**inputs)[0].detach()
        initialize_compression(model, torch.eye(32), torch.randn(32))
        torch.testing.assert_close(model(**inputs)[0], before, rtol=1e-4, atol=1e-5)
        build_param_groups(model, ComponentLRs(), 1e-5)
        model(**inputs)[0].square().mean().backward()
        for name, p in model.named_parameters():
            if compressed_parameter(name):
                self.assertIsNotNone(p.grad, name)
        with tempfile.TemporaryDirectory() as folder:
            cfg = SimpleNamespace(
                is_lora=False,
                quant=SimpleNamespace(mode="none"),
                train=SimpleNamespace(run_name="tiny", save_native=True),
            )
            export_checkpoint(model, folder, "tiny", cfg, torch.float32, 0)
            loaded = load_components(
                "/missing",
                transformer_path=Path(folder) / "tiny.safetensors",
                dtype=torch.float32,
                load_vae=False,
                load_text_encoder=False,
                load_tokenizers=False,
            ).transformer
            torch.testing.assert_close(loaded(**inputs)[0], model(**inputs)[0])
            self.assertEqual(loaded.params.modulation_rank, 32)
        exact_factors = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if compressed_parameter(n)
        }
        model.to(dtype=torch.bfloat16)
        for name, p in model.named_parameters():
            self.assertEqual(
                p.dtype,
                torch.float32 if compressed_parameter(name) else torch.bfloat16,
                name,
            )
            if name in exact_factors:
                self.assertTrue(torch.equal(p, exact_factors[name]), name)
        # Saving a BF16 trunk must preserve the factors' original FP32 bits, too.
        with tempfile.TemporaryDirectory() as folder:
            export_checkpoint(model, folder, "mixed", cfg, torch.bfloat16, 0)
            loaded = load_components(
                "/missing",
                transformer_path=Path(folder) / "mixed.safetensors",
                load_vae=False,
                load_text_encoder=False,
                load_tokenizers=False,
            ).transformer
            for name, p in loaded.named_parameters():
                if name in exact_factors:
                    self.assertTrue(torch.equal(p, exact_factors[name]), name)
        build_param_groups(model, ComponentLRs(adaln=0), 1e-5)
        for name, p in model.named_parameters():
            if compressed_parameter(name):
                self.assertFalse(p.requires_grad, name)

    def test_lora_keeps_compressed_modulation_frozen(self):
        model = MageFlow(
            MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], False, modulation_rank=8)
        )
        apply_adapter(model, AdapterConfig(rank=2, alpha=2))
        for name, p in model.named_parameters():
            if compressed_parameter(name):
                self.assertFalse(p.requires_grad, name)


if __name__ == "__main__":
    unittest.main()
