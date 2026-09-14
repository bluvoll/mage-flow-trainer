import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from trainer.modeling.mage_flow import MageFlow, MageFlowParams
from trainer.training.params import (
    AdapterConfig,
    ComponentLRs,
    apply_adapter,
    build_adapter_param_groups,
    build_param_groups,
    classify,
)
from trainer.training.quant import QuantConfig, resolve_quantized_matmul
from trainer.modeling.export import export_checkpoint
from trainer.modeling.loader import load_components


def tiny():
    torch.manual_seed(42)
    model = MageFlow(MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], False))
    model.configure_execution(False)
    return model


def inputs(device="cpu", dtype=torch.float32):
    return dict(
        hidden_states=torch.randn(2, 128, 1, 3, 4, device=device, dtype=dtype),
        timestep=torch.tensor([0.2, 0.7], device=device, dtype=dtype),
        encoder_hidden_states=(
            torch.randn(2, 5, 24, device=device, dtype=dtype),
            torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], device=device).bool(),
        ),
    )


class MageFlowTests(unittest.TestCase):
    def test_lora_never_trains_adaln(self):
        with self.assertRaisesRegex(ValueError, "AdaLN must remain frozen"):
            AdapterConfig(components=["image_attn", "adaln"])
        cfg = AdapterConfig(rank=2, alpha=1)
        cfg.components.append("adaln")
        with self.assertRaisesRegex(ValueError, "AdaLN must remain frozen"):
            apply_adapter(tiny(), cfg)

        m = tiny()
        apply_adapter(m, AdapterConfig(rank=2, alpha=1))
        adaln = {n: p for n, p in m.named_parameters() if classify(n) == "adaln"}
        self.assertTrue(adaln)
        self.assertFalse(any("lora_" in n or p.requires_grad for n, p in adaln.items()))
        before = {n: p.detach().clone() for n, p in adaln.items()}
        # Even a nonzero component LR cannot unfreeze AdaLN in an adapter run.
        report = build_adapter_param_groups(m, ComponentLRs(adaln=1.0), 1e-3)
        self.assertNotIn("adaln", report.counts)
        opt = torch.optim.AdamW(report.groups)
        m(**inputs())[0].square().mean().backward()
        opt.step()
        for n, p in adaln.items():
            self.assertIsNone(p.grad)
            self.assertTrue(torch.equal(p, before[n]), n)

    def test_padding_and_checkpoint_gradients(self):
        a = tiny()
        b = copy.deepcopy(a)
        b.configure_execution(True, [0])
        x = inputs()
        y = a(**x)[0]
        z = b(**x)[0]
        torch.testing.assert_close(y, z)
        y.square().mean().backward()
        z.square().mean().backward()
        for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            self.assertEqual(p.grad is None, q.grad is None, n)
            torch.testing.assert_close(p.grad, q.grad)
        changed = dict(x)
        text, mask = x["encoder_hidden_states"]
        text = text.clone()
        text[0, 2:] = 10000
        changed["encoder_hidden_states"] = text, mask
        torch.testing.assert_close(a(**changed)[0], y)

    def test_full_export_strict_reload(self):
        m = tiny()
        cfg = SimpleNamespace(
            is_lora=False,
            quant=QuantConfig(),
            train=SimpleNamespace(run_name="test", save_native=False),
        )
        with tempfile.TemporaryDirectory() as d:
            export_checkpoint(m, Path(d), "test", cfg, torch.float32, 3)
            loaded = load_components(d, torch.float32, False, False, False).transformer
            x = inputs()
            torch.testing.assert_close(m(**x)[0], loaded(**x)[0])

    def test_lora_export_and_gradients(self):
        m = tiny()
        cfg = AdapterConfig(rank=2, alpha=1)
        apply_adapter(m, cfg)
        y = m(**inputs())[0]
        y.square().mean().backward()
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for n, p in m.named_parameters()
                if "lora_B" in n
            )
        )
        self.assertFalse(
            any(p.requires_grad for n, p in m.named_parameters() if "lora_" not in n)
        )
        with tempfile.TemporaryDirectory() as d:
            c = SimpleNamespace(
                is_lora=True,
                quant=QuantConfig(),
                train=SimpleNamespace(run_name="test"),
            )
            export_checkpoint(m, Path(d), "test", c, torch.float32, 1)
            self.assertEqual(
                json.loads((Path(d) / "test.json").read_text())["lora_alpha"], 1
            )

    def test_full_finetune_trains_all_components_by_default(self):
        m = tiny()
        report = build_param_groups(m, ComponentLRs(), 5e-6)
        self.assertTrue(all(p.requires_grad for p in m.parameters()))
        self.assertFalse(report.frozen)
        self.assertGreater(report.counts["adaln"], 0)
        self.assertGreater(report.counts["base"], 0)
        self.assertTrue(all(g["lr"] == 5e-6 for g in report.groups))

    def test_full_finetune_preserves_explicit_component_freezes(self):
        m = tiny()
        report = build_param_groups(m, ComponentLRs(adaln=0.0, base=0.0), 5e-6)
        self.assertGreater(report.frozen["adaln"], 0)
        self.assertGreater(report.frozen["base"], 0)
        self.assertNotIn("adaln", report.counts)
        self.assertNotIn("base", report.counts)
        self.assertEqual(ComponentLRs(adaln=0.0).explicit(), {"adaln": 0.0})

    def test_quant_policy_and_components(self):
        m = tiny()
        report = build_param_groups(m, ComponentLRs(), 1e-5)
        self.assertEqual(
            sum(report.counts.values()) + sum(report.frozen.values()),
            sum(p.numel() for p in m.parameters()),
        )
        q = QuantConfig(skip_policy="all_adaln")
        self.assertIn("img_mod", q.skip_keys())
        self.assertIn("txt_mod", q.skip_keys())
        self.assertIn("img_in", q.skip_keys())
        self.assertFalse(resolve_quantized_matmul(q, 100000))

    def test_text_wrapper_and_prefix(self):
        from trainer.modeling.loader import encode_prompts

        tok = Mock(
            return_value=SimpleNamespace(
                input_ids=torch.ones(2, 39, dtype=torch.long),
                attention_mask=torch.ones(2, 39, dtype=torch.long),
            )
        )
        te = Mock(return_value=SimpleNamespace(hidden_states=[torch.randn(2, 39, 24)]))
        c = SimpleNamespace(tokenizer=tok, text_encoder=te)
        hidden, mask = encode_prompts(c, ["", "a cat"], "cpu", 5)
        self.assertEqual(hidden.shape, (2, 5, 24))
        self.assertEqual(mask.shape, (2, 5))
        self.assertEqual(te.call_args.kwargs["logits_to_keep"], 1)
        self.assertFalse(te.call_args.kwargs["use_cache"])
        self.assertIn("Describe the image", tok.call_args.args[0][0])

    @unittest.skipUnless(os.environ.get("MAGE_GPU_TEST") == "1", "GPU opt-in")
    def test_torch_varlen_forward_backward(self):
        a = tiny().to("cuda", torch.bfloat16)
        b = copy.deepcopy(a)
        b.configure_execution(True, [0], attention_backend="torch_varlen")
        x = inputs("cuda", torch.bfloat16)
        y = a(**x)[0]
        z = b(**x)[0]
        torch.testing.assert_close(y, z, atol=0.02, rtol=0.03)
        y.float().square().mean().backward()
        z.float().square().mean().backward()
        for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-4, rtol=0.08, msg=n)


class PackedResolutionTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("MAGE_GPU_TEST") == "1", "GPU opt-in")
    def test_native_packing_matches_independent_images(self):
        a = tiny().to("cuda", torch.bfloat16)
        b = copy.deepcopy(a)
        b.configure_execution(True, [0], attention_backend="torch_varlen")
        x = inputs("cuda", torch.bfloat16)
        images = [
            x["hidden_states"][0:1],
            x["hidden_states"][1:2, :, :, :2, :3].contiguous(),
        ]
        ctx = x["encoder_hidden_states"]
        ys = [
            a(
                hidden_states=im,
                timestep=x["timestep"][i : i + 1],
                encoder_hidden_states=(ctx[0][i : i + 1], ctx[1][i : i + 1]),
            )[0]
            for i, im in enumerate(images)
        ]
        zs = b(hidden_states=images, timestep=x["timestep"], encoder_hidden_states=ctx)[
            0
        ]
        for y, z in zip(ys, zs):
            torch.testing.assert_close(y, z, atol=0.02, rtol=0.03)
        sum(y.float().square().mean() for y in ys).backward()
        sum(z.float().square().mean() for z in zs).backward()
        for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=4e-4, rtol=0.08, msg=n)


class QuantizedTrainingTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("MAGE_GPU_TEST") == "1", "GPU opt-in")
    def test_quantized_update_export_and_state(self):
        from trainer.training.quant import (
            quantize_module,
            quantized_layer_report,
            dequantize_state_dict,
        )
        from trainer.training.optim import build_optimizer
        from trainer.training.config import OptimizerConfig

        m = MageFlow(MageFlowParams(128, 128, 64, 256, 4, 2, [16, 24, 24], False)).to(
            torch.bfloat16
        )
        q = QuantConfig(
            mode="training", skip_policy="all_adaln", use_quantized_matmul=False
        )
        m = quantize_module(m, q, torch.device("cuda"), torch.bfloat16, False).to(
            "cuda"
        )
        m.configure_execution(True, attention_backend="torch_varlen")
        self.assertGreater(quantized_layer_report(m)[0], 0)
        groups = build_param_groups(m, ComponentLRs(), 0.01)
        opt = build_optimizer(
            groups.groups,
            OptimizerConfig(
                kind="adamw8bit", lr=0.01, quantize_state=True, offload_state=True
            ),
        )
        x = inputs("cuda", torch.bfloat16)
        x["encoder_hidden_states"] = (
            torch.randn(2, 5, 64, device="cuda", dtype=torch.bfloat16),
            x["encoder_hidden_states"][1],
        )
        before = dequantize_state_dict(m.state_dict())[
            "transformer_blocks.0.attn.to_q.weight"
        ].clone()
        loss = m(**x)[0].float().square().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        after = dequantize_state_dict(m.state_dict())[
            "transformer_blocks.0.attn.to_q.weight"
        ]
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(torch.isfinite(after).all())
        with tempfile.TemporaryDirectory() as d:
            cfg = SimpleNamespace(
                is_lora=False,
                quant=q,
                train=SimpleNamespace(run_name="quant", save_native=False),
            )
            export_checkpoint(m, Path(d), "quant", cfg, torch.bfloat16, 1)
            loaded = load_components(d, torch.bfloat16, False, False, False).transformer
            torch.testing.assert_close(
                loaded.state_dict()["transformer_blocks.0.attn.to_q.weight"],
                after.cpu(),
            )
            state = Path(d) / "resume.pt"
            torch.save({"model": m.state_dict(), "optimizer": opt.state_dict()}, state)
            saved = torch.load(state, weights_only=False)
            m.load_state_dict(saved["model"])
            opt.load_state_dict(saved["optimizer"])
            m(**x)[0].float().square().mean().backward()
            opt.step()

    @unittest.skipUnless(os.environ.get("MAGE_GPU_TEST") == "1", "GPU opt-in")
    def test_compiled_varlen_backward(self):
        m = tiny().to("cuda", torch.bfloat16)
        m.configure_execution(
            True, [0], compile_mode="default", attention_backend="torch_varlen"
        )
        x = inputs("cuda", torch.bfloat16)
        loss = m(**x)[0].float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        m.zero_grad(set_to_none=True)
        x["hidden_states"] = [
            x["hidden_states"][0:1],
            x["hidden_states"][1:2, :, :, :2, :3].contiguous(),
        ]
        outputs = m(**x)[0]
        packed_loss = sum(y.float().square().mean() for y in outputs)
        packed_loss.backward()
        self.assertTrue(torch.isfinite(packed_loss))


if __name__ == "__main__":
    unittest.main()
