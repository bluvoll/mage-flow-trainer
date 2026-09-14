import os
import unittest
import torch
from test_mageflow import tiny, inputs
from trainer.training.params import AdapterConfig, apply_adapter
from trainer.training.lycoris import (
    ALGORITHMS,
    DECOMPOSED,
    lycoris_state_dict,
    load_lycoris_state_dict,
)
from trainer.training.quant import QuantConfig, quantize_module


class AlgorithmTests(unittest.TestCase):
    def test_lokr_factor_changes_actual_factorization(self):
        shapes = {}
        for factor in (-1, 2, 128):
            model = tiny()
            apply_adapter(model, AdapterConfig(kind="lycoris_lora", lycoris_algo="lokr",
                                              rank=2, alpha=2, lokr_factor=factor))
            adapter = model.transformer_blocks[0].attn.to_q.lycoris_adapter
            shapes[factor] = tuple(adapter.lokr_w1.shape)
            model(**inputs())[0].square().mean().backward()
        self.assertEqual(shapes[2], (2, 2))
        self.assertNotEqual(shapes[-1], shapes[2])

    def test_compile_defaults_and_explicit_off(self):
        import tempfile
        from pathlib import Path
        from trainer.training.config import load_config

        for algo in ALGORITHMS:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "config.toml"
                text = f'[adapter]\nkind = "lycoris_lora"\nlycoris_algo = "{algo}"\n'
                text += '[dataset]\npath = "/path/to/images"\n'
                path.write_text(text)
                self.assertEqual(load_config(path).train.compile, "default")
                path.write_text(text + "[train]\ncompile = false\n")
                self.assertIsNone(load_config(path).train.compile)

    def test_all_algorithms_gradients_and_reload(self):
        for algo in (
            os.environ.get("MAGE_ALGOS", "").split(",")
            if os.environ.get("MAGE_ALGOS")
            else ALGORITHMS
        ):
            with self.subTest(algo=algo):
                m = tiny()
                cfg = AdapterConfig(
                    kind="lycoris_lora", lycoris_algo=algo, rank=4, alpha=4
                )
                apply_adapter(m, cfg)
                adapters = [
                    v for n, v in m.named_modules() if n.endswith("lycoris_adapter")
                ]
                self.assertTrue(
                    all(v.bypass_mode == (algo not in DECOMPOSED) for v in adapters)
                )
                x = inputs()
                out = m(**x)[0]
                out.square().mean().backward()
                self.assertTrue(
                    any(
                        p.grad is not None and p.grad.abs().sum() > 0
                        for p in m.parameters()
                        if p.requires_grad
                    )
                )
                state = lycoris_state_dict(m)
                load_lycoris_state_dict(m, state)
                self.assertTrue(torch.isfinite(m(**x)[0]).all())

    @unittest.skipUnless(os.environ.get("MAGE_ALGO_GPU") == "1", "GPU opt-in")
    def test_quantized_compiled_algorithms(self):
        for algo in (
            os.environ.get("MAGE_ALGOS", "").split(",")
            if os.environ.get("MAGE_ALGOS")
            else ALGORITHMS
        ):
            with self.subTest(algo=algo):
                print("GPU algorithm:", algo, flush=True)
                torch._dynamo.reset()
                m = tiny().to("cuda", torch.bfloat16)
                m = quantize_module(
                    m,
                    QuantConfig(mode="frozen", use_quantized_matmul=False),
                    torch.device("cuda"),
                    torch.bfloat16,
                    False,
                )
                apply_adapter(
                    m,
                    AdapterConfig(
                        kind="lycoris_lora",
                        lycoris_algo=algo,
                        rank=4,
                        alpha=4,
                        dtype="bfloat16",
                    ),
                )
                x = inputs("cuda", torch.bfloat16)
                eager = m(**x)[0]
                eager.float().square().mean().backward()
                grads = {
                    n: p.grad.detach().clone()
                    for n, p in m.named_parameters()
                    if p.requires_grad and p.grad is not None
                }
                m.zero_grad(set_to_none=True)
                m.configure_execution(
                    True,
                    compile_mode="default",
                    compile_dynamic=False,
                    attention_backend="torch_varlen",
                )
                out = m(**x)[0]
                out.float().square().mean().backward()
                torch.testing.assert_close(eager, out, atol=0.03, rtol=0.04)
                for n, p in m.named_parameters():
                    if n in grads:
                        self.assertTrue(torch.isfinite(p.grad).all(), n)
                        torch.testing.assert_close(
                            p.grad, grads[n], atol=0.002, rtol=0.2, msg=n
                        )
                del m, out, eager, grads
                torch.cuda.empty_cache()
