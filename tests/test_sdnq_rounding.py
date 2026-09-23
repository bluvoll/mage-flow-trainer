"""Small signed updates must survive integer writeback in expectation."""
import pytest
import torch


@pytest.fixture
def quantizers():
    import sdnq.quantizer as q
    from trainer.training.sdnq_rounding import enable_unbiased_integer_rounding
    original = q.quantize_weight
    assert enable_unbiased_integer_rounding()
    fixed = q.quantize_weight
    assert enable_unbiased_integer_rounding()
    assert q.quantize_weight is fixed
    yield original, fixed
    q.quantize_weight = original


@pytest.mark.parametrize('dtype', ['int8', 'uint8'])
def test_sub_bin_updates_are_unbiased(quantizers, dtype):
    _, fixed = quantizers
    torch.manual_seed(51)
    # Fixed extrema make the quantization step exactly one. Repeated positive
    # and negative sub-bin values detect a dead zone around integer codes.
    values = torch.tensor([-0.05, 0.05, 0.1, -0.1]).repeat(50000)
    bounds = [-127., 127.] if dtype == 'int8' else [-128., 127.]
    x = torch.cat([torch.tensor(bounds), values]).reshape(1, -1)
    q, s, z = fixed(x, -1, dtype, torch.float32, True)
    reconstructed = q.float() * s
    if z is not None:
        reconstructed += z
    means = reconstructed.flatten()[2:].reshape(-1, 4).mean(0)
    torch.testing.assert_close(means, torch.tensor([-0.05, 0.05, 0.1, -0.1]), atol=.004, rtol=0)


@pytest.mark.parametrize('dtype', ['int8', 'uint8'])
def test_deterministic_quantization_unchanged(quantizers, dtype):
    original, fixed = quantizers
    x = torch.randn(16, 128)
    a = original(x, -1, dtype, torch.float32, False)
    b = fixed(x, -1, dtype, torch.float32, False)
    for u, v in zip(a, b):
        if u is None:
            assert v is None
        else:
            assert torch.equal(u, v)


@pytest.mark.parametrize('dtype', ['int8', 'uint8'])
def test_zero_groups_are_finite(quantizers, dtype):
    _, fixed = quantizers
    q, s, z = fixed(torch.zeros(4, 128), -1, dtype, torch.float32, True)
    result = q.float() * s
    if z is not None:
        result += z
    assert torch.equal(result, torch.zeros_like(result))
