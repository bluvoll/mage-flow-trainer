"""Compatibility fix for SDNQ's biased integer stochastic requantization.

Install before training quantization/compilation. This also fixes integer optimizer
buffers in that process; deterministic and floating-point quantization are unchanged.
No installed package files are modified.
"""

import inspect
import logging


def enable_unbiased_integer_rounding():
    import torch
    import sdnq.quantizer as quantizer
    from sdnq.common import dtype_dict
    from sdnq.quant_utils import get_scale_asymmetric, get_scale_symmetric

    original = quantizer.quantize_weight
    if getattr(original, "_mageflow_unbiased_integer_rounding", False):
        return True
    # Do not override a future upstream implementation with different semantics.
    source = inspect.getsource(original)
    if "torch.randn_like(quantized_weight), alpha=0.1" not in source:
        return False

    def quantize_weight(weight, dim, weights_dtype, dtype=None,
                        use_stochastic_rounding=False):
        if not use_stochastic_rounding or weights_dtype not in ("int8", "uint8"):
            return original(weight, dim, weights_dtype, dtype, use_stochastic_rounding)
        if weight.dtype != torch.float64:
            weight = weight.to(torch.float32)
        spec = dtype_dict[weights_dtype]
        if spec["is_unsigned"]:
            scale, zero_point = get_scale_asymmetric(weight, dim, weights_dtype)
            if dtype is not None:
                scale, zero_point = scale.to(dtype), zero_point.to(dtype)
            values = (weight - zero_point) / scale
        else:
            scale = get_scale_symmetric(weight, dim, weights_dtype)
            if dtype is not None:
                scale = scale.to(dtype)
            zero_point = None
            values = weight / scale
        # For x=n+f, choose n+1 with probability f (also correct for x<0).
        # Constant groups have zero scale; their codes can safely be zero.
        values = values.nan_to_num(nan=0.0, posinf=spec["max"], neginf=spec["min"])
        codes = (values + torch.rand_like(values)).floor_()
        codes = codes.clamp_(spec["min"], spec["max"]).to(spec["torch_dtype"])
        return codes, scale, zero_point

    quantize_weight._mageflow_unbiased_integer_rounding = True
    quantizer.quantize_weight = quantize_weight
    logging.getLogger(__name__).warning(
        "SDNQ training: enabled unbiased INT8/UINT8 stochastic rounding "
        "to preserve small weight and optimizer-buffer updates."
    )
    return True
