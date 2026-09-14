


from megatron.bridge.training.comm_overlap import *
from megatron.bridge.training.mixed_precision import (
    bf16_mixed,
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_fp8_subchannel_scaling_mixed,
    bf16_with_mxfp8_mixed,
    bf16_with_nvfp4_mixed,
)


def get_precision_config(compute_dtype: str):
    """Get the precision configs for the given compute dtype and FP8 recipe."""
    if compute_dtype == "fp8_cs":
        current_scaling_cfg = bf16_with_fp8_current_scaling_mixed()
        # Disable BF16 Transformer layers in the performance config
        current_scaling_cfg.first_last_layers_bf16 = False
        return current_scaling_cfg
    elif compute_dtype == "fp8_mx":
        return bf16_with_mxfp8_mixed()
    elif compute_dtype == "fp8_sc":
        return bf16_with_fp8_subchannel_scaling_mixed()
    elif compute_dtype == "bf16":
        return bf16_mixed()
    elif compute_dtype == "nvfp4":
        fp4_precision_cfg = bf16_with_nvfp4_mixed()
        return fp4_precision_cfg
    else:
        raise ValueError(f"Invalid compute dtype: {compute_dtype}")
