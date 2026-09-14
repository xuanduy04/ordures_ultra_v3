

import torch


def apply_torch_aten_alias_tensor_patch():
    """Register a sharding rule for `torch.ops.aten.alias.default`.

    Work around 'NotImplementedError: Operator aten.alias.default does not have a sharding strategy registered'
    in PyTorch 2.9. See https://github.com/pytorch/pytorch/pull/166867 for the upstream fix.
    We can remove this patch when we upgrade torch to include this fix.
    """
    if not torch.__version__.startswith("2.9.0"):
        return
    try:
        from torch.distributed.tensor._ops._tensor_ops import (
            propagate_single_input_strategy,
        )
        from torch.distributed.tensor._ops.utils import register_op_strategy

        register_op_strategy(torch.ops.aten.alias.default)(
            propagate_single_input_strategy
        )
    except Exception as e:
        print(f"Error applying torch.ops.aten.alias.default patch: {e}")
