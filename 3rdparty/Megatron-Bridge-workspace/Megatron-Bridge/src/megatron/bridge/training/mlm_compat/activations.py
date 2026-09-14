

import torch
import torch.nn.functional as F
from megatron.core.jit import jit_fuser


@jit_fuser
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU activation function."""
    return torch.pow(F.relu(x), 2)
