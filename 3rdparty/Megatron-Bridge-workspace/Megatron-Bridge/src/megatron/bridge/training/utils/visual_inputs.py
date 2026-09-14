

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch


@dataclass
class Qwen2_5_VLVisualInputs:
    """Container for Qwen2/Qwen2.5-VL visual modality tensors.

    Fields mirror the processor outputs for Qwen2/Qwen2.5-VL. Shapes may be
    normalized for model consumption via normalized_for_model().
    """

    # Image tensors, e.g., Qwen2.5-VL processor output.
    pixel_values: Optional[torch.Tensor] = None

    # Per-image temporal/spatial grid metadata (T, H, W) for videos, Qwen2.5-VL.
    image_grid_thw: Optional[torch.Tensor] = None

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        """Return a mapping of non-None fields suitable for model forward kwargs."""
        result: dict[str, torch.Tensor] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                result[f.name] = value
        return result

    def normalized_for_model(self) -> dict[str, torch.Tensor]:
        """Return non-None fields with shapes normalized for model expectations.

        - pixel_values: [B, N, C, H, W] -> [B*N, C, H, W]
        - image_grid_thw: [B, N, 3] -> [B*N, 3]
        """
        kwargs = self.as_model_kwargs()

        pixel_values = kwargs.get("pixel_values")
        if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 5:
            b, n, c, h, w = pixel_values.shape
            kwargs["pixel_values"] = pixel_values.view(b * n, c, h, w)

        image_grid_thw = kwargs.get("image_grid_thw")
        if isinstance(image_grid_thw, torch.Tensor) and image_grid_thw.dim() == 3:
            kwargs["image_grid_thw"] = image_grid_thw.view(-1, image_grid_thw.size(-1))

        return kwargs
