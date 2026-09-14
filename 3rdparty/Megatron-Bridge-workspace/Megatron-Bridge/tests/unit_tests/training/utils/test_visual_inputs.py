

import torch

from megatron.bridge.training.utils.visual_inputs import Qwen2_5_VLVisualInputs


def test_normalized_for_model_shapes():
    # pixel_values: [B, N, C, H, W] -> [B*N, C, H, W]
    pixel_values = torch.randn(2, 3, 4, 5, 6)
    # image_grid_thw: [B, N, 3] -> [B*N, 3]
    image_grid_thw = torch.randint(0, 10, (2, 3, 3))

    vi = Qwen2_5_VLVisualInputs(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    kwargs = vi.normalized_for_model()

    assert "pixel_values" in kwargs
    assert "image_grid_thw" in kwargs
    assert kwargs["pixel_values"].shape == (2 * 3, 4, 5, 6)
    assert kwargs["image_grid_thw"].shape == (2 * 3, 3)


def test_as_model_kwargs_filters_none():
    vi = Qwen2_5_VLVisualInputs(pixel_values=None, image_grid_thw=None)
    kwargs = vi.as_model_kwargs()
    assert kwargs == {}
