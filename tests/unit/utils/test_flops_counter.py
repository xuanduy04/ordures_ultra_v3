

import pytest

from nemo_rl.utils.flops_tracker import FLOPTracker, get_default_hf_config


@pytest.mark.parametrize(
    "model_name, gbs, seqlen, expected_flops",
    [
        ("meta-llama/Llama-2-7b-hf", 128, 4096, 2.25e16),
        ("meta-llama/Llama-2-13b-hf", 128, 4096, 4.17e16),
        ("meta-llama/Llama-2-70b-hf", 128, 4096, 2.25e17),
        ("meta-llama/Meta-Llama-3-8B", 128, 8192, 5.31e16),
        ("meta-llama/Llama-3.1-70B-Instruct", 128, 8192, 4.71e17),
        ("meta-llama/Llama-3.1-405B-Instruct", 128, 8192, 2.65e18),
        ("Qwen/Qwen3-30B-A3B", 128, 4096, 9.37e15),
        ("Qwen/Qwen3-235B-A22B", 128, 4096, 6.21e16),
        ("deepseek-ai/DeepSeek-V3", 1, 4096, 1.023e15),
        ("moonshotai/Moonlight-16B-A3B-Instruct", 1, 4096, 6.45e13),
    ],
)
def test_flops_counter(model_name, gbs, seqlen, expected_flops):
    model_config = get_default_hf_config(model_name)
    flops_tracker = FLOPTracker.from_config(model_name, model_config)
    flops_tracker.track(gbs, seqlen)

    # check within 5% relative difference
    assert abs(flops_tracker.total_flops - expected_flops) / expected_flops <= 0.05, (
        f"Expected {expected_flops} flops, got {flops_tracker.total_flops}"
    )
