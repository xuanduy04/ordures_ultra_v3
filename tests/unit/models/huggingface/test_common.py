

import pytest

from nemo_rl.models.huggingface.common import ModelFlag, is_gemma_model


@pytest.mark.hf_gated
@pytest.mark.parametrize(
    "model_name",
    [
        "google/gemma-2-2b",
        "google/gemma-2-9b",
        "google/gemma-2-27b",
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        "google/gemma-3-1b-pt",
        "google/gemma-3-4b-pt",
        "google/gemma-3-12b-pt",
        "google/gemma-3-27b-pt",
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
    ],
)
def test_gemma_models(model_name):
    assert is_gemma_model(model_name)
    assert ModelFlag.VLLM_LOAD_FORMAT_AUTO.matches(model_name)


@pytest.mark.hf_gated
@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ],
)
def test_non_gemma_models(model_name):
    assert not is_gemma_model(model_name)
    assert not ModelFlag.VLLM_LOAD_FORMAT_AUTO.matches(model_name)
