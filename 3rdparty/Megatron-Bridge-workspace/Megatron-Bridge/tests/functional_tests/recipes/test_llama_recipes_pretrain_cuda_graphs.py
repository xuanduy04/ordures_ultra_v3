

"""Functional smoke tests for LLaMA recipe configurations."""

import pytest

from megatron.bridge.recipes.llama import (
    llama32_1b_pretrain_config as llama32_1b_config,
)
from tests.functional_tests.recipes.utils import run_pretrain_recipe_perf_test


LLAMA_PRETRAIN_RECIPES = [
    # (config_func, name, config_overrides)
    (
        llama32_1b_config,
        "llama32_1b",
        {
            "model": {
                "num_layers": 2,
                "cuda_graph_impl": "local",
                "cuda_graph_scope": ["full_iteration"],
                "check_for_nan_in_grad": False,
                "use_te_rng_tracker": True,
            },
            "rerun_state_machine": {"check_for_nan_in_loss": False},
            "ddp": {"check_for_nan_in_grad": False},
        },
    ),
    (
        llama32_1b_config,
        "llama32_1b",
        {
            "model": {
                "num_layers": 2,
                "cuda_graph_impl": "transformer_engine",
                "cuda_graph_scope": ["attn"],
                "check_for_nan_in_grad": False,
                "use_te_rng_tracker": True,
            },
            "rerun_state_machine": {"check_for_nan_in_loss": False},
            "ddp": {"check_for_nan_in_grad": False},
        },
    ),
]


class TestLlamaCudaGraphRecipes:
    """Test class for LLaMA recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,config_overrides", LLAMA_PRETRAIN_RECIPES)
    def test_llama_pretrain_recipes(self, config_func, recipe_name, config_overrides):
        """Functional test for LLaMA recipes with appropriate parallelism configurations."""
        run_pretrain_recipe_perf_test(
            config_func,
            recipe_name,
            config_overrides=config_overrides,
        )
