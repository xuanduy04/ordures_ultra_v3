

"""Functional smoke tests for DeepSeek recipe configurations."""

import pytest

from megatron.bridge.recipes.deepseek import (
    deepseek_v2_lite_pretrain_config as deepseek_v2_lite_config,
)
from tests.functional_tests.recipes.utils import run_pretrain_recipe_test


DEEPSEEK_PRETRAIN_RECIPES = [
    # (config_func, name, parallelism_overrides, model_overrides)
    (
        deepseek_v2_lite_config,
        "deepseek_v2_lite",
        {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1, "expert_model_parallel_size": 1},
        {"num_layers": 2, "num_moe_experts": 8, "moe_router_topk": 1, "moe_layer_freq": [0, 1]},
    ),
    # (
    #     deepseek_v3_config,
    #     "deepseek_v3",
    #     {"tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 1, "expert_model_parallel_size": 1},
    #     {
    #         "num_layers": 2,
    #         "num_moe_experts": 8,
    #         "moe_router_topk": 1,
    #         "moe_layer_freq": [0, 1],
    #         "pipeline_model_parallel_layout": [["embedding"] + ["decoder"] * 2 + ["mtp", "loss"]],
    #     },
    # ),
]


class TestDeepSeekRecipes:
    """Test class for DeepSeek recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "config_func,recipe_name,parallelism_overrides,model_overrides", DEEPSEEK_PRETRAIN_RECIPES
    )
    def test_deepseek_pretrain_recipes(
        self, config_func, recipe_name, parallelism_overrides, model_overrides, tmp_path
    ):
        """Functional test for DeepSeek recipes with appropriate parallelism configurations."""
        run_pretrain_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            **parallelism_overrides,
        )
