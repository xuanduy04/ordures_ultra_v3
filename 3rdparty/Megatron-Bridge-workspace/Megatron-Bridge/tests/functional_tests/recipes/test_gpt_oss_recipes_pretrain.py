

"""Functional smoke tests for GPT-OSS recipe configurations."""

import pytest

from megatron.bridge.recipes.gpt_oss.gpt_oss import gpt_oss_20b_pretrain_config
from tests.functional_tests.recipes.utils import run_pretrain_recipe_test


GPT_OSS_PRETRAIN_RECIPES = [
    # (config_func, name, parallelism_overrides, model_overrides)
    (
        gpt_oss_20b_pretrain_config,
        "gpt_oss_20b",
        {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1, "expert_model_parallel_size": 1},
        {"num_layers": 2, "sequence_parallel": False},
    ),
]


class TestGPTOSSRecipes:
    """Test class for GPT-OSS recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,parallelism_overrides,model_overrides", GPT_OSS_PRETRAIN_RECIPES)
    def test_gpt_oss_pretrain_recipes(
        self, config_func, recipe_name, parallelism_overrides, model_overrides, tmp_path
    ):
        """Functional test for GPT-OSS recipes with appropriate parallelism configurations."""
        run_pretrain_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            **parallelism_overrides,
        )
