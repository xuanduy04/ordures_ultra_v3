

"""Functional smoke tests for Gemma3-VL recipe configurations."""

import pytest

from megatron.bridge.recipes.gemma3_vl.gemma3_vl import (
    gemma3_vl_4b_finetune_config,
)
from tests.functional_tests.recipes.utils import run_pretrain_vl_recipe_test


GEMMA3_VL_FINETUNE_RECIPES = [
    # Small model, only use 2 layers
    (
        gemma3_vl_4b_finetune_config,
        "gemma3_vl_4b",
        {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1, "num_layers": 2},
    ),
]

GEMMA3_VL_FINETUNE_PACKED_RECIPES = [
    # Small model with packed sequences, only use 2 layers
    (
        gemma3_vl_4b_finetune_config,
        "gemma3_vl_4b_packed",
        {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1, "num_layers": 2},
        {"pack_sequences_in_batch": True},
    ),
]


class TestGemma3VLRecipes:
    """Test class for Gemma3-VL recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,model_overrides", GEMMA3_VL_FINETUNE_RECIPES)
    def test_gemma3_vl_finetune_recipes(self, config_func, recipe_name, model_overrides, tmp_path):
        """Functional test for Gemma3-VL recipes with appropriate parallelism configurations."""
        run_pretrain_vl_recipe_test(config_func, recipe_name, tmp_path, model_overrides=model_overrides)

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "config_func,recipe_name,model_overrides,dataset_overrides", GEMMA3_VL_FINETUNE_PACKED_RECIPES
    )
    def test_gemma3_vl_finetune_packed_recipes(
        self, config_func, recipe_name, model_overrides, dataset_overrides, tmp_path
    ):
        """Functional test for Gemma3-VL recipes with packed sequences enabled."""
        run_pretrain_vl_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            dataset_overrides=dataset_overrides,
        )
