

"""Functional smoke tests for LLaMA recipe configurations."""

import pytest

from megatron.bridge.recipes.llama import (
    llama32_3b_pretrain_config as llama32_3b_config,
)
from tests.functional_tests.recipes.utils import run_pretrain_config_override_test, run_pretrain_recipe_test


LLAMA_PRETRAIN_RECIPES = [
    # (config_func, name, parallelism_overrides)
    (llama32_3b_config, "llama32_3b", {}),  # Small model, use recipe defaults
]


class TestLlamaRecipes:
    """Test class for LLaMA recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,parallelism_overrides", LLAMA_PRETRAIN_RECIPES)
    def test_llama_pretrain_recipes(self, config_func, recipe_name, parallelism_overrides, tmp_path):
        """Functional test for LLaMA recipes with appropriate parallelism configurations."""
        run_pretrain_recipe_test(config_func, recipe_name, tmp_path, **parallelism_overrides)

    @pytest.mark.parametrize("config_func,recipe_name,parallelism_overrides", LLAMA_PRETRAIN_RECIPES)
    def test_pretrain_config_override_after_instantiation(self, config_func, recipe_name, parallelism_overrides):
        """Functional test for overriding LLaMA recipes from CLI"""
        run_pretrain_config_override_test(config_func)
