

"""Functional smoke tests for Qwen2.5-VL recipe configurations."""

import pytest

from megatron.bridge.recipes.qwen_vl.qwen25_vl import qwen25_vl_3b_finetune_config
from tests.functional_tests.recipes.utils import run_pretrain_vl_recipe_test


QWEN_VL_PRETRAIN_RECIPES = [
    # (config_func, name, parallelism_overrides)
    # Two-GPU TP for local/CI multi-GPU runs
    (
        qwen25_vl_3b_finetune_config,
        "qwen25_vl_3b",
        {"tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 1},
        {"num_layers": 2},
    ),
]

QWEN_VL_PRETRAIN_PACKED_RECIPES = [
    # (config_func, name, parallelism_overrides, model_overrides, dataset_overrides)
    # Two-GPU TP with packed sequences
    (
        qwen25_vl_3b_finetune_config,
        "qwen25_vl_3b_packed",
        {"tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 1},
        {"num_layers": 2},
        {"pack_sequences_in_batch": True},
    ),
]


class TestQwenVLRecipes:
    """Test class for Qwen2.5-VL recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,parallelism_overrides,model_overrides", QWEN_VL_PRETRAIN_RECIPES)
    def test_qwen25_vl_pretrain_recipes(
        self, config_func, recipe_name, parallelism_overrides, model_overrides, tmp_path
    ):
        """Functional test for Qwen2.5-VL recipes with appropriate parallelism configurations."""
        run_pretrain_vl_recipe_test(
            config_func, recipe_name, tmp_path, model_overrides=model_overrides, **parallelism_overrides
        )

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "config_func,recipe_name,parallelism_overrides,model_overrides,dataset_overrides",
        QWEN_VL_PRETRAIN_PACKED_RECIPES,
    )
    def test_qwen25_vl_pretrain_packed_recipes(
        self, config_func, recipe_name, parallelism_overrides, model_overrides, dataset_overrides, tmp_path
    ):
        """Functional test for Qwen2.5-VL recipes with packed sequences enabled."""
        run_pretrain_vl_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            dataset_overrides=dataset_overrides,
            **parallelism_overrides,
        )
