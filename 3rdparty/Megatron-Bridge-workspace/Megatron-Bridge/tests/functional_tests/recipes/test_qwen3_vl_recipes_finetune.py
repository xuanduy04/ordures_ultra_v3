

"""Functional smoke tests for Qwen3-VL finetuning recipes.

This test ensures that:
1. Qwen3-VL model forward pass works with all required parameters (including loss_mask)
2. Training loop completes without errors
3. Checkpoints are saved correctly

This catches regressions like missing parameters in the forward pass signature.

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/functional_tests/recipes/test_qwen3_vl_recipes_peft.py -v
"""

import pytest

from megatron.bridge.recipes.qwen_vl.qwen3_vl import qwen3_vl_8b_finetune_config
from tests.functional_tests.recipes.utils import run_pretrain_vl_recipe_test


QWEN3_VL_FINETUNE_RECIPES = [
    # (config_func, recipe_name, parallelism_overrides, model_overrides)
    # Qwen3-VL 8B finetune - uses TP=2 for 2-GPU CI
    # Note: deepstack_visual_indexes must have len <= num_layers
    (
        qwen3_vl_8b_finetune_config,
        "qwen3_vl_8b_finetune",
        {"tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 1},
        {"num_layers": 4, "deepstack_visual_indexes": [0, 1, 2]},
    ),
]

QWEN3_VL_FINETUNE_PACKED_RECIPES = [
    # (config_func, recipe_name, parallelism_overrides, model_overrides, dataset_overrides)
    # Qwen3-VL 8B finetune with packed sequences
    (
        qwen3_vl_8b_finetune_config,
        "qwen3_vl_8b_finetune_packed",
        {"tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 1},
        {"num_layers": 4, "deepstack_visual_indexes": [0, 1, 2]},
        {"pack_sequences_in_batch": True},
    ),
]


class TestQwen3VLFinetuneRecipes:
    """Test class for Qwen3-VL finetune recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "config_func,recipe_name,parallelism_overrides,model_overrides",
        QWEN3_VL_FINETUNE_RECIPES,
    )
    def test_qwen3_vl_finetune_recipes(
        self,
        config_func,
        recipe_name,
        parallelism_overrides,
        model_overrides,
        tmp_path,
    ):
        """Functional test for Qwen3-VL finetune recipes.

        This test runs a minimal training session to verify that:
        1. The config loads correctly
        2. Model forward pass accepts all required parameters (loss_mask, etc.)
        3. Training completes without errors
        4. Checkpoints are created
        """
        run_pretrain_vl_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            **parallelism_overrides,
        )

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "config_func,recipe_name,parallelism_overrides,model_overrides,dataset_overrides",
        QWEN3_VL_FINETUNE_PACKED_RECIPES,
    )
    def test_qwen3_vl_finetune_packed_recipes(
        self,
        config_func,
        recipe_name,
        parallelism_overrides,
        model_overrides,
        dataset_overrides,
        tmp_path,
    ):
        """Functional test for Qwen3-VL finetune recipes with packed sequences enabled."""
        run_pretrain_vl_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            dataset_overrides=dataset_overrides,
            **parallelism_overrides,
        )
