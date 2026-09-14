

"""Functional tests for context parallelism on attention layers.

These tests validate that attention layers produce identical forward outputs
and gradients when using different context parallel sizes with packed sequences.
"""

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "context_parallel"
CP_QWEN3_MOE_ATTENTION_TEST_FILENAME = "L2_CP_Qwen3MoE_Attention_Test.sh"
CP_DEEPSEEK_V3_MLA_TEST_FILENAME = "L2_CP_DeepSeekV3_MLA_Test.sh"


class TestContextParallelAttention:
    """Test suite for context parallel attention layers."""

    def test_cp_qwen3_moe_attention(self):
        """Test Qwen3MoeAttention layer with CP=1 vs CP=2."""
        run_test_script(TEST_FOLDER, CP_QWEN3_MOE_ATTENTION_TEST_FILENAME)

    def test_cp_deepseek_v3_mla(self):
        """Test DeepSeek V3 MLA layer with CP=1 vs CP=2."""
        run_test_script(TEST_FOLDER, CP_DEEPSEEK_V3_MLA_TEST_FILENAME)
