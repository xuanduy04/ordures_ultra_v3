

"""Functional tests for gradient clipping with various parallelism configurations.

These tests validate that _clip_grad_norm_impl works correctly with
TP, PP, EP, and combinations thereof.
"""

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "clip_grad_norm"
TEST_FILENAME = "L2_ClipGradNorm_Test.sh"


class TestClipGradNorm:
    """Test suite for gradient clipping with various parallelism strategies."""

    def test_clip_grad_norm_all_configs(self):
        """Test gradient clipping with all parallelism configurations."""
        run_test_script(TEST_FOLDER, TEST_FILENAME)
