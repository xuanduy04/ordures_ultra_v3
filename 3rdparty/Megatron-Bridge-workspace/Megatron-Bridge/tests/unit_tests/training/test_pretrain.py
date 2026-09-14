

"""Unit tests for pretrain module process group cleanup."""

from unittest.mock import patch

from megatron.bridge.training.pretrain import _maybe_destroy_process_group


class TestDestroyProcessGroupIfNeeded:
    """Test process group destruction logic."""

    @patch("megatron.bridge.training.pretrain.dist")
    def test_destroy_when_should_destroy_and_initialized(self, mock_dist):
        """Test process group is destroyed when both conditions are met."""
        mock_dist.is_initialized.return_value = True

        _maybe_destroy_process_group(should_destroy=True)

        mock_dist.barrier.assert_called_once()
        mock_dist.destroy_process_group.assert_called_once()

    @patch("megatron.bridge.training.pretrain.dist")
    def test_no_destroy_when_should_not_destroy(self, mock_dist):
        """Test no destruction when should_destroy is False."""
        mock_dist.is_initialized.return_value = True

        _maybe_destroy_process_group(should_destroy=False)

        mock_dist.barrier.assert_not_called()
        mock_dist.destroy_process_group.assert_not_called()

    @patch("megatron.bridge.training.pretrain.dist")
    def test_no_destroy_when_not_initialized(self, mock_dist):
        """Test no destruction when process group is not initialized."""
        mock_dist.is_initialized.return_value = False

        _maybe_destroy_process_group(should_destroy=True)

        mock_dist.barrier.assert_not_called()
        mock_dist.destroy_process_group.assert_not_called()

    @patch("megatron.bridge.training.pretrain.dist")
    def test_no_destroy_when_neither_condition_met(self, mock_dist):
        """Test no destruction when both conditions are false."""
        mock_dist.is_initialized.return_value = False

        _maybe_destroy_process_group(should_destroy=False)

        mock_dist.barrier.assert_not_called()
        mock_dist.destroy_process_group.assert_not_called()
