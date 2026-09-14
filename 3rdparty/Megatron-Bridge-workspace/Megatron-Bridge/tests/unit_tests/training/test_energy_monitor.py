

from unittest.mock import Mock, patch

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.state import GlobalState


class TestEnergyMonitorIntegration:
    """Test suite for energy monitor integration in the training pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock configs for all required ConfigContainer fields
        mock_train = Mock()
        mock_train.micro_batch_size = 1
        mock_train.train_iters = 100
        mock_train.exit_signal = None

        mock_model = Mock()
        mock_optimizer = Mock()
        mock_scheduler = Mock()
        mock_dataset = Mock()
        mock_logger = Mock()
        mock_logger.log_energy = False
        mock_logger.log_interval = 1
        mock_logger.tensorboard_log_interval = 1
        mock_logger.log_timers_to_tensorboard = False
        mock_logger.log_loss_scale_to_tensorboard = False
        mock_logger.log_world_size_to_tensorboard = False
        mock_logger.log_memory_to_tensorboard = False
        mock_logger.log_params_norm = False
        mock_logger.tensorboard_dir = None  # Set to None to avoid creating real tensorboard logger
        mock_logger.tensorboard_queue_size = 1000

        mock_tokenizer = Mock()
        mock_checkpoint = Mock()

        # Create ConfigContainer with all required fields
        self.config = ConfigContainer(
            train=mock_train,
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            dataset=mock_dataset,
            logger=mock_logger,
            tokenizer=mock_tokenizer,
            checkpoint=mock_checkpoint,
        )
        self.config.data_parallel_size = 1

    def test_energy_monitor_lazy_initialization_disabled(self):
        """Test that energy monitor is not created when log_energy is False."""
        self.config.logger.log_energy = False
        global_state = GlobalState()
        global_state.cfg = self.config

        assert global_state.energy_monitor is None
        assert global_state._energy_monitor_created is False

    def test_energy_monitor_lazy_initialization_enabled(self):
        """Test that energy monitor is created when log_energy is True."""
        self.config.logger.log_energy = True

        with patch("megatron.bridge.training.state.EnergyMonitor") as mock_energy_monitor:
            global_state = GlobalState()
            global_state.cfg = self.config

            # Access the property to trigger lazy initialization
            energy_monitor = global_state.energy_monitor

            assert energy_monitor is not None
            assert global_state._energy_monitor_created is True
            mock_energy_monitor.assert_called_once()

    def test_energy_monitor_single_initialization(self):
        """Test that energy monitor is only created once."""
        self.config.logger.log_energy = True

        with patch("megatron.bridge.training.state.EnergyMonitor") as mock_energy_monitor:
            global_state = GlobalState()
            global_state.cfg = self.config

            # Access the property multiple times
            energy_monitor1 = global_state.energy_monitor
            energy_monitor2 = global_state.energy_monitor

            assert energy_monitor1 is energy_monitor2
            mock_energy_monitor.assert_called_once()

    def test_energy_monitor_no_config(self):
        """Test that energy monitor is not created when no config is set."""
        global_state = GlobalState()
        # Don't set config

        assert global_state.energy_monitor is None

    def test_energy_monitor_config_change(self):
        """Test energy monitor behavior when config changes."""
        # Start with disabled
        self.config.logger.log_energy = False
        global_state = GlobalState()
        global_state.cfg = self.config

        assert global_state.energy_monitor is None

        # Change to enabled
        self.config.logger.log_energy = True

        with patch("megatron.bridge.training.state.EnergyMonitor") as mock_energy_monitor:
            # Access after config change
            energy_monitor = global_state.energy_monitor

            assert energy_monitor is not None
            mock_energy_monitor.assert_called_once()
