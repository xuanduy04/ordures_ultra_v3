

from megatron.bridge.models.distillation_provider import DistillationProvider
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step_modelopt
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.utils.decorators import experimental_fn


@experimental_fn
def distill(
    config: ConfigContainer,
) -> None:
    """Main function to run knowledge distillation (KD).

    Args:
        config: The main configuration container holding all necessary parameters.

    Warnings:
        This is an experimental API and is subject to change in backwards
        incompatible ways without notice.
    """
    assert isinstance(config.model, DistillationProvider), "Distillation requires a DistillationProvider"

    return pretrain(config, forward_step_modelopt)
