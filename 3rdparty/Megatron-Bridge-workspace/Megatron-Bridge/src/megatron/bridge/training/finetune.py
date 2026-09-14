

from megatron.bridge.training.callbacks import Callback, CallbackManager
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.forward_step_func_types import ForwardStepCallable
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.utils.decorators import experimental_fn


@experimental_fn
def finetune(
    config: ConfigContainer,
    forward_step_func: ForwardStepCallable,
    callbacks: list[Callback] | CallbackManager | None = None,
) -> None:
    """Main function to run the finetuning.

    Args:
        config: The main configuration container holding all necessary parameters.
        forward_step_func: A callable (function or functor) that performs a single
                          forward and backward step, returning the loss and any computed
                          metrics. Supports the following signatures:
                          - 2 args: (data_iterator, model)
                          - 3 args: (data_iterator, model, return_schedule_plan=False)
                                   OR (state: GlobalState, data_iterator, model)
                          - 4 args: (state: GlobalState, data_iterator, model, return_schedule_plan=False)
        callbacks: Optional list of Callback instances, a CallbackManager, or None.

    Note:
        Use the signature with GlobalState type hint for full access to configuration, timers, and training state.
        State injection is automatic based on type hints or parameter names.
        Functors (classes with __call__) are fully supported.

    Warnings:
        This is an experimental API and is subject to change in backwards
        incompatible ways without notice.
    """
    assert config.checkpoint.pretrained_checkpoint is not None or config.checkpoint.load is not None, (
        "Finetuning requires a loading from a pretrained checkpoint or resuming from a checkpoint"
    )
    return pretrain(config, forward_step_func, callbacks=callbacks)
