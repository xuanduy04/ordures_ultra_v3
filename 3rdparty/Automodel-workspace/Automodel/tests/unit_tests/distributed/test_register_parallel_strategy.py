

import pytest
import torch.nn as nn

from nemo_automodel.components.distributed import parallelizer as p
from nemo_automodel.components.distributed.parallelizer import (
    ParallelizationStrategy,
    get_parallelization_strategy,
    register_parallel_strategy,
)


def test_register_parallel_strategy_decorator_registers_and_resolves_by_model_name():
    original_registry = dict(p.PARALLELIZATION_STRATEGIES)
    try:

        @register_parallel_strategy(name="OutOfTreeModel")
        class OutOfTreeModelStrat(ParallelizationStrategy):
            def parallelize(self, model: nn.Module, *args, **kwargs) -> nn.Module:
                return model

        # Registered under string key
        assert "OutOfTreeModel" in p.PARALLELIZATION_STRATEGIES

        # Shadow with a model class of the same name; lookup should resolve by name
        class OutOfTreeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

        model = OutOfTreeModel()
        strategy = get_parallelization_strategy(model)

        assert strategy is p.PARALLELIZATION_STRATEGIES["OutOfTreeModel"]
        assert isinstance(strategy, ParallelizationStrategy)
        assert strategy.parallelize(model) is model
    finally:
        # Restore registry to avoid test ordering side effects
        p.PARALLELIZATION_STRATEGIES.clear()
        p.PARALLELIZATION_STRATEGIES.update(original_registry)


def test_register_parallel_strategy_decorator_raises_error_if_not_a_strategy():
    with pytest.raises(ValueError):

        @register_parallel_strategy
        class NotAParallelizationStrategy:
            def parallelize(self, model: nn.Module, *args, **kwargs) -> nn.Module:
                return model


def test_register_parallel_strategy_with_custom_name_registers_under_provided_key():
    original_registry = dict(p.PARALLELIZATION_STRATEGIES)
    try:

        @register_parallel_strategy(name="CustomKeyName")
        class SomeStrategy(ParallelizationStrategy):
            def parallelize(self, model: nn.Module, *args, **kwargs) -> nn.Module:
                return model

        assert "CustomKeyName" in p.PARALLELIZATION_STRATEGIES
        assert isinstance(p.PARALLELIZATION_STRATEGIES["CustomKeyName"], ParallelizationStrategy)

        class CustomKeyName(nn.Module):
            def __init__(self) -> None:
                super().__init__()

        model = CustomKeyName()
        strategy = get_parallelization_strategy(model)
        assert strategy is p.PARALLELIZATION_STRATEGIES["CustomKeyName"]
        assert strategy.parallelize(model) is model
    finally:
        p.PARALLELIZATION_STRATEGIES.clear()
        p.PARALLELIZATION_STRATEGIES.update(original_registry)
