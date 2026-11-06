from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, Generic, Sequence, Mapping, Self, ClassVar, get_args, get_origin, cast


from nemo_rl.algorithms.advantage_estimator import BaseAdvantageEstimator, ADVANTAGE_ESTIMATOR_MAPPING

if TYPE_CHECKING:
    from torch import Tensor

    from nemo_rl.distributed.batched_data_dict import BatchedDataDict


DEFAULT_ADVANTAGE_PENALTY_NORMALIZER_NAME = "grpo"
DEFAULT_ADVANTAGE_PENALTY_WEIGHT = 0.1
DEFAULT_MAX_ADVANTAGE_PENALTY = 6767

DEFAULT_CONFIG_KEYS = frozenset(
    {"default_weight", "default_advantage_penalty_normalizer", "default_max_advantage_penalty"}
)

_PENALTY_NORMALIZER_DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "reinforce_plus_plus": {"minus_baseline": False},
    "reinforceplusplus": {"minus_baseline": False},
    "reinforce++": {"minus_baseline": False},
    "grpo": {"normalize_rewards": True, "use_leave_one_out_baseline": False},
    "reinforce": {},
}


@dataclass(frozen=True, kw_only=True)
class BaseAdvantagePenaltyDefinition:
    name: str  # Penalty name used in logging
    config_key: str  # ``advantage_penalty`` sub-block to enable and configure the penalty


@dataclass(frozen=True, kw_only=True)
class BaseAdvantagePenaltySettings:
    weight: float  # Scale factor for the penalty magnitude (sign is ignored)
    normalizer: BaseAdvantageEstimator = field(compare=False, repr=False)  # Advantage normalizer for the penalty signal
    max_advantage_penalty: float  # Magnitude bounding the per-rollout penalty

    @classmethod
    def from_config(cls, name: str, block: Mapping[str, Any], cfg: Mapping[str, Any], loss_config: dict) -> Self:
        """Resolve settings for penalty ``name`` from its sub-block, falling back to ``cfg``'s default_* keys."""
        return cls(**_resolve_common_settings(name, block, cfg, loss_config))


def _resolve_common_settings(
    name: str, block: Mapping[str, Any], cfg: Mapping[str, Any], loss_config: dict
) -> dict[str, Any]:
    normalizer_name = block.get(
        "advantage_penalty_normalizer",
        cfg.get("default_advantage_penalty_normalizer", DEFAULT_ADVANTAGE_PENALTY_NORMALIZER_NAME),
    )
    # Validate against the dict actually indexed below (fixes the latent KeyError)
    if normalizer_name not in _PENALTY_NORMALIZER_DEFAULT_CONFIGS:
        raise ValueError(
            f"Invalid advantage_penalty_normalizer {normalizer_name!r} for penalty {name!r}. "
            f"Supported: {sorted(_PENALTY_NORMALIZER_DEFAULT_CONFIGS)}"
        )
    normalizer = ADVANTAGE_ESTIMATOR_MAPPING[normalizer_name](
        {**_PENALTY_NORMALIZER_DEFAULT_CONFIGS[normalizer_name], "name": normalizer_name}, loss_config
    )

    weight = abs(block.get("weight", cfg.get("default_weight", DEFAULT_ADVANTAGE_PENALTY_WEIGHT)))
    max_advantage_penalty = abs(
        block.get("max_advantage_penalty", cfg.get("default_max_advantage_penalty", DEFAULT_MAX_ADVANTAGE_PENALTY))
    )

    return {"weight": weight, "normalizer": normalizer, "max_advantage_penalty": max_advantage_penalty}


D = TypeVar("D", bound="BaseAdvantagePenaltyDefinition")
S = TypeVar("S", bound="BaseAdvantagePenaltySettings")


@dataclass(frozen=True, kw_only=True)
class AdvantagePenaltySpec(Generic[D, S]):
    definition: D
    settings: S

    @property
    def name(self) -> str:
        return self.definition.name


class BaseAdvantagePenalty(ABC, Generic[D, S]):
    """Base class for advantage penalties.

    A penalty is a pseudo-reward: it builds a per-rollout severity signal from the batch, has it normalized by the
    shared advantage-estimator normalizer, and places the weighted penalty into the offending messages' token
    advantages.
    """

    DEFINITIONS: ClassVar[tuple[BaseAdvantagePenaltyDefinition, ...]]
    _settings_type: ClassVar[type[BaseAdvantagePenaltySettings]]

    # Functions for subclassing and inheritance
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        args = next(
            (get_args(b) for b in getattr(cls, "__orig_bases__", ()) if get_origin(b) is BaseAdvantagePenalty), None
        )
        if args is None:
            return  # not a direct concrete subclass (e.g. an intermediate generic base)
        definition_type, settings_type = args

        definitions = getattr(cls, "DEFINITIONS", None)
        if not definitions:
            raise TypeError(f"{cls.__name__} must declare a non-empty DEFINITIONS")
        if bad := [d for d in definitions if not isinstance(d, definition_type)]:
            raise TypeError(f"{cls.__name__}.DEFINITIONS entries must be {definition_type.__name__}, got {bad}")
        for attr in ("name", "config_key"):
            values = [getattr(d, attr) for d in definitions]
            if len(values) != len(set(values)):
                raise TypeError(f"{cls.__name__}: duplicate definition {attr}s: {values}")

        cls._settings_type = settings_type

    @classmethod
    def definitions(cls) -> tuple[D, ...]:
        # Safe: __init_subclass__ checks every entry is an instance of this class's D.
        return cast("tuple[D, ...]", cls.DEFINITIONS)

    @classmethod
    def settings_type(cls) -> type[S]:
        return cast("type[S]", cls._settings_type)

    # Functions to initialize penalties from the config
    @classmethod
    def config_keys(cls) -> frozenset[str]:
        return frozenset(d.config_key for d in cls.definitions())

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], loss_config: dict) -> Self | None:
        settings_type = cls.settings_type()
        specs = [
            AdvantagePenaltySpec(
                definition=definition, settings=settings_type.from_config(definition.name, block, cfg, loss_config)
            )
            for definition in cls.definitions()
            if (block := cfg.get(definition.config_key) or {}).get("enabled", False)
        ]
        return cls(specs) if specs else None

    # Functions to actually use the penalties
    def __init__(self, penalties: Sequence[AdvantagePenaltySpec[D, S]]):
        self.penalties: tuple[AdvantagePenaltySpec[D, S], ...] = tuple(penalties)

    @abstractmethod
    def compute_signal(self, batch: BatchedDataDict) -> dict[str, Tensor]:
        """Extract the per-rollout severity signal for every penalty.

        Args:
            batch: The rollout batch containing the message logs.

        Returns:
            Mapping of penalty name to a tensor of shape [batch_size] with non-negative severities (higher = worse).
        """
        ...

    @abstractmethod
    def apply(self, advantages: Tensor, batch: BatchedDataDict, penalties: dict[str, Tensor]) -> None:
        """Add weighted per-rollout penalties to the offending messages' tokens.

        Args:
            advantages: Token-level advantages of shape [batch_size, seq_len]. Modified in-place.
            batch: The rollout batch containing the message logs.
            penalties: Mapping of a penalty name to the normalized and weighted per-rollout penalty values of shape
                [batch_size]. NOTE: ``settings.weight`` already applied, implementations must not scale again.
        """
        ...
