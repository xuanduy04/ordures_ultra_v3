
import importlib

from .package_info import __package_name__, __version__

__all__ = [
    "recipes",
    "shared",
    "components",
    "__version__",
    "__package_name__",
]

# Promote NeMoAutoModelForCausalLM, AutoModelForImageTextToText into the top level
# to enable: `from nemo_automodel import NeMoAutoModelForCausalLM`
try:
    # adjust this import path if your class lives somewhere else
    from nemo_automodel._transformers.auto_model import (
        NeMoAutoModelForCausalLM,
        NeMoAutoModelForImageTextToText,
        NeMoAutoModelForSequenceClassification,
        NeMoAutoModelForTextToWaveform,
    )  # noqa: I001

    globals()["NeMoAutoModelForCausalLM"] = NeMoAutoModelForCausalLM
    globals()["NeMoAutoModelForImageTextToText"] = NeMoAutoModelForImageTextToText
    globals()["NeMoAutoModelForSequenceClassification"] = NeMoAutoModelForSequenceClassification
    globals()["NeMoAutoModelForTextToWaveform"] = NeMoAutoModelForTextToWaveform
    __all__.append("NeMoAutoModelForCausalLM")
    __all__.append("NeMoAutoModelForImageTextToText")
    __all__.append("NeMoAutoModelForSequenceClassification")
    __all__.append("NeMoAutoModelForTextToWaveform")
except:
    # optional dependency might be missing,
    # leave the name off the module namespace so other imports still work
    pass


def __getattr__(name: str):
    """
    Lazily import and cache submodules listed in __all__ when accessed.

    Raises:
        AttributeError if the name isn’t in __all__.
    """
    if name in __all__:
        # import submodule on first access
        module = importlib.import_module(f"{__name__}.{name}")
        # cache it in globals() so future lookups do not re-import
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """
    Expose the names of all available submodules for auto-completion.
    """
    return sorted(__all__)
