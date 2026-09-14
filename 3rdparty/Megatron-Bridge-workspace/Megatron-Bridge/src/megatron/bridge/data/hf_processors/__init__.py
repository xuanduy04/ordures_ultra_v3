

"""
HuggingFace dataset processors for use with HFDatasetBuilder.

This module contains processing functions that conform to the ProcessExampleFn protocol
and are designed to work with the HFDatasetConfig and HFDatasetBuilder classes.
"""

from .squad import process_squad_example


__all__ = [
    "process_squad_example",
]
