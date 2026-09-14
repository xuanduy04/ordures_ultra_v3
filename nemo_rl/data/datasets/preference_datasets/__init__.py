
from nemo_rl.data import PreferenceDatasetConfig
from nemo_rl.data.datasets.preference_datasets.binary_preference_dataset import (
    BinaryPreferenceDataset,
)
from nemo_rl.data.datasets.preference_datasets.helpsteer3 import HelpSteer3Dataset
from nemo_rl.data.datasets.preference_datasets.preference_dataset import (
    PreferenceDataset,
)
from nemo_rl.data.datasets.preference_datasets.tulu3 import Tulu3PreferenceDataset

DATASET_REGISTRY = {
    # built-in datasets
    "HelpSteer3": HelpSteer3Dataset,
    "Tulu3Preference": Tulu3PreferenceDataset,
    # load from local JSONL file or HuggingFace
    "BinaryPreferenceDataset": BinaryPreferenceDataset,
    "PreferenceDataset": PreferenceDataset,
}


def load_preference_dataset(data_config: PreferenceDatasetConfig):
    """Loads preference dataset."""
    dataset_name = data_config["dataset_name"]

    # load dataset
    if dataset_name in DATASET_REGISTRY:
        dataset_class = DATASET_REGISTRY[dataset_name]
        dataset = dataset_class(
            **data_config  # pyrefly: ignore[missing-argument]  `data_path` is required for some classes
        )
    else:
        raise ValueError(
            f"Unsupported {dataset_name=}. "
            "Please either use a built-in dataset "
            "or set dataset_name in {'BinaryPreferenceDataset', 'PreferenceDataset'} to load from local JSONL file or HuggingFace."
        )

    # bind prompt and system prompt
    dataset.set_task_spec(data_config)

    return dataset


__all__ = [
    "BinaryPreferenceDataset",
    "HelpSteer3Dataset",
    "PreferenceDataset",
    "Tulu3PreferenceDataset",
    "load_preference_dataset",
]
