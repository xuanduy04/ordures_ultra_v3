
from typing import Optional

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path


class PreferenceDataset(RawDataset):
    """Dataset class for preference data which can be loaded from a JSON file.

    This class handles loading of preference data for DPO and RM training.
    The input JSONL files should contain valid JSON objects formatted like this:
    {
        "context": list[dict],              # The prompt message (including previous turns, if any)
        "completions": [                    # The list of completions
            {
                "rank": 0,                  # The rank of the completion (lower rank is preferred)
                "completion": list[dict],   # The completion message(s)
            },
            {
                "rank": 1,                  # The rank of the completion (lower rank is preferred)
                "completion": list[dict],   # The completion message(s)
            },
            ...                             # More completions can be added if needed
        ]
    }
    Please refer to https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/dpo.md#datasets for more details.

    Args:
        data_path: Path to the dataset JSON file
        subset: Optional subset name for the dataset, used for HuggingFace datasets
        split: Optional split name for the dataset, used for HuggingFace datasets
    """

    def __init__(
        self,
        data_path: str,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        **kwargs,
    ):
        self.task_name = "-".join(data_path.split("/")[-2:]).split(".")[0]
        if self.task_name[0] == "-":
            self.task_name = self.task_name[1:]

        # load from local or huggingface
        self.dataset = load_dataset_from_path(data_path, subset, split)

        # format the dataset
        self.dataset = self.dataset.add_column(
            "task_name", [self.task_name] * len(self.dataset)
        )
