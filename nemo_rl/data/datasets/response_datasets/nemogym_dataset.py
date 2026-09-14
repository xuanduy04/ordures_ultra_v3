

from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class NemoGymDataset(RawDataset):
    """Simple wrapper around the Nemo Gym dataset.

    Args:
        data_path: Path to the dataset JSONL file
        repeat: Number of times to repeat the dataset, default is 1
    """

    def __init__(self, data_path: str, repeat: int = 1, **kwargs) -> None:
        self.task_name = "-".join(data_path.split("/")[-2:]).split(".")[0]
        if self.task_name[0] == "-":
            self.task_name = self.task_name[1:]

        # load raw line from jsonl
        # will use `json.loads` to load to dict format at `nemo_gym_data_processor` later since `Dataset` cannot handle nested structure well
        with open(data_path) as f:
            self.dataset = [raw_line for raw_line in f]

        # format the dataset
        self.dataset = Dataset.from_dict(
            {
                "extra_env_info": self.dataset,
                "task_name": [self.task_name] * len(self.dataset),
            }
        )

        # repeat the dataset
        if repeat > 1:
            self.dataset = self.dataset.repeat(repeat)
