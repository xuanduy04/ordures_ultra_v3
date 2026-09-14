

"""Local math dataset."""

import os
from typing import Any, Literal, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class LocalMathDataset:
    def __init__(
        self,
        data_path: str,
        problem_key: str,
        solution_key: str,
        split: Optional[str] = None,
        file_format: Literal["csv", "json"] = "csv",
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        ds = load_dataset(file_format, data_files=data_path)
        if split is not None:
            ds = ds[split]
        self._problem_key = problem_key
        self._solution_key = solution_key
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name=os.path.basename(data_path).split(".")[0],
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.math_data_processor

    def _rekey(self, data: dict[str, Any]):
        return {
            "problem": data[self._problem_key],
            "expected_answer": data[self._solution_key],
        }
