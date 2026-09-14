

import torch

from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def create_mock_batch_with_responses(
    num_samples: int,
    response_lengths: list[int],
    initial_rewards: list[float],
    task_names: list[str] = None,
) -> BatchedDataDict[DatumSpec]:
    """Helper function to create a mock batch with specified response lengths and initial rewards."""
    if task_names is None:
        task_names = ["math"] * num_samples

    message_logs = []
    for i, length in enumerate(response_lengths):
        # Create dummy token_ids for assistant response with specified length
        assistant_tokens = torch.arange(length, dtype=torch.long)
        user_tokens = torch.tensor([100, 101, 102], dtype=torch.long)

        message_log = [
            {"role": "user", "content": f"Question {i}", "token_ids": user_tokens},
            {
                "role": "assistant",
                "content": f"Response {i}",
                "token_ids": assistant_tokens,
            },
        ]
        message_logs.append(message_log)

    return BatchedDataDict[DatumSpec](
        {
            "task_name": task_names,
            "message_log": message_logs,
            "extra_env_info": [{} for _ in range(num_samples)],
            "loss_multiplier": torch.ones(num_samples),
            "total_reward": torch.tensor(initial_rewards),
        }
    )


def create_mock_batch(
    num_samples: int,
    task_names: list[str],
    message_logs: list[LLMMessageLogType],
    extra_env_info: list[dict] = None,
) -> BatchedDataDict[DatumSpec]:
    """Helper function to create a mock batch for testing."""
    if extra_env_info is None:
        extra_env_info = [{} for _ in range(num_samples)]

    return BatchedDataDict[DatumSpec](
        {
            "task_name": task_names,
            "message_log": message_logs,
            "extra_env_info": extra_env_info,
            "loss_multiplier": torch.ones(num_samples),
        }
    )
