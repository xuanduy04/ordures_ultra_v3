"""Single-turn general_qa contract, independent of Ray and tensor libraries."""

from copy import deepcopy
import math


def prepare_example(example, index, max_output_tokens):
    """Keep labels outside generation input and constrain the supported task."""
    row = deepcopy(example)
    reward_metric_valid(row)
    if row["agent_ref"]["name"] not in (
        "general_qa_simple_agent", "general_qa_simple_agent_reasoning_off"
    ):
        raise ValueError("Initial Gym X-Token support is general_qa only")
    if row["should_use_judge"] is not False:
        raise ValueError("LLM judge is disabled for this KD-only integration")
    params = row["responses_create_params"]
    if set(params) - {"input", "temperature", "top_p", "max_output_tokens"}:
        raise ValueError("Unsupported generation override in Gym example")
    if not isinstance(row["expected_answer"], str) or not row["expected_answer"].strip():
        raise ValueError("general_qa requires a nonempty text answer")
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be a positive integer")
    messages = params["input"]
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("Expected one user turn; tool/multiturn adaptation is separate")
    if not isinstance(messages[0]["content"], str) or not messages[0]["content"].strip():
        raise ValueError("Empty or non-text prompt")
    if "tools" in params or "instructions" in params or "previous_response_id" in params:
        raise ValueError("Tool/system/history inputs require a separate reviewed adapter")
    params["temperature"] = 1.0
    params["top_p"] = 1.0
    params["max_output_tokens"] = max_output_tokens
    row["_rowidx"] = index
    return row, [{"role": "user", "content": messages[0]["content"]}]


def reward_metric_valid(example):
    """Exclude QA scores for code labels; this never changes the KD mask."""
    if "xtoken_data" not in example:
        return True
    metadata = example["xtoken_data"]
    task, valid = metadata["task_type"], metadata["reward_metric_valid"]
    if task not in ("math", "code") or type(valid) is not bool or valid != (task == "math"):
        raise ValueError("Invalid math/code QA reward semantics")
    return valid


def unpack_result(result, expected_prompt_ids):
    """Use the Gym-provided sampled IDs; never reconstruct them from text."""
    full = result["full_result"]
    instance = full["instance_config"] if "instance_config" in full else {}
    masked = "mask_sample" in instance and bool(instance["mask_sample"])
    messages = result["message_log"]
    if masked or len(messages) == 1:
        return list(expected_prompt_ids), 0.0, None
    if len(messages) != 2 or [m["role"] for m in messages] != ["user", "assistant"]:
        raise ValueError("Expected exactly one model call in Gym rollout")
    prompt = messages[0]["token_ids"]
    response = messages[1]["token_ids"]
    prompt = prompt.tolist() if hasattr(prompt, "tolist") else list(prompt)
    response = response.tolist() if hasattr(response, "tolist") else list(response)
    if prompt != expected_prompt_ids:
        raise ValueError("Gym prompt IDs differ from the configured student chat template")
    if not all(type(token) is int and token >= 0 for token in prompt + response):
        raise ValueError("Invalid sampled token IDs")
    reward = float(full["reward"])
    if not math.isfinite(reward):
        raise ValueError("Gym returned a non-finite reward")
    return prompt + response, float(bool(response)), reward


def restore_order(results, size):
    """Reject missing/duplicate results rather than pairing the wrong samples."""
    ordered = [None] * size
    for index, result in results:
        if type(index) is not int or not 0 <= index < size or ordered[index] is not None:
            raise ValueError("Invalid or duplicate Gym row index")
        ordered[index] = result
    if any(result is None for result in ordered):
        raise ValueError("Missing Gym rollout result")
    return ordered
