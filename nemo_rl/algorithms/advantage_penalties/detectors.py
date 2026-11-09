from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from nemo_rl.algorithms.advantage_penalties.flag_advantage_penalty import DEFAULT_INVALID_TOOL_CALL_PATTERNS
from nemo_rl.algorithms.advantage_penalties.tool_call_expectations import expects_tool_call


# ========================= Main detection function ========================= #
def apply_penalty_detection(
    results: list[dict], master_config: Mapping[str, Any] | None, nemo_gym_rows: list[dict] | None = None
) -> dict[str, int]:
    """Run each enabled ``grpo.advantage_penalty.*`` detector, which annotates the message logs in place.

    NOTE: These detections assume Gym-path message_log structure where roles strictly alternate "user" → "assistant".
    Tool responses are folded into user prompt tokens by _postprocess_nemo_gym_to_nemo_rl_result and never appear as
    separate message_log entries. Do not call from non-Gym rollout paths.

    Returns:
        Per-penalty counts of flagged samples, keyed by penalty name.
    """
    counts = {
        "empty_final_answer": 0,
        "eos_token": 0,
        "malformed_thinking": 0,
        "n_gram_repetition": 0,
        "invalid_tool_call": 0,
        "missing_tool_call": 0,
    }
    if not master_config or not results:
        return counts

    grpo_cfg = master_config.get("grpo") or {}
    advantage_penalty_cfg = grpo_cfg.get("advantage_penalty") or {}

    enabled_detections = [
        key
        for key in (
            "empty_final_answer",
            "eos_token",
            "malformed_thinking",
            "n_gram_repetition",
            "invalid_tool_call",
            "missing_tool_call",
        )
        if advantage_penalty_cfg.get(key, {}).get("enabled", False)
    ]

    # Guard: detections rely on Gym-path message_log (strictly alternating user/assistant roles).
    # Non-Gym paths may have "environment", "tool", or "system" roles which these checks don't handle.
    if enabled_detections:
        for result in results:
            roles = {msg.get("role") for msg in result["message_log"]}
            assert roles <= {"user", "assistant"}, (
                f"apply_penalty_detection requires Gym-path message_log with only 'user' and 'assistant' roles, "
                f"but found roles: {roles}. These detections are not supported for non-Gym rollout paths."
            )

    if "empty_final_answer" in enabled_detections:
        counts["empty_final_answer"] = _detect_empty_final_answer(results)
    if "eos_token" in enabled_detections:
        counts["eos_token"] = _detect_eos_token(results, master_config)
    if "malformed_thinking" in enabled_detections:
        counts["malformed_thinking"] = _detect_malformed_thinking(results, master_config)
    if "n_gram_repetition" in enabled_detections:
        counts["n_gram_repetition"] = _detect_n_gram_repetition(results, advantage_penalty_cfg)
    detect_invalid = "invalid_tool_call" in enabled_detections
    detect_missing = "missing_tool_call" in enabled_detections
    if detect_missing:
        assert nemo_gym_rows is not None and len(nemo_gym_rows) == len(results), (
            "missing_tool_call detection requires the raw NeMo-Gym rows (one per result), "
            f"got {None if nemo_gym_rows is None else len(nemo_gym_rows)} rows for {len(results)} results."
        )
    if detect_invalid or detect_missing:
        counts["invalid_tool_call"], counts["missing_tool_call"] = _detect_tool_call_issues(
            results, nemo_gym_rows, advantage_penalty_cfg, detect_invalid, detect_missing
        )

    return counts


# ========================= Flag-based ========================= #


# --------------- chat template + empty answers --------------- #


def _detect_empty_final_answer(results: list[dict]) -> int:
    """Mark samples whose final answer is empty with has_empty_final_answer on the last assistant message."""
    count = 0
    for result in results:
        output_items = result["full_result"].get("response", {}).get("output", [])
        # Skip if the last output item is a function_call — it is legit for model to
        # produce reasoning and then a function_call as the last output item in PivotRL
        if output_items and output_items[-1].get("type") == "function_call":
            continue
        final_answer_text = None
        for item in reversed(output_items):
            # Skip items without content (function_call, function_call_output, etc.)
            if "content" not in item:
                continue
            content = item["content"]
            if isinstance(content, list) and content and "text" in content[0]:
                final_answer_text = content[0]["text"].strip()
                break
            elif isinstance(content, str):
                final_answer_text = content.strip()
                break
        if final_answer_text is None or final_answer_text.strip() == "":
            count += 1
            for msg in reversed(result["message_log"]):
                if msg["role"] == "assistant":
                    msg["has_empty_final_answer"] = True
                    break
    return count


def _detect_malformed_thinking(results: list[dict], master_config: Mapping[str, Any]) -> int:
    """Mark samples with malformed think tags (token-ID + string checks) with has_malformed_thinking."""
    token_ids_cfg = master_config.get("token_ids", {})
    think_open_token_id = token_ids_cfg.get("think_open", 12)
    think_close_token_id = token_ids_cfg.get("think_close", 13)
    count = 0
    for result in results:
        has_violation = False
        # 4a) Token ID check per (user, assistant) turn pair
        # Infer thinking mode from prompt token counts:
        #   enable_thinking=True:  prompt has open=close+1 (trailing <think>), expect asst: 0 open, 1 close
        #   enable_thinking=False: prompt has open=close (balanced), expect asst: 0 open, 0 close
        msgs = result["message_log"]
        for i in range(len(msgs) - 1):
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                user_ids = msgs[i]["token_ids"]
                asst_ids = msgs[i + 1]["token_ids"]
                prompt_open = (user_ids == think_open_token_id).sum().item()
                prompt_close = (user_ids == think_close_token_id).sum().item()
                asst_open = (asst_ids == think_open_token_id).sum().item()
                asst_close = (asst_ids == think_close_token_id).sum().item()
                if prompt_open == prompt_close:
                    # enable_thinking=False: both tags in prompt, none in generation
                    expected_open, expected_close = 0, 0
                elif prompt_open == prompt_close + 1:
                    # enable_thinking=True: trailing <think> in prompt, expect </think> in generation
                    expected_open, expected_close = 0, 1
                else:
                    # Unexpected prompt pattern — flag as violation
                    has_violation = True
                    msgs[i + 1]["has_malformed_thinking"] = True
                    continue
                if asst_open != expected_open or asst_close != expected_close:
                    has_violation = True
                    msgs[i + 1]["has_malformed_thinking"] = True
        # 4b) String check on generation_str per output item
        output_items = result["full_result"].get("response", {}).get("output", [])
        assistant_msgs = [msg for msg in msgs if msg["role"] == "assistant"]
        gen_item_idx = 0
        for item in output_items:
            gen_str = item.get("generation_str", "")
            if not gen_str:
                continue
            if gen_str.count("<think>") > 0 or gen_str.count("</think>") > 1:
                has_violation = True
                if gen_item_idx < len(assistant_msgs):
                    assistant_msgs[gen_item_idx]["has_malformed_thinking"] = True
            gen_item_idx += 1  # increment idx only for generation turns
        if has_violation:
            count += 1
    return count


# --------------- tool call issues --------------- #


def _is_invalid_tool_call_item(output_item: dict, invalid_tool_call_patterns: Sequence[str]) -> bool:
    """Return whether a turn's final output item spells out a tool call in its text instead of a structured call."""
    # If the final item is a message, check its text content. If it's a reasoning item, the model output only
    # reasoning (no content/tool calls), so check its summary text.
    is_output_message = (
        "content" in output_item
        and isinstance(output_item["content"], list)
        and len(output_item["content"]) > 0
        and "text" in output_item["content"][0]
    )
    is_reasoning_message = (
        "type" in output_item
        and output_item["type"] == "reasoning"
        and len(output_item["summary"]) > 0
        and "text" in output_item["summary"][0]
    )
    if is_output_message:
        text = output_item["content"][0]["text"]
    elif is_reasoning_message:
        text = output_item["summary"][0]["text"]
    else:
        return False
    return any(pattern in text for pattern in invalid_tool_call_patterns)


def _detect_tool_call_issues(
    results: list[dict],
    nemo_gym_rows: list[dict] | None,
    advantage_penalty_cfg: Mapping[str, Any],
    detect_invalid: bool,
    detect_missing: bool,
) -> tuple[int, int]:
    """Mark invalid (is_invalid_tool_call) and missing (is_missing_tool_call) tool calls per assistant turn.

    Returns:
        Number of samples with at least one invalid tool call turn, and with at least one missing tool call turn.
    """
    if (
        "invalid_tool_call" in advantage_penalty_cfg
        and "invalid_tool_call_patterns" in advantage_penalty_cfg["invalid_tool_call"]
    ):
        invalid_tool_call_patterns = tuple(advantage_penalty_cfg["invalid_tool_call"]["invalid_tool_call_patterns"])
    else:
        invalid_tool_call_patterns = DEFAULT_INVALID_TOOL_CALL_PATTERNS

    invalid_count = 0
    missing_count = 0
    for sample_idx, result in enumerate(results):
        sample_expects_tool_call = (
            detect_missing and nemo_gym_rows is not None and expects_tool_call(nemo_gym_rows[sample_idx])
        )
        assistant_msgs = [msg for msg in result["message_log"] if msg["role"] == "assistant"]
        output_items = result["full_result"]["response"]["output"]

        # Split output items into turns: _postprocess_nemo_gym_to_nemo_rl_result attaches generation_str to exactly
        # the final (token-bearing) output item of each model call, i.e. one item per assistant message.
        turns: list[tuple[dict, bool]] = []  # (final output item, turn contains a function_call)
        turn_has_function_call = False
        for item in output_items:
            if "type" in item and item["type"] == "function_call":
                turn_has_function_call = True
            if "generation_str" in item:
                turns.append((item, turn_has_function_call))
                turn_has_function_call = False
        assert len(turns) == len(assistant_msgs), (
            f"Expected one generation_str-bearing output item per assistant message, "
            f"got {len(turns)} turns for {len(assistant_msgs)} assistant messages."
        )

        sample_invalid = False
        sample_missing = False
        for (final_item, has_function_call), msg in zip(turns, assistant_msgs):
            is_invalid = _is_invalid_tool_call_item(final_item, invalid_tool_call_patterns)
            if detect_invalid and is_invalid:
                msg["is_invalid_tool_call"] = True
                sample_invalid = True
            if sample_expects_tool_call and not has_function_call and not is_invalid:
                msg["is_missing_tool_call"] = True
                sample_missing = True
        invalid_count += sample_invalid
        missing_count += sample_missing
    return invalid_count, missing_count


# --------------- eos token in generation --------------- #


def _detect_eos_token(results: list[dict], master_config: Mapping[str, Any]) -> int:
    """Mark samples whose assistant generations contain EOS with has_eos_token from the first EOS to the end."""
    token_ids_cfg = master_config.get("token_ids", {})
    eos_cfg = token_ids_cfg.get("eos", "2,11")
    if isinstance(eos_cfg, str):
        eos_token_ids: set[int] = set()
        for part in eos_cfg.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                eos_token_ids.add(int(part))
            except ValueError:
                raise ValueError(
                    f"Invalid token_ids.eos entry {part!r} in {eos_cfg!r}; expected comma-separated integer token IDs."
                ) from None
        if not eos_token_ids:
            raise ValueError(f"token_ids.eos value {eos_cfg!r} contains no token IDs.")
    else:
        try:
            eos_token_ids: set[int] = {int(eos_cfg)}
        except ValueError:
            raise ValueError(f"Invalid token_ids.eos entry '{eos_cfg!r}'; expected integer token IDs.") from None

    count = 0
    for result in results:
        first_eos_idx = None
        for idx, msg in enumerate(result["message_log"]):
            if msg["role"] == "assistant" and any(eos_token_id in msg["token_ids"] for eos_token_id in eos_token_ids):
                first_eos_idx = idx
                count += 1
                break
        if first_eos_idx is not None:
            # Punish from the literal first EOS token in
            # generation until the final token, assistant turns only.
            for msg in result["message_log"][first_eos_idx:]:
                if msg["role"] == "assistant":
                    msg["has_eos_token"] = True
    return count


# ========================= N-Gram Repetition ========================= #


def _n_gram_repetition_mask(token_ids: torch.Tensor, n_gram_size: int, repetition_threshold: int) -> torch.Tensor:
    """Per-token overlap-count mask for repeated N-grams via rolling hash.

    Each position holds how many N-gram windows, whose occurrence rank is at least `repetition_threshold`, cover it. The
    first `repetition_threshold - 1` occurrences of each distinct N-gram are exempt.

    Uses a single-modulus polynomial hash (2^61 - 1) for O(T) counting.
    """
    _N_GRAM_HASH_BASE = 1_000_003
    _N_GRAM_HASH_MODULUS = (1 << 61) - 1

    token_id_list = token_ids.tolist()
    num_tokens = len(token_id_list)
    repetition_mask = torch.zeros(num_tokens)
    if num_tokens < n_gram_size:
        return repetition_mask
    rolling_hash = 0
    for token_id in token_id_list[:n_gram_size]:
        rolling_hash = (rolling_hash * _N_GRAM_HASH_BASE + token_id) % _N_GRAM_HASH_MODULUS
    oldest_token_multiplier = pow(_N_GRAM_HASH_BASE, n_gram_size - 1, _N_GRAM_HASH_MODULUS)
    occurrence_counts = {}
    for window_end in range(n_gram_size - 1, num_tokens):
        occurrence_counts[rolling_hash] = occurrence_counts.get(rolling_hash, 0) + 1
        if occurrence_counts[rolling_hash] >= repetition_threshold:
            repetition_mask[window_end - n_gram_size + 1 : window_end + 1] += 1.0
        if window_end < num_tokens - 1:
            rolling_hash = (
                (rolling_hash - token_id_list[window_end - n_gram_size + 1] * oldest_token_multiplier)
                * _N_GRAM_HASH_BASE
                + token_id_list[window_end + 1]
            ) % _N_GRAM_HASH_MODULUS
    return repetition_mask


def _detect_n_gram_repetition(results: list[dict], advantage_penalty_cfg: Mapping[str, Any]) -> int:
    """Attach per-token N-gram repetition masks to assistant messages and count offending samples."""
    n_gram_cfg = advantage_penalty_cfg.get("n_gram_repetition", {})
    n_gram_size = int(n_gram_cfg.get("size", 12))
    n_gram_threshold = int(n_gram_cfg.get("threshold", 3))
    count = 0
    for result in results:
        sample_repeated = False
        for msg in result["message_log"]:
            if msg["role"] != "assistant":
                continue
            repeat_mask = _n_gram_repetition_mask(msg["token_ids"], n_gram_size, n_gram_threshold)
            msg["n_gram_repetition_mask"] = repeat_mask
            if bool(repeat_mask.any()):
                sample_repeated = True
        if sample_repeated:
            count += 1
    return count
