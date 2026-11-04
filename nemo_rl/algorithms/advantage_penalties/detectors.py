from __future__ import annotations

from typing import Any, Mapping

import torch


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


def apply_penalty_detection(results: list[dict], master_config: Mapping[str, Any] | None) -> dict[str, int]:
    """Detect advantage-level penalty triggers and annotate the rollout message logs.

    Every detection is gated by its ``grpo.advantage_penalty.*`` sub-block and, when enabled, attaches the message
    flag/mask consumed at advantage time by FlagAdvantagePenalty/NGramAdvantagePenalty. Returns a dict of per-penalty
    sample counts keyed by penalty name for rollout metrics.

    NOTE: These detections assume Gym-path message_log structure where roles strictly alternate "user" → "assistant".
    Tool responses are folded into user prompt tokens by _postprocess_nemo_gym_to_nemo_rl_result and never appear as
    separate message_log entries. Do not call from non-Gym rollout paths.

    Detections:
      1. empty_final_answer (text-based) Walks response["output"] in reverse to find the last message-type item. If no
         message item exists or its content text is empty, the model failed to produce a final answer. Skipped when the
         last output item is a function_call (model was mid-agentic-loop, not producing an empty answer). Marks the
         final assistant message with has_empty_final_answer. Data: full_result["response"]["output"] — message items
         have content[0]["text"].

      2. eos_token (token-based) The EOS token(s) (default ids "2,11", configurable via token_ids.eos as a single
         integer or comma-separated integers, e.g. "2,177") should never appear in any assistant generation. Marks every
         assistant message from the first EOS-bearing message to the final message with has_eos_token. Data:
         message_log[i]["token_ids"] where role == "assistant".

      3. malformed_thinking (token-based + string-based) Two complementary checks to catch malformed think tags: a)
         Token ID check: infers thinking mode from prompt token counts. If prompt has open==close:
         enable_thinking=False, expect 0 open and 0 close in generation. If prompt has open==close+1:
         enable_thinking=True, expect 0 open and 1 close in generation. Any other prompt pattern or mismatched
         generation counts is a violation. Token IDs configurable via token_ids.think_open / token_ids.think_close. b)
         String check: the model can spell out <think>/</think> with piecemeal regular tokens (e.g. "<", "/", "thi",
         "nk", ">") that bypass special token IDs. Checks generation_str (decoded generation text) per output item:
         "<think>" count must be 0 (always in prompt, never generated), "</think>" count must be 0 or 1. Marks each
         violative assistant message with has_malformed_thinking. Data: message_log pairs for token IDs, full_result
         output items for strings.

      4. n_gram_repetition (token-based) Counts repeated N-gram windows (grpo.advantage_penalty.n_gram_repetition.size /
         grpo.advantage_penalty.n_gram_repetition.threshold) within each assistant generation via rolling hash and
         attaches a per-token overlap-count mask (n_gram_repetition_mask) to each assistant message. Data:
         message_log[i]["token_ids"] where role == "assistant".
    """
    counts = {"empty_final_answer": 0, "eos_token": 0, "malformed_thinking": 0, "n_gram_repetition": 0}
    if not master_config or not results:
        return counts

    grpo_cfg = master_config.get("grpo") or {}
    advantage_penalty_cfg = grpo_cfg.get("advantage_penalty") or {}

    enabled_detections = [
        key
        for key in ("empty_final_answer", "eos_token", "malformed_thinking", "n_gram_repetition")
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

    return counts


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


def _detect_n_gram_repetition(results: list[dict], advantage_penalty_cfg: Mapping[str, Any]) -> int:
    """Attach per-token N-gram repetition masks to assistant messages and count offending samples."""
    n_gram_cfg = advantage_penalty_cfg.get("n_gram_repetition", {})
    n_gram_size = int(n_gram_cfg.get("size", 4))
    n_gram_threshold = int(n_gram_cfg.get("threshold", 2))
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
