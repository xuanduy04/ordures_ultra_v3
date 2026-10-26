"""Exercise the Gym phase adapter with real CPU tensors and mocked Ray I/O."""

from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms.x_token.gym_rollout import NemoGymOnPolicyRollout


def harness(monkeypatch):
    from nemo_rl.algorithms.x_token import gym_rollout

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.apply_chat_template.return_value = [3, 4]
    generation = MagicMock()
    generation.prepare_for_generation.return_value = True
    generation.finish_generation.return_value = True
    environment = MagicMock()
    payload = {"full_result": {"reward": 0.75}, "message_log": [
        {"role": "user", "token_ids": torch.tensor([3, 4])},
        {"role": "assistant", "token_ids": torch.tensor([7, 8])}]}
    environment.run_rollouts.remote.side_effect = lambda *args: iter([(0, payload, None)])
    monkeypatch.setattr(gym_rollout.ray, "get", lambda reference, **kwargs: reference)
    monkeypatch.setattr(gym_rollout.ray, "wait", lambda references, **kwargs: (references, []))
    collator = MagicMock(return_value={"token_mask": torch.tensor([[0, 0, 1, 1]])})
    refit = MagicMock()
    rollout = NemoGymOnPolicyRollout(
        MagicMock(), generation, tokenizer, collator,
        {"max_new_tokens": 32}, {"max_prompt_tokens": 128, "refit_buffer_size_gb": 1},
        0, refit, environment=environment, gym_config={"result_timeout_s": 2})
    batch = {"gym_examples": [{"agent_ref": {"name": "general_qa_simple_agent_reasoning_off"},
                              "responses_create_params": {"input": [{"role": "user", "content": "2+3?"}]},
                              "question": "2+3?", "expected_answer": "5", "should_use_judge": False}],
             "sample_mask": torch.tensor([1.0]), "idx": [12]}
    return rollout, batch, generation, collator, refit


def test_sampled_ids_survive_two_distinct_versions(monkeypatch):
    rollout, batch, generation, collator, refit = harness(monkeypatch)
    rollout.generate(batch, training=True)
    sent, sampled, prompt_ids, version, rollout_id = collator.call_args.args
    assert sampled["output_ids"].tolist() == [[3, 4, 7, 8]]
    assert prompt_ids == [[3, 4]] and version == 0 and rollout_id == 1
    assert sent["sample_mask"].tolist() == [1.0]
    with pytest.raises(RuntimeError, match="Previous Gym rollout"):
        rollout.generate(batch, training=True)
    rollout.after_update()
    rollout.generate(batch, training=True)
    assert collator.call_args.args[3:] == (1, 2)
    assert refit.call_count == 2 and generation.finish_generation.call_count == 2


def test_stream_deadline_cancels_only_owned_request_and_releases_generation(monkeypatch):
    from nemo_rl.algorithms.x_token import gym_rollout

    rollout, batch, generation, _, _ = harness(monkeypatch)
    monkeypatch.setattr(gym_rollout.ray, "wait", lambda references, **kwargs: ([], references))
    cancel = MagicMock()
    monkeypatch.setattr(gym_rollout.ray, "cancel", cancel)
    with pytest.raises(TimeoutError):
        rollout.generate(batch, training=True)
    cancel.assert_called_once()
    generation.finish_generation.assert_called_once()
    assert rollout.pending_rollout is None


def test_release_failure_cannot_lease_an_update(monkeypatch):
    rollout, batch, generation, _, _ = harness(monkeypatch)
    generation.finish_generation.return_value = False
    with pytest.raises(RuntimeError, match="release student"):
        rollout.generate(batch, training=True)
    assert rollout.pending_rollout is None
