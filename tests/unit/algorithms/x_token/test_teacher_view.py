import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from nemo_rl.algorithms import async_utils, grpo
from nemo_rl.algorithms.advantage_estimator import OPDAdvantageEstimator
from nemo_rl.algorithms.x_token import teacher_view
from nemo_rl.algorithms.x_token.teacher_view import TeacherView, XTokenTeacherViewBuilder
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience import rollouts

NAN = float("nan")
XTOKEN_CFG = {"enabled": True, "teacher_tokenizer": {"name": "fixture"}, "alignment_max_combination_length": 3}
# Policy row "user hello assistant a b <eos>"; the teacher sees "teacher user hello assistant a b".
STUDENT_ROW = [3, 5, 4, 6, 7, 1]
TEACHER_ROW = [3, 4, 6, 5, 7, 8]


def word_tokenizer(words, template):
    core = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>", chat_template=template
    )


@pytest.fixture
def rollout_group(monkeypatch):
    """Student tokenizer and a two-rollout NeMo-Gym group; the teacher tokenizer is served by ``get_tokenizer``."""
    student = word_tokenizer(
        ["<pad>", "<eos>", "<unk>", "user", "assistant", "hello", "a", "b"],
        "user {{ messages[0]['content'] }} assistant ",
    )
    teacher = word_tokenizer(
        ["<pad>", "<eos>", "<unk>", "teacher", "user", "assistant", "hello", "a", "b"],
        "teacher user {{ messages[0]['content'] }} assistant ",
    )
    monkeypatch.setattr(teacher_view, "get_tokenizer", lambda tokenizer_cfg: teacher)
    prompt = [{"role": "user", "content": "hello"}]
    messages = [
        {"role": "user", "token_ids": torch.tensor(STUDENT_ROW[:3]), "content": "hello"},
        {"role": "assistant", "token_ids": torch.tensor(STUDENT_ROW[3:]), "generation_logprobs": torch.zeros(3)},
    ]
    batch = BatchedDataDict(
        message_log=[deepcopy(messages), deepcopy(messages)],
        full_result=[{"responses_create_params": {"input": deepcopy(prompt)}} for _ in range(2)],
        agent_ref=[{"name": "qa"}] * 2,
    )
    return student, batch


def test_teacher_scores_its_own_tokenization_aligned_to_the_policy_tokens(rollout_group):
    student, batch = rollout_group
    view = XTokenTeacherViewBuilder(student, XTOKEN_CFG, {"stop_token_ids": None})(batch, len(STUDENT_ROW))
    assert view.input_ids.tolist() == [TEACHER_ROW, TEACHER_ROW]
    assert view.input_lengths.tolist() == [6, 6]
    # "a" and "b" are one chunk each; the prompt and the stripped <eos> belong to none.
    assert view.student_chunk_id.tolist() == [[-1, -1, -1, 0, 1, -1]] * 2
    assert view.teacher_chunk_id.tolist() == [[-1, -1, -1, -1, 0, 1]] * 2


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("tools", "tools_or_instructions"),
        ("multi_turn", "multi_turn"),
        ("prompt", "prompt_not_chat_template"),
        ("empty_response", "empty_response"),
        ("special_token", "student_special_token"),
        ("empty_log", "empty_message_log"),
    ],
)
def test_rollouts_the_teacher_cannot_reproduce_are_skipped(rollout_group, capsys, mutation, reason):
    student, batch = rollout_group
    if mutation == "tools":
        batch["full_result"][0]["responses_create_params"]["tools"] = []
    elif mutation == "multi_turn":
        batch["message_log"][0].append({"role": "tool", "token_ids": torch.tensor([6])})
    elif mutation == "prompt":
        batch["message_log"][0][0]["token_ids"][0] = 5
    elif mutation == "empty_response":
        batch["message_log"][0][1]["token_ids"] = torch.tensor([1])
    elif mutation == "special_token":
        # A <pad> inside the sampled response is structural, not content.
        batch["message_log"][0][1]["token_ids"] = torch.tensor([6, 0, 7, 1])
    else:
        batch["message_log"][0] = []
    view = XTokenTeacherViewBuilder(student, XTOKEN_CFG, {"stop_token_ids": None})(batch, 7)
    assert view.input_lengths.tolist() == [0, 6]
    assert (view.student_chunk_id[0] == -1).all()
    assert view.student_chunk_id[1].tolist() == [-1, -1, -1, 0, 1, -1, -1]
    assert f"0:{reason}" in capsys.readouterr().out


def test_each_policy_token_gets_its_share_of_the_chunk_teacher_logprob():
    view = TeacherView(
        input_ids=torch.zeros(2, 5, dtype=torch.long),
        input_lengths=torch.tensor([5, 0]),
        teacher_chunk_id=torch.tensor([[-1, 0, 1, 1, -1], [-1] * 5]),
        student_chunk_id=torch.tensor([[-1, 0, 0, 1, 2], [-1] * 5]),
    )
    mapped = view.to_student(torch.tensor([[0.0, -1.0, -2.0, -3.0, -9.0], [NAN] * 5]))
    # Chunk 0: one teacher token over two policy tokens. Chunk 1: two teacher tokens over one policy token.
    # Chunk 2 has no teacher token, and the unscored row gets nothing.
    expected = torch.tensor([[NAN, -0.5, -0.5, -5.0, NAN], [NAN] * 5])
    torch.testing.assert_close(mapped, expected, equal_nan=True)


def make_collector(student, xtoken_cfg, replay_buffer=None):
    """A local (non-Ray) trajectory collector with OPD on and the given ``on_policy_distillation.xtoken``."""
    opd_cfg = {
        "enabled": True,
        "teacher_model_by_agent_name": {"qa": {"url": "http://fixture", "model": "teacher"}},
        "xtoken": xtoken_cfg,
    }
    master_config = {
        "grpo": {"async_grpo": {"max_trajectory_age_steps": 1}},
        "policy": {"generation": {"stop_token_ids": None}},
        "on_policy_distillation": opd_cfg,
    }
    collector_cls = async_utils.AsyncTrajectoryCollector.__ray_metadata__.modified_class
    return collector_cls(None, student, {}, master_config, replay_buffer, 0, opd_cfg)


def test_disabled_xtoken_keeps_the_same_tokenizer_teacher(rollout_group):
    student, _ = rollout_group
    collector = make_collector(student, {"enabled": False})
    assert collector._teacher_logprob_client is not None
    assert collector._xtoken_teacher_views is None


def test_collector_turns_the_teacher_scores_into_opd_advantages(rollout_group, monkeypatch):
    student, batch = rollout_group
    pushed = []

    async def push(group, *versions):
        pushed.append(group)
        return "success"

    collector = make_collector(student, XTOKEN_CFG, SimpleNamespace(push_with_wait_signal=SimpleNamespace(remote=push)))
    collector.running = True
    scored = []

    async def score_group(teacher, input_ids, input_lengths):
        scored.append((teacher, input_ids.tolist(), input_lengths.tolist()))
        # Teacher logprob of the token at position j is -(j + 1) / 10.
        return -(torch.arange(input_ids.shape[1], dtype=torch.float32) + 1).expand(input_ids.shape) / 10

    async def nemo_gym_rollout(**kwargs):
        yield SimpleNamespace(final_batch=batch, rollout_metrics={}, ng_task_index=None)

    monkeypatch.setattr(collector._teacher_logprob_client, "score_group", score_group)
    monkeypatch.setattr(grpo, "_should_use_nemo_gym", lambda master_config: True)
    monkeypatch.setattr(rollouts, "run_async_nemo_gym_rollout", nemo_gym_rollout)
    asyncio.run(collector._run_prompt_group_worker(batch, 0, 1, 0, 2, 0))

    assert scored == [({"url": "http://fixture", "model": "teacher"}, [TEACHER_ROW] * 2, [6, 6])]
    (group,) = pushed
    assert group["rollout_metrics"]["teacher_logprob_skipped"] == 0.0
    teacher_logprobs = group["batch"]["teacher_reference_logprobs"]
    # Teacher "a" (position 4) and "b" (position 5) land on the policy's "a" and "b"; the rest gets nothing.
    expected = torch.tensor([[NAN, NAN, NAN, -0.5, -0.6, NAN]] * 2)
    torch.testing.assert_close(teacher_logprobs, expected, equal_nan=True)

    estimator = OPDAdvantageEstimator({"opd_advantage_weight": 1.0}, {})
    response_mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]] * 2)
    advantages = estimator.compute_advantage(
        torch.zeros(2), torch.zeros(2), response_mask, teacher_logprobs=teacher_logprobs, prev_logprobs=torch.full((2, 6), -0.2)
    )
    # Every aligned response token gets teacher minus policy logprob; the unscored <eos> gets zero.
    torch.testing.assert_close(advantages, torch.tensor([[0.0, 0.0, 0.0, -0.3, -0.4, 0.0]] * 2))
