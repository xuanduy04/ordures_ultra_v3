import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
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
    "gym_input",
    [
        "hello",
        [{"role": "user", "content": [{"type": "input_text", "text": "hello"}], "type": "message"}],
    ],
)
def test_responses_api_text_prompts_are_scored_like_chat_messages(rollout_group, gym_input):
    student, batch = rollout_group
    for result in batch["full_result"]:
        result["responses_create_params"]["input"] = deepcopy(gym_input)
    view = XTokenTeacherViewBuilder(student, XTOKEN_CFG, {"stop_token_ids": None})(batch, len(STUDENT_ROW))
    assert view.input_ids.tolist() == [TEACHER_ROW, TEACHER_ROW]
    assert view.student_chunk_id.tolist() == [[-1, -1, -1, 0, 1, -1]] * 2


def test_policy_prompt_may_use_other_generation_prompt_settings():
    student = word_tokenizer(
        ["<pad>", "<eos>", "<unk>", "user", "assistant", "hello", "a", "b"],
        "user {{ messages[0]['content'] }} {% if add_generation_prompt %}assistant {% if think %}b {% endif %}{% endif %}",
    )
    builder = SimpleNamespace(student_tokenizer=student)
    matches = XTokenTeacherViewBuilder._prompt_matches_chat_template
    prompt = [{"role": "user", "content": "hello"}]
    think = {"metadata": {"chat_template_kwargs": '{"think": true}'}}
    assert matches(builder, {}, prompt, [3, 5, 4])
    # The model server's own kwargs only change the generation prompt.
    assert matches(builder, {}, prompt, [3, 5, 4, 7])
    # Per-request kwargs are applied.
    assert matches(builder, think, prompt, [3, 5, 4, 7])
    # Other messages, or extra structural tokens after them, do not match.
    assert not matches(builder, {}, prompt, [3, 6, 4])
    assert not matches(builder, {}, prompt, [3, 5, 4, 1])


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("tools", "tools_or_instructions"),
        ("multi_turn", "multi_turn"),
        ("prompt", "prompt_not_chat_template"),
        ("image", "non_text_prompt"),
        ("reasoning_item", "non_text_prompt"),
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
    elif mutation == "image":
        batch["full_result"][0]["responses_create_params"]["input"][0]["content"] = [
            {"type": "input_image", "image_url": "data:image/png;base64,"}
        ]
    elif mutation == "reasoning_item":
        batch["full_result"][0]["responses_create_params"]["input"] = [{"type": "reasoning", "summary": []}]
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


def byte_level_tokenizer(pieces, template):
    """Byte-level BPE without merges whose vocabulary is ``pieces``."""
    core = Tokenizer(models.BPE({piece: i for i, piece in enumerate(pieces)}, [], unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>", chat_template=template
    )


# Sampled pieces of a byte-level response, the response text the teacher scores, and how many policy tokens it covers.
UTF8_CASES = [
    # A character split over two tokens is valid.
    (["a", "Â", "¢"], "a¢", 3),
    # A literal U+FFFD is valid UTF-8 and is scored.
    (["a", "ï", "¿", "½"], "a�", 4),
    # A lead byte that never completes, at the end or followed by more tokens.
    (["a", "ð"], "a", 1),
    (["a", "ð", "!"], "a", 1),
    (["a", "ï", "¿", "½", "ð", "!"], "a�", 4),
    # The prefix cannot end inside "¢", whose last byte is in the token of the bad byte.
    (["a", "Â", "¢ð", "!"], "a", 1),
    # A token that starts valid and ends with a bad byte is cut whole.
    (["a", "að", "!"], "a", 1),
    # Without a valid prefix, the rollout is skipped.
    (["ð", "a"], "", 0),
]


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_byte_level_policy_with_invalid_utf8_keeps_the_valid_prefix(monkeypatch, capsys, backend):
    if backend == "numba":
        pytest.importorskip("numba")
    vocab = ["<unk>", "<pad>", "<eos>", "p", "q", "Ġ", "a", "!", "Â", "¢", "ï", "¿", "½", "ð", "¢ð", "að"]
    student = byte_level_tokenizer(vocab, "{{ messages[0]['content'] }} ")
    teacher = byte_level_tokenizer(vocab[::-1], "q ")
    monkeypatch.setattr(teacher_view, "get_tokenizer", lambda tokenizer_cfg: teacher)
    prompt = [{"role": "user", "content": "p"}]
    prompt_ids = list(student.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))
    responses = [student.convert_tokens_to_ids([*pieces, "<eos>"]) for pieces, _, _ in UTF8_CASES]
    batch = BatchedDataDict(
        message_log=[
            [
                {"role": "user", "token_ids": torch.tensor(prompt_ids)},
                {"role": "assistant", "token_ids": torch.tensor(ids)},
            ]
            for ids in responses
        ],
        full_result=[{"responses_create_params": {"input": prompt}} for _ in UTF8_CASES],
    )
    width = len(prompt_ids) + max(map(len, responses))
    config = {**XTOKEN_CFG, "alignment_backend": backend}
    view = XTokenTeacherViewBuilder(student, config, {"stop_token_ids": None})(batch, width)

    start = len(prompt_ids)
    for index, (_, text, count) in enumerate(UTF8_CASES):
        expected = teacher.encode("q " + text, add_special_tokens=False) if count else []
        assert view.input_ids[index, : view.input_lengths[index]].tolist() == expected
        # Every policy token of the prefix is aligned; the ones after it and the stripped <eos> get no chunk.
        chunk_ids = view.student_chunk_id[index].tolist()
        assert [pos for pos, chunk in enumerate(chunk_ids) if chunk >= 0] == list(range(start, start + count))
    out = capsys.readouterr().out
    assert "valid UTF-8 prefix of 5/8 rollouts; 10 policy token(s)" in out
    assert "skips 1/8 rollouts" in out and "7:student_undecodable" in out


def test_numba_preserves_sampled_token_teacher_logprobs(rollout_group):
    pytest.importorskip("numba")
    student, batch = rollout_group
    batch["full_result"][0]["responses_create_params"]["input"] = "hello"
    batch["full_result"][1]["responses_create_params"]["input"] = [{"role": "user", "content": [{"type": "input_image"}]}]
    reference = XTokenTeacherViewBuilder(student, XTOKEN_CFG, {})(batch, len(STUDENT_ROW))
    optimized = XTokenTeacherViewBuilder(student, {**XTOKEN_CFG, "alignment_backend": "numba"}, {})(
        batch, len(STUDENT_ROW)
    )
    for name in ("input_ids", "input_lengths", "teacher_chunk_id", "student_chunk_id"):
        assert torch.equal(getattr(reference, name), getattr(optimized, name))
    logprobs = torch.arange(reference.input_ids.numel(), dtype=torch.float32).reshape_as(reference.input_ids) / -10
    torch.testing.assert_close(optimized.to_student(logprobs), reference.to_student(logprobs), rtol=0, atol=0, equal_nan=True)
