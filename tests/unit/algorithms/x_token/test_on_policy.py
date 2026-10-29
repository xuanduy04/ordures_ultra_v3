"""Discriminative tests for sampled IDs, response masks and rollout freshness."""

from types import SimpleNamespace
import json
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms.x_token.on_policy import OnPolicyRollout, validate_on_policy_config
from nemo_rl.algorithms.x_token.token_aligner import TokenAligner
from nemo_rl.data.xtoken_rollout_collate import PromptCollator, RolloutCrossTokenizerCollator
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class Tokenizer:
    pad_token_id = 0
    all_special_ids = [0, 1]
    is_fast = True

    def __init__(self, prefix):
        self.prefix = prefix
        self.vocab = {0: "<pad>", 1: "<eos>", 2: "a", 3: "b", 4: "ab", 5: prefix}

    def decode(self, ids, **kwargs):
        return "".join(self.vocab[index] for index in ids)

    def convert_ids_to_tokens(self, ids):
        return [self.vocab[int(index)] for index in ids]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict=None):
        assert add_generation_prompt
        if tokenize:
            assert return_dict is False
        return [5] if tokenize else self.prefix

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens and return_offsets_mapping
        assert text.startswith(self.prefix)
        # Intentionally re-encodes student token 4 ('ab') into two teacher tokens.
        chars = text[len(self.prefix):]
        return {"input_ids": [5] + [2 if char == "a" else 3 for char in chars],
                "offset_mapping": [(0, len(self.prefix))] +
                [(len(self.prefix) + index, len(self.prefix) + index + 1) for index in range(len(chars))]}


def fixture():
    student, teacher = Tokenizer("S:"), Tokenizer("TEACHER:")
    aligner = TokenAligner.__new__(TokenAligner)
    aligner.student_tokenizer = student
    aligner.teacher_tokenizer = teacher
    aligner.max_combination_len = 4
    collator = RolloutCrossTokenizerCollator(student, [teacher], [aligner], 16, [16], 4, [4], [1])
    prompts = PromptCollator("raw_text")([
        {"message_log": [{"role": "assistant", "content": "question"}], "loss_multiplier": 1.0, "idx": 7},
    ])
    generation = {"output_ids": torch.tensor([[5, 4, 1]]), "unpadded_sequence_lengths": torch.tensor([3])}
    return student, collator, prompts, generation


def test_preserves_sampled_ids_and_excludes_prompt_eos_and_padding():
    _, collator, prompts, generation = fixture()
    batch = collator(prompts, generation, [[5]], 3, 9)
    assert batch["input_ids"].tolist() == [[5, 4, 1, 0]]
    assert batch["token_mask"].tolist() == [[0, 1, 0, 0]]
    assert batch["teacher_0_input_ids"].tolist() == [[5, 2, 3, 0]]
    assert batch["teacher_0_token_mask"].tolist() == [[0, 1, 1, 0]]
    assert batch["alignment_0_student_chunk_id"][0, 0] == -1
    assert batch["alignment_0_student_chunk_id"][0, 1] == 0
    assert batch["alignment_0_student_chunk_id"][0, 2] == -1
    assert batch["alignment_0_pair_valid"].sum() == 1
    assert batch["behavior_version"].item() == 3
    assert batch["rollout_id"].item() == 9
    # Shifted labels retain the first response token at its prompt predictor.
    assert batch["token_mask"][:, 1:].tolist() == [[1, 0, 0]]


def test_behavior_logprobs_keep_token_positions_and_allow_extra_generation_padding():
    _, collator, prompts, generation = fixture()
    generation["behavior_logprobs"] = torch.tensor([[0., -.25, -.75, float("nan"), 0., 0., 0., 0.]])
    batch = collator(prompts, generation, [[5]], 0, 1)
    assert batch["behavior_logprobs"].shape == batch["input_ids"].shape
    assert batch["behavior_logprobs"][0, 1] == -.25
    assert batch["token_mask"][0, 2] == 0
    generation["behavior_logprobs"][0, 1] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite sampled-token"):
        collator(prompts, generation, [[5]], 0, 1)


def test_variant_byte_atoms_preserve_native_groups_and_response_offsets():
    from nemo_rl.algorithms.x_token.variant_alignment import SynchronizedAligner

    student, collator, prompts, generation = fixture()
    teacher = collator.teacher_tokenizers[0]
    for tokenizer in (student, teacher):
        tokenizer.backend_tokenizer = SimpleNamespace(pre_tokenizer="ByteLevel", decoder="ByteLevel")
    collator.aligners = [SynchronizedAligner(student, teacher, collator.aligners[0])]
    batch = collator(prompts, generation, [[5]], 0, 1)
    assert batch["alignment_0_native_group_id"].tolist() == [[0]]
    assert batch["alignment_0_student_chunk_id"].tolist() == [[-1, 0, -1, -1]]
    assert batch["alignment_0_teacher_chunk_id"].tolist() == [[-1, 0, 0, -1]]


@pytest.mark.parametrize("output,pattern", [([2, 4, 1], "changed"), ([5, 1], "non-empty"), ([5, 0, 1], "control")])
def test_bad_or_empty_response_cannot_update(output, pattern):
    _, collator, prompts, _ = fixture()
    generation = {"output_ids": torch.tensor([output]), "unpadded_sequence_lengths": torch.tensor([len(output)])}
    with pytest.raises(ValueError, match=pattern):
        collator(prompts, generation, [[5]], 0, 1)


def test_replacement_character_and_overflow_fail_explicitly():
    student, collator, prompts, generation = fixture()
    student.vocab[4] = "\ufffd"
    with pytest.raises(ValueError, match="replacement"):
        collator(prompts, generation, [[5]], 0, 1)
    student.vocab[4] = "ab"
    collator.student_limit = 2
    with pytest.raises(ValueError, match="no silent truncation"):
        collator(prompts, generation, [[5]], 0, 1)


def test_mixed_replacement_character_batch_masks_only_unrepresentable_sample():
    student, collator, _, _ = fixture()
    student.vocab[4] = "\ufffd"
    prompts = BatchedDataDict(prompts=[[{"role": "user", "content": "q"}]] * 2,
                             sample_mask=torch.tensor([1., 1.]), idx=[7, 8])
    sampled = torch.tensor([[5, 4, 1], [5, 2, 1]])
    generation = {"output_ids": sampled, "unpadded_sequence_lengths": torch.tensor([3, 3])}
    batch = collator(prompts, generation, [[5], [5]], 66, 67)
    assert batch["input_ids"][:, :3].tolist() == sampled.tolist()
    assert batch["sample_mask"].tolist() == [0., 1.]
    assert not batch["token_mask"][0].any()
    assert not batch["teacher_0_token_mask"][0].any()
    assert not batch["alignment_0_pair_valid"][0].any()


def test_same_tokenizer_keeps_ids_even_if_decode_would_be_lossy():
    student, collator, prompts, generation = fixture()
    student.vocab[4] = "\ufffd"
    collator.aligners = [None]
    batch = collator(prompts, generation, [[5]], 0, 1)
    assert batch["sample_mask"].tolist() == [1.]
    assert batch["token_mask"].sum() == 1


class ThinkingTokenizer(Tokenizer):
    def __init__(self, prefix, atomic_markers=True, reasoning_is_special=True):
        super().__init__(prefix)
        self.atomic_markers = atomic_markers
        self.vocab.update({6: "<think>", 7: "</think>", 8: "<|im_start|>", 9: "<|im_start|>"})
        self.all_special_ids = [0, 1, 8] + ([6, 7] if reasoning_is_special else [])
        self.characters = {char: index + 20 for index, char in enumerate(sorted(set("ab<think>/|_msart")))}
        self.vocab.update({token: char for char, token in self.characters.items()})

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens and return_offsets_mapping
        assert text.startswith(self.prefix)
        ids, offsets, position = [5], [(0, len(self.prefix))], len(self.prefix)
        while position < len(text):
            markers = [("<|im_start|>", 8)] + ([("<think>", 6), ("</think>", 7)] if self.atomic_markers else [])
            for piece, token in markers:
                if text.startswith(piece, position):
                    break
            else:
                piece = text[position]
                token = self.characters[piece]
            ids.append(token)
            offsets.append((position, position + len(piece)))
            position += len(piece)
        return {"input_ids": ids, "offset_mapping": offsets}


def thinking_fixture(cross, response, *, atomic_teacher=False, reasoning_is_special=True):
    student = ThinkingTokenizer("S:", reasoning_is_special=reasoning_is_special)
    teacher = ThinkingTokenizer("T:", atomic_markers=atomic_teacher) if cross else student
    aligner = None
    if cross:
        aligner = TokenAligner.__new__(TokenAligner)
        aligner.student_tokenizer, aligner.teacher_tokenizer = student, teacher
        aligner.max_combination_len = 32
    collator = RolloutCrossTokenizerCollator(student, [teacher], [aligner], 128, [128], 1, [1], [1])
    prompts = BatchedDataDict(prompts=[[{"role": "user", "content": "q"}]],
                             sample_mask=torch.ones(1), idx=[103])
    sampled = torch.tensor([[5, *response]])
    generation = {"output_ids": sampled, "unpadded_sequence_lengths": torch.tensor([sampled.shape[1]])}
    return collator, prompts, generation


@pytest.mark.parametrize("cross,atomic_teacher", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("reasoning_is_special", [False, True])
@pytest.mark.parametrize("response", [[6, 4, 7, 3, 1], [6, 4], [4, 7, 3, 1], [6, 4, 7, 3, 1, 1]])
def test_thinking_preserves_sampled_ids_context_and_masks(cross, atomic_teacher, reasoning_is_special, response):
    collator, prompts, generation = thinking_fixture(cross, response,
        atomic_teacher=atomic_teacher, reasoning_is_special=reasoning_is_special)
    original = generation["output_ids"].clone()
    behavior = torch.arange(original.numel(), dtype=torch.float32).reshape_as(original) / -10
    generation["behavior_logprobs"] = behavior.clone()
    batch = collator(prompts, generation, [[5]], 102, 103)
    assert torch.equal(batch["input_ids"], original)
    assert torch.equal(generation["output_ids"], original)
    assert torch.equal(batch["behavior_logprobs"], behavior)
    assert batch["sample_mask"].tolist() == [1.]
    content = list(response)
    while content and content[-1] == 1:
        content.pop()
    assert batch["token_mask"].tolist() == [[0, *([1] * len(content)), *([0] * (len(response) - len(content)))]]
    if cross:
        teacher = collator.teacher_tokenizers[0]
        teacher_ids = batch["teacher_0_input_ids"][0, 1:].tolist()
        assert teacher.decode(teacher_ids) == collator.student_tokenizer.decode(content)
        assert batch["alignment_0_pair_valid"].any()
        assert not batch["alignment_0_student_exact_partition_mask"][0, 0]
        assert not batch["alignment_0_student_exact_partition_mask"][0, 1 + len(content):].any()


@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("control", [0, 1, 8])
def test_unknown_control_masks_sample_without_joining_or_changing_ids(cross, control, capsys):
    collator, prompts, _ = thinking_fixture(cross, [6, 4, 7, 3, 1])
    prompts = BatchedDataDict(prompts=prompts["prompts"] * 2, sample_mask=torch.ones(2), idx=[103, 104])
    sampled = torch.tensor([[5, 6, 4, 7, 3, 1], [5, 6, control, 7, 3, 1]])
    batch = collator(prompts, {"output_ids": sampled, "unpadded_sequence_lengths": torch.tensor([6, 6])}, [[5], [5]], 102, 103)
    assert torch.equal(batch["input_ids"], sampled)
    assert batch["sample_mask"].tolist() == [1., 0.]
    assert not batch["token_mask"][1].any()
    if cross:
        assert not batch["teacher_0_token_mask"][1].any()
        assert not batch["alignment_0_pair_valid"][1].any()
    log = json.loads(next(line.split("=", 1)[1] for line in capsys.readouterr().out.splitlines()
                          if line.startswith("XTOKEN_CONTROL_TOKENS=")))
    event = next(event for event in log["tokens"] if event["action"] == "mask_sample")
    assert event["token_id"] == control and event["sequence_position"] == 2
    assert event["response_position"] == 1 and event["source_index"] == 104
    assert event["side"] == "student" and log["sampled_ids_preserved"]


def test_teacher_control_masks_sample_for_all_teachers_and_logs_teacher_id(capsys):
    collator, prompts, _ = thinking_fixture(True, [4, 1])
    collator.teacher_tokenizers *= 2
    collator.aligners *= 2
    collator.teacher_limits *= 2
    collator.teacher_divisors *= 2
    prompts = BatchedDataDict(prompts=prompts["prompts"] * 2, sample_mask=torch.ones(2), idx=[103, 104])
    sampled = torch.tensor([[5, 4, 1], [5, 9, 1]])
    batch = collator(prompts, {"output_ids": sampled, "unpadded_sequence_lengths": torch.tensor([3, 3])}, [[5], [5]], 0, 1)
    assert torch.equal(batch["input_ids"], sampled)
    assert batch["sample_mask"].tolist() == [1., 0.]
    for index in range(2):
        assert not batch[f"teacher_{index}_token_mask"][1].any()
        assert not batch[f"alignment_{index}_pair_valid"][1].any()
    logs = capsys.readouterr().out.splitlines()
    log = json.loads(next(line.split("=", 1)[1] for line in logs
                          if line.startswith("XTOKEN_CONTROL_TOKENS=")))
    events = [event for event in log["tokens"] if event["action"] == "mask_sample"]
    assert {event["teacher_index"] for event in events} == {0, 1}
    assert all(event["side"] == "teacher" and event["token"] == "<|im_start|>" for event in events)
    masks = json.loads(next(line.split("=", 1)[1] for line in logs if line.startswith("XTOKEN_RESPONSE_MASK=")))
    assert masks["masked_count"] == 1 and len(masks["samples"]) == 2


@pytest.mark.parametrize("cross", [False, True])
def test_all_control_batch_logs_before_refusing_empty_update(cross, capsys):
    collator, prompts, generation = thinking_fixture(cross, [6, 8, 7, 1])
    with pytest.raises(ValueError, match="control_token.*no optimizer update"):
        collator(prompts, generation, [[5]], 0, 1)
    assert '"token_id": 8' in capsys.readouterr().out


@pytest.mark.parametrize("cross", [False, True])
def test_thinking_loss_backward_has_no_gradient_for_masked_control_sample(cross, tmp_path):
    from nemo_rl.algorithms.x_token.topk_tail import ProjectedTopKTailLoss

    collator, prompts, _ = thinking_fixture(cross, [6, 4, 7, 3, 1])
    prompts = BatchedDataDict(prompts=prompts["prompts"] * 2, sample_mask=torch.ones(2), idx=[103, 104])
    sampled = torch.tensor([[5, 6, 4, 7, 3, 1], [5, 6, 8, 7, 3, 1]])
    batch = collator(prompts, {"output_ids": sampled, "unpadded_sequence_lengths": torch.tensor([6, 6])}, [[5], [5]], 0, 1)
    path = None
    selected = 2
    if cross:
        path = str(tmp_path / "thinking-projection.pt")
        selected = collator.teacher_tokenizers[0].characters["a"]
        indices = torch.stack([torch.arange(64), torch.full((64,), -1)], dim=1)
        weights = torch.stack([torch.ones(64), torch.zeros(64)], dim=1)
        indices[4] = torch.tensor([selected, collator.teacher_tokenizers[0].characters["b"]])
        weights[4] = .5
        torch.save({"indices": indices, "likelihoods": weights}, path)
    cfg = {"temperature": 1., "gold_loss": False, "xtoken_loss": False,
           "kd_loss_mode": "sum", "dynamic_loss_scaling": False, "ce_loss_scale": 0.,
           "sum_weights_metric": None, "normalize_teacher_by_vocab": False,
           "teacher_gold_loss": [None], "teacher_xtoken_loss": [None],
           "projection_matrix_paths": [path], "student_vocab_size": 64, "teacher_vocab_sizes": [64],
           "teacher_weights": [1.], "kl_loss_weight": 1., "tail_epsilon": 1e-8, "reverse_kl": False,
           "exact_token_match_only": True}
    shape = batch["teacher_0_input_ids"].shape if cross else batch["input_ids"].shape
    batch["teacher_0_topk_ids"] = torch.full((*shape, 1), selected, dtype=torch.long)
    batch["teacher_0_topk_logprobs"] = torch.full((*shape, 1), -.5)
    batch["teacher_0_topk_valid"] = torch.ones((*shape, 1), dtype=torch.bool)
    logits = torch.randn(2, 6, 64, requires_grad=True)
    loss, _ = ProjectedTopKTailLoss(cfg)(logits, batch, torch.tensor(1.), batch["token_mask"].sum())
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0].abs().sum() > 0
    assert logits.grad[1].eq(0).all()
    assert logits.grad[0, 4:].eq(0).all()


def test_teacher_boundary_straddling_token_excluded():
    _, collator, prompts, generation = fixture()
    teacher = collator.teacher_tokenizers[0]
    class BoundaryTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            # Token 2 consumes the final prefix byte and response 'a'.
            return {"input_ids": [5, 2, 3], "offset_mapping": [(0, 7), (7, 9), (9, 10)]}
    collator.teacher_tokenizers[0] = BoundaryTokenizer(teacher.prefix)
    # 'ab' no longer has a byte-exact teacher chunk after excluding the boundary token.
    with pytest.raises(ValueError, match="no exact response chunk"):
        collator(prompts, generation, [[5]], 0, 1)


def test_refit_before_sampling_every_updated_version_and_single_use_lease():
    student, collator, prompts, generation = fixture()
    engine = MagicMock()
    engine.generate.return_value = generation
    refit = MagicMock()
    driver = OnPolicyRollout(MagicMock(), engine, student, collator, {},
                             {"max_prompt_tokens": 4, "refit_buffer_size_gb": 1}, 12, refit)
    batch = driver.generate(prompts, training=True)
    assert batch["behavior_version"].item() == 12
    assert refit.call_count == 1
    with pytest.raises(RuntimeError, match="Previous rollout"):
        driver.generate(prompts, training=True)
    driver.after_update()
    with pytest.raises(RuntimeError, match="No fresh rollout"):
        driver.after_update()
    driver.generate(prompts, training=True)
    assert refit.call_count == 2
    assert driver.synced_version == 13
    driver.after_update()
    driver.generate(prompts, training=False)
    assert driver.pending_rollout is None
    version = driver.policy_version
    driver.generate(prompts, training=False)
    assert driver.policy_version == version
    assert refit.call_count == 3


def test_generation_failure_always_offloads_engine():
    student, collator, prompts, _ = fixture()
    engine = MagicMock()
    engine.generate.side_effect = RuntimeError("sampler failed")
    driver = OnPolicyRollout(MagicMock(), engine, student, collator, {},
                             {"max_prompt_tokens": 4, "refit_buffer_size_gb": 1}, 0, MagicMock())
    with pytest.raises(RuntimeError, match="sampler failed"):
        driver.generate(prompts, training=True)
    engine.finish_generation.assert_called_once()
    assert driver.pending_rollout is None


@pytest.mark.parametrize("knob,value", [("temperature", 0.7), ("top_p", 0.9), ("top_k", 8)])
def test_strict_opd_rejects_changed_sampling_distribution(knob, value):
    generation = {"backend": "vllm", "colocated": {"enabled": True}, "temperature": 1.0,
                  "top_p": 1.0, "top_k": None, "max_new_tokens": 4}
    generation[knob] = value
    config = SimpleNamespace(policy={"generation": generation, "train_global_batch_size": 2,
                                      "max_total_sequence_length": 8},
                             distillation={"num_prompts_per_step": 2},
                             on_policy={"max_prompt_tokens": 4, "include_eos_loss": False, "refit_buffer_size_gb": 1},
                             loss_fn={"dynamic_loss_scaling": False, "ce_loss_scale": 0})
    with pytest.raises(ValueError, match="Strict OPD"):
        validate_on_policy_config(config)


@pytest.mark.parametrize("name", ["off", "on"])
def test_shipped_yaml_resolves_and_preserves_nested_runtime_fields(name):
    from omegaconf import OmegaConf
    from nemo_rl.algorithms.xtoken_off_policy_distillation import MasterConfig
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    path = f"examples/configs/xtoken_{name}_policy_distillation.yaml"
    config = MasterConfig(**OmegaConf.to_container(load_config(path), resolve=True))
    assert config.data["train"]["data_files"] == "/data/xtoken/train.parquet"
    assert config.policy["generation"]["vllm_cfg"]["tensor_parallel_size"] == 1
    assert len(config.teachers) == 1
    if name == "on":
        validate_on_policy_config(config)


def test_dataloader_resume_rejects_changed_data_configuration(tmp_path):
    import yaml
    from nemo_rl.algorithms.x_token.compat import load_dataloader_state

    config = {"train": {"data_files": "/data/a.parquet"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"data": config}))
    torch.save({"step": 3}, tmp_path / "train_dataloader.pt")
    loader = MagicMock()
    load_dataloader_state(loader, tmp_path, config)
    loader.load_state_dict.assert_called_once_with({"step": 3})
    with pytest.raises(ValueError, match="different data configuration"):
        load_dataloader_state(loader, tmp_path, {"train": {"data_files": "/data/b.parquet"}})
