"""Discriminative tests for sampled IDs, response masks and rollout freshness."""

from types import SimpleNamespace
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
