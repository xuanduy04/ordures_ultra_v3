import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from nemo_rl.algorithms.x_token import opd_integration as integration
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _tokenizer(pieces, template):
    vocab = {piece: idx for idx, piece in enumerate(pieces)}
    backend = Tokenizer(models.BPE(vocab, [], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>", chat_template=template
    )


@pytest.mark.parametrize("alignment_backend", ["python", "numba"])
def test_native_topk_utf8_prefix_masks_teacher_and_gradient(tmp_path, monkeypatch, alignment_backend):
    if alignment_backend == "numba":
        pytest.importorskip("numba")
    vocab = ["<unk>", "<pad>", "<eos>", "p", "q", "Ġ", "a", "!", "Â", "¢", "ï", "¿", "½", "ð", "¢ð", "að"]
    student = _tokenizer(vocab, "{{ messages[0]['content'] }} ")
    teacher = _tokenizer(vocab[::-1], "q ")
    projection = tmp_path / "projection.pt"
    mapping = torch.tensor([[teacher.convert_tokens_to_ids(piece)] for piece in vocab])
    torch.save({"indices": mapping, "likelihoods": torch.ones_like(mapping, dtype=torch.float)}, projection)
    xtoken = {
        "projection_matrix_path": str(projection),
        "alignment_max_combination_length": 3,
        "alignment_backend": alignment_backend,
        "teacher_max_sequence_length": 32,
        "teacher_micro_batch_size": 1,
        "kl_loss_weight": 1.0,
        "reverse_kl": False,
        "tail_epsilon": 1e-8,
        "remote": {"model_vocab_size": len(teacher), "top_k": 2, "max_score_elements": 1_000},
    }
    config = {
        "policy": {"generation": {}, "max_total_sequence_length": 32, "make_sequence_length_divisible_by": 4},
        "grpo": {"async_grpo": {"max_trajectory_age_steps": 0}},
        "on_policy_distillation": {
            "xtoken": xtoken,
            "teacher_model_by_agent_name": {"qa": {"url": "http://fixture", "model": "teacher"}},
        },
    }
    cases = [
        (["a", "Â", "¢"], "a¢", 3),
        (["a", "ï", "¿", "½"], "a\ufffd", 4),
        (["a", "ð"], "a", 1),
        (["a", "ð", "!"], "a", 1),
        (["a", "ï", "¿", "½", "ð", "!"], "a\ufffd", 4),
        (["a", "Â", "¢ð", "!"], "a", 1),
        (["a", "að", "!"], "a", 1),
        (["ð", "a"], "", 0),
    ]
    prompt = [{"role": "user", "content": "p"}]
    prompt_ids = student.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    response_ids = [student.convert_tokens_to_ids(pieces + ["<eos>"]) for pieces, _, _ in cases]
    batch = BatchedDataDict(
        message_log=[
            [
                {"role": "user", "token_ids": torch.tensor(prompt_ids)},
                {"role": "assistant", "token_ids": torch.tensor(ids), "generation_logprobs": torch.zeros(len(ids))},
            ]
            for ids in response_ids
        ],
        full_result=[{"responses_create_params": {"input": prompt}} for _ in cases],
        agent_ref=[{"name": "qa"}] * len(cases),
        loss_multiplier=torch.ones(len(cases)),
    )
    teacher_rows = []

    class Teacher:
        def __init__(self, cfg, *, opd_delivery):
            assert opd_delivery

        async def score_group(self, spec, ids, lengths):
            teacher_rows.append(ids[0, : lengths[0]].tolist())
            shape = (*ids.shape, 2)
            return {
                "topk_ids": torch.tensor(teacher.convert_tokens_to_ids(["a", "!"])).expand(shape),
                "topk_logprobs": torch.tensor([.6, .2]).log().expand(shape),
                "topk_valid": torch.ones(shape, dtype=torch.bool),
            }

        async def close(self):
            pass

    monkeypatch.setattr(integration, "XTokenTopKLogprobClient", Teacher)
    data = integration.XTokenOPDBatchBuilder(config, student, teacher)(batch, 0)
    assert teacher_rows == [teacher.encode("q " + text, add_special_tokens=False) for _, text, _ in cases]
    assert data["sample_mask"].tolist() == [float(count > 0) for _, _, count in cases]
    start = len(prompt_ids)
    for index, ((_, _, count), ids) in enumerate(zip(cases, response_ids, strict=True)):
        assert data["input_ids"][index, : start + len(ids)].tolist() == prompt_ids + ids
        assert data["token_mask"][index].nonzero().flatten().tolist() == list(range(start, start + count))
        assert (data["alignment_0_student_chunk_id"][index, start + count :] == -1).all()

    torch.manual_seed(19)
    logits = torch.randn(*data["input_ids"].shape, len(student), requires_grad=True)
    valid_tokens = (data["token_mask"][:, 1:] * data["sample_mask"][:, None]).sum()
    loss, _ = integration.build_xtoken_loss(xtoken, len(student))(logits, data, data["sample_mask"].sum(), valid_tokens)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    for index, (_, _, count) in enumerate(cases):
        rows_with_gradient = logits.grad[index].abs().sum(dim=-1).nonzero().flatten().tolist()
        assert rows_with_gradient == list(range(start - 1, start - 1 + count))
