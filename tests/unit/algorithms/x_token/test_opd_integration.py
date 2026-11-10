import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from nemo_rl.algorithms import opd_outsource
from nemo_rl.algorithms.x_token import opd_integration as integration
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.fixture
def setup(tmp_path):
    def tokenizer(words, template):
        core = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
        core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        return PreTrainedTokenizerFast(tokenizer_object=core, unk_token="<unk>", pad_token="<pad>",
                                      eos_token="<eos>", chat_template=template)
    student = tokenizer(["<pad>", "<eos>", "<unk>", "user", "assistant", "hello", "a", "b"],
                        "user {{ messages[0]['content'] }} assistant ")
    teacher = tokenizer(["<pad>", "<eos>", "<unk>", "teacher", "user", "assistant", "hello", "a", "b"],
                        "teacher user {{ messages[0]['content'] }} assistant ")
    path = str(tmp_path / "projection.pt")
    # Vocabulary dictionary insertion order is not the ID order.
    mapping = torch.tensor([[teacher.get_vocab()[student.convert_ids_to_tokens(i)]] for i in range(len(student))])
    torch.save({"indices": mapping, "likelihoods": torch.ones_like(mapping, dtype=torch.float)}, path)
    cfg = {
        "policy": {"dtensor_cfg": {"enabled": True, "_v2": True, "tensor_parallel_size": 1,
                                    "context_parallel_size": 1},
                   "generation": {"colocated": {"enabled": False}, "temperature": 1., "top_p": 1., "top_k": None},
                   "dynamic_batching": {"enabled": False}, "sequence_packing": {"enabled": False},
                   "train_global_batch_size": 2, "train_micro_batch_size": 1,
                   "max_total_sequence_length": 32, "make_sequence_length_divisible_by": 4},
        "grpo": {"async_grpo": {"enabled": True, "max_trajectory_age_steps": 0,
                                  "in_flight_weight_updates": False},
                 "num_prompts_per_step": 2, "num_generations_per_prompt": 1,
                 "reward_shaping": {"enabled": False}},
        "loss_fn": {"reference_policy_kl_penalty": 0},
        "on_policy_distillation": {"enabled": True, "mode": "xtoken_topk",
            "teacher_model_by_agent_name": {"qa": {"url": "http://fixture", "model": "teacher"}},
            "xtoken": {"teacher_tokenizer": {"name": "fixture"}, "projection_matrix_path": path,
                       "alignment_max_combination_length": 3, "teacher_max_sequence_length": 32,
                       "teacher_micro_batch_size": 1, "kl_loss_weight": 1., "reverse_kl": False,
                       "tail_epsilon": 1e-8,
                       "remote": {"model_vocab_size": len(teacher), "top_k": 2,
                                  "max_score_elements": 10000, "max_response_bytes": 100000,
                                  "request_timeout_s": 2., "max_attempts": 1, "retry_backoff_s": 0.,
                                  "normalization_atol": .001, "api_key_env": None}}},
    }
    prompt = [{"role": "user", "content": "hello"}]
    ids = student.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    messages = [{"role": "user", "token_ids": torch.tensor(ids)},
                {"role": "assistant", "token_ids": torch.tensor([6, 7, 1]),
                 "generation_logprobs": torch.zeros(3)}]
    batch = BatchedDataDict(message_log=[deepcopy(messages), deepcopy(messages)],
                            full_result=[{"responses_create_params": {"input": prompt}} for _ in range(2)],
                            agent_ref=[{"name": "qa"}] * 2, loss_multiplier=torch.tensor([1., 0.]))
    return cfg, student, teacher, batch


class Teacher:
    instances = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.calls = []
        self.closed = False
        self.instances.append(self)

    async def score_group(self, spec, ids, lengths):
        assert ids.shape[0] == 1
        assert spec["model"] == "teacher"
        assert ids[0, :4].tolist() == [3, 4, 6, 5]
        self.calls.append((ids.clone(), lengths.clone()))
        shape = (*ids.shape, 2)
        return {"topk_ids": torch.tensor([7, 8]).expand(shape),
                "topk_logprobs": torch.tensor([.6, .2]).log().expand(shape),
                "topk_valid": torch.ones(shape, dtype=torch.bool)}

    async def close(self):
        self.closed = True


def test_mode_defaults_and_rejection():
    assert opd_outsource.get_opd_mode({}) is None
    assert opd_outsource.get_opd_mode({"on_policy_distillation": {"enabled": True}}) == "sampled_token"
    with pytest.raises(ValueError, match="Unsupported OPD mode"):
        opd_outsource.get_opd_mode({"on_policy_distillation": {"enabled": True, "mode": "typo"}})


@pytest.mark.parametrize("mutation", ["negative_age", "inflight", "colocated", "multiple_updates", "reference_kl", "teacher_model", "nan_weight", "temperature", "top_p", "top_k"])
def test_rejects_unsupported_contract(setup, mutation):
    cfg, *_ = setup
    integration.validate_xtoken_mode(cfg)
    if mutation == "negative_age":
        cfg["grpo"]["async_grpo"]["max_trajectory_age_steps"] = -1
    elif mutation == "inflight":
        cfg["grpo"]["async_grpo"]["in_flight_weight_updates"] = True
    elif mutation == "colocated":
        cfg["policy"]["generation"]["colocated"]["enabled"] = True
    elif mutation == "multiple_updates":
        cfg["policy"]["train_global_batch_size"] = 1
    elif mutation == "reference_kl":
        cfg["loss_fn"]["reference_policy_kl_penalty"] = .1
    elif mutation == "teacher_model":
        cfg["on_policy_distillation"]["teacher_model_by_agent_name"]["other"] = {"url": "http://fixture", "model": "different"}
    elif mutation in {"temperature", "top_p", "top_k"}:
        cfg["policy"]["generation"][mutation] = .5
    else:
        cfg["on_policy_distillation"]["xtoken"]["kl_loss_weight"] = float("nan")
    with pytest.raises(ValueError):
        integration.validate_xtoken_mode(cfg)


def test_native_batch_to_two_optimizer_steps_with_real_autograd(setup, monkeypatch):
    cfg, student, teacher, batch = setup
    monkeypatch.setattr(integration, "XTokenTopKLogprobClient", Teacher)
    builder = integration.XTokenOPDBatchBuilder(cfg, student, teacher)
    loss_fn = integration.build_xtoken_loss(cfg["on_policy_distillation"]["xtoken"], len(student))
    torch.manual_seed(4)
    logits = torch.nn.Parameter(torch.randn(2, 8, len(student)))
    optimizer = torch.optim.SGD([logits], lr=.2)
    values = []
    for version in range(2):
        data = builder(batch, version)
        assert data["input_ids"][0, :6].tolist() == [3, 5, 4, 6, 7, 1]
        assert data["token_mask"][0].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
        assert data["teacher_0_input_ids"].shape[1] == 6
        assert data["behavior_version"].tolist() == [version, version]
        assert Teacher.instances[-1].closed and len(Teacher.instances[-1].calls) == 2
        denominator = (data["token_mask"][:, 1:] * data["sample_mask"][:, None]).sum()
        optimizer.zero_grad()
        loss, _ = loss_fn(logits, data, torch.tensor(1.), denominator)
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        assert logits.grad[1].abs().sum() == 0
        assert logits.grad[0, :2].abs().sum() == 0
        assert logits.grad[0, 4:].abs().sum() == 0
        assert logits.grad[0, 2:4].abs().sum() > 0
        values.append(loss.item())
        optimizer.step()
    assert values[1] < values[0]


@pytest.mark.parametrize("mutation", ["prompt", "multiturn", "tools", "empty", "teacher_limit"])
def test_batch_rejects_unsafe_text_adaptation(setup, monkeypatch, mutation):
    cfg, student, teacher, batch = setup
    monkeypatch.setattr(integration, "XTokenTopKLogprobClient", Teacher)
    if mutation == "prompt":
        batch["message_log"][0][0]["token_ids"][0] = 0
    elif mutation == "multiturn":
        batch["message_log"][0].append({"role": "tool"})
    elif mutation == "tools":
        batch["full_result"][0]["responses_create_params"]["tools"] = []
    elif mutation == "empty":
        batch["loss_multiplier"].zero_()
    else:
        cfg["on_policy_distillation"]["xtoken"]["teacher_max_sequence_length"] = 5
    with pytest.raises(ValueError):
        integration.XTokenOPDBatchBuilder(cfg, student, teacher)(batch, 0)


def test_teacher_failure_always_closes_session(setup, monkeypatch):
    cfg, student, teacher, batch = setup
    class FailingTeacher(Teacher):
        async def score_group(self, *args):
            raise RuntimeError("fixture failure")
    monkeypatch.setattr(integration, "XTokenTopKLogprobClient", FailingTeacher)
    with pytest.raises(RuntimeError, match="fixture failure"):
        integration.XTokenOPDBatchBuilder(cfg, student, teacher)(batch, 0)
    assert Teacher.instances[-1].closed


def test_fresh_target_is_current_version_after_every_refit(setup, monkeypatch):
    from nemo_rl.algorithms import async_utils
    cfg, *_ = setup
    monkeypatch.setattr(async_utils.ray, "get", lambda value: value)
    cls = async_utils.AsyncTrajectoryCollector.__ray_metadata__.modified_class
    collector = object.__new__(cls)
    collector.master_config = cfg
    collector._fresh_xtoken = True
    collector.initial_weight_version = 0
    collector._generation_check_lock = threading.Lock()
    collector._generating_targets = set()
    collector.replay_buffer = SimpleNamespace(
        get_last_target_weight_already_generated=SimpleNamespace(remote=lambda: -1),
        get_trajectories_needed=SimpleNamespace(remote=lambda target, count: count))
    for version in range(3):
        collector.current_weight_version = version
        collector._generating_targets.clear()
        assert collector._calculate_target_weights(version) == [version]
        assert not collector._should_pause_for_generation_limits()
        assert collector._get_next_target_for_generation(version) == version
        assert collector._get_next_target_for_generation(version) is None
        assert collector._should_pause_for_generation_limits()


def test_replay_sample_with_zero_age_evicts_stale_groups():
    from nemo_rl.algorithms.async_utils import ReplayBuffer
    cls = ReplayBuffer.__ray_metadata__.modified_class
    buffer = cls(max_size=4)
    buffer.push_with_wait_signal({"batch": "stale"}, 0, 1)
    buffer.push_with_wait_signal({"batch": "fresh"}, 1, 1)
    result = buffer.sample(1, 1, 0)
    assert result["avg_trajectory_age"] == 0
    assert result["trajectories"][0]["batch"] == "fresh"


def test_sequence_check_skips_only_declared_teacher_and_pair_axes():
    source = Path(__file__).resolve().parents[4] / "nemo_rl/models/automodel/data.py"
    tree = ast.parse(source.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "check_sequence_dim")
    scope = {"torch": torch, "BatchedDataDict": BatchedDataDict, "Any": object, "Tuple": tuple}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    check = scope["check_sequence_dim"]
    data = BatchedDataDict(input_ids=torch.ones(1, 8), token_mask=torch.ones(1, 8),
                          teacher_0_input_ids=torch.ones(1, 11), alignment_0_pair_valid=torch.ones(1, 2))
    with pytest.raises(AssertionError):
        check(data)
    assert check(data, skip_keys=integration.xtoken_non_student_seq_keys()) == (1, 8)
    data["alignment_0_student_chunk_id"] = torch.ones(1, 7)
    with pytest.raises(AssertionError):
        check(data, skip_keys=integration.xtoken_non_student_seq_keys())


@pytest.mark.parametrize("versions,targets", [([0], [1]), ([1], [2]), ([1, 1], [1])])
def test_replay_restore_rejects_stale_future_or_malformed_metadata(versions, targets):
    state = {"trajectories": [{}], "trajectory_versions": versions, "target_weight_versions": targets}
    with pytest.raises(ValueError):
        integration.validate_xtoken_replay_state(state, 1)
    integration.validate_xtoken_replay_state({"trajectories": [{}], "trajectory_versions": [1], "target_weight_versions": [1]}, 1)


def test_global_score_cap_fails_before_allocating_http_client(setup, monkeypatch):
    cfg, student, teacher, batch = setup
    cfg["on_policy_distillation"]["xtoken"]["remote"]["max_score_elements"] = 1
    monkeypatch.setattr(integration, "XTokenTopKLogprobClient", lambda cfg: pytest.fail("HTTP client must not be allocated"))
    with pytest.raises(ValueError, match="Global top-k batch"):
        integration.XTokenOPDBatchBuilder(cfg, student, teacher)(batch, 0)


@pytest.mark.parametrize("key,value", [("top_k", 1.5), ("max_attempts", 1.5), ("model_vocab_size", True),
                                        ("request_timeout_s", float("nan")), ("retry_backoff_s", float("inf"))])
def test_remote_limits_reject_nonfinite_or_noninteger_configuration(setup, key, value):
    from nemo_rl.algorithms.x_token.remote_teacher import XTokenTopKLogprobClient
    cfg, *_ = setup
    remote = cfg["on_policy_distillation"]["xtoken"]["remote"]
    remote[key] = value
    with pytest.raises(ValueError):
        XTokenTopKLogprobClient(remote)


def test_health_probe_uses_the_configured_api_key_environment(monkeypatch):
    import json
    monkeypatch.setenv("XTOKEN_FIXTURE_KEY", "fixture-only-key")
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return json.dumps({"data": [{"id": "teacher"}]}).encode()
    def open_request(request, **kwargs):
        assert request.get_header("Authorization") == "Bearer fixture-only-key"
        return Response()
    monkeypatch.setattr(opd_outsource.urllib.request, "urlopen", open_request)
    opd_outsource.probe_teacher_serve("http://fixture", {"teacher"}, api_key_env="XTOKEN_FIXTURE_KEY")


def test_microbatch_accumulation_uses_global_student_token_denominator(setup, monkeypatch):
    cfg, student, teacher, batch = setup
    batch["loss_multiplier"].fill_(1.)
    monkeypatch.setattr(integration, "XTokenTopKLogprobClient", Teacher)
    data = integration.XTokenOPDBatchBuilder(cfg, student, teacher)(batch, 0)
    fn = integration.build_xtoken_loss(cfg["on_policy_distillation"]["xtoken"], len(student))
    logits = torch.randn(2, 8, len(student), requires_grad=True)
    denominator = (data["token_mask"][:, 1:] * data["sample_mask"][:, None]).sum()
    full, _ = fn(logits, data, torch.tensor(2.), denominator)
    whole_gradient = torch.autograd.grad(full, logits, retain_graph=True)[0]
    parts = [fn(logits[index:index+1], data.slice(index, index+1), torch.tensor(2.), denominator)[0] for index in range(2)]
    torch.testing.assert_close(sum(parts), full)
    torch.testing.assert_close(torch.autograd.grad(sum(parts), logits)[0], whole_gradient)


@pytest.mark.parametrize("success", [True, False])
def test_fresh_collector_clears_cache_before_resuming(setup, success):
    from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector
    cfg, *_ = setup
    cls = AsyncTrajectoryCollector.__ray_metadata__.modified_class
    collector = object.__new__(cls)
    collector.master_config = cfg
    collector._fresh_xtoken = True
    collector._collection_error = None
    collector._refit_pause_cleared = threading.Event()
    calls = []
    collector.policy_generation = SimpleNamespace(invalidate_kv_cache=lambda: calls.append("reset") or success)
    if success:
        collector.resume_after_refit()
        assert collector._refit_pause_cleared.is_set()
    else:
        with pytest.raises(RuntimeError, match="cache invalidation"):
            collector.resume_after_refit()
        assert not collector._refit_pause_cleared.is_set()
    assert calls == ["reset"]
