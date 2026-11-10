import __future__
import ast
from copy import deepcopy
import gc
import os
from pathlib import Path
from types import SimpleNamespace
import time
from typing import cast

import numpy as np
import pytest
import torch

from nemo_rl.algorithms import async_utils, opd_outsource
from nemo_rl.algorithms.x_token import opd_integration as integration
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.timer import Timer
from .test_opd_integration import setup, Teacher


def load_definitions(path, names, scope):
    tree = ast.parse(path.read_text())
    nodes = [item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), scope)


class Actor:
    def __init__(self, instance):
        self.instance = instance

    def __getattr__(self, name):
        return SimpleNamespace(remote=getattr(self.instance, name))


class Factory:
    def __init__(self, cls):
        self.cls = cls

    def options(self, **kwargs):
        return self

    def remote(self, **kwargs):
        return Actor(self.cls(**kwargs))


@pytest.mark.parametrize("mode", ["xtoken_topk", "sampled_token"])
def test_native_driver_runs_two_steps_and_refits_before_fresh_collection(setup, monkeypatch, mode):
    cfg, student, teacher, batch = setup
    cfg["on_policy_distillation"]["mode"] = mode
    cfg["grpo"].update({"max_num_steps": 2, "val_period": 0, "val_at_start": False,
                        "val_at_end": False, "overlong_filtering": False})
    cfg["loss_fn"].update({"use_importance_sampling_correction": True, "force_on_policy_ratio": True})
    cfg["checkpointing"] = {"enabled": False, "checkpoint_must_save_by": None,
                            "ft_save_period": None, "save_period": 100}
    cfg["cluster"] = {"num_nodes": 1, "gpus_per_node": 1}
    cfg["policy"]["generation"].update({"backend": "vllm", "vllm_cfg": {"async_engine": True}})
    batch["length"] = torch.tensor([3, 3])
    batch["total_reward"] = torch.tensor([1., 0.])
    for messages in batch["message_log"]:
        messages[0]["content"] = "hello"
        messages[1]["content"] = "a b"
    trace, penalty_inputs = [], []
    real_buffer = async_utils.ReplayBuffer.__ray_metadata__.modified_class
    class Collector:
        def __init__(self, replay_buffer, start_step, **kwargs):
            self.buffer = replay_buffer
            self.version = start_step
            self.started = False

        def fill(self):
            for index in range(2):
                group = {"batch": deepcopy(batch.slice(index, index + 1)), "rollout_metrics": {}}
                assert self.buffer.push_with_wait_signal.remote(group, self.version, self.version) == "success"

        def start_collection(self, dataloader):
            trace.append("collect")
            self.started = True
            self.fill()

        def set_weight_version(self, version):
            self.version = version
            if self.started and version > 0:
                self.fill()

        def prepare_for_refit(self):
            trace.append("pause")

        def resume_after_refit(self):
            trace.append("resume")

        def get_status(self):
            return {"running": True, "data_exhausted": False, "inflight_workers": 0, "error": None}

        def get_efficiency_metrics(self):
            return {}

    monkeypatch.setattr(async_utils, "ReplayBuffer", Factory(real_buffer))
    monkeypatch.setattr(async_utils, "AsyncTrajectoryCollector", Factory(Collector))
    monkeypatch.setattr(integration, "XTokenTopKLogprobClient", Teacher)
    logged = []
    logits = torch.nn.Parameter(torch.randn(2, 8, len(student)))
    optimizer = torch.optim.SGD([logits], lr=.1)
    def train(data, loss_fn, **kwargs):
        optimizer.zero_grad()
        valid_tokens = (data["token_mask"][:, 1:] * data["sample_mask"][:, None]).sum()
        if mode == "xtoken_topk":
            assert kwargs["check_dim_skip_keys"] == integration.xtoken_non_student_seq_keys()
            loss, metrics = loss_fn(logits, data, data["sample_mask"].sum(), valid_tokens)
        else:
            assert "check_dim_skip_keys" not in kwargs
            loss, metrics = logits.sum() * 0 + 1, {"gen_kl_error": 0.}
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        norm = logits.grad.norm()
        optimizer.step()
        trace.append("train")
        return {"loss": torch.tensor([loss.item()]), "grad_norm": norm.reshape(1),
                "all_mb_metrics": {"global_valid_toks": [valid_tokens.item()],
                                   "global_valid_seqs": [data["sample_mask"].sum().item()],
                                   **{key: [value] for key, value in metrics.items() if key != "loss"}}}
    policy = SimpleNamespace(sharding_annotations=SimpleNamespace(get_axis_size=lambda name: 1),
                             prepare_for_training=lambda: None, train=train, shutdown=lambda: None)
    generation = SimpleNamespace(clear_logger_metrics=lambda: None, get_logger_metrics=lambda: {},
                                 shutdown=lambda: None)
    checkpoint = SimpleNamespace(get_latest_checkpoint_path=lambda: None, shutdown=lambda: None)
    timeout = SimpleNamespace(start_iterations=lambda: None, mark_iteration=lambda: None, check_save=lambda: False)
    def penalties(**kwargs):
        penalty_inputs.append(kwargs["penalties"])
        assert isinstance(kwargs["penalties"], list)
        return {"fixture_penalty": torch.zeros_like(kwargs["advantages"])}
    scope = {
        "torch": torch, "np": np, "gc": gc, "os": os, "time": time, "cast": cast,
        "BatchedDataDict": BatchedDataDict, "ClippedPGLossDataDict": dict,
        "opd_module": opd_outsource, "validate_xtoken_mode": integration.validate_xtoken_mode,
        "XTokenOPDBatchBuilder": integration.XTokenOPDBatchBuilder,
        "xtoken_non_student_seq_keys": integration.xtoken_non_student_seq_keys,
        "Timer": Timer, "TimeoutChecker": lambda **kwargs: timeout,
        "get_actor_python_env": lambda name: "/usr/local/bin/python",
        "get_tokenizer": lambda cfg: teacher,
        "ray": SimpleNamespace(get=lambda value, **kwargs: value, kill=lambda actor: None),
        "refit_policy_generation": lambda *args: trace.append("refit"),
        "create_advantage_estimator": lambda cfg: SimpleNamespace(compute_advantage=lambda **kwargs: torch.zeros_like(kwargs["mask"])),
        "create_advantage_penalties": lambda *args: [], "apply_advantage_penalties": penalties,
        "_calculate_single_metric": lambda *args, **kwargs: {},
        "_malloc_trim": lambda: None, "_log_rss": lambda label: None,
        "maybe_gpu_profile_step": lambda *args: None,
        "print_performance_metrics": lambda *args: {}, "print_efficiency_summary": lambda *args: {},
        "_should_log_nemo_gym_responses": lambda cfg: False,
    }
    repo = Path(__file__).resolve().parents[4]
    scope.update({"Tensor": torch.Tensor, "PackedTensor": PackedTensor, "FlatMessagesType": dict})
    load_definitions(repo / "nemo_rl/data/llm_message_utils.py",
                     ["message_log_to_flat_messages", "batched_message_log_to_flat_message",
                      "_pad_tensor", "_validate_tensor_consistency"], scope)
    load_definitions(repo / "nemo_rl/algorithms/grpo.py",
                     ["async_grpo_train", "extract_initial_prompt_messages", "_distribution_stats_str",
                      "_should_use_async_rollouts"], scope)
    loss_fn = integration.build_xtoken_loss(cfg["on_policy_distillation"]["xtoken"], len(student))
    state = {"current_step": 0, "consumed_samples": 0, "total_valid_tokens": 0}
    logger = SimpleNamespace(log_metrics=lambda metrics, step, **kwargs: logged.append((step, kwargs["prefix"], metrics)))
    scope["async_grpo_train"](policy, generation, object(), None, student, loss_fn, {}, None,
                              logger, checkpoint, state, cfg, max_trajectory_age_steps=0)
    assert trace.count("train") == 2
    assert trace.count("refit") == 3
    if mode == "xtoken_topk":
        assert trace.index("refit") < trace.index("collect")
        assert penalty_inputs == []
        assert all("final_advantages" not in metrics for _, _, metrics in logged)
    else:
        assert len(penalty_inputs) == 2


@pytest.mark.parametrize("raw", [True, False])
def test_actual_automodel_forward_preserves_raw_kd_logits_and_legacy_temperature(raw):
    class Capture:
        requires_raw_logits = raw

        def __call__(self, logits, *args):
            return logits.clone(), {}
    scope = {"torch": torch, "model_forward": lambda *args, **kwargs: torch.tensor([[[2., 4.]]]),
             "extract_logits": lambda model, outputs: outputs,
             "LogprobsPostProcessor": type("LP", (), {}), "TopkLogitsPostProcessor": type("TopK", (), {}),
             "ScorePostProcessor": type("Score", (), {})}
    repo = Path(__file__).resolve().parents[4]
    load_definitions(repo / "nemo_rl/models/automodel/train.py",
                     ["forward_with_post_processing_fn", "apply_temperature_scaling", "LossPostProcessor"], scope)
    processor = scope["LossPostProcessor"](Capture(), {}, None, None, None, 1, 1)
    output, _, _ = scope["forward_with_post_processing_fn"](
        None, {"generation": {"temperature": .5}}, processor,
        SimpleNamespace(data_dict={}, processed_inputs=None),
    )
    torch.testing.assert_close(output, torch.tensor([[[2., 4.]]]) if raw else torch.tensor([[[4., 8.]]]))
