"""Offline profile/topology checks without installing Torch locally."""

from copy import deepcopy
import importlib.util
from pathlib import Path, PurePosixPath

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("variant_pilot_fixtures", Path(__file__).with_name("test_pilot.py"))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


@pytest.mark.parametrize("gpus", [2, 3, 4])
@pytest.mark.parametrize("arm", ["cross", "same"])
def test_variant_recipe_has_integral_per_rank_microbatches_and_preserves_old_pilot(gpus, arm):
    settings = yaml.safe_load((ROOT / "tools/xtoken_all/pilot_algorithm.yaml").read_text())
    settings["runtime"]["gpu_indices"] = list(range(gpus))
    settings["student"].update(model_path=None, tokenizer_path=None)
    old = fixtures.baseline()
    old["policy"]["dtensor_cfg"] = {"tensor_parallel_size": 2, "context_parallel_size": 2}
    old["loss_fn"] = {}
    original = deepcopy(old)
    cfg = fixtures.pilot.pilot_config(old, settings, arm, PurePosixPath("/old/run"),
        PurePosixPath("/new/run"), PurePosixPath("/old/source"), PurePosixPath("/new/source"), "host:6379")
    assert old == original
    assert cfg["policy"]["train_global_batch_size"] == cfg["distillation"]["num_prompts_per_step"] == 12
    assert 12 % gpus == 0 and cfg["policy"]["train_micro_batch_size"] == 1
    assert cfg["policy"]["dtensor_cfg"] == {"tensor_parallel_size": 1, "context_parallel_size": 1}
    assert cfg["teachers"][0]["train_global_batch_size"] == 12
    assert cfg["loss_fn"]["variant"]["mode"] == "dpca"
    assert cfg["loss_fn"]["teacher_distribution_mode"] == "topk_tail_kl"
    assert cfg["distillation"]["max_num_steps"] == 200
    assert cfg["policy"]["scheduler"][0]["kwargs"]["total_iters"] == 20
    assert cfg["checkpointing"]["save_period"] == 10
    assert "variant" not in fixtures.spec() and "variant" not in fixtures.baseline()


def test_full_score_variant_does_not_silently_keep_top20_transport():
    settings = yaml.safe_load((ROOT / "tools/xtoken_all/pilot_algorithm.yaml").read_text())
    settings["student"].update(model_path=None, tokenizer_path=None)
    settings["variant"].update(mode="simct_full", score_mode="full")
    old = fixtures.baseline()
    old["policy"]["dtensor_cfg"] = {}
    old["loss_fn"] = {}
    cfg = fixtures.pilot.pilot_config(old, settings, "same", PurePosixPath("/old/run"),
        PurePosixPath("/new/run"), PurePosixPath("/old/source"), PurePosixPath("/new/source"), "host:6379")
    assert cfg["teachers"][0]["remote"]["score_mode"] == "full"
    assert cfg["loss_fn"]["teacher_distribution_mode"] == "full_logprobs"
    assert cfg["loss_fn"]["vocab_topk"] == 151936
