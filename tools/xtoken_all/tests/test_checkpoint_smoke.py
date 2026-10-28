"""Checkpoint scheduling and file closure, without tensor serialization."""

import ast
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NotRequired, TypedDict
from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import TypeAdapter


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/xtoken_all"))
spec = importlib.util.spec_from_file_location("checkpoint_smoke_checker", ROOT / "tools/xtoken_all/check_result.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.mark.parametrize("completed_steps", [0, 1])
def test_existing_training_save_branch_runs_after_first_and_second_update(tmp_path, completed_steps):
    tree = ast.parse((ROOT / "nemo_rl/algorithms/xtoken_off_policy_distillation.py").read_text())
    assignments = {}
    save_branch = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("should_save_by_step", "should_save_by_timeout"):
                assignments[node.targets[0].id] = node
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("master_config.checkpointing['enabled']"):
            save_branch = node
    assert save_branch is not None
    checkpointer = MagicMock()
    checkpointer.save_optimizer = True
    checkpointer.init_tmp_checkpoint.return_value = str(tmp_path / ('tmp_step_' + str(completed_steps + 1)))
    cfg = {"enabled": True, "save_period": 1, "metric_name": None}
    student, dataloader, torch = MagicMock(), MagicMock(), MagicMock()
    namespace = {"os": os, "torch": torch, "is_last_step": completed_steps == 1,
        "total_steps": completed_steps, "current_epoch": 0, "current_step": completed_steps,
        "total_valid_tokens": 123, "consumed_samples": (completed_steps + 1) * 4,
        "master_config": SimpleNamespace(checkpointing=cfg), "off_policy_distillation_state": {},
        "timeout": SimpleNamespace(check_save=lambda: False), "student_policy": student,
        "checkpointer": checkpointer, "timer": MagicMock(), "val_metrics": None, "dataloader": dataloader}
    body = [assignments["should_save_by_step"], assignments["should_save_by_timeout"], save_branch]
    exec(compile(ast.Module(body=body, type_ignores=[]), "actual_checkpoint_branch", "exec"), namespace)
    assert namespace["off_policy_distillation_state"]["total_steps"] == completed_steps + 1
    saved = student.save_checkpoint.call_args.kwargs
    assert saved["optimizer_path"].endswith(os.path.join('policy', 'optimizer'))
    assert saved["tokenizer_path"].endswith(os.path.join('policy', 'tokenizer'))
    assert saved["checkpointing_cfg"] is cfg
    torch.save.assert_called_once()
    checkpointer.finalize_checkpoint.assert_called_once_with(checkpointer.init_tmp_checkpoint.return_value)


@pytest.mark.parametrize("failure", [None, "optimizer", "weights", "training_info", "dataloader"])
@pytest.mark.parametrize('step_field', ['total_steps', 'current_step'])
def test_result_checker_requires_finalized_step_one_payloads(tmp_path, failure, step_field):
    step = tmp_path / 'step_1'
    step.mkdir()
    metadata = {'total_steps': 1} if step_field == 'total_steps' else {'total_steps': 0, 'current_step': 1}
    (step / 'training_info.json').write_text(json.dumps(metadata))
    (step / 'config.yaml').write_text('checkpointing: {}\n')
    (step / 'train_dataloader.pt').write_bytes(b'fixture-only')
    for name in ('weights', 'optimizer', 'tokenizer'):
        path = step / 'policy' / name
        path.mkdir(parents=True)
        (path / ('tokenizer.json' if name == 'tokenizer' else 'shard.distcp')).write_bytes(b'fixture-only')
    if failure in ('optimizer', 'weights'):
        (step / 'policy' / failure / 'shard.distcp').unlink()
    elif failure == 'training_info':
        (step / 'training_info.json').write_text('{broken')
    elif failure == 'dataloader':
        (step / 'train_dataloader.pt').unlink()
    result = checker.inspect_checkpoint(tmp_path, 1, step_field=step_field)
    assert result['success'] is (failure is None)
    if failure != 'training_info':
        assert result['training_info'] == metadata
        assert len(result['training_info_sha256']) == 64
    assert checker.inspect_checkpoint(tmp_path, 2, step_field=step_field)['success'] is False
    if failure is None and step_field == 'current_step':
        assert checker.inspect_checkpoint(tmp_path, 1)['success'] is False


def test_smoke_and_train_have_separate_save_cadences_and_budgets():
    smoke = yaml.safe_load((ROOT / 'examples/configs/xtoken_gym_qwen3_gemma4.yaml').read_text())
    train = yaml.safe_load((ROOT / 'examples/configs/xtoken_gym_qwen3_gemma4_train.yaml').read_text())
    assert smoke['checkpointing']['enabled'] is True
    assert smoke['checkpointing']['save_period'] == 1
    assert smoke['checkpointing']['save_optimizer'] is True
    assert smoke['checkpointing']['is_async'] is False
    assert smoke['policy']['max_total_sequence_length'] == 256
    assert train['policy']['max_total_sequence_length'] == 8192
    assert train['on_policy']['teacher_max_total_sequence_length'] == 16384
    assert train['checkpointing']['save_period'] == 10
    assert train['policy']['train_global_batch_size'] == train['distillation']['num_prompts_per_step']


def test_company_async_checkpoint_updates_current_step_instead_of_total_steps():
    tree = ast.parse((ROOT / 'nemo_rl/algorithms/grpo.py').read_text())
    async_train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'async_grpo_train')
    assignments = [node for node in ast.walk(async_train) if isinstance(node, ast.Assign)
                   and ast.unparse(node.targets[0]) == "grpo_save_state['current_step']"]
    assert len(assignments) == 1
    for completed in (0, 1):
        state = {'current_step': completed, 'total_steps': 0}
        namespace = {'step': completed, 'grpo_save_state': state}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), 'actual_company_checkpoint_counter', 'exec'), namespace)
        assert state == {'current_step': completed + 1, 'total_steps': 0}


def test_checkpoint_async_setting_survives_actual_config_type_validation():
    tree = ast.parse((ROOT / 'nemo_rl/algorithms/xtoken_off_policy_distillation.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CheckpointingConfig')
    namespace = {'ProductionCheckpointingConfig': TypedDict('ProductionCheckpointingConfig', {}),
                 'NotRequired': NotRequired, '__name__': __name__}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'actual_checkpoint_type', 'exec'), namespace)
    validated = TypeAdapter(namespace['CheckpointingConfig']).validate_python({'save_optimizer': True, 'is_async': False})
    assert validated['is_async'] is False
