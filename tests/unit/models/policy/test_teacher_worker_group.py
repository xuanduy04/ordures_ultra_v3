import pytest


def test_teacher_config_defaults():
    from nemo_rl.models.policy.teacher_worker_group import TeacherConfig
    cfg = TeacherConfig(alias="math", model_name="/ckpt/math", tensor_model_parallel_size=4)
    assert cfg["alias"] == "math"
    assert cfg["tensor_model_parallel_size"] == 4


def test_create_teacher_configs_homogeneous():
    from nemo_rl.models.policy.teacher_worker_group import create_teacher_configs_from_opd_config
    configs = create_teacher_configs_from_opd_config({
        "teacher_model_by_agent_name": {"math": "/ckpt/math", "code": "/ckpt/code"},
        "non_colocated_teachers": {"default_teacher_cfg": {"tensor_model_parallel_size": 4}},
    })
    assert len(configs) == 2
    assert all(c["tensor_model_parallel_size"] == 4 for c in configs)


def test_create_teacher_configs_heterogeneous_override():
    from nemo_rl.models.policy.teacher_worker_group import create_teacher_configs_from_opd_config
    configs = create_teacher_configs_from_opd_config({
        "teacher_model_by_agent_name": {"math": "/ckpt/math", "code": "/ckpt/code"},
        "non_colocated_teachers": {
            "default_teacher_cfg": {"tensor_model_parallel_size": 4},
            "teacher_overrides": {"code": {"tensor_model_parallel_size": 8}},
        },
    })
    code_cfg = [c for c in configs if c["alias"] == "code"][0]
    assert code_cfg["tensor_model_parallel_size"] == 8


def test_create_teacher_configs_deduplicates():
    from nemo_rl.models.policy.teacher_worker_group import create_teacher_configs_from_opd_config
    configs = create_teacher_configs_from_opd_config({
        "teacher_model_by_agent_name": {"math": "/shared", "code": "/shared", "rlhf": "/rlhf"},
        "deduplicate_shared_teacher_checkpoints": True,
        "non_colocated_teachers": {"default_teacher_cfg": {"tensor_model_parallel_size": 2}},
    })
    assert len(configs) == 2
