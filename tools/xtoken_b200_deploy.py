import argparse
import json
import os
from pathlib import Path
import sys


def main() -> None:
    """Apply the B200 pilot overrides to an existing native OPD config."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher-tokenizer", type=Path, required=True)
    parser.add_argument("--teacher-vocab-size", type=int, required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-url", required=True)
    parser.add_argument("--agent", action="append", required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--teacher-max-sequence-length", type=int, required=True)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("The number of training steps must be positive")
    if not 0 < args.max_new_tokens < args.max_sequence_length:
        raise ValueError("Generation length must be positive and shorter than the sequence limit")

    from omegaconf import OmegaConf

    from nemo_rl.algorithms.x_token.opd_integration import validate_xtoken_mode
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    base = load_config(args.config)
    base_policy = base["policy"]
    if not base_policy["megatron_cfg"]["enabled"]:
        raise ValueError("Use the existing Megatron OPD configuration as the base")
    n = base["grpo"]["num_generations_per_prompt"]
    if n <= 0 or 128 % n:
        raise ValueError("Generations per prompt must divide the global batch of 128")
    if any(not path.exists() for path in (args.student, args.teacher_tokenizer, args.projection)):
        raise ValueError("Student, teacher tokenizer and projection must be staged before launch")
    teacher = {"url": args.teacher_url, "model": args.teacher_model}
    xtoken = {
        "teacher_tokenizer": {
            "name": str(args.teacher_tokenizer),
            "chat_template": "default",
            "chat_template_kwargs": {"enable_thinking": True},
        },
        "projection_matrix_path": str(args.projection),
        "alignment_max_combination_length": 4,
        "teacher_max_sequence_length": args.teacher_max_sequence_length,
        "teacher_micro_batch_size": 1,
        "kl_loss_weight": 1.0,
        "reverse_kl": False,
        "tail_epsilon": 1e-8,
        "remote": {
            "model_vocab_size": args.teacher_vocab_size,
            "top_k": 20,
            "max_score_elements": 50_000_000,
            "max_response_bytes": 64_000_000,
            "request_timeout_s": 120.0,
            "max_attempts": 3,
            "retry_backoff_s": 1.0,
            "normalization_atol": 0.001,
            "api_key_env": None,
        },
    }
    # Single-node pilot: the Ultra base config sets segment_size=16 to pin
    # NVLink domains across a multi-node cluster. With num_nodes=1, grpo.py
    # still requires train_nodes + inference_nodes alive nodes even though both
    # roles share one node, so pinning must be off or the run aborts before the
    # first step. Keep it for real multi-node launches.
    cluster_override = {"num_nodes": 1, "gpus_per_node": 3}
    if cluster_override["num_nodes"] == 1:
        cluster_override["segment_size"] = None
    overrides = {
        "cluster": cluster_override,
        "policy": {
            "model_name": str(args.student),
            "tokenizer": {
                "name": str(args.student),
                "chat_template_kwargs": {"enable_thinking": True},
            },
            "train_global_batch_size": 128,
            "max_total_sequence_length": args.max_sequence_length,
            "dtensor_cfg": {"enabled": False},
            "megatron_cfg": {
                "tensor_model_parallel_size": 1,
                "expert_tensor_parallel_size": 1,
                "expert_model_parallel_size": 1,
                "context_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "sequence_parallel": False,
            },
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": None,
                "colocated": {"enabled": False, "resources": {"num_nodes": 1, "gpus_per_node": 1}},
                "vllm_cfg": {
                    "async_engine": True,
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "expert_parallel_size": 1,
                    "max_model_len": args.max_sequence_length,
                },
            },
        },
        "grpo": {
            "max_num_steps": args.steps,
            "num_prompts_per_step": 128 // n,
            "async_grpo": {
                "enabled": True,
                "max_trajectory_age_steps": 0,
                "in_flight_weight_updates": False,
            },
        },
        "loss_fn": {"reference_policy_kl_penalty": 0.0},
        "logger": {
            "log_dir": str(args.run_dir / "logs"),
            "wandb_enabled": False,
            "swanlab_enabled": False,
            "mlflow_enabled": False,
        },
        "checkpointing": {
            "enabled": True,
            "checkpoint_dir": str(args.run_dir / "checkpoints"),
            "save_period": 1,
            "metric_name": None,
            "keep_top_k": None,
            "ft_keep_latest_k": None,
        },
    }
    config = OmegaConf.merge(base, overrides)
    if "_teachers" in config and OmegaConf.is_missing(config._teachers, "general"):
        config._teachers.general = args.teacher_model
    # The Ultra recipe's Nano-specific parsers cannot process Qwen3 rollouts.
    config.policy.generation.vllm_cfg.http_server_serving_chat_kwargs = {}
    config["on_policy_distillation"] = OmegaConf.create(
        {
            "enabled": True,
            "mode": "xtoken_topk",
            "teacher_model_by_agent_name": {agent: teacher for agent in args.agent},
            "xtoken": xtoken,
        }
    )
    resolved = OmegaConf.to_container(config, resolve=True)
    validate_xtoken_mode(resolved)
    if args.check_only:
        print(f"XTOKEN_DEPLOY_CONFIG_OK batch=128 steps={args.steps} save_period=1 age=0 thinking=both")
        return
    if (args.run_dir / "checkpoints").exists():
        raise FileExistsError("Use a fresh run directory to start at step zero")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    effective = args.run_dir / "effective.json"
    fd = os.open(effective, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(resolved, stream, indent=2)
    from examples.nemo_gym.run_grpo_nemo_gym import main as train

    sys.argv = [sys.argv[0], "--config", str(effective)]
    train()


if __name__ == "__main__":
    main()
