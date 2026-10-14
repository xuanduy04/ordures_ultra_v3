import argparse
import os
import pprint
import time

# Increase the W&B single object size warning threshold. Initially 100_000 (100 KB) -> 10_000_000 (10 MB)
import wandb.util

wandb.util.VALUE_BYTES_LIMIT = 10_000_000

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import (
    MasterConfig,
    _should_use_nemo_gym,
    async_grpo_train,
    setup,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import setup_nemo_gym_config
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir, log_container_init_timing
from nemo_rl.utils.timer import Timer


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def main() -> None:
    """Main entry point."""
    main_start = time.perf_counter()
    log_container_init_timing()

    rl_init_timer = Timer(context={"worker": "driver"})

    # Parse arguments
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "grpo_workplace_assistant_nemotron_nano_v2_9b.yaml",
        )

    with rl_init_timer.time("config"):
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")

        if overrides:
            print(f"Overrides: {overrides}")
            config = parse_hydra_overrides(config, overrides)

        config: MasterConfig = OmegaConf.to_container(config, resolve=True)
        print("Applied CLI overrides")

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    # setup tokenizer
    with rl_init_timer.time("tokenizer"):
        tokenizer = get_tokenizer(config["policy"]["tokenizer"])
        assert config["policy"]["generation"] is not None, (
            "A generation config is required for GRPO"
        )
        config["policy"]["generation"] = configure_generation_config(
            config["policy"]["generation"], tokenizer
        )

        # NeMo-Gym specific config setup.
        setup_nemo_gym_config(config, tokenizer)

    # We assert here since this is right after the final config has been materialized.
    assert _should_use_nemo_gym(config)

    # NeMo-Gym environment needs to get dp_openai_server_base_urls from policy_generation, so we don't setup env here.
    # Validation dataset config setup.
    with rl_init_timer.time("data"):
        print("\n▶ Setting up data...")
        train_dataset, _ = setup_response_data(
            tokenizer, config["data"], env_configs=None
        )

    # Print config
    print("Final config:")
    pprint.pprint(config)

    with rl_init_timer.time("ray_connect"):
        init_ray()

    with rl_init_timer.time("setup"):
        (
            policy,
            policy_generation,
            nemo_gym_env,
            cluster,
            dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            grpo_state,
            master_config,
        ) = setup(config, tokenizer, train_dataset)

    rl_init_timer.record("total", time.perf_counter() - main_start)

    rl_init_metrics = rl_init_timer.get_timing_metrics(reduction_op="sum")
    print("\n" + "=" * 60)
    print(" " * 14 + "RL INIT TIMING BREAKDOWN")
    for label, value in sorted(rl_init_metrics.items()):
        if isinstance(value, (int, float)):
            print(f"  {label}: {value:.1f}s")
    print("=" * 60 + "\n", flush=True)

    task_to_env = {"nemo_gym": nemo_gym_env}
    val_task_to_env = task_to_env

    # Check if async mode is enabled
    if "async_grpo" in config["grpo"] and config["grpo"]["async_grpo"]["enabled"]:
        # Async GRPO does not support dynamic sampling (implementation issue with async), reward scaling (already natively enforced)
        unsupported_features = [
            "use_dynamic_sampling",
            "reward_scaling",
        ]

        for feature in unsupported_features:
            if feature not in config["grpo"]:
                continue

            if (
                config["grpo"][feature]
                if isinstance(config["grpo"][feature], bool)
                else config["grpo"][feature].get("enabled", False)
            ):
                raise NotImplementedError(
                    f"{feature} is not supported with async GRPO"
                )

        print("🚀 Running async GRPO training")

        async_config = config["grpo"]["async_grpo"]
        # Run async GRPO training
        async_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=None,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=async_config["max_trajectory_age_steps"],
        )
    else:
        raise NotImplementedError("Synchronous GRPO has been removed; enable grpo.async_grpo.enabled and policy.generation.vllm_cfg.async_engine.")


if __name__ == "__main__":
    main()
