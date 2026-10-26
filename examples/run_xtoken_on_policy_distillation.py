"""Run X-Token with fresh student rollouts; requires an on_policy config."""

from examples.run_xtoken_off_policy_distillation import main


if __name__ == "__main__":
    main(require_on_policy=True)
