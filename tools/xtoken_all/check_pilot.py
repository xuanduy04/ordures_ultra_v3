"""Check a completed pilot mechanically; no held-out quality claim."""

import argparse
import json
import math
from pathlib import Path

import yaml

from check_result import inspect_checkpoint


def assess(run):
    cfg = yaml.safe_load((run / "effective.yaml").read_text())
    maximum = cfg["distillation"]["max_num_steps"]
    lines = (run / "driver.log").read_text(errors="replace").splitlines()
    def events(prefix):
        return [json.loads(line[len(prefix):]) for line in lines if line.startswith(prefix)]
    steps, rollouts, scores = events("XTOKEN_TRAIN_STEP="), events("XTOKEN_ROLLOUT="), events("XTOKEN_TEACHER_SCORE=")
    exits = json.loads((run / "process-exit.json").read_text())
    reasons = []
    if exits["driver_exit_code"] != 0 or exits["tee_exit_code"] != 0:
        reasons.append("Driver or log capture failed")
    if [item["step"] for item in steps] != list(range(1, maximum + 1)):
        reasons.append("Missing/duplicate training updates")
    if not all(math.isfinite(item["loss"]) and math.isfinite(item["grad_norm"]) for item in steps):
        reasons.append("Non-finite training metrics")
    training = [item for item in rollouts if item["training"]]
    if [item["policy_version"] for item in training] != list(range(maximum)):
        reasons.append("Rollouts are missing, stale or duplicated")
    if not all(item["backend"] == "nemo_gym" and item["response_tokens"] > 0 and item["reward_used_in_loss"] is False for item in training):
        reasons.append("Invalid Gym rollout contract")
    if not scores or not all(item["representation"] == "topk_logprobs" and item["top_k"] == 20 for item in scores):
        reasons.append("Wrong/missing teacher representation")
    if not any("Max steps reached, stopping training." in line for line in lines):
        reasons.append("Missing normal stop marker")
    checkpoint = inspect_checkpoint(Path(cfg["checkpointing"]["checkpoint_dir"]), maximum)
    reasons.extend(checkpoint["reasons"])
    return {"success": not reasons, "reasons": reasons, "steps_observed": len(steps), "steps_expected": maximum,
        "final_checkpoint": checkpoint, "scope": "mechanical pilot completion; no quality or resume-parity evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.run_dir)
    (args.run_dir / "pilot-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
