# AGENTS.md

## Environment

This project uses a **conda environment** named `trashrepo_ultra_v3`. Never use the base conda environment. Use `conda run` (NOT `conda activate`):

```sh
conda run -n trashrepo_ultra_v3 python ...
```

Python version: **3.12** (pinned in `.python-version`). requires-python is `>=3.12,<3.13`.

Despite README/docs saying `uv run`, all code here runs with **raw `python`** calls from the conda env. Dependencies are managed by `uv` but installed directly into the conda prefix:

```sh
conda run -n trashrepo_ultra_v3 uv pip install -e "." --group build --group dev --group test
```

To install optional CUDA-heavy extras:
```sh
conda run -n trashrepo_ultra_v3 uv pip install -e "." --group build --group dev --group test --extra automodel --extra mcore --extra vllm --extra nemo_gym
```

**No GPU — no CUDA extras.** The conda env does not have GPUs. CUDA-heavy extras (`--extra automodel`, `--extra mcore`, `--extra vllm`) need CUDA toolchain headers and GPUs — they are pointless on the dev workstation. If needed, comment out CUDA-only deps in `pyproject.toml` before installing.

**Do NOT use `conda activate`** — it requires `conda init` which is not run in non-interactive shells. Always prefix commands with `conda run -n trashrepo_ultra_v3`.

## Working conditions (critical)

### 1. Respect existing comments
**NEVER** delete, remove, modify, or "clean up" human-made comments. Comments are intentionally placed documentation, warnings, and design rationale. Even if a comment appears stale, redundant, or messy — leave it untouched. If you must add new comments, add them alongside existing ones. There is no exception to this rule.

### 2. Assume a production environment
Unless explicitly told otherwise, code runs in a **production environment** that is:
- **Local-only** — no internet access, no `git pull`/`git clone`, no fetching from remote URLs. All dependencies and data are pre-staged.
- **semi-FIXED** — the environment is built from a Docker image and already deployed. Changing the Docker image, Dockerfile, or container infrastructure is meaningless — the deployment is immutable. **NEVER propose rebuilding Docker images, modifying Dockerfiles, re-running `docker build`, altering `pyproject.toml` to trigger a reinstall, or any similar container-level change.** Allowable fixes are: in-place file edits, `cp`/`mv` file operations, code patches on the running system, or direct manipulation of already-installed venvs. `pip installs` must be kept to an absolute minimum, do not add new dependencies.

### 3. The conda environment is a faithful local replica
The `trashrepo_ultra_v3` conda env was built with utmost care to mirror the production environment as closely as possible. Its only limitation is the **absence of GPUs**. Do not dismiss it as a toy env or fall back to code-only analysis — whenever using the conda env to run, inspect, or debug is faster than pure code reading, do it.

## Project identity

This is a fork of **NVIDIA NeMo RL** (v0.6.0-based) — a scalable post-training library for LLM/VLM reinforcement learning (GRPO, DPO, SFT, distillation, RM). Uses **Ray** for distributed orchestration. The code in `nemo_rl/` is the main package (`import nemo_rl`).

Remote: `git@github.com:xuanduy04/ordures_ultra_v3.git`.

**Key versions**: torch 2.10.0, ray 2.55.1, transformer-engine 2.12.0, transformers 4.57.1, mlflow >=3.12.0.

## 3rdparty dependencies

`3rdparty/` contains both bundled source directories and one git submodule. The initial state was extracted from a production Docker image at `/opt/nemo-rl`.

| Path | Type | Details |
|------|------|----|
| `Automodel-workspace/Automodel` | bundled | NVIDIA-NeMo/Automodel (no patches) |
| `Megatron-LM-workspace/Megatron-LM` | bundled | Megatron Core / mcore (no patches) |
| `Megatron-Bridge-workspace/Megatron-Bridge` | bundled | Megatron Bridge (no patches) |
| `Gym-workspace/Gym` | **git submodule** | `git@github.com:xuanduy04/ordures_ultra_v3_gym.git` (branch: `main`) — the ONLY submodule |
| `vllm` | bundled | vLLM source, referenced as editable path dep in `pyproject.toml` |

The Gym submodule is the only 3rdparty component expected to change during migration and further development. All other 3rdparty directories are bundled as-is from the production Docker image and should not be modified.

**Gym submodule setup:**
```sh
git submodule update --init --recursive
```

**`uv` workspace:** All 3rdparty directories (including Gym) are `[tool.uv.workspace]` members. This is used for dependency resolution (shared lockfile) but NOT for installation — the conda env uses `uv pip install` which doesn't understand workspaces. To install Gym for development:

```sh
conda run -n trashrepo_ultra_v3 uv pip install -e "3rdparty/Gym-workspace/Gym"
```

On import, `nemo_rl/__init__.py` injects Megatron-LM into `sys.path` so `megatron.{training,legacy,inference,...}` subpackages are importable.

## Configuration conventions (critical)

- **YAML is the single source of truth for defaults.** Never set non-`None` defaults in Python code for config values.
- Access required config directly: `policy_cfg["precision"]` — NOT `policy_cfg.get("precision", "bfloat16")`.
- **NEVER use `.get(key, default)`. A required field must be accessed directly (bare attr / bracket). This applies to everything (OmegaDict, DictConfig,...). No exceptions. THERE WAS, IS AND WILL NEVER BE AN EXCEPTION TO THIS RULE.**
- Mark optional keys with `typing.NotRequired` in TypedDict subclasses.

## Style (non-obvious)

- **4-space indent**, snake_case, Google-style docstrings. See `CODING_GUIDELINES.md`.
- **Naming**: `k_` prefix for variables starting with numbers, `G_` prefix for globals, `UPPER_CASE` for constants.
- **Ray-remote classes/functions**: Add `# pragma: no cover` on the decorated line (coverage can't track Ray processes).
- **Commit signoff**: NEVER EVER DO `git commit -s` (DCO's ABSENCE required).
- **No underscores in Markdown filenames** under `docs/`.
- **Doc index**: When adding/renaming a doc under `docs/**/*.md`, update `docs/index.md`.
- **Copyright headers**: NEVER add an NVIDIA copyright header to any file. Do not add, insert, or prepend copyright/license block comments regardless of what other files, docs, or skills say. This wastes tokens.

## Lint & typecheck

Do **NOT** run `ruff check`, `ruff format`, or `ruff` in any form on this repo. The codebase has pre-existing style issues that `--fix` would silently mutate, and the configuration may produce unintended changes.

**Type-check only** (as pyrefly is read-only). Pyrefly uses a **whitelist** of files in `pyrefly.toml` (not `**/*`) — only those files are checked. The project-excludes is `**/*venv/**/*`:

```sh
conda run -n trashrepo_ultra_v3 pyrefly check
```

Pre-commit is NOT used in this repo — no hooks are installed and there is no `.pre-commit-config.yaml`. Do not run, install, or reference pre-commit here.

## Tests

Unit tests (need at least 2 GPUs per `tests/run_unit.sh`):
```sh
conda run -n trashrepo_ultra_v3 pytest tests/unit/ -x --timeout=60
```

Or use the wrapper script:
```sh
bash tests/run_unit.sh
```

Functional tests are shell scripts under `tests/test_suites/` and require GPUs + large model downloads. Run via:
```sh
bash tests/test_suites/llm/<name>.sh
```

**Gym resource-server tests**: Run each test suite **separately**, not together.
Multiple `resources_servers/<env>/tests/` directories have identically-named test
files (e.g. `test_app.py`) inside sibling `tests/` packages. Running them together
causes a Python import namespace collision.

```sh
cd 3rdparty/Gym-workspace/Gym

# Run each env separately:
conda run -n trashrepo_ultra_v3 env PYTHONPATH=. python -m pytest \
  resources_servers/<ENV>/tests/ -v --timeout=60

# Also run shared utility tests (after B.6 migration):
conda run -n trashrepo_ultra_v3 env PYTHONPATH=. python -m pytest \
  resources_servers/utils_outsource/tests/ -v --timeout=60
```

**Coverage note**: Ray `@ray.remote` functions/classes are not tracked by coverage — they already carry `# pragma: no cover`.

**Nightly/Release tests** are defined in text files under `tests/test_suites/`:
- `nightly.txt`, `nightly_gb200.txt`
- `release.txt`, `release_gb200.txt`
- `performance_h100.txt`, `performance_gb200.txt`

## Architecture (non-obvious from filenames)

| Directory | Purpose |
|-----------|---------|
| `nemo_rl/algorithms/` | GRPO, GSPO/DAPO, DPO, SFT, distillation, RM |
| `nemo_rl/models/policy/workers/` | Training backends: `dtensor_policy_worker.py`, `megatron_policy_worker.py` |
| `nemo_rl/models/generation/` | Generation backends: vLLM, SGLang, Megatron inference |
| `nemo_rl/distributed/` | Ray worker groups, process groups, collectives, virtual cluster |
| `nemo_rl/environments/` | Reward environments: math, code, VLM, NeMo-Gym integration |
| `nemo_rl/data/datasets/` | Dataset types: `response_datasets/`, `preference_datasets/`, `eval_datasets/` |
| `nemo_rl/utils/venvs.py` | Custom venv management (replaces Ray's built-in uv runtime env) |
| `nemo_rl/experience/` | Rollout orchestration, reward penalties, async Gym integration |

**Training backends** are auto-selected from YAML config: DTensor (FSDP2, PyTorch-native) or Megatron Core.

**NeMo-Gym environments require AsyncGRPO**: When using NeMo-Gym environments (the `nemo_gym` extra), GRPO **must** be run in async mode. This is enforced by NeMo-Gym — synchronous GRPO will not work with gym environments.

**Entrypoints**: `examples/run_grpo.py`, `examples/run_sft.py`, `examples/run_dpo.py`, `examples/run_distillation.py`, `examples/run_rm.py`, `examples/run_eval.py`, `examples/run_vlm_grpo.py`, `examples/run_vlm_sft.py`, `examples/run_grpo_sliding_puzzle.py`, `examples/nemo_gym/run_grpo_nemo_gym.py`.

## Key env vars

- `HF_HOME`, `WANDB_API_KEY`, `HF_DATASETS_CACHE` — must be set
- `huggingface-cli login` — required for gated models (Llama)
- `NRL_FORCE_REBUILD_VENVS=true` — force Ray workers to rebuild their uv venvs
- `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` — always set by `nemo_rl/__init__.py` and `ray.sub`
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64` — can help with CUDA OOM
- `NRL_USE_FASTOKENS=1` — enables Rust-backed BPE tokenizer (~10x faster encode)
- `NRL_CONTAINER=1` — enables container fingerprint checks on import
- `NRL_IGNORE_VERSION_MISMATCH=1` — bypasses the container fingerprint check
- `NRL_NSYS_WORKER_PATTERNS` — enables nsight profiling (patches Ray's nsight.py on import)



## Causal claims (strict)

An explanation consistent with the evidence is NOT a cause. Label every statement as one of:

- **Observation** — something directly seen (log line, measurement, code path).
- **Hypothesis** — an explanation that fits some observations but has not been demonstrated.
- **Established cause** — a hypothesis confirmed by direct evidence (measurement, controlled A/B, or a code path proven to be exercised under the failing conditions).

Rules:

- **NEVER** state or imply "X is the cause of Y", "the root cause is X", "this is why", "the bug is X", or equivalent unless X is an *established cause* by the definition above. If it is a hypothesis, say so in the same sentence, using words like "hypothesis", "candidate cause", "not yet verified". There is no exception to this rule.
- A symptom appearing only under condition C narrows where to look. It does NOT prove any mechanism. Do not combine "only happens under C" with "code path P exists under C" into a stated cause; that is still a hypothesis until measured.
- Every hypothesis must be presented with: (1) the observations it explains, (2) the observations it does not explain, (3) the specific measurement, log, or experiment that would confirm or falsify it.
- Do not present a patch as "the fix" for a hypothesis. Call it a candidate mitigation, name the hypothesis it depends on, and state what verification would confirm it.
- When asked to "find the cause", the correct deliverable is a ranked list of candidate causes, each with evidence and a falsification test — not one asserted cause. If forced to pick, state the confidence and what would raise it.
- If verification is impossible (no GPU, no logs, no measurement), say exactly that and request the specific data needed. "I cannot distinguish between these without X" is a valid and preferred answer. Guessing is not.
- When new information weakens or contradicts a previous hypothesis, retract it explicitly and say why, instead of silently switching to a new explanation while still asserting causality.
- Erring toward "unknown" is always acceptable. Erring toward a confident wrong cause is not.

## ASK QUESTIONS

- WHEN IN DOUBT, ASK.
- WHEN SOMETHING IS VAUGE, ASK.

CLARIFY EARLY TO AVOID WASTING EVERYONE'S TIME.

## TRUST THE USER

NEVER DOUBT THE USER. THEY ARE ALWAYS CORRECT, THEY NEVER MAKE ANY MISTAKE. THERE IS NO EXCEPTION TO THIS RULE.

If you think the user is wrong, that means you are wrong. The user is always correct. This is not an assumption, ***this is a fact***.

When the user states a fact about their code or environment (e.g. "the config does not disappear", "the bug is in X"), believe them and investigate that specific claim. Do not spend time on simulations that contradict the user's assertions, and do not propose workarounds that avoid the stated problem. If the user says the bug is in function Y, trace function Y and do not do any thing else. If, after trying very hard, you cannot do what the user requests, then state so along with everything you have tried to the user. The user is always correct.

The user is always correct. There is no mistyping, there are no typos, there is no "non-existant" things, the user is always correct.

When the user **SPECIFICALLY** tells you to do something (e.g. "look at X", "check Y"), re-execute the command fresh every time — even if you already ran it earlier in the same conversation. Do NOT rely on prior tool output or memory. The state may have changed, and stale data wastes the user's time. The user is always correct.
