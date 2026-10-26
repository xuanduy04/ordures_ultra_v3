# X-Token source port for the production v0.6 fork

This branch ports NeMo RL v0.7.0 X-Token from commit
`81aa43dda4765b0429cf31dab44441e4e4383911` onto production source
`da929379634d7dc1e8546019e85b9b76839479af`. It adds an experimental synchronous
on-policy data path. It does not change the dependency image or bundled third-party code.
The audited upstream file hashes are in `tools/xtoken/upstream-port.json`.

## Two execution modes

Off-policy reads raw text, tokenizes it separately for student and teacher,
aligns chunks, exports full teacher logits through node-local CUDA IPC and trains
the student with the upstream X-Token loss. Multi-teacher aggregation and the
P-KL, gold and relaxed-gold objectives remain available.

On-policy prefetches prompts only. Immediately before each update it synchronizes
the current student weights into colocated vLLM through the existing production
`refit_policy_generation` IPC/ZMQ protocol. It samples one response per prompt,
keeps the original sampled student IDs and prepares teacher context using the
teacher's own chat template. Alignment operates on responses. A rollout lease
permits exactly one global optimizer update; only a successful update advances
the policy version. The next version forces a refit before sampling.

Prompt, padding, terminal stop tokens and non-matching alignment chunks have
zero loss. The first response token remains a valid next-token target. Teacher
tokens crossing the prompt/response boundary are excluded. Empty batches,
context overflow, non-terminal control tokens and replacement-character responses
fail explicitly. This conservative path can reject otherwise valid Unicode text
containing a literal U+FFFD. It never silently re-encodes sampled student IDs.

The on-policy recipe changes trajectory sampling, while retaining the X-Token
surrogate objective. Its default is forward KL for a controlled comparison with
off-policy; set `loss_fn.reverse_kl=true` for reverse KL. This is not a claim of
exact cross-tokenizer sequence KL or an unbiased full-trajectory gradient.
The upstream microbatch-global top-k support selection is retained, including
its dependence on context logits. It is not a response-only support selector.
Self-imitation CE and dynamic CE-based loss scaling are disabled in the OPD recipe.

## Source changes

| Area | Files and purpose |
| --- | --- |
| Algorithm | `algorithms/xtoken_off_policy_distillation.py`: shared setup, serial teacher export, update, validation and checkpoints; optional rollout hook |
| Loss/alignment | `algorithms/x_token/`: upstream loss, alignment, differentiable TP/CP helpers, persistent IPC buffers; isolated from existing losses |
| On-policy | `algorithms/x_token/on_policy.py`, `data/xtoken_rollout_collate.py`: refit/version lease, exact sampled IDs, response masks and per-teacher contexts |
| Data | `data/cross_tokenizer_collate.py`, `arrow_text_dataset.py`, processor/registry additions: offline raw-text data |
| Workers | `models/policy/lm_policy.py`, `dtensor_policy_worker_v2.py`: full-logit IPC and sequence-axis exclusions |
| Automodel | `models/automodel/train.py`, `data.py`: raw-logit dispatch, no double temperature scaling, CP alignment/mask sharding |
| Compatibility | `x_token/compat.py`: original checkpoint layout and data-config resume guard; `automodel_checkpoint.py`: Transformers 4/5 cache import compatibility |
| Validation | `tests/unit/algorithms/x_token/`, `tools/xtoken/modal_validate.py`: CPU, real CUDA/NCCL/IPC, native worker and tiny end-to-end stages |

All paths in the table except `tests`/`tools` are relative to `nemo_rl/`.
Existing training entrypoints retain their normal loss and temperature paths.

## Run inside the existing company runtime

Pre-stage both model/tokenizer directories, a compatible projection matrix and
the dataset. No model weights, tokenizers or datasets are downloaded by this port.
The sample YAML uses `/models/Llama-3.1-8B-Instruct`, `/models/Qwen3-235B-A22B`,
`/data/projections/llama31-qwen3.pt` and `/data/xtoken/train.parquet` as placeholders.
The 235B recipe is not a qualified full-model memory/performance configuration.

The raw-text parquet/arrow/json/txt dataset supplies a `text` column. In off-policy
mode this is training text. In on-policy `prompt_source=raw_text` it is a user
prompt, with no reference answer appended. For pre-existing message datasets,
use `kd_data_processor` and `on_policy.prompt_source=messages`; the final message
must be a user message. Disable character packing for prompts.

```bash
# On a workstation with the documented replica environment:
conda run -n trashrepo_ultra_v3 python -m examples.run_xtoken_off_policy_distillation \
  --config examples/configs/xtoken_off_policy_distillation.yaml

conda run -n trashrepo_ultra_v3 python -m examples.run_xtoken_on_policy_distillation \
  --config examples/configs/xtoken_on_policy_distillation.yaml
```

Inside a container, invoke its already-installed NeMo driver interpreter instead
of the workstation conda prefix. Copy only source/config files over the checkout
used by its editable installation. Preserve the deployed `pyproject.toml`, lock,
native venvs and third-party directories. Do not install another runtime to make
an import test pass.

The projection accepts the upstream top-k format: `indices` and `likelihoods`
with shape `[student_vocab, k]`, or its sparse `(student_id, teacher_id) -> count`
format. It must have been generated for the exact tokenizers in this run. A null
projection means direct same-vocab KL and requires equal token-to-ID mappings;
OPD additionally requires matching chat templates and template kwargs.

The port currently requires DTensor V2, fixed microbatches and no sequence packing
for all policies. Teacher/student global batch sizes must match and tile their
DP/MBS grids. Strict OPD requires temperature 1, top-p 1 and no top-k filtering.
It supports one generated assistant response per prompt, not a multi-turn agent
loop. Resume restores the training/data position and forces a new weight refit;
bitwise rollout/trajectory equivalence across restarts has not been qualified.

## Modal verification and image identity

The company custom image reference/digest is still required for production
qualification. The supplied public NGC v0.6.0 tag resolved to AMD64 image:

```text
nvcr.io/nvidia/nemo-rl@sha256:060a3337d167e66230ce8b3dac9e9b733ad25bc997f8a0121eebccf81cbf1b23
```

Its inventory is Python 3.13.13, Torch 2.10.0+cu129, Ray 2.54.0 and Transformers
5.3.0. These differ from the production replica described in `AGENTS.md`.
Passing on this stock image does not qualify the custom company runtime.
The Modal runner mounts source only, preserving native dependency fingerprints.

```bash
export MODAL_PROFILE=no1ceboy
export XTOKEN_IMAGE='<immutable company image reference>'
export XTOKEN_TEST_VOLUME='ordures-xtoken-port-20261023'
# For a private registry only: existing Modal Secret name, never its value.
# export XTOKEN_REGISTRY_SECRET='<registry secret name>'
modal run tools/xtoken/modal_validate.py --stage probe
modal run tools/xtoken/modal_validate.py --stage compat
modal run tools/xtoken/modal_validate.py --stage cpu
modal run tools/xtoken/modal_validate.py --stage math1
modal run tools/xtoken/modal_validate.py --stage math2
modal run tools/xtoken/modal_validate.py --stage gpu1
modal run tools/xtoken/modal_validate.py --stage gpu2
modal run tools/xtoken/modal_validate.py --stage off
modal run tools/xtoken/modal_validate.py --stage opd
```

Run the billing guard before allocating GPUs. `math1`/`math2` isolate CUDA math
and transport tests from the native backend. `gpu1`/`gpu2` also require the native
worker optimizer smoke; `off`/`opd` require full entrypoint training. Those four
stages run a CPU API preflight and refuse GPU allocation if it fails.
The initial stock-image run failed because its
`FSDP2Manager(config, device_mesh, ...)` constructor differs from production's
constructor. `models/automodel/runtime_compat.py` now translates the legacy
arguments into the installed config and device meshes, preserving the legacy
constructor when available. It also handles the removed, unused
`peft_init_method=None` checkpoint argument. Non-null PEFT initialization and
explicit HF TP plans fail if the installed API cannot honor them. No bundled
dependency or container fingerprint is changed. Passing the constructor preflight
alone still does not establish successful training.

GPU stages use B200 and B200:2,
bounded at 480/600 seconds. `off`/`opd` use two B200s and tiny offline Llama
fixtures with permuted teacher token IDs, not Llama-8B/Qwen-235B weights.
The OPD end-to-end gate requires two successful updates with rollout versions
0 then 1; a zero exit alone is insufficient. Logs, JUnit and JSON receipts are
persisted in the specified Volume. Local console logs live under the existing
ignored `internal_only/xtoken/` directory. See `tools/xtoken/validation.json` for
the latest recorded outcome and unresolved gates.

## An existing vLLM teacher on another node

Use `xtoken_remote_off_policy_distillation.yaml` or
`xtoken_remote_on_policy_distillation.yaml`. The student node allocates no teacher
GPU workers and loads only the teacher tokenizer and projection. Set
`teachers.0.remote.url`, its served `model` name, the actual lm-head vocabulary
size, and local model/data/projection paths. For authentication set
`remote.api_key_env` to the name of an already-provided environment variable;
never put its value into YAML. The OPD recipe uses all eight student-node GPUs
for training and colocated student generation, serially.

`XTokenFullLogprobClient` subclasses the existing `VLLMTeacherLogprobClient` from
`opd_outsource.py`, which came from the `vllm_teacher` branch. It reuses its worker
thread, event loop, HTTP session and shutdown lifecycle. Unlike sampled-token OPD,
it sends **teacher-tokenizer IDs** with `echo=true`, `max_tokens=0`,
`prompt_logprobs=-1`, `logprobs=-1`, `return_token_ids=true` and
`return_tokens_as_token_ids=true`. Full prompt distributions are indexed by
numeric IDs. API position `i` predicts input token `i`, so the adapter moves it
to X-Token predictor position `i-1`; the last predictor is unused. It validates
the echoed IDs, positions, vocabulary coverage, normalization and response size.
Missing scores, decoded-string keys, top-k-only output or HTTP rejection are
errors. Requests have bounded retries and timeouts.

The completions API returns **full log-probabilities**, not raw logits. These are
equivalent for full-vocabulary KL, including temperature scaling, because a
constant added to every logit at a position cancels in softmax. They are not
equivalent for choosing a microbatch-global top-k by maximum raw logit across
positions. The remote P-KL recipe therefore uses the full vocabulary and rejects
a smaller `vocab_topk`. It does not silently change upstream top-k semantics.

The company server is neither changed nor restarted. Its existing vLLM 0.31
configuration must already permit returning the entire vocabulary. If it does
not, this client cannot recover the missing probabilities; a top-k response is
insufficient for this recipe. Full JSON responses also cost substantial RAM,
bandwidth and parsing time: 2,048 positions times 151,936 scores are about 311
million numbers, or 1.24 GB in float32 before JSON overhead. Start with short
contexts and microbatch 1; response/element limits are upper bounds, not a claim
that a long-context response fits the available host RAM. Production throughput
has not been qualified.

The isolated Modal HTTP test uses a stock vLLM **0.31.0** teacher image pinned to
its Linux AMD64 digest:

```text
vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
```

```bash
export XTOKEN_TEACHER_IMAGE='vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'
export XTOKEN_TEST_VOLUME='ordures-xtoken-http-20261026'
modal run tools/xtoken/modal_validate.py --stage http-both
```

This creates an authenticated temporary test teacher on one B200 and runs the
student on another B200, with tiny offline Llama fixtures, permuted tokenizer IDs
and different chat templates. It terminates the test server in a `finally`
block. The company endpoint is never contacted. The gate checks dense-loss CUDA
gradients, two finite training steps with nonzero gradients, complete teacher
score receipts and, for OPD, fresh rollout versions 0 then 1. It does not qualify
Llama-8B/Qwen-235B weights, eight-GPU full-model training, company-server
capabilities, checkpoint trajectory parity or production throughput.
