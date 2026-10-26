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
ignored `internal_only/xtoken/` directory. `tools/xtoken/validation.json` preserves
the initial port outcome; see `tools/xtoken/http-validation.json` for the later
external-teacher pipeline checks and unresolved gates.

## An existing vLLM teacher on another node

Use `xtoken_remote_off_policy_distillation.yaml` or
`xtoken_remote_on_policy_distillation.yaml`. The student node allocates no teacher
GPU workers and loads only the teacher tokenizer and projection. Set
`teachers.0.remote.url`, its served `model` name, the actual lm-head vocabulary
size, and local model/data/projection paths. For authentication set
`remote.api_key_env` to the name of an already-provided environment variable;
never put its value into YAML. The OPD recipe uses all eight student-node GPUs
for training and colocated student generation, serially. Start-up defaults are
global batch 8 and context 256, with 128 prompt/128 generated tokens for OPD.

`XTokenFullLogprobClient` subclasses the existing `VLLMTeacherLogprobClient` from
`opd_outsource.py`, which came from the `vllm_teacher` branch. It reuses its worker
thread, event loop, HTTP session and shutdown lifecycle. Unlike sampled-token OPD,
it sends **teacher-tokenizer IDs** with `echo=true`, `max_tokens=0`,
`prompt_logprobs=-1`, `logprobs=null`, `return_token_ids=true` and
`return_tokens_as_token_ids=true`. Full prompt distributions are indexed by
numeric IDs. API position `i` predicts input token `i`, so the adapter moves it
to X-Token predictor position `i-1`; the last predictor is unused. It validates
the echoed IDs, positions, vocabulary coverage, normalization and response size.
Missing scores, decoded-string keys, top-k-only output or HTTP rejection are
errors. Requests have bounded retries and timeouts.
Only `prompt_logprobs` requests the dense distribution; the ordinary `logprobs`
field is null to avoid serializing a second copy in `top_logprobs`.

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
The element limit applies to the whole padded global teacher batch, as well as
each HTTP request. The client preallocates the final tensor once instead of
retaining all microbatch tensors and concatenating another full copy.

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

## Approximation for an unchanged top-20 endpoint

The company endpoint was observed to reject a full-vocabulary request: requested
151,936 scores, maximum allowed 20. This prevents the original P-KL/Gold/H-KL
transport, but does not prevent a separate top-k objective. The approximate
recipes are `xtoken_topk_off_policy_distillation.yaml` and
`xtoken_topk_on_policy_distillation.yaml`. They request `prompt_logprobs=20` and
never request full scores. The server is neither changed nor restarted.

The client preserves the original, globally normalized log-probabilities and
numeric IDs. vLLM may additionally return the actual input token outside top-k;
that token is retained as another explicit bucket. At teacher position `t`,
let `S_t` be returned token IDs reachable through the projection. The teacher
distribution used by the loss is:

```text
p_bar_t = [p_t(v) for v in S_t] + [1 - sum_{v in S_t} p_t(v)]
```

The last bucket is OTHER. It includes unreturned tokens and returned IDs that
cannot be reached through the projection. Student mapping rows are normalized
to sum to one when mapped; missing student rows contribute to OTHER. Student
probabilities are averaged over the aligned response chunk, projected into the
teacher vocabulary, then gathered on the same `S_t` with their own OTHER mass.
The loss is KL between these two coarse distributions. `reverse_kl` selects
its direction. Teacher-position losses are averaged within each chunk; chunks
are weighted by their number of student response tokens and normalized by the
global valid student-token count across data-parallel workers.

This is **a separate approximation**, not original P-KL, Gold, H-KL, nor an
unbiased estimator of their full-vocabulary losses. It does not reconstruct
relative probabilities within OTHER. With one-to-one token alignment, complete
projection coverage and all vocabulary IDs returned, it reduces to ordinary
projected KL. Tests check the finite top-k case against an independent dense
reference, including unequal chunk lengths, prompt masking, dropped samples,
missing projection rows, forward/reverse gradients and zero tail mass.

The top-k-plus-tail construction has prior art in
[Tail-Aware Top-k OPD](https://arxiv.org/abs/2608.14728). This implementation adds
cross-tokenizer projection/chunk handling; it does not claim to reproduce that
paper's experiments or its sampled-token correction. No quality improvement is
established by a two-step pipeline test.

Initial restrictions: TP=CP=1 with data parallelism, temperature 1, static teacher
weights, CE=0 and no dynamic scaling. Temperature cannot be changed after scores
arrive because the hidden tail cannot be re-tempered exactly. Gold/H-KL flags
are rejected in this mode. The student still computes its full distribution and
sparse projection locally; only teacher transport is sparse.

For the four free company GPUs in this run, start from the existing driver
environment, with source/config already copied into its editable checkout:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/nemo_rl_venv/bin/python \
  -m examples.run_xtoken_on_policy_distillation \
  --config examples/configs/xtoken_topk_on_policy_distillation.yaml \
  cluster.gpus_per_node=4 cluster.num_nodes=1 \
  teachers.0.remote.url=http://teacher.internal:8000 \
  teachers.0.remote.model=Qwen3-235B-A22B-Instruct \
  policy.model_name=/models/Llama-3.1-8B-Instruct \
  policy.tokenizer.name=/models/Llama-3.1-8B-Instruct \
  teachers.0.tokenizer.name=/models/exact-teacher-tokenizer \
  teachers.0.projection_matrix_path=/data/projections/llama31-qwen3.pt \
  data.train.data_files=/data/xtoken/train.parquet \
  distillation.max_num_steps=2
```

Replace the endpoint and local paths with the actual staged resources. Do not
copy dependency manifests, bundled third-party code or a replacement venv into
the deployed image. The company runtime inspection reported Python 3.12.12,
Torch 2.10.0+cu129, Transformers 4.57.1 and Automodel 0.2.0, with the legacy FSDP
constructor and public `to_hf` adapter. Those observed APIs are handled by the
source compatibility layer; a real four-GPU company run remains a separate gate.

The Modal `http-topk-both` stage uses an isolated vLLM 0.31.0 teacher started with
the same maximum of 20 scores, one teacher B200 and one student B200. Its tiny
checkpoint has nontrivial tail probability. It verifies off-policy and on-policy
entrypoints, finite gradients and fresh rollout/refit versions. Logs and receipts
identify the exact source and images used.

## Current target: Qwen3-4B student and Gemma 4 teacher

The recorded CPU suite has 169 passing tests and six GPU-only skips. Separate
CUDA tests check the top-k loss and its backward pass under autocast. Tiny Llama
and Qwen3 students completed two off-policy and two on-policy updates through
an isolated vLLM 0.31.0 top-20 teacher. OPD receipts require rollout versions
0 then 1. These checks establish the tested mechanics, not convergence or
full-model production readiness.

`source-manifest` compares the mounted code against the checkout. The tested
stock NGC overlay matches all 159 local Python files but retains 24 image-only
modules, including an `algorithms/loss` package. The JSON report records these
paths; this mixed source tree is another reason exact company-image validation
remains a separate gate. No company runtime files were modified by these tests.

Use `xtoken_qwen3_gemma4_off_policy_distillation.yaml` or
`xtoken_qwen3_gemma4_on_policy_distillation.yaml` for the updated pair. These
recipes use four student GPUs, a separate HTTP teacher, top-20 plus OTHER KL,
and non-thinking chat prefixes for both models. They require a new
Qwen3-to-Gemma4 projection; the Llama-to-Qwen projection is incompatible.

The official [Gemma 4 26B A4B configuration](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json)
sets its text lm-head vocabulary to 262,144. Its
[tokenizer metadata](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/tokenizer_config.json)
uses `GemmaTokenizer` and a list in `extra_special_tokens`, whereas
[Transformers 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/tokenization_utils_base.py)
expects named entries in that field. The recipe passes an explicit named-token
mapping through `tokenizer.from_pretrained_kwargs` while keeping the video token
in `additional_special_tokens`. This changes loading arguments in memory, not
the staged files or installed Transformers. The standard loader still applies
the configured chat template and `enable_thinking=false` kwargs.

The schema bridge and recipe parsing are unit-tested. Loading the actual Gemma 4
tokenizer files on the company runtime, checking offsets and comparing encoded
IDs against the newly hosted teacher remain separate integration gates. The
named metadata tokens must retain their original IDs; do not accept a tokenizer
that adds new IDs beyond the teacher's lm-head vocabulary.

The `http-topk-qwen3` Modal stage changes the tiny student fixture to the actual
Qwen3 architecture while retaining a tiny Llama HTTP teacher. This tests the
student backend/refit path and cross-tokenizer protocol without downloading
Qwen3-4B or Gemma4-26B weights. It does not qualify those full checkpoints.
The NGC HF-5 fallback initializes Qwen3 parameters from config on `meta` and
delegates weight loading to the existing Checkpointer. Nonpersistent buffers,
including RoPE frequencies, retain real data. Weight tying happens after the
empty-parameter context. This avoids a second weight-loading attempt inside
Accelerate's patch and prevents meta-buffer transfers to CUDA. HF-4 and custom
Automodel model initialization keep their existing path.

The official [Qwen3-4B configuration](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json)
ties input embeddings and the output head. The CPU native check covers both
tied and untied construction. `http-full-qwen3-opd` exercises full-score OPD
with a tied Qwen3 fixture, including vLLM refit. Receipts record the architecture,
weight-tying setting and tested source hash; earlier untied fixtures remain
separate evidence.

After staging the exact tokenizer, projection and dataset files and replacing
the teacher URL/served name in YAML:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/nemo_rl_venv/bin/python \
  -m examples.run_xtoken_on_policy_distillation \
  --config examples/configs/xtoken_qwen3_gemma4_on_policy_distillation.yaml \
  distillation.max_num_steps=2
```

The example alias `gemma4-teacher` must match `/v1/models` from the new server.
No change to the old Qwen teacher is needed. Use the new server's actual score
capability; its limits cannot be inferred from the previous Qwen deployment.
