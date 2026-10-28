# Real-model Gym qualification

This optional Modal harness tests Qwen3-4B (nonthinking) against isolated real
Gemma-4-26B-A4B and Qwen3-32B teachers. It does not contact company servers or
modify the existing production launcher, pilot YAMLs, or historical runs.

## Contract

`tools/xtoken_all/gym_real.yaml` pins models, revisions, the public assets ZIP,
algorithm selection, and the small canary budget. Each arm runs five independent
fresh students: `topk_tail_kl`, `dpca`, `ctopd_topk`, `simct_topk`, and
`grass_chunk_fixed_sigma`. These are the existing explicit modes described in
[Algorithm ports](algorithm-ports.md); top-k variants are approximations.

Each mode requires two real Gym HTTP rollouts, policy versions 0 and 1, teacher
top-20 prompt scores, finite losses and nonzero finite gradients, normal training
completion, and finalized step-1 and step-2 checkpoints. The existing
`check_result.py` judges these receipts. Checkpoint file closure does not prove
resume parity. This eight-example smoke does not measure quality or convergence.

The entrypoint is unchanged: `examples/run_xtoken_on_policy_distillation.py`.
The data path uses the fixed eight-example DAPO smoke pool from the reviewed
assets, converted by Super's `_convert_entry` through `convert_general_qa.py`.
Answers remain verifier metadata. Reward is not mixed into the distillation loss.

The default test allocation is two student GPUs plus one separate teacher GPU.
Changing to three student GPUs requires a divisible global batch and explicitly
selects the existing `hf_sharded` backend. Two/four use `sharded`. Actual 2/3/4-GPU
qualification must be reported separately; a two-GPU Gym pass does not qualify
three/four-GPU Gym orchestration.

## Runtime boundary

The learner uses `IMAGE_DIGEST` pinned in `modal_gym_pipeline.py`: Python 3.13.13,
Torch 2.10.0+cu129, Transformers 5.3.0, and generation vLLM 0.17.1. It is a test
runtime, not a byte-for-byte restoration of the company's Python 3.12 runtime.
The canonical Gym source is extracted from the supplied Super ZIP; all 60 files
in `super-source-contract.json` must match. Existing source and bundled native
packages are not overwritten.

The NGC image lacks the custom service environments and has Ray 2.54.0. Remote
CPU preparation installs only the explicitly pinned supplemental packages with
`--no-deps` into the task's scratch volume. Every learner role imports the same
Ray 2.55.1 supplement. Three real thin service venvs expose the existing Gym actor
packages and this supplement; import probes verify their actual prefixes and apps.
Super calls `uvicorn.run(timeout_worker_healthcheck=30)` even for one worker.
The harness pins Uvicorn 0.37.0, the first release with this API, and binds that
keyword during CPU preflight. An import-only pass is insufficient. The earlier
0.35.0 pin failed this check before another GPU launch; see the
[Uvicorn release notes](https://uvicorn.dev/release-notes/#0370-september-23-2025).
No Torch, CUDA, Transformers, or learner vLLM replacement is performed.

The separate teacher image contains vLLM 0.31.0, Torch 2.13.0+cu130 and
Transformers 5.17.0. Modal needs an added Python 3.12 control interpreter and an
empty image entrypoint. The teacher's existing `/usr/local/bin/vllm` CLI still
uses its original `/usr/bin/python3`; `teacher_inventory` checks this before GPU
allocation. Learner and teacher exchange HTTP scores, not native tensors/ABI.

Super QA requests `/v1/models` without authentication at startup. The ephemeral
test gateway forwards real metadata without client auth, while completion scoring
and tokenization require the random per-run bearer key. The actual vLLM server
remains authenticated. The key is never written into YAML or retained logs. The
gateway does not alter token IDs, scores, or upstream payloads.

## Operation and cost

Use the existing Modal SDK and explicit authorized profile. The local controller
expects `XTOKEN_GYM_OVERLAY` to contain the reviewed `Gym/` and `nltk_data/`
directories. `XTOKEN_GYM_CONFIG_JSON` is a JSON export of `gym_real.yaml`, made
with an already-installed YAML reader, so the local Modal environment needs no
new dependency. Models, datasets, and dependency downloads occur remotely.

Run `modal run tools/xtoken_all/modal_gym_pipeline.py` with required arguments
`--phase stage|imports|gpu --arm cross|same --output-dir <evidence-directory>`.
GPU calls also accept explicit `--accelerator B200` or `H100`.
Both `stage` and `imports` must finish before `gpu`. Use one controller per
profile; a locked local reservation ledger prevents concurrent GPU overspend.

The hard ceiling is cumulative USD20 per profile, including prior calendar-month
reported usage. This is not a claim about remaining credits or exact billing-cycle
alignment. Fresh provider billing, conservative outstanding reservations, and a
USD2 lag reserve must all fit. GPU reservation uses the requested topology and
bounded teacher/student lifetimes, startup and teardown. Unknown billing or a
failed guard stops submission. Failed/queued requests retain an audit entry;
settlement uses conservative elapsed-time bounds rather than calling them bills.

Both roles have bounded allocation waits. The controller always stops its own
teacher and student; it never stops unrelated apps or company Ray processes.
GPU calls have hard resource/lifetime limits and no automatic retries. H100
fallback requires an explicit new call and its own guard, and is labeled H100.
Modal can automatically allocate H200 for an H100 request at the H100 price;
reports retain both requested and actual hardware. See the
[Modal GPU contract](https://modal.com/docs/guide/gpu#automatic-upgrades-to-h200s).

Bulk canary checkpoints are temporary remote files. After checkpoint closure is
recorded, only this harness's resolved `/tmp/xtoken-real/<arm>/<mode>` directory
is removed. Sanitized logs/config/results remain on the dedicated scratch volume
and in local evidence; no model weights or optimizer tensors are downloaded to
the Windows workstation. Clean up only the task-created volume after collecting
evidence to avoid retaining unnecessary remote storage.

## Evidence

Current staging/import receipts and run logs are under
`D:/dev/codex/research_vdt/artifacts/gym-real-20261028` on the operator workstation.
GPU results must be read from their actual attempt receipts; staging/import
success alone is not a full-pipeline pass. The test runner records exact mode,
teacher arm, actual GPUs, training/rollout/teacher-score receipts, and checkpoint
closure. Scientific comparisons require a separate controlled evaluation.
