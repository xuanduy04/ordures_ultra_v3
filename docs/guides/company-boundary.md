# Company GRPO boundary and isolated X-Token entrypoint

The `tung/algo/minimal-company` branch narrows the integration around the
company's `vllm_teacher` commit `d92dc0967696a8982c6f226b97cd74957fb5c83d`.
This is the explicitly pinned Git baseline, not an assertion that every supplied
Super ZIP or the latest VAI deployment has identical source. The earlier
`tung/algo/operator-reconcile` branch and its GPU evidence remain intact.

## Protected company modules

These eight files match that baseline (normalizing only LF/CRLF):

- `nemo_rl/algorithms/grpo.py`
- `nemo_rl/algorithms/reward_functions.py`
- `nemo_rl/algorithms/async_utils.py`
- `nemo_rl/algorithms/opd.py`
- `examples/nemo_gym/run_grpo_nemo_gym.py`
- `nemo_rl/experience/rollouts.py`
- `custom_utils/utils_data/filter_overlong_token.py`
- `custom_utils/utils_training/use_custom_opt.sh`

All eight are restored, including the original external-teacher OPD module
name and its corresponding GRPO entrypoint. Old
GRPO field names, advantage/reward rules and logging in those files are retained.
The independent `opd_outsource.py` used by the X-Token client is still present;
restoring company OPD does not replace the X-Token remote-teacher transport.
Imported Super GRPO behavior remains available in the previous branch rather
than silently becoming the default behavior of this company-baseline branch.

`tools/xtoken_all/super-runtime-lineage.json` records the baseline commit and
SHA256 of the eight normalized files. `integrity.source_snapshot` checks this
contract before render/preflight records a source fingerprint, without Git,
network or native imports. A changed protected file causes an error; an old
bundle without this contract is not retrospectively relabeled as protected.
The baseline's existing trailing whitespace is preserved to retain its source
identity instead of editing company code solely to satisfy a whitespace check.

## Algorithm-owned behavior

The shell still calls the dedicated X-Token entrypoint, which owns its config,
training loop, teacher scoring, alignment, losses and rollout/update schedule.
It does not run the GRPO training loop.

`x_token/refit.py` now owns X-Token's weight-sync helper; it was extracted from
the previously qualified X-Token branch, including its 4GiB IPC buffer path.
`x_token/on_policy.py` no longer imports the GRPO training module for refit.
GRPO continues to call its original helper inside its original module.

`setup_response_data` has one optional keyword-only callback,
`dataset_transform=None`. With no callback, it follows the original company
data path and does not apply the imported automatic Gym prompt-length filter.
Only the X-Token entrypoint supplies `x_token/data.py`'s callback, preserving
X-Token's existing train/validation length filtering, order, and configuration.
No existing GRPO YAML gains a required key or has to opt out of X-Token behavior.

The shared tokenizer helper has one optional constructor callback,
`pretrained_factory=None`. With no callback it retains the company's loading
kwargs and template/pad behavior. Only `x_token/data.py`'s loader consumes the
new `from_pretrained_kwargs`, retaining Gemma special-token handling for the
X-Token entrypoint, pilot preparation, preflight and scoring probe. Company
callers do not acquire new tokenizer-loading behavior from a shared YAML key.

The Gym postprocessor selects the adapter's bounded-prefix/maskable-padding
handling only when `retain_truncated_token_ids` is explicitly present. Without
that new adapter field, it preserves `vllm_teacher`'s handling of truncated
responses and its empty `message_log`/empty prompt-token placeholder. Explicit
true/false adapter settings retain their existing respective meanings.
The independent `skip_load_from_dotenv` option is still explicitly selected by
the X-Token adapter; old callers without it keep the SDK's existing behavior.

## Remaining integration and verification limits

This branch does not claim that every company file is byte-identical. Worker
logit/head-input hooks, collators, additive processor/registry entries, the two
optional data/tokenizer callbacks, refit allocation logs and Automodel API/backend compatibility
remain. They support the existing X-Token/DPCA/CTOPD/SimCT/GRASS paths; this change
does not delete algorithms, change YAML defaults or replace the fixed runtime.
In particular, raw-logit/head capture is loss-selected and three-GPU HF sharding
remains explicitly configured. These are integration points to review separately
from the protected company training modules, not evidence of whole-stack parity.

No deployed company source, runtime, teacher or completed run is modified by
this branch. Deploy into a new checkout and let the operator establish fresh
allocation, runtime and teacher evidence. Do not overwrite a running checkout
or reuse an old source receipt after changing source.

Validation covers normalized source identity/tamper rejection, actual refit
control flow with stub policy/generation interfaces, old/new dataset callback
behavior, and old/explicit-adapter Gym postprocessor branches. Tensor-free
fixtures verify control flow; they are not NCCL, native imports or GPU training.
The existing two-update/200-step receipts belong to their recorded earlier
source commits. They do not qualify this restored branch automatically. A fresh
short native GRPO/company-config check and X-Token Gym GPU canary are still
needed before claiming complete deployed backward compatibility.

The 2026-10-29 local check completed with 262 passed and 15 skipped on Windows;
the separate Linux/Bash runtime-setup suite completed with 18 passed. The skipped
tests and native/GPU checks are not included in the pass count. AST/JSON parsing,
shell syntax, protected-file comparison against the pinned Git blobs and
whitespace checks outside the restored company files also passed. No paid remote
job, model download or company-node action was performed for this change.

## Native qualification harness

The subsequent GPU campaign also exercises the unchanged company entrypoint
with `adv_estimator: opd`, not a substituted reward-only GRPO test.
`tools/xtoken_all/gym_legacy.yaml` retains the OPD/loss/penalty settings from
the supplied `vt_super.yaml` SHA256
`46f7d06294762398ba5a564d8f7e60bde749ffdbc2dcd6efccf38252021c818e`.
Only the canary model/topology, batches, token budgets, steps, validation and
local logging are overridden; EOS/thinking token IDs come from the real Qwen
tokenizer. Native CPU preflight imports the actual legacy entrypoint and
constructs its rendered loss and advantage estimator before GPU allocation.

Legacy acceptance requires two finite losses and nonzero finite gradient norms
from real TensorBoard events, successful actual teacher scoring requests, normal
completion and finalized checkpoints 1/2 including replay-buffer/rollout state.
The raw TensorBoard files are retained with hashes. This does not establish
quality or resume parity. The five X-Token modes retain their existing two-update
acceptance criteria, with fresh students/optimizers per mode.

For the tight cumulative USD20/profile ceiling, the harness can reserve teacher
and learner roles on different authorized profiles. `--phase teacher` writes a
0600 ephemeral descriptor inside a private WSL `/tmp` directory; it is never
printed, committed or included in evidence. `--phase learner` consumes it and
signals the owned teacher to stop in cleanup. Each role receives its own fresh
billing guard, hard GPU/CPU/RAM/lifetime reservation and audit entry. Old usage
and reservations remain; CPU reconciliation explicitly excludes GPU role entries.
The USD2/profile billing-lag reserve is retained. Bounds use the public
[Modal resource prices](https://modal.com/pricing).

The old OPD client has no API-key option. A test-only relay bound exclusively to
the learner container's loopback interface forwards its unchanged score-only
requests to the real authenticated isolated teacher. It preserves request and
response payloads and records status/count receipts; it does not synthesize
scores or expose an unauthenticated public scorer. Protected company files are
not patched for this transport. No company teacher endpoint is contacted.

Campaign startup/preparation or a configured harness is not a GPU pass. Read the
actual new attempt receipts for results and the recorded source payload hash.
