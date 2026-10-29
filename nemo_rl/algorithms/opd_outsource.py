"""On-policy distillation (OPD) helpers for async GRPO.

Teacher routing, config helpers, and remote teacher serve resolution.
Advantage computation lives in advantage_estimator.OPDAdvantageEstimator.
IS truncation lives in loss_functions.ClippedPGLoss (ICE-POP mode).

The OPD teacher is served by an externally-managed vLLM endpoint (each
``_teachers.<alias>`` entry carries a ``url`` and a ``model``). This module
scores trajectories by POSTing the student-generated token sequences to the
OpenAI-compatible ``/v1/completions`` endpoint with ``echo=true`` and
``max_tokens=0`` (score-only prefill, no decode). The response carries the
logprob of the actual next token at every position, which matches the
convention used by the rest of the training loop: the logprob of input token
``i`` is stored at position ``i``, and position 0 is 0.
"""

# Compatibility name only: retain one implementation and one set of exception
# classes. Module identity also preserves monkeypatches of private constants and
# helper functions, which a copied export namespace would not forward.
import sys

from nemo_rl.algorithms import opd as _opd

sys.modules[__name__] = _opd
