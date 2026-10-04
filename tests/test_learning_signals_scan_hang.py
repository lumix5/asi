"""Scan sequence-length ceiling for the public learning-signal scan (hang guard).

``LearningSignalEstimator.scan`` validated its leading step axis only against
``1 <= n <= _INT32_MAX`` and then handed the ``member_means`` sequence straight
to ``jax.lax.scan``.  A signed-int32 shape bound is not a compute budget: the
same hang/OOM class was closed for the Horde learning loop, PrototypeAgent
scans, actor-critic and world-model array loops, dream rollouts, Step 4, and
Step 7, but the shared ``ScanBudget`` consolidation missed this module.  On
current ``main`` an ``INT32_MAX``-legal sequence of a few megabytes (small
``ensemble_size``/``target_dim``) forces JAX to trace a scan of that length,
hanging the process well before any step executes.

``LearningSignalEstimator.scan`` is exported public API
(``alberta_framework.__init__``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from alberta_framework.core.learning_signals import (
    _LEARNING_SIGNAL_SCAN_BUDGET,
    LearningSignalEstimator,
    LearningSignalEstimatorConfig,
)


def _estimator() -> LearningSignalEstimator:
    return LearningSignalEstimator(
        LearningSignalEstimatorConfig(ensemble_size=2, target_dim=1)
    )


def _sequence(num_steps: int) -> tuple[Array, Array, Array, Array]:
    means = jnp.zeros((num_steps, 2, 1), dtype=jnp.float32)
    variances = jnp.ones_like(means)
    targets = jnp.ones((num_steps, 1), dtype=jnp.float32)
    losses = jnp.ones((num_steps,), dtype=jnp.float32)
    return means, variances, targets, losses


def _spy_scan(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []

    def spy(fn, init, xs, **kwargs):  # type: ignore[no-untyped-def]
        first = xs[0] if isinstance(xs, tuple) else xs
        seen.append(int(first.shape[0]))
        raise AssertionError(f"jax.lax.scan must not run: T={first.shape[0]}")

    monkeypatch.setattr("alberta_framework.core.learning_signals.jax.lax.scan", spy)
    return seen


def test_budget_matches_the_shared_documented_ceiling() -> None:
    assert _LEARNING_SIGNAL_SCAN_BUDGET.maximum_steps == 10_000
    assert _LEARNING_SIGNAL_SCAN_BUDGET.label == "learning-signal scan"


@pytest.mark.parametrize("num_steps", [10_001, 200_000])
def test_oversized_rejected_before_scan(
    num_steps: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _spy_scan(monkeypatch)
    estimator = _estimator()
    with pytest.raises(
        ValueError,
        match=r"scan sequence length must be an integer in \[1, 10000\] "
        r"for the learning-signal scan budget",
    ):
        estimator.scan(estimator.init(), *_sequence(num_steps))
    assert seen == []


def test_zero_step_sequence_keeps_its_non_empty_rejection() -> None:
    # Validation order is unchanged: rank, then non-empty, then the budget.
    estimator = _estimator()
    with pytest.raises(ValueError, match="member_means sequence must be non-empty"):
        estimator.scan(estimator.init(), *_sequence(0))


def test_last_fit_length_still_scans_end_to_end() -> None:
    estimator = _estimator()
    num_steps = _LEARNING_SIGNAL_SCAN_BUDGET.maximum_steps
    state, signals = estimator.scan(estimator.init(), *_sequence(num_steps))
    for value in jax.tree_util.tree_leaves(signals):
        assert value.shape == (num_steps,)
    assert int(state.step_count) == num_steps
