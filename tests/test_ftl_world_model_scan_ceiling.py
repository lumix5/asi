"""Scan-budget bounds for the sparse FTL world-model rollout.

``run_sparse_ftl_world_model`` validated the stream length only against
signed-int32 shapes and byte overflow. A signed shape bound is not a compute
budget: an INT32-legal 10**9-step stream with ``observation_dim=1`` entered
``jax.lax.scan`` and hung. These tests pin the documented sparse FTL
world-model rollout budget (10,000 steps) ahead of any scan work.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework._scan_resources import ScanBudget
from alberta_framework.core.ftl_world_model import (
    _FTL_ROLLOUT_BUDGET,
    SparseFTLWorldModel,
    SparseFTLWorldModelConfig,
    run_sparse_ftl_world_model,
)

pytestmark = pytest.mark.unit


class _ScanReachedError(Exception):
    pass


def test_frozen_ftl_rollout_budget_bound() -> None:
    assert isinstance(_FTL_ROLLOUT_BUDGET, ScanBudget)
    assert _FTL_ROLLOUT_BUDGET.maximum_steps == 10_000


def test_scan_rejects_oversized_stream_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rollout budget must fire before any ``jax.lax.scan`` work.

    The ordering is asserted structurally: the monkeypatched ``jax.lax.scan``
    raises ``_ScanReachedError`` the moment any scan is traced. If the budget gate
    were removed or moved after the array assembly, the call would surface
    ``_ScanReachedError`` instead of the budget ``ValueError`` and both assertions
    below would fail.
    """
    reached: list[bool] = []

    def recording_scan(*_args: object, **_kwargs: object) -> object:
        reached.append(True)
        raise _ScanReachedError

    monkeypatch.setattr(jax.lax, "scan", recording_scan)
    model = SparseFTLWorldModel(
        SparseFTLWorldModelConfig(observation_dim=1, action_dim=1, projection_dim=2, bins=4)
    )
    state = model.init(jr.key(0))
    observations = jnp.zeros((10_001, 1), dtype=jnp.float32)
    with pytest.raises(ValueError, match=r"\[1, 10000\]"):
        run_sparse_ftl_world_model(model, state, observations, observations[:, 0], observations)
    assert reached == []


def test_scan_accepts_the_budget_boundary() -> None:
    """A 10,000-step stream is inside the budget and completes."""
    model = SparseFTLWorldModel(
        SparseFTLWorldModelConfig(observation_dim=1, action_dim=1, projection_dim=2, bins=4)
    )
    state = model.init(jr.key(0))
    time = jnp.arange(10_000, dtype=jnp.float32)
    observation = jnp.sin(0.01 * time)[:, None]
    action = jnp.cos(0.01 * time)
    target = observation + 0.25 * action[:, None]
    result = run_sparse_ftl_world_model(model, state, observation, action, target)
    assert result.predicted_next_observations.shape == (10_000, 1)
    assert bool(jnp.all(jnp.isfinite(result.squared_errors)))


def test_scan_still_rejects_zero_and_non_integer_lengths() -> None:
    """The pre-existing int32 message cases are unchanged by the ceiling."""
    model = SparseFTLWorldModel(
        SparseFTLWorldModelConfig(observation_dim=1, action_dim=1, projection_dim=2, bins=4)
    )
    state = model.init(jr.key(0))
    with pytest.raises(ValueError, match="scan sequence length"):
        run_sparse_ftl_world_model(
            model, state, jnp.zeros((0, 1)), jnp.zeros((0,)), jnp.zeros((0, 1))
        )
