"""Scan sequence-length budget for the PrototypeAgent scan entrypoints.

``PrototypeAgent.scan`` and ``PrototypeAgent.scan_transitions`` hand their
caller-supplied transition sequences straight to ``jax.lax.scan`` with no
leading-axis bound beyond ``1 <= n <= _INT32_MAX``. A signed-int32 shape bound
is not a compute budget: the maintainer rejection of PR #2041 said exactly
this for both of these entrypoints ("``_INT32_MAX`` is not a meaningful
scan/resource ceiling. Both PrototypeAgent scan paths ... still accept
sequences with hundreds of thousands or millions of steps, which can trigger
pathological tracing even though the values fit int32"). Merged PR #2060
consolidated every other scan path onto the shared ``ScanBudget`` contract in
``alberta_framework._scan_resources`` but missed this module.

With a small observation dimension a hostile or mistaken caller can pass a
multi-million-step array of a few megabytes; tracing and compiling a scan of
that length hangs the process well before any step executes. Both entrypoints
now require ``1 <= n <= 10_000`` (the shared scan ceiling) through
``require_scan_steps`` before any JAX work, exactly like
``core.dreaming._DREAM_ROLLOUT_BUDGET`` and the other consolidated budgets.

The oversize regressions monkeypatch ``jax.lax.scan`` with a spy that fails
the test if tracing is ever reached, so the failing-first run on an unfixed
tree is fast instead of hanging.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.prototype_agent import (
    _PROTOTYPE_SCAN_BUDGET,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeTransition,
)

_OVERSIZE_PATTERN = (
    rf"must be an integer in \[1, {_PROTOTYPE_SCAN_BUDGET.maximum_steps}\] "
    rf"for the {_PROTOTYPE_SCAN_BUDGET.label} budget"
)


def _spy_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail immediately if ``jax.lax.scan`` is ever reached."""

    def spy(fn, init, xs, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("jax.lax.scan must not run for an oversized sequence")

    monkeypatch.setattr("alberta_framework.core.prototype_agent.jax.lax.scan", spy)


def _agent_and_state() -> tuple[PrototypeAgent, object]:
    config = PrototypeAgentConfig()
    agent = PrototypeAgent(config)
    key = jr.key(123)
    state = agent.init(key)
    state = agent.start(state, jnp.zeros(config.oak.observation_dim, dtype=jnp.float32))
    return agent, state


def _transitions(num_steps: int, observation_dim: int) -> PrototypeTransition:
    return PrototypeTransition(
        observation=jnp.zeros((num_steps, observation_dim), dtype=jnp.float32),
        action=jnp.zeros((num_steps,), dtype=jnp.int32),
        decision_id=jnp.zeros((num_steps, 4), dtype=jnp.uint32),
        reward=jnp.zeros((num_steps,), dtype=jnp.float32),
        discount=jnp.ones((num_steps,), dtype=jnp.float32),
        terminated=jnp.zeros((num_steps,), dtype=jnp.bool_),
        truncated=jnp.zeros((num_steps,), dtype=jnp.bool_),
        next_observation=jnp.zeros((num_steps, observation_dim), dtype=jnp.float32),
        next_decision_observation=jnp.zeros((num_steps, observation_dim), dtype=jnp.float32),
    )


def test_budget_matches_shared_scan_ceiling() -> None:
    assert _PROTOTYPE_SCAN_BUDGET.maximum_steps == 10_000
    assert _PROTOTYPE_SCAN_BUDGET.label == "Prototype agent transition scan"


def test_scan_rejects_oversized_sequence_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, state = _agent_and_state()
    _spy_scan(monkeypatch)
    obs_dim = agent.config.oak.observation_dim
    rewards = jnp.zeros((_PROTOTYPE_SCAN_BUDGET.maximum_steps + 1,), dtype=jnp.float32)
    next_obs = jnp.zeros((_PROTOTYPE_SCAN_BUDGET.maximum_steps + 1, obs_dim), dtype=jnp.float32)
    with pytest.raises(ValueError, match=_OVERSIZE_PATTERN):
        agent.scan(state, rewards, next_obs)  # type: ignore[arg-type]


def test_scan_transitions_rejects_oversized_transitions_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, state = _agent_and_state()
    _spy_scan(monkeypatch)
    transitions = _transitions(
        _PROTOTYPE_SCAN_BUDGET.maximum_steps + 1, agent.config.oak.observation_dim
    )
    with pytest.raises(ValueError, match=_OVERSIZE_PATTERN):
        agent.scan_transitions(state, transitions)  # type: ignore[arg-type]


def test_scan_rejects_zero_steps_with_budget_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, state = _agent_and_state()
    _spy_scan(monkeypatch)
    obs_dim = agent.config.oak.observation_dim
    with pytest.raises(ValueError, match=_OVERSIZE_PATTERN):
        agent.scan(
            state,
            jnp.zeros((0,), dtype=jnp.float32),
            jnp.zeros((0, obs_dim), dtype=jnp.float32),
        )


def test_scan_transitions_rejects_zero_steps_with_budget_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, state = _agent_and_state()
    _spy_scan(monkeypatch)
    transitions = _transitions(0, agent.config.oak.observation_dim)
    with pytest.raises(ValueError, match=_OVERSIZE_PATTERN):
        agent.scan_transitions(state, transitions)


def test_scan_still_runs_small_sequences() -> None:
    agent, state = _agent_and_state()
    obs_dim = agent.config.oak.observation_dim
    rewards = jnp.zeros((3,), dtype=jnp.float32)
    next_obs = jnp.zeros((3, obs_dim), dtype=jnp.float32)
    result = agent.scan(state, rewards, next_obs)
    assert result.actions.shape == (3,)


def test_scan_transitions_still_runs_small_sequences() -> None:
    agent, state = _agent_and_state()
    transitions = _transitions(3, agent.config.oak.observation_dim)
    result = agent.scan_transitions(state, transitions)
    assert result.actions.shape == (3,)
