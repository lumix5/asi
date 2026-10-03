"""Scan sequence-length ceiling for public actor-critic array loops (hang guard).

``run_actor_critic_from_arrays`` and ``run_continuous_actor_critic_from_arrays``
bound their caller-supplied leading step axis only by ``_INT32_MAX`` and then
charge a signed-int32 working-set budget. With small ``feature_dim`` /
``n_actions`` the budget admits tens of millions of steps, so a hostile or
mistaken caller drives ``jax.lax.scan`` with a sequential loop of that length
and hangs the process well before any step executes -- the same hang class
already fixed for the sibling array loops (``core/sarsa.py``,
``core/horde.py``, ``core/horde_actor_critic.py``, ``core/average_reward.py``,
``core/off_policy_td.py``, ``core/dreaming.py``).

Both functions are exported public API (``alberta_framework.__init__``).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array

from alberta_framework import (
    ActorCriticAgent,
    ActorCriticConfig,
    ContinuousActorCriticAgent,
    ContinuousActorCriticConfig,
    run_actor_critic_from_arrays,
    run_continuous_actor_critic_from_arrays,
)
from alberta_framework.core.actor_critic import (
    _ACTOR_CRITIC_SEQUENCE_MAX_STEPS,
    ActorCriticState,
    ContinuousActorCriticState,
)


class _ScanReachedError(AssertionError):
    """Raised by the scan spy: the scan body was traced with a given length."""


def _spy_scan(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []

    def spy(fn, init, xs, **kwargs):  # type: ignore[no-untyped-def]
        first = xs[0] if isinstance(xs, tuple) else xs
        seen.append(int(first.shape[0]))
        raise _ScanReachedError(f"jax.lax.scan reached: T={first.shape[0]}")

    monkeypatch.setattr("alberta_framework.core.actor_critic.jax.lax.scan", spy)
    return seen


def _discrete_agent(
    feature_dim: int = 2,
) -> tuple[ActorCriticAgent, ActorCriticState]:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2))
    return agent, agent.init(feature_dim=feature_dim, key=jr.key(3))


def _continuous_agent(
    feature_dim: int = 2,
) -> tuple[ContinuousActorCriticAgent, ContinuousActorCriticState]:
    agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=2))
    return agent, agent.init(feature_dim=feature_dim, key=jr.key(3))


def _discrete_arrays(
    num_steps: int, feature_dim: int = 2
) -> tuple[Array, Array, Array, Array]:
    observations = jnp.zeros((num_steps, feature_dim), dtype=jnp.float32)
    rewards = jnp.zeros((num_steps,), dtype=jnp.float32)
    terminated = jnp.zeros((num_steps,), dtype=jnp.bool_)
    next_observations = jnp.zeros((num_steps, feature_dim), dtype=jnp.float32)
    return observations, rewards, terminated, next_observations


def _continuous_arrays(
    num_steps: int, feature_dim: int = 2, action_dim: int = 2
) -> tuple[Array, Array, Array, Array, Array]:
    observations = jnp.zeros((num_steps, feature_dim), dtype=jnp.float32)
    rewards = jnp.zeros((num_steps,), dtype=jnp.float32)
    terminated = jnp.zeros((num_steps,), dtype=jnp.bool_)
    next_observations = jnp.zeros((num_steps, feature_dim), dtype=jnp.float32)
    actions = jnp.zeros((num_steps, action_dim), dtype=jnp.float32)
    return observations, rewards, terminated, next_observations, actions


_CEILING_ERROR = rf"1 <= num_steps <= {_ACTOR_CRITIC_SEQUENCE_MAX_STEPS}"


# =============================================================================
# run_actor_critic_from_arrays
# =============================================================================


class TestRunActorCriticFromArraysSequenceLengthGuard:
    def test_oversized_rejected_before_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _spy_scan(monkeypatch)
        agent, state = _discrete_agent()
        observations, rewards, terminated, next_observations = _discrete_arrays(
            _ACTOR_CRITIC_SEQUENCE_MAX_STEPS + 1
        )
        with pytest.raises(ValueError, match=_CEILING_ERROR):
            run_actor_critic_from_arrays(
                agent, state, observations, rewards, terminated, next_observations
            )
        assert seen == []

    def test_origin_hang_class_sized_rejected_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Mirrors the hang class this guard closes: a leading axis the int32
        # working-set budget admits (small feature_dim/n_actions) must be
        # rejected long before it reaches lax.scan.
        seen = _spy_scan(monkeypatch)
        agent, state = _discrete_agent()
        observations, rewards, terminated, next_observations = _discrete_arrays(200_000)
        with pytest.raises(ValueError, match=_CEILING_ERROR):
            run_actor_critic_from_arrays(
                agent, state, observations, rewards, terminated, next_observations
            )
        assert seen == []

    def test_exact_ceiling_reaches_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _spy_scan(monkeypatch)
        agent, state = _discrete_agent()
        observations, rewards, terminated, next_observations = _discrete_arrays(
            _ACTOR_CRITIC_SEQUENCE_MAX_STEPS
        )
        with pytest.raises(_ScanReachedError):
            run_actor_critic_from_arrays(
                agent, state, observations, rewards, terminated, next_observations
            )
        assert seen == [_ACTOR_CRITIC_SEQUENCE_MAX_STEPS]

    def test_last_fit_length_runs_end_to_end(self) -> None:
        agent, state = _discrete_agent()
        observations, rewards, terminated, next_observations = _discrete_arrays(5)
        result = run_actor_critic_from_arrays(
            agent, state, observations, rewards, terminated, next_observations
        )
        assert result.actions.shape == (5,)
        assert int(result.state.step_count) == 5


# =============================================================================
# run_continuous_actor_critic_from_arrays
# =============================================================================


class TestRunContinuousActorCriticFromArraysSequenceLengthGuard:
    def test_oversized_rejected_before_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _spy_scan(monkeypatch)
        agent, state = _continuous_agent()
        observations, rewards, terminated, next_observations, actions = _continuous_arrays(
            _ACTOR_CRITIC_SEQUENCE_MAX_STEPS + 1
        )
        with pytest.raises(ValueError, match=_CEILING_ERROR):
            run_continuous_actor_critic_from_arrays(
                agent,
                state,
                observations,
                rewards,
                terminated,
                next_observations,
                actions=actions,
            )
        assert seen == []

    def test_origin_hang_class_sized_rejected_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        agent, state = _continuous_agent()
        observations, rewards, terminated, next_observations, actions = _continuous_arrays(
            200_000
        )
        with pytest.raises(ValueError, match=_CEILING_ERROR):
            run_continuous_actor_critic_from_arrays(
                agent,
                state,
                observations,
                rewards,
                terminated,
                next_observations,
                actions=actions,
            )
        assert seen == []

    def test_exact_ceiling_reaches_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _spy_scan(monkeypatch)
        agent, state = _continuous_agent()
        observations, rewards, terminated, next_observations, actions = _continuous_arrays(
            _ACTOR_CRITIC_SEQUENCE_MAX_STEPS
        )
        with pytest.raises(_ScanReachedError):
            run_continuous_actor_critic_from_arrays(
                agent,
                state,
                observations,
                rewards,
                terminated,
                next_observations,
                actions=actions,
            )
        assert seen == [_ACTOR_CRITIC_SEQUENCE_MAX_STEPS]

    def test_last_fit_length_runs_end_to_end(self) -> None:
        agent, state = _continuous_agent()
        observations, rewards, terminated, next_observations, actions = _continuous_arrays(5)
        result = run_continuous_actor_critic_from_arrays(
            agent,
            state,
            observations,
            rewards,
            terminated,
            next_observations,
            actions=actions,
        )
        assert result.actions.shape == (5, 2)
        assert int(result.state.step_count) == 5
