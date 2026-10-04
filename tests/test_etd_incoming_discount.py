"""ETD(lambda) must advance its follow-on trace and emphatic trace on the
discount of the *incoming* transition.

Sutton, Mahmood & White (2016) eqs. 17-20 pair the follow-on advance

    F_t = rho_{t-1} * gamma_t * F_{t-1} + i_t

with the *previous* call's ratio **and** the *previous* call's discount —
``gamma_t`` is the discount of the transition into ``S_t`` (the discount that
multiplied ``V(S_t)`` in the previous call's TD error), exactly the convention
the sibling per-decision and gradient learners in the same module already
document and implement through the stored ``previous_gamma`` (Sutton & Barto
2nd ed., eqs. 12.23/12.25). The emphatic trace decays by the same incoming
``gamma_t * lambda`` and is scaled by the current call's ratio; only the TD
error bootstrap uses the current call's discount.

``ETDLinearLearner.update`` advertises ``gamma: State-dependent discount gamma
(0 at terminal)``, so these pairings are exercised by every episodic runner:
at a terminal the accumulated follow-on must carry into the terminal update,
and the follow-on and trace must reset for the new episode's first update.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.off_policy_td import ETDLinearLearner, ETDState, ETDUpdateResult

_X0 = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
_X1 = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
_ZERO_REWARD = jnp.float32(0.0)
_UNIT_RHO = jnp.float32(1.0)


def _step(
    learner: ETDLinearLearner,
    state: ETDState,
    observation: jnp.ndarray,
    gamma: float,
    rho: float = 1.0,
    interest: float = 1.0,
) -> ETDUpdateResult:
    return learner.update(
        state,
        observation,
        _ZERO_REWARD,
        observation,
        jnp.float32(gamma),
        jnp.float32(rho),
        jnp.float32(interest),
    )


def test_follow_on_advances_on_the_incoming_transition_pair() -> None:
    """F_2 must pair the prior call's rho with the prior call's gamma."""
    learner = ETDLinearLearner(step_size=0.1, trace_decay=0.0)
    state = learner.init(2)

    # First call: the seeded previous (gamma=1, rho=1) is inert on F_0 = 0.
    first = _step(learner, state, _X0, gamma=0.5, rho=0.8)
    np.testing.assert_allclose(float(first.state.follow_on_trace), 1.0, rtol=1e-6)

    # Second call: F_2 = gamma_1 * rho_1 * F_1 + i with the PRIOR call's
    # (gamma=0.5, rho=0.8): 0.5 * 0.8 * 1.0 + 1.0 = 1.4. Using the current
    # call's gamma=0.9 instead would give 1.72.
    second = _step(learner, first.state, _X1, gamma=0.9, rho=1.2)
    np.testing.assert_allclose(float(second.state.follow_on_trace), 1.4, rtol=1e-6)


def test_terminal_keeps_follow_on_and_reset_does_not_leak() -> None:
    """An episodic boundary must carry F into the terminal update and reset it."""
    learner = ETDLinearLearner(step_size=0.1, trace_decay=0.0)
    state = learner.init(2)

    mid_episode = _step(learner, state, _X0, gamma=0.9)
    np.testing.assert_allclose(float(mid_episode.state.follow_on_trace), 1.0, rtol=1e-6)

    # Terminal call (gamma=0): F must advance on the incoming mid-episode
    # discount: 0.9 * 1.0 * 1.0 + 1.0 = 1.9. Collapsing to i_t = 1.0 would
    # silently drop the accumulated emphatic weight of the whole episode.
    terminal = _step(learner, mid_episode.state, _X1, gamma=0.0)
    np.testing.assert_allclose(float(terminal.state.follow_on_trace), 1.9, rtol=1e-6)

    # First call of the next episode: the incoming discount is the terminal
    # call's gamma=0, so F resets to the fresh interest 1.0. Anything else
    # leaks the dead episode's follow-on across the boundary.
    next_episode = _step(learner, terminal.state, _X0, gamma=0.9)
    np.testing.assert_allclose(float(next_episode.state.follow_on_trace), 1.0, rtol=1e-6)


def test_emphatic_trace_decays_by_the_incoming_discount() -> None:
    """The trace decay pairs with gamma_t, not the current call's gamma."""
    learner = ETDLinearLearner(step_size=0.1, trace_decay=1.0)
    state = learner.init(2)

    first = _step(learner, state, _X0, gamma=0.5)
    np.testing.assert_allclose(
        np.asarray(first.state.eligibility_traces), [1.0, 0.0], rtol=1e-6
    )

    # With lambda=1 the emphasis is the interest (M = i = 1), so the second
    # call's trace must be 0.5 * e_1 + M * x_1 = [0.5, 1.0]. Decaying by the
    # current call's gamma=0.9 would give [0.9, 1.0].
    second = _step(learner, first.state, _X1, gamma=0.9)
    np.testing.assert_allclose(
        np.asarray(second.state.eligibility_traces), [0.5, 1.0], rtol=1e-6
    )


def test_terminal_update_keeps_credit_assignment_and_resets_across_episodes() -> None:
    """The terminal TD error must still reach past features."""
    learner = ETDLinearLearner(step_size=0.1, trace_decay=1.0)
    state = learner.init(2)

    first = _step(learner, state, _X0, gamma=0.5)

    # Terminal call: the trace keeps the incoming discount's weighting
    # 0.5 * e_1 + M * x_1 = [0.5, 1.0]; flushing it to [0.0, 1.0] would
    # destroy the terminal error's credit assignment to past features.
    terminal = _step(learner, first.state, _X1, gamma=0.0)
    np.testing.assert_allclose(
        np.asarray(terminal.state.eligibility_traces), [0.5, 1.0], rtol=1e-6
    )

    # New episode: the incoming discount is the terminal gamma=0, so the
    # trace resets to the fresh feature instead of carrying [0.5, 1.0].
    next_episode = _step(learner, terminal.state, _X0, gamma=0.9)
    np.testing.assert_allclose(
        np.asarray(next_episode.state.eligibility_traces), [1.0, 0.0], rtol=1e-6
    )


def test_first_call_follow_on_is_interest_regardless_of_current_gamma() -> None:
    """The seeded previous (gamma, rho) pair must be inert on F_0 = 0."""
    learner = ETDLinearLearner(step_size=0.1, trace_decay=0.0)
    for gamma in (0.0, 0.3, 0.7, 1.0):
        result = _step(learner, learner.init(2), _X0, gamma=gamma)
        np.testing.assert_allclose(float(result.state.follow_on_trace), 1.0, rtol=1e-6)
        assert bool(result.update_applied)


def test_state_carries_and_validates_previous_gamma() -> None:
    """The prior call's discount is part of the state contract."""
    learner = ETDLinearLearner(step_size=0.1, trace_decay=0.0)
    state = learner.init(2)
    result = _step(learner, state, _X0, gamma=0.5)
    np.testing.assert_allclose(float(result.state.previous_gamma), 0.5, rtol=1e-6)

    # The strict state contract must reject a non-scalar previous_gamma.
    corrupted = result.state.replace(
        previous_gamma=jnp.zeros((1,), dtype=jnp.float32),
    )  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="state.previous_gamma"):
        learner.predict(corrupted, _X0)
