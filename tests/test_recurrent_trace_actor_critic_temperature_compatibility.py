"""Compiled temperature-scaling regressions for the RTU-RTRL actor-critic.

Under ``jax.jit`` on XLA:CPU, ``softmax(x / T)`` and ``log_softmax(x / T)``
with a non-power-of-two scale return NaN for finite inputs once ``|x / T|``
reaches roughly ``2**31``: XLA recomputes the product inside both softmax
fusions and the contracted FMA leaves a rounding residual at the argmax.
Issue #2886 measured this reachable from ``init`` through the public
``RecurrentTraceActorCriticAgent`` API.  These tests pin the corrected
construction: finite inputs keep a finite policy, an accepted finite update,
and finite gradients.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.recurrent_trace_actor_critic import (
    RecurrentTraceActorCriticAgent,
    RecurrentTraceActorCriticConfig,
    RecurrentTraceActorCriticState,
)


def _agent(**overrides: object) -> RecurrentTraceActorCriticAgent:
    config = RecurrentTraceActorCriticConfig(
        n_actions=2,
        normalize_observations=False,
        normalize_rewards=False,
        **overrides,  # type: ignore[arg-type]
    )
    return RecurrentTraceActorCriticAgent(config)


def _with_constant_logits(
    state: RecurrentTraceActorCriticState,
    head_bias: tuple[float, float],
) -> RecurrentTraceActorCriticState:
    """Replace the actor head with a constant, head-bias-only logit function."""
    actor_params = state.actor_params._replace(
        head_weights=jnp.zeros_like(state.actor_params.head_weights),
        head_bias=jnp.asarray(head_bias, dtype=jnp.float32),
    )
    return state.replace(actor_params=actor_params)


def test_low_temperature_start_policy_stays_finite_eager_and_compiled() -> None:
    """temperature=1e-12 passes validation and must not emit a NaN policy."""
    agent = _agent(temperature=1e-12)
    state = agent.init(1, jr.key(0))

    compiled_start = agent.start(state, jnp.asarray([1.0]))
    started, _action, start_probabilities = compiled_start
    np.testing.assert_array_equal(np.isfinite(np.asarray(start_probabilities)), True)

    compiled_policy = agent.policy(started)
    np.testing.assert_array_equal(np.isfinite(np.asarray(compiled_policy)), True)
    np.testing.assert_allclose(np.asarray(compiled_policy).sum(), 1.0, rtol=1e-5)


def test_compiled_update_applies_with_huge_shared_logits_at_heated_temperature() -> None:
    """A finite transition stays applied when both logits sit near float32 max.

    The two logits differ by far more than one ulp, so the direction is real,
    while ``|logits / temperature|`` is far past the ``2**31`` NaN onset.  The
    compiled path must keep reporting, sampling from, and accepting this
    policy instead of livelocking on a NaN one.
    """
    agent = _agent(temperature=0.7)
    state = agent.init(1, jr.key(1))
    started, _, _ = agent.start(state, jnp.asarray([1.0]))
    started = _with_constant_logits(started, (3e38, 3e38 - 1e32))

    result = agent.update(started, reward=jnp.asarray(0.25), observation=jnp.asarray([1.0]))

    assert bool(result.update_applied)
    np.testing.assert_array_equal(np.isfinite(np.asarray(result.policy)), True)
    np.testing.assert_array_equal(np.isfinite(np.asarray(result.entropy)), True)
    assert int(result.action) in (0, 1)

    next_probabilities = agent.policy(result.state)
    np.testing.assert_array_equal(np.isfinite(np.asarray(next_probabilities)), True)


def test_large_temperature_preserves_opposite_finite_logit_separation() -> None:
    """Opposite finite extreme logits survive an extreme temperature intact.

    At ``temperature=1e38`` the reciprocal ``1/T = 1e-38`` is a subnormal
    float32, so the historical ``logits / temperature`` expression collapses
    both logits to ``±0`` and reports a uniform ``[0.5, 0.5]`` policy.  The
    stabilized construction halves both operands first, which brings the
    effective reciprocal back into the normal range and must return the true
    ``softmax([3, -3])`` distribution instead.
    """
    agent = _agent(temperature=1e38)
    state = agent.init(1, jr.key(2))
    started, _, _ = agent.start(state, jnp.asarray([1.0]))
    started = _with_constant_logits(started, (3e38, -3e38))

    probabilities = agent.policy(started)
    np.testing.assert_allclose(
        np.asarray(probabilities),
        np.asarray(jax.nn.softmax(jnp.asarray([3.0, -3.0], dtype=jnp.float32))),
        rtol=1e-5,
    )

    result = agent.update(started, reward=jnp.asarray(0.0), observation=jnp.asarray([1.0]))
    assert bool(result.update_applied)


def test_sampler_and_policy_agree_on_stabilized_extreme_logits() -> None:
    """``select_action`` samples the same distribution ``policy`` reports."""
    agent = _agent(temperature=0.7)
    state = agent.init(1, jr.key(3))
    started, _, _ = agent.start(state, jnp.asarray([1.0]))
    started = _with_constant_logits(started, (3e38, 3e38 - 1e32))

    probabilities = agent.policy(started)
    action, _next_key, sampled_probabilities = agent.select_action(started)

    np.testing.assert_array_equal(np.isfinite(np.asarray(sampled_probabilities)), True)
    assert int(action) == int(np.argmax(np.asarray(probabilities)))


@pytest.mark.parametrize("temperature", [1.0, 0.7, 1e38])
def test_moderate_logits_match_the_historical_expression(temperature: float) -> None:
    """Ordinary in-range policies are unchanged up to float32 rounding."""
    agent = _agent(temperature=temperature)
    state = agent.init(1, jr.key(4))
    started, _, _ = agent.start(state, jnp.asarray([1.0]))
    started = _with_constant_logits(started, (0.75, -1.25))

    probabilities = agent.policy(started)
    expected = jax.nn.softmax(
        jnp.asarray([0.75, -1.25], dtype=jnp.float32) / jnp.float32(temperature)
    )
    np.testing.assert_allclose(np.asarray(probabilities), np.asarray(expected), rtol=1e-5)
