"""Compiled temperature-scaling regressions for the linear actor-critic agent.

Under ``jax.jit`` on XLA:CPU, ``softmax(x / T)`` with a non-power-of-two scale
returns NaN for finite inputs once ``|x / T|`` reaches roughly ``2**31``: XLA
recomputes the product inside both softmax fusions and the contracted FMA
leaves a rounding residual at the argmax.  Issue #2886 measured this at
``ActorCriticAgent.policy``, where ``select_action`` takes
``argmax(gumbel + log(probs))`` so a NaN policy silently wins the sample and
the learned policy is replaced by whichever index the NaN lands on.  These
tests pin the corrected construction: finite inputs keep a finite,
eager-consistent policy, an applied finite update stays applied with finite
committed state, and ordinary in-range policies are unchanged up to float32
rounding.

Every probe below is an unbatched compiled public call: a batch of four or
more rows was measured to mask this defect entirely, so batched sweeps are
not reliable evidence for it.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.actor_critic import ActorCriticAgent, ActorCriticConfig

_OBSERVATION = jnp.asarray([1.0], dtype=jnp.float32)
_LARGE_REWARD = jnp.asarray(3e10, dtype=jnp.float32)


def _agent(temperature: float = 0.7) -> ActorCriticAgent:
    return ActorCriticAgent(
        ActorCriticConfig(
            n_actions=2,
            actor_step_size=1.0,
            temperature=temperature,
        )
    )


def _start_on_second_action(agent: ActorCriticAgent) -> tuple:
    """``start`` from init on the first root key whose sample picks action 1.

    Only public calls are used: the state comes from ``init`` and ``start``,
    never from parameter injection.  The sampled action is what the applied
    update reinforces, and the sampled-action-one case is the one where a NaN
    policy is observable as a wrong sample instead of an accidentally
    matching index.
    """
    for seed in range(16):
        state = agent.init(1, jr.key(seed))
        state, action, _ = agent.start(state, _OBSERVATION)
        if int(action) == 1:
            return state, action
    raise AssertionError("no start key sampled action 1 in 16 tries")


def _eager_policy(state, temperature: float):  # type: ignore[no-untyped-def]
    logits = state.actor_weights @ _OBSERVATION + state.actor_bias
    return jax.nn.softmax(logits / jnp.float32(temperature))


def test_large_reward_transition_applies_with_finite_eager_consistent_policy() -> None:
    """A finite reward the agent must accept is applied with a finite policy.

    One applied update from ``init`` drives ``|logits / 0.7|`` to about
    ``3e10``, far past the compiled NaN onset.  The compiled policy must stay
    finite, agree with the eager policy on the same committed weights, and
    report the reinforced action.
    """
    agent = _agent()
    state, action = _start_on_second_action(agent)

    result = agent.update(state, _LARGE_REWARD, _OBSERVATION)

    np.testing.assert_array_equal(np.isfinite(np.asarray(result.policy)), True)
    committed = agent.policy(result.state, _OBSERVATION)
    np.testing.assert_array_equal(np.isfinite(np.asarray(committed)), True)
    np.testing.assert_allclose(float(np.sum(np.asarray(committed))), 1.0, rtol=1e-6)
    expected = _eager_policy(result.state, 0.7)
    np.testing.assert_allclose(np.asarray(committed), np.asarray(expected), rtol=1e-5)
    assert int(np.argmax(np.asarray(committed))) == int(action)


def test_sampling_matches_eager_policy_after_large_update() -> None:
    """Compiled sampling on eight distinct keys matches the eager policy.

    A NaN policy wins ``argmax(gumbel + log(probs))`` on every key, so the
    compiled agent collapses to action 0 while the eager policy samples the
    reinforced action.  Each actual call advances the committed state key via
    the ``(action, new_rng_key, probabilities)`` return; the earlier revision
    reused one immutable state, so all eight iterations repeated a single
    stored draw and the loop could pass on one sample.  The post-update
    policy is a near-one-hot distribution, so the eager categorical sample
    equals the eager argmax on every non-adversarial key, and each distinct
    actual key must reproduce it.
    """
    agent = _agent()
    state, _ = _start_on_second_action(agent)
    state = agent.update(state, _LARGE_REWARD, _OBSERVATION).state

    expected_probabilities = _eager_policy(state, 0.7)
    eager_action = int(np.argmax(np.asarray(expected_probabilities)))
    for _ in range(8):
        sampled_action, key, probabilities = agent.select_action(state, _OBSERVATION)
        state = state.replace(rng_key=key)
        np.testing.assert_array_equal(np.isfinite(np.asarray(probabilities)), True)
        assert int(sampled_action) == eager_action


def test_consecutive_large_reward_transitions_keep_applying_finite_state() -> None:
    """Repeated valid large-reward transitions keep finite committed state.

    The linear agent has no update finiteness gate, so a NaN policy inside one
    update silently commits NaN actor weights and every later step inherits
    them.  Once the policy computation is stable, consecutive transitions all
    apply with finite state.
    """
    agent = _agent()
    state, _ = _start_on_second_action(agent)

    for _ in range(3):
        result = agent.update(state, _LARGE_REWARD, _OBSERVATION)
        np.testing.assert_array_equal(np.isfinite(np.asarray(result.policy)), True)
        np.testing.assert_array_equal(
            np.isfinite(np.asarray(result.state.actor_bias)), True
        )
        np.testing.assert_array_equal(
            np.isfinite(np.asarray(result.state.actor_weights)), True
        )
        state = result.state


def test_opposite_extreme_logits_stay_finite_at_heating_temperature() -> None:
    """Opposite finite extremes under a heating temperature keep a finite policy.

    With float32 logits ``[3.2e38, -0.4e38]`` and ``temperature=2.0`` the
    plain expression stays finite (``logits / temperature`` is in range), so
    the corrected construction must not overflow the pre-scale centering
    either: halving both operands first keeps the subtraction representable.
    """
    agent = _agent(temperature=2.0)
    state = agent.init(1, jr.key(0))
    state = state.replace(actor_bias=jnp.asarray([3.2e38, -0.4e38], dtype=jnp.float32))

    probabilities = agent.policy(state, _OBSERVATION)
    expected = jax.nn.softmax(
        jnp.asarray([1.6e38, -0.2e38], dtype=jnp.float32),
    )
    np.testing.assert_array_equal(np.isfinite(np.asarray(probabilities)), True)
    np.testing.assert_allclose(np.asarray(probabilities), np.asarray(expected), rtol=1e-6)


def test_large_temperature_preserves_opposite_finite_logit_separation() -> None:
    """Opposite finite extreme logits survive an extreme temperature intact.

    At ``temperature=1e38`` the reciprocal ``1 / T`` is a subnormal float32,
    so the historical ``logits / temperature`` expression loses the logit
    separation.  The corrected construction halves both operands first, which
    keeps the effective reciprocal in the normal range and must return the
    true ``softmax([3, -3])`` distribution.
    """
    agent = _agent(temperature=1e38)
    state = agent.init(1, jr.key(1))
    state = state.replace(actor_bias=jnp.asarray([3e38, -3e38], dtype=jnp.float32))

    probabilities = agent.policy(state, _OBSERVATION)
    np.testing.assert_allclose(
        np.asarray(probabilities),
        np.asarray(jax.nn.softmax(jnp.asarray([3.0, -3.0], dtype=jnp.float32))),
        rtol=1e-5,
    )


@pytest.mark.parametrize("temperature", [1.0, 0.5, 0.7, 4.0])
def test_moderate_logits_match_the_historical_expression(temperature: float) -> None:
    """Ordinary in-range policies are unchanged up to float32 rounding."""
    agent = _agent(temperature=temperature)
    state = agent.init(1, jr.key(2))
    state = state.replace(actor_bias=jnp.asarray([0.75, -1.25], dtype=jnp.float32))

    probabilities = agent.policy(state, _OBSERVATION)
    expected = jax.nn.softmax(
        jnp.asarray([0.75, -1.25], dtype=jnp.float32) / jnp.float32(temperature)
    )
    np.testing.assert_allclose(np.asarray(probabilities), np.asarray(expected), rtol=1e-5)
