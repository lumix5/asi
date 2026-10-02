"""Compiled temperature-scaling regressions for the Horde-backed actors.

Under ``jax.jit`` on XLA:CPU, ``softmax(x / T)`` and ``log_softmax(x / T)``
with a non-power-of-two scale return NaN for finite inputs once ``|x / T|``
reaches roughly ``2**31``: XLA recomputes the product inside both softmax
fusions and the contracted FMA leaves a rounding residual at the argmax.
Issue #2886 measured this across the four Horde actor-critic agents, where a
finite large-reward transition is either rejected forever by the finiteness
gates or committed with a NaN policy.  These tests pin the corrected
construction: finite inputs keep a finite policy, an accepted finite update,
and finite committed state.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.horde_actor_critic import (
    HordeActorCriticAgent,
    HordeActorCriticConfig,
    NonlinearHordeActorCriticAgent,
    NonlinearHordeActorCriticConfig,
    NonlinearQHordeActorCriticAgent,
    NonlinearQHordeActorCriticConfig,
    QHordeActorCriticAgent,
    QHordeActorCriticConfig,
)
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec

_OBSERVATION = jnp.asarray([1.0], dtype=jnp.float32)
_LARGE_REWARD = jnp.asarray(3e10, dtype=jnp.float32)


def _value_critic() -> HordeLearner:
    """One prediction value head, linear, as the scalar Horde critic."""
    return HordeLearner(
        create_horde_spec(
            [
                GVFSpec(
                    name="value",
                    demon_type=DemonType.PREDICTION,
                    gamma=0.9,
                    lamda=0.8,
                    cumulant_index=-1,
                )
            ]
        ),
        hidden_sizes=(),
        step_size=0.1,
        use_layer_norm=False,
    )


def _qhorde_agent(temperature: float = 0.7) -> QHordeActorCriticAgent:
    critic = HordeLearner(
        create_horde_spec(
            [
                GVFSpec(
                    name=f"q_{idx}",
                    demon_type=DemonType.CONTROL,
                    gamma=0.0,
                    lamda=0.0,
                    cumulant_index=-1,
                )
                for idx in range(2)
            ]
        ),
        hidden_sizes=(),
        step_size=0.1,
        use_layer_norm=False,
    )
    return QHordeActorCriticAgent(
        QHordeActorCriticConfig(
            n_actions=2,
            gamma=0.9,
            actor_step_size=0.05,
            actor_lamda=0.7,
            temperature=temperature,
        ),
        critic=critic,
    )


def _horde_agent(temperature: float = 0.7) -> HordeActorCriticAgent:
    return HordeActorCriticAgent(
        HordeActorCriticConfig(
            n_actions=2,
            actor_step_size=0.05,
            actor_lamda=0.7,
            temperature=temperature,
        ),
        critic=_value_critic(),
    )


def _nlhac_agent() -> NonlinearHordeActorCriticAgent:
    return NonlinearHordeActorCriticAgent(
        NonlinearHordeActorCriticConfig(
            n_actions=2,
            hidden_sizes=(),
            temperature=0.7,
            actor_epsilon=0.0,
        ),
        critic=_value_critic(),
    )


def _nlqhac_agent() -> NonlinearQHordeActorCriticAgent:
    critic = HordeLearner(
        create_horde_spec(
            [
                GVFSpec(
                    name=f"q_{idx}",
                    demon_type=DemonType.CONTROL,
                    gamma=0.0,
                    lamda=0.0,
                    cumulant_index=-1,
                )
                for idx in range(2)
            ]
        ),
        hidden_sizes=(),
        step_size=0.1,
        use_layer_norm=False,
    )
    return NonlinearQHordeActorCriticAgent(
        NonlinearQHordeActorCriticConfig(
            n_actions=2,
            hidden_sizes=(),
            temperature=0.7,
        ),
        critic=critic,
    )


def _extreme_head_weights() -> jnp.ndarray:
    """Two finite logits 1e10 apart: far past the compiled NaN onset."""
    return jnp.asarray([[1e10], [0.0]], dtype=jnp.float32)


def test_qhorde_large_reward_transition_applies_with_finite_policy() -> None:
    """A finite reward the gates must accept is applied with a finite policy.

    One applied update from ``init`` drives ``|logits / 0.7|`` to about
    ``3e9``.  Every gate input stays finite, so the transition is valid; the
    compiled NaN inside the gate's own policy computation must not reject it,
    and the committed state must not report or sample from a NaN policy.
    """
    agent = _qhorde_agent()
    state = agent.init(1, jr.key(0))
    state, _, _ = agent.start(state, _OBSERVATION)

    result = agent.update(state, _LARGE_REWARD, _OBSERVATION, jnp.asarray(False))

    assert bool(result.update_applied)
    np.testing.assert_array_equal(np.isfinite(np.asarray(result.policy)), True)

    committed = agent.policy(result.state, _OBSERVATION)
    np.testing.assert_array_equal(np.isfinite(np.asarray(committed)), True)
    np.testing.assert_allclose(float(np.sum(np.asarray(committed))), 1.0, rtol=1e-6)
    logits = result.state.actor_weights @ _OBSERVATION + result.state.actor_bias
    np.testing.assert_array_equal(
        np.argmax(np.asarray(committed)),
        np.argmax(np.asarray(jax.nn.softmax(logits / jnp.float32(0.7)))),
    )


def test_qhorde_consecutive_large_reward_transitions_keep_applying() -> None:
    """Repeated valid large-reward transitions must not livelock the agent.

    A rejected update leaves the state unchanged, so the identical transition
    is rejected again forever.  Once the policy computation is stable, every
    consecutive transition applies.
    """
    agent = _qhorde_agent()
    state = agent.init(1, jr.key(0))
    state, _, _ = agent.start(state, _OBSERVATION)

    for _ in range(3):
        result = agent.update(state, _LARGE_REWARD, _OBSERVATION, jnp.asarray(False))
        assert bool(result.update_applied)
        np.testing.assert_array_equal(np.isfinite(np.asarray(result.policy)), True)
        state = result.state


def test_horde_large_reward_transition_applies_with_finite_policy() -> None:
    """The linear value-critic actor accepts the same finite transition."""
    agent = _horde_agent()
    state = agent.init(1, jr.key(0))
    state, _, _ = agent.start(state, _OBSERVATION)

    result = agent.update(state, _LARGE_REWARD, _OBSERVATION)

    assert bool(result.update_applied)
    np.testing.assert_array_equal(np.isfinite(np.asarray(result.policy)), True)
    committed = agent.policy(result.state, _OBSERVATION)
    np.testing.assert_array_equal(np.isfinite(np.asarray(committed)), True)


@pytest.mark.parametrize("agent_factory", [_nlhac_agent, _nlqhac_agent])
def test_nonlinear_policy_stays_finite_on_extreme_logits(
    agent_factory: object,
) -> None:
    """Extreme finite head weights keep a finite, eager-consistent policy."""
    agent = agent_factory()
    state = agent.init(1, jr.key(0))
    state = state.replace(actor_head_w=_extreme_head_weights())

    probabilities = agent.policy(state, _OBSERVATION)
    np.testing.assert_array_equal(np.isfinite(np.asarray(probabilities)), True)
    np.testing.assert_allclose(float(np.sum(np.asarray(probabilities))), 1.0, rtol=1e-6)


def test_nonlinear_extreme_logit_update_applies_with_finite_state() -> None:
    """The ``jax.grad`` policy-gradient path accepts the extreme transition.

    The update differentiates ``log_softmax`` through the stabilized
    construction; the gradient stays finite, the finiteness gates accept the
    transition, and the committed head weights stay finite.
    """
    agent = _nlhac_agent()
    state = agent.init(1, jr.key(0))
    state = state.replace(
        actor_head_w=_extreme_head_weights(),
        last_observation=_OBSERVATION,
        last_action=jnp.asarray(0, dtype=jnp.int32),
    )

    result = agent.update(state, jnp.asarray(1.0), _OBSERVATION)

    assert bool(result.update_applied)
    np.testing.assert_array_equal(
        np.isfinite(np.asarray(result.state.actor_head_w)), True
    )


def test_large_temperature_preserves_opposite_finite_logit_separation() -> None:
    """Opposite finite extreme logits survive an extreme temperature intact.

    At ``temperature=1e38`` the reciprocal ``1 / T`` is a subnormal float32,
    so the historical ``logits / temperature`` expression loses the logit
    separation.  The stabilized construction halves both operands first,
    which keeps the effective reciprocal in the normal range and must return
    the true ``softmax([3, -3])`` distribution.
    """
    agent = _qhorde_agent(temperature=1e38)
    state = agent.init(1, jr.key(1))
    state, _, _ = agent.start(state, _OBSERVATION)
    state = state.replace(actor_bias=jnp.asarray([3e38, -3e38], dtype=jnp.float32))

    probabilities = agent.policy(state, _OBSERVATION)
    np.testing.assert_allclose(
        np.asarray(probabilities),
        np.asarray(jax.nn.softmax(jnp.asarray([3.0, -3.0], dtype=jnp.float32))),
        rtol=1e-5,
    )


@pytest.mark.parametrize("temperature", [1.0, 0.7, 4.0])
def test_moderate_logits_match_the_historical_expression(temperature: float) -> None:
    """Ordinary in-range policies are unchanged up to float32 rounding."""
    agent = _qhorde_agent(temperature=temperature)
    state = agent.init(1, jr.key(2))
    state, _, _ = agent.start(state, _OBSERVATION)
    state = state.replace(actor_bias=jnp.asarray([0.75, -1.25], dtype=jnp.float32))

    probabilities = agent.policy(state, _OBSERVATION)
    expected = jax.nn.softmax(
        jnp.asarray([0.75, -1.25], dtype=jnp.float32) / jnp.float32(temperature)
    )
    np.testing.assert_allclose(np.asarray(probabilities), np.asarray(expected), rtol=1e-5)
