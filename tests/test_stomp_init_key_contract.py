"""STOMP ``init`` must enforce the typed-key contract its states are held to.

``STOMPState`` validity requires a scalar typed JAX PRNG key
(``state.rng_key.shape == ()`` with a ``prng_key`` subtype).  ``init`` used to
accept any array — including a legacy ``jax.random.PRNGKey(0)`` whose shape is
``(2,)`` — after which every later transaction failed ``state_valid`` and was
silently dropped: the agent returned its prior state forever with no error at
any boundary.  ``init`` now rejects such keys eagerly, so the accepted key set
equals the set the state contract accepts.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import random as jr

from alberta_framework.core.options import STOMPAgent, STOMPConfig, SubtaskSpec


def _agent() -> STOMPAgent:
    config = STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),),
        observation_dim=1,
        n_primitive_actions=2,
    )
    return STOMPAgent(config)


def test_init_rejects_legacy_prng_key_with_explicit_error() -> None:
    """A legacy ``(2,)`` uint32 key wedges every later transaction; reject it."""
    agent = _agent()
    with pytest.raises(TypeError, match="typed JAX PRNG key"):
        agent.init(jax.random.PRNGKey(0))


def test_init_rejects_raw_word_arrays_and_wrong_shapes() -> None:
    """Arbitrary arrays are not keys and must not silently prime the agent."""
    agent = _agent()
    for bad in (
        jnp.zeros((2,), dtype=jnp.uint32),
        jnp.zeros((), dtype=jnp.float32),
        jnp.zeros((4,), dtype=jnp.uint32),
    ):
        with pytest.raises(TypeError, match="typed JAX PRNG key"):
            agent.init(bad)


def test_init_accepts_exactly_the_keys_the_state_contract_accepts() -> None:
    """Every key ``init`` accepts must produce a state ``state_valid`` passes."""
    agent = _agent()
    for key in (jr.key(0), jr.split(jr.key(1))[0], jr.fold_in(jr.key(2), 7)):
        state = agent.init(key)
        assert bool(agent.state_valid(state))


def test_typed_key_init_primes_and_applies_a_transaction() -> None:
    """The documented typed-key path stays fully functional."""
    agent = _agent()
    state = agent.init(jr.key(0))
    observation = jnp.asarray([1.0], dtype=jnp.float32)
    state = agent.start(state, observation)
    result = agent.update(state, jnp.float32(1.0), observation)
    assert bool(result.update_applied)
