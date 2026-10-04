"""Compiled finiteness regressions for the behavior-model temperature softmax.

Under ``jax.jit`` on XLA:CPU, ``jax.nn.softmax(logits / temperature)`` in
``core/behavior_model.py`` returns NaN for finite inputs once ``|logits /
temperature|`` reaches about ``2**31``: XLA recomputes the product inside both
softmax fusions and LLVM contracts the multiply-then-subtract into one FMA, so
the argmax entry carries the rounding residual of the product instead of an
exact zero (SlopDotCash/asi#2886, behavior_model sites).  The default
temperature 1.0 is immune, but any non-power-of-two temperature such as 0.7 is
reachable from ``init`` through the public API alone: one applied
``update(obs=[3e5], a=0)`` moves the weights to about ``+-1.1e4`` and the next
compiled ``predict_probabilities`` call then produces ``[nan, nan]`` while
eager evaluation returns ``[1, 0]``.

Measured consequences pinned here, all on unbatched compiled calls (batches of
four or more rows were measured to mask the defect entirely):

- ``predict_probabilities`` stays finite and matches eager after the measured
  update instead of handing a NaN policy to the caller;
- the following ``update`` at the measured scale stays applied instead of
  being rejected by the finiteness gate on every subsequent transition;
- ``input_loss_gradient`` stays valid with a finite gradient instead of
  reporting ``valid=False`` and dropping the bridge;
- ``action_probability`` is not silently replaced by the uniform coin flip
  that ``floor_and_renormalize_probabilities`` produces from a NaN row;
- ordinary finite-scale rows keep matching eager evaluation, so the
  construction only repairs the compiled contraction artifact.
"""

import jax
import jax.numpy as jnp
import numpy as np

from alberta_framework.core.behavior_model import BehaviorModel, BehaviorModelConfig

_TEMPERATURE = 0.7
_MEASURED_OBSERVATION = jnp.asarray([3.0e5], dtype=jnp.float32)
_MEASURED_ACTION = jnp.asarray(0, dtype=jnp.int32)


def _model() -> BehaviorModel:
    """Return the two-action model at the measured non-power-of-two temperature."""
    return BehaviorModel(BehaviorModelConfig(n_actions=2, step_size=0.05, temperature=_TEMPERATURE))


def _state_after_measured_update() -> tuple[BehaviorModel, object]:
    """Reach the huge-weight state through the public API only (no injection)."""
    model = _model()
    state = model.init(feature_dim=1, key=jax.random.key(0))
    first = model.update(state, _MEASURED_OBSERVATION, _MEASURED_ACTION)
    assert bool(first.update_applied)
    return model, first.state


def _eager_probabilities(model: BehaviorModel, state, observation) -> jnp.ndarray:
    """Plain eager ``softmax(logits / T)`` reference (no jit, no centering)."""
    logits = jnp.asarray(state.weights) @ jnp.asarray(observation) + jnp.asarray(state.bias)
    return jax.nn.softmax(logits / _TEMPERATURE)


def test_compiled_predict_probabilities_finite_and_match_eager_after_measured_update() -> None:
    """The measured path must not return a NaN policy under jit."""
    model, state = _state_after_measured_update()
    compiled = model.predict_probabilities(state, _MEASURED_OBSERVATION)
    assert bool(jnp.all(jnp.isfinite(compiled)))
    eager = _eager_probabilities(model, state, _MEASURED_OBSERVATION)
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager), rtol=1e-5, atol=1e-8)


def test_second_update_at_measured_scale_stays_applied_and_matches_eager() -> None:
    """The finiteness gate must keep accepting what eager evaluation accepts."""
    model, state = _state_after_measured_update()
    second = model.update(state, _MEASURED_OBSERVATION, _MEASURED_ACTION)
    assert bool(second.update_applied)
    eager_logits = jnp.asarray(state.weights) @ _MEASURED_OBSERVATION + jnp.asarray(state.bias)
    eager_p = jax.nn.softmax(eager_logits / _TEMPERATURE)
    one_hot = jax.nn.one_hot(_MEASURED_ACTION, 2, dtype=jnp.float32)
    logit_error = (one_hot - eager_p) / _TEMPERATURE
    step_size = 0.05
    eager_weights = jnp.asarray(state.weights) + step_size * (
        logit_error[:, None] * _MEASURED_OBSERVATION[None, :]
    )
    eager_bias = jnp.asarray(state.bias) + 0.05 * logit_error
    np.testing.assert_allclose(
        np.asarray(second.state.weights), np.asarray(eager_weights), rtol=1e-4, atol=1e-3
    )
    np.testing.assert_allclose(
        np.asarray(second.state.bias), np.asarray(eager_bias), rtol=1e-4, atol=1e-6
    )


def test_compiled_input_loss_gradient_valid_and_matches_eager_after_measured_update() -> None:
    """The causal bridge must stay valid instead of silently reporting invalid."""
    model, state = _state_after_measured_update()
    result = model.input_loss_gradient(state, _MEASURED_OBSERVATION, _MEASURED_ACTION)
    assert bool(result.valid)
    assert bool(jnp.all(jnp.isfinite(result.gradient)))
    eager_logits = jnp.asarray(state.weights) @ _MEASURED_OBSERVATION + jnp.asarray(state.bias)
    eager_p = jax.nn.softmax(eager_logits / _TEMPERATURE)
    one_hot = jax.nn.one_hot(_MEASURED_ACTION, 2, dtype=jnp.float32)
    eager_gradient = jnp.asarray(state.weights).T @ ((eager_p - one_hot) / _TEMPERATURE)
    np.testing.assert_allclose(
        np.asarray(result.gradient), np.asarray(eager_gradient), rtol=1e-4, atol=1e-6
    )


def test_action_probability_not_silently_uniform_after_measured_update() -> None:
    """A NaN row must not collapse into the floor-renormalized coin flip."""
    model, state = _state_after_measured_update()
    compiled = model.predict_probabilities(state, _MEASURED_OBSERVATION)
    argmax = int(jnp.argmax(compiled))
    probability = model.action_probability(
        state, _MEASURED_OBSERVATION, jnp.asarray(argmax, dtype=jnp.int32)
    )
    assert float(probability) > 0.99


def test_ordinary_rows_keep_matching_eager_across_temperatures() -> None:
    """The construction must not perturb finite-scale behavior (drift guard)."""
    for temperature in (1.0, 0.5, 0.3, 0.7, 0.99):
        model = BehaviorModel(
            BehaviorModelConfig(n_actions=3, step_size=0.05, temperature=temperature)
        )
        state = model.init(feature_dim=4, key=jax.random.key(7))
        observation = jnp.asarray([1.5, -2.0, 0.25, 3.0], dtype=jnp.float32)
        compiled = model.predict_probabilities(state, observation)
        logits = jnp.asarray(state.weights) @ observation + jnp.asarray(state.bias)
        eager = jax.nn.softmax(logits / temperature)
        np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager), rtol=2e-5, atol=1e-7)
