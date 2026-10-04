# mypy: disable-error-code="attr-defined"
"""Regression pins for the associative-memory cross-entropy logit contraction.

`row_step` forms ``row_logits = logit_scale * old_row`` and hands the product
to ``_cross_entropy_from_logits``, which subtracts ``max(logits)``. A backend
whose compiler re-forms the scaled product inside the max-subtraction fusion
(measured on arm64 XLA:CPU) turns the difference at the maximum into the
rounding error of the product instead of an exact zero, so the per-feature
loss shifts, the feature utility shifts with it, and the ±8 clip can commit a
large wrong-but-finite utility that no finiteness gate can see (#2885).

The pins here carry two layers, matching the platform notes on the issue:

- **Deterministic shift-invariance guards.** Cross entropy is exactly shift
  invariant. Centering the operand before the ``logit_scale`` product makes
  that exactness bitwise for rows whose entries all lie in ``[0, 1)``, where
  an exact ``+1`` shift is representable for every entry and both centering
  subtractions are exact (every intermediate difference is a multiple of
  ``2**-23`` in ``(-1, 1)``). These discriminate through ordinary rounding on
  every IEEE-754 backend, independent of contraction, and assert the losses
  are *bit-identical* — a property of the exact construction, not of this
  host's outputs.
- **Large-row compiled guards.** The two-identical-writes scenario from the
  issue, where every per-feature loss equals the aggregate loss so the exact
  utility is zero. On a contracting backend an unpatched tree commits
  ``±8``-clipped utilities while reporting ``update_applied=True``; on a
  non-contracting backend these pass either way. They keep the exact-answer
  pin in place so the suite cannot silently regress in either direction.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core import associative_memory as associative_memory_module
from alberta_framework.core.associative_memory import (
    AssociativeMemoryConfig,
    AssociativeMemoryLearner,
)

# Rows with every entry in [0, 1): an exact +1 shift is exactly representable
# for each entry, so the shifted row equals the base row plus exactly one in
# exact arithmetic. Any loss difference is pure rounding of the scaled product
# path. Measured discriminating on x86_64 (JAX 0.11.0) and reported
# discriminating on arm64 against unpatched main.
_SHIFT_INVARIANT_ROWS = (
    jnp.asarray([[0.125, 0.5, 0.0625, 0.375]], dtype=jnp.float32),
    jnp.asarray([[0.25, 0.75, 0.5, 0.0]], dtype=jnp.float32),
    jnp.asarray([[0.0, 0.9375, 0.3125, 0.1875]], dtype=jnp.float32),
)
_NON_POWER_OF_TWO_SCALES = (0.3, 0.7, 0.9, 0.99)


def _feature_loss(row_values: jax.Array, logit_scale: float, label: int) -> jax.Array:
    config = AssociativeMemoryConfig(
        vocab_size=int(row_values.shape[1]),
        block_size=2,
        suffix_length=2,
        logit_scale=logit_scale,
    )
    learner = AssociativeMemoryLearner(config)
    weights = jnp.ones((row_values.shape[0],), dtype=jnp.float32)
    return learner._weighted_feature_loss(
        row_values, weights, jnp.asarray(label, dtype=jnp.int32)
    )


@pytest.mark.parametrize("logit_scale", _NON_POWER_OF_TWO_SCALES)
@pytest.mark.parametrize("label", range(4))
@pytest.mark.parametrize("row_index", range(len(_SHIFT_INVARIANT_ROWS)))
def test_weighted_feature_loss_is_bitwise_shift_invariant(
    row_index: int, label: int, logit_scale: float
) -> None:
    """The scaled per-feature loss must not depend on an exact +1 logit shift.

    Both losses have the same exact value because cross entropy is shift
    invariant; the operand-side centering makes them bitwise equal because the
    centered operands have identical bits before any product is formed.
    """

    row = _SHIFT_INVARIANT_ROWS[row_index]
    base = _feature_loss(row, logit_scale, label)
    shifted = _feature_loss(row + jnp.float32(1.0), logit_scale, label)
    assert np.asarray(base).tobytes() == np.asarray(shifted).tobytes()


@pytest.mark.parametrize("label", range(4))
def test_direct_cross_entropy_is_bitwise_shift_invariant(label: int) -> None:
    """Property pin for the unscaled loss on an exactly shifted row.

    Non-discriminating by itself on any measured backend (the plain form is
    already exact for an exactly representable shift), but it pins the shift
    invariance that the product-side centering inside
    ``_cross_entropy_from_logits`` must preserve.
    """

    row = _SHIFT_INVARIANT_ROWS[0][0]
    base = associative_memory_module._cross_entropy_from_logits(
        row, jnp.asarray(label, dtype=jnp.int32)
    )
    shifted = associative_memory_module._cross_entropy_from_logits(
        row + jnp.float32(1.0), jnp.asarray(label, dtype=jnp.int32)
    )
    assert np.asarray(base).tobytes() == np.asarray(shifted).tobytes()


def _two_identical_writes_utility(logit_scale: float) -> jax.Array:
    """Run the issue's two-identical-writes scenario and return the utility.

    Both rows predict the written label exactly after the first write, so
    every per-feature loss equals the aggregate loss and the exact utility
    after the second write is zero for every slot.
    """

    config = AssociativeMemoryConfig(
        vocab_size=2,
        block_size=2,
        suffix_length=2,
        feature_family="position_token",
        max_features=4,
        logit_scale=logit_scale,
        write_lr=1.0e12 / logit_scale,
    )
    learner = AssociativeMemoryLearner(config)
    state = learner.init()
    context = jnp.asarray([0, 1], dtype=jnp.int32)
    label = jnp.asarray(0, dtype=jnp.int32)
    for _ in range(2):
        result = learner.update(state, context, label)
        assert bool(jnp.all(result.update_applied))
        state = result.state
    return state.utility


@pytest.mark.parametrize("logit_scale", _NON_POWER_OF_TWO_SCALES)
def test_compiled_large_row_write_utilities_stay_exactly_zero(
    logit_scale: float,
) -> None:
    """A compiled large-row write must commit the exact float32 answer.

    Zero is the exact answer, not merely the eager one: with logits ``[L, 0]``
    and label 0 the cross entropy is ``L - L = 0``. A contracting backend that
    re-forms the product inside the fusion moves the per-feature loss and the
    clip can commit ``±8`` instead.
    """

    compiled = _two_identical_writes_utility(logit_scale)
    expected = jnp.zeros((4,), dtype=jnp.float32)
    assert np.asarray(compiled).tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize("logit_scale", _NON_POWER_OF_TWO_SCALES)
def test_compiled_and_eager_large_row_writes_agree_bitwise(
    logit_scale: float,
) -> None:
    """Compiled execution must match ``jax.disable_jit()`` bit for bit."""

    compiled = _two_identical_writes_utility(logit_scale)
    with jax.disable_jit():
        eager = _two_identical_writes_utility(logit_scale)
    assert np.asarray(compiled).tobytes() == np.asarray(eager).tobytes()
