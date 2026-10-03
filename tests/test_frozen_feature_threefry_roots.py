"""Frozen-extractor RNG roots must not depend on the ambient PRNG implementation.

The RanDumb/RanPAC/PROL arms' frozen extractors (``_key_from_params``) and the
registered ``rff_rls`` arm's frozen ``Omega``/``b`` draw both derive their
feature maps from model-parameter bits folded into a fixed domain key, and
both describe the result as frozen: a recorded (arm, seed, hyperparameters,
source) identity is supposed to pin the measured feature map. Creating the
domain root with ambient ``jax.random.key(...)`` instead makes the same
parameter bytes produce a different feature map whenever
``jax.config.jax_default_prng_impl`` changes, silently breaking that promise.

Both roots are bound to explicit ``threefry2x32``: byte-identical under the
default ambient implementation, environment-independent otherwise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

from alberta_framework.benchmarks.ipmnist_screening import screening_spec
from alberta_framework.benchmarks.replay_frozen_ipmnist import _key_from_params
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    init_mlp_params,
)

pytestmark = pytest.mark.unit

_SMALL = IPMNISTConfig(
    n_tasks=3, task_length=30, input_dim=12, hidden1=8, hidden2=6, n_classes=5
)

_RANPAC_DOMAIN = 0x52504143


def _projection_sample(params: dict[str, Array]) -> Array:
    return jr.normal(_key_from_params(params, _RANPAC_DOMAIN), (16,), jnp.float32)


def _rff_omega(params: dict[str, Array]) -> Array:
    init_fn, _ = screening_spec("rff_rls").factory(screening_spec("rff_rls").hyperparameters)
    return jax.device_get(init_fn(params).omega)


def _with_ambient_prng_impl(impl: str, sample: object) -> Array:
    """Sample under a temporarily non-default ambient PRNG implementation."""
    previous = str(jax.config.jax_default_prng_impl)
    jax.config.update("jax_default_prng_impl", impl)
    try:
        return jax.device_get(sample())
    finally:
        jax.config.update("jax_default_prng_impl", previous)


def test_frozen_feature_root_is_ambient_impl_independent() -> None:
    params = init_mlp_params(jr.key(7), _SMALL)
    default = _with_ambient_prng_impl("threefry2x32", lambda: _projection_sample(params))
    rbg = _with_ambient_prng_impl("rbg", lambda: _projection_sample(params))
    assert isinstance(default, np.ndarray) and isinstance(rbg, np.ndarray)
    np.testing.assert_array_equal(default, rbg)


def test_rff_rls_frozen_omega_is_ambient_impl_independent() -> None:
    params = init_mlp_params(jr.key(7), _SMALL)
    default = _with_ambient_prng_impl("threefry2x32", lambda: _rff_omega(params))
    rbg = _with_ambient_prng_impl("rbg", lambda: _rff_omega(params))
    assert isinstance(default, np.ndarray) and isinstance(rbg, np.ndarray)
    np.testing.assert_array_equal(default, rbg)


def test_frozen_feature_root_is_unchanged_under_default_ambient() -> None:
    """The explicit-impl root stays byte-identical to the historical default."""
    params = init_mlp_params(jr.key(7), _SMALL)
    bits = jax.lax.bitcast_convert_type(params["w1"].reshape(-1), jnp.uint32)
    historical = jr.fold_in(
        jr.fold_in(
            jr.key(jnp.uint32(_RANPAC_DOMAIN), impl="threefry2x32"), bits[0]
        ),
        bits[-1],
    )
    pinned = _key_from_params(params, _RANPAC_DOMAIN)
    np.testing.assert_array_equal(
        jax.device_get(jr.key_data(historical)), jax.device_get(jr.key_data(pinned))
    )
