"""Scan sequence-length ceiling for public world-model array loops (hang guard).

``run_world_model_learning_loop`` and
``run_action_conditioned_world_model_learning_loop`` hand their caller-supplied
step arrays (``observations``, ``actions``, ``rewards``, ``next_observations``,
``discounts``) straight to ``jax.lax.scan`` with no bound on the leading (step)
axis beyond the signed-int32 working-set budget. With small
``observation_dim``/``n_heads`` that budget admits tens of millions of steps --
e.g. ``observation_dim=1`` admits ~52M steps -- so a hostile or mistaken caller
traces a scan of that length and hangs the process well before any step
executes. ``run_latent_world_model_learning_loop`` has the same exposure
through its ``_require_int32`` upper bound. This is the same hang class already
fixed for the sibling array loops (``core/sarsa.py``, ``core/horde.py``,
``core/horde_actor_critic.py``, ``core/average_reward.py``,
``core/off_policy_td.py``, ``core/dreaming.py``, ``core/behavior_model.py``,
``core/actor_critic.py``).

All three functions are exported public API (``alberta_framework.__init__``).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array

from alberta_framework.core.latent_world_model import (
    _LATENT_WORLD_MODEL_SEQUENCE_MAX_STEPS,
    LatentWorldModel,
    LatentWorldModelConfig,
    LatentWorldModelState,
    run_latent_world_model_learning_loop,
)
from alberta_framework.core.world_model import (
    _WORLD_MODEL_SEQUENCE_MAX_STEPS,
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
    ActionConditionedWorldModelState,
    OneStepWorldModel,
    WorldModelConfig,
    WorldModelState,
    run_action_conditioned_world_model_learning_loop,
    run_world_model_learning_loop,
)

_OVERSIZE_PATTERN = rf"1 <= num_steps <= {_WORLD_MODEL_SEQUENCE_MAX_STEPS}"
_LATENT_OVERSIZE_PATTERN = (
    rf"scan sequence length must be an integer in \[1, {_LATENT_WORLD_MODEL_SEQUENCE_MAX_STEPS}\]"
)


def _spy_scan(monkeypatch: pytest.MonkeyPatch, module: str) -> list[int]:
    seen: list[int] = []

    def spy(fn, init, xs, **kwargs):  # type: ignore[no-untyped-def]
        first = xs[0] if isinstance(xs, tuple) else xs
        seen.append(int(first.shape[0]))
        raise AssertionError(f"jax.lax.scan must not run: T={first.shape[0]}")

    monkeypatch.setattr(f"alberta_framework.core.{module}.jax.lax.scan", spy)
    return seen


def _one_step_model() -> tuple[OneStepWorldModel, WorldModelState]:
    model = OneStepWorldModel(WorldModelConfig(observation_dim=1, hidden_sizes=()))
    return model, model.init(jr.key(0))


def _action_conditioned_model() -> tuple[
    ActionConditionedWorldModel, ActionConditionedWorldModelState
]:
    model = ActionConditionedWorldModel(
        ActionConditionedWorldModelConfig(observation_dim=1, n_actions=2, hidden_sizes=())
    )
    return model, model.init(jr.key(0))


def _latent_model() -> tuple[LatentWorldModel, LatentWorldModelState]:
    model = LatentWorldModel(
        LatentWorldModelConfig(observation_dim=4, n_actions=2, latent_dim=4, hidden_sizes=(8,))
    )
    return model, model.init(jr.key(1))


def _one_step_arrays(
    num_steps: int, observation_dim: int = 1
) -> tuple[Array, Array, Array, Array]:
    key = jr.key(42)
    k1, k2, k3, k4 = jr.split(key, 4)
    observations = jr.normal(k1, (num_steps, observation_dim))
    actions = jnp.zeros((num_steps,), dtype=jnp.int32)
    rewards = jr.normal(k3, (num_steps,))
    next_observations = jr.normal(k4, (num_steps, observation_dim))
    return observations, actions, rewards, next_observations


def _latent_arrays(num_steps: int) -> tuple[Array, Array, Array, Array]:
    return _one_step_arrays(num_steps, 4)


# =============================================================================
# run_world_model_learning_loop
# =============================================================================


class TestRunWorldModelLearningLoopSequenceLengthGuard:
    def test_oversized_rejected_before_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _spy_scan(monkeypatch, "world_model")
        model, state = _one_step_model()
        arrays = _one_step_arrays(_WORLD_MODEL_SEQUENCE_MAX_STEPS + 1)
        with pytest.raises(ValueError, match=_OVERSIZE_PATTERN):
            run_world_model_learning_loop(model, state, *arrays)
        assert seen == []

    def test_origin_hang_class_sized_rejected_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch, "world_model")
        model, state = _one_step_model()
        arrays = _one_step_arrays(200_000)
        with pytest.raises(ValueError, match=_OVERSIZE_PATTERN):
            run_world_model_learning_loop(model, state, *arrays)
        assert seen == []

    def test_last_fit_length_still_runs(self) -> None:
        model, state = _one_step_model()
        result = run_world_model_learning_loop(model, state, *_one_step_arrays(16))
        assert result.reward_predictions.shape[0] == 16


# =============================================================================
# run_action_conditioned_world_model_learning_loop
# =============================================================================


class TestRunActionConditionedWorldModelLearningLoopSequenceLengthGuard:
    def test_oversized_rejected_before_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _spy_scan(monkeypatch, "world_model")
        model, state = _action_conditioned_model()
        arrays = _one_step_arrays(_WORLD_MODEL_SEQUENCE_MAX_STEPS + 1)
        with pytest.raises(ValueError, match=_OVERSIZE_PATTERN):
            run_action_conditioned_world_model_learning_loop(model, state, *arrays)
        assert seen == []

    def test_origin_hang_class_sized_rejected_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch, "world_model")
        model, state = _action_conditioned_model()
        arrays = _one_step_arrays(200_000)
        with pytest.raises(ValueError, match=_OVERSIZE_PATTERN):
            run_action_conditioned_world_model_learning_loop(model, state, *arrays)
        assert seen == []

    def test_last_fit_length_still_runs(self) -> None:
        model, state = _action_conditioned_model()
        result = run_action_conditioned_world_model_learning_loop(
            model, state, *_one_step_arrays(16)
        )
        assert result.prediction_errors.shape[0] == 16


# =============================================================================
# run_latent_world_model_learning_loop
# =============================================================================


class TestRunLatentWorldModelLearningLoopSequenceLengthGuard:
    def test_oversized_rejected_before_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _spy_scan(monkeypatch, "latent_world_model")
        model, state = _latent_model()
        observations, actions, rewards, next_observations = _latent_arrays(
            _LATENT_WORLD_MODEL_SEQUENCE_MAX_STEPS + 1
        )
        with pytest.raises(
            ValueError, match=_LATENT_OVERSIZE_PATTERN
        ):
            run_latent_world_model_learning_loop(
                model, state, observations, actions, rewards, next_observations
            )
        assert seen == []

    def test_origin_hang_class_sized_rejected_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch, "latent_world_model")
        model, state = _latent_model()
        observations, actions, rewards, next_observations = _latent_arrays(200_000)
        with pytest.raises(
            ValueError, match=_LATENT_OVERSIZE_PATTERN
        ):
            run_latent_world_model_learning_loop(
                model, state, observations, actions, rewards, next_observations
            )
        assert seen == []

    def test_last_fit_length_still_runs(self) -> None:
        model, state = _latent_model()
        observations, actions, rewards, next_observations = _latent_arrays(16)
        result = run_latent_world_model_learning_loop(
            model, state, observations, actions, rewards, next_observations
        )
        assert result.latent_predictions.shape[0] == 16
