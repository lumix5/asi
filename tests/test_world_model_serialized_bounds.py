"""Serialized ``hidden_sizes`` cardinality bounds for the world-model configs.

``ActionConditionedWorldModelConfig`` and ``WorldModelConfig`` walked a payload's
``hidden_sizes`` element by element (``from_config`` also copies it into a tuple
via ``_serialized_sequence``) with no ceiling on the number of layers, so a
hostile or mistaken multi-million-entry list cost seconds-to-minutes of
pure-Python walking and a multi-hundred-megabyte copy inside the
deserialization boundary before any rejection — or, when every element was
valid, was silently accepted as a multi-million-layer MLP configuration. The
configs now enforce the house serialized-sequence ceiling — the 4096-item bound
already merged for working-memory decay lists (#2220), stacked-horde demons,
UPGD-memory hidden sizes, and ``types._MAX_HORDE_DEMONS`` — before any element
is read and before the payload is copied, and they reject non-exact list/tuple
payloads without touching their ``__len__``/``__iter__`` hooks.
"""

from __future__ import annotations

from typing import Any

import pytest

from alberta_framework.core.world_model import (
    _MAX_HIDDEN_SIZES as _WORLD_MODEL_HIDDEN_SIZES_CEILING,
)
from alberta_framework.core.world_model import (
    ActionConditionedWorldModelConfig,
    WorldModelConfig,
)


class _HostileHiddenSizes(list):
    """A ``list`` subclass whose length and iteration hooks are booby-trapped."""

    def __len__(self) -> int:
        raise RuntimeError("len hook must not be reached")

    def __iter__(self):  # type: ignore[override]
        raise RuntimeError("iter hook must not be reached")


def test_house_ceiling_value() -> None:
    """The module pins the same 4096-item serialized-sequence ceiling."""
    assert _WORLD_MODEL_HIDDEN_SIZES_CEILING == 4_096


# -----------------------------------------------------------------------------
# ActionConditionedWorldModelConfig constructor
# -----------------------------------------------------------------------------


def test_action_config_constructor_rejects_oversized_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        ActionConditionedWorldModelConfig(
            observation_dim=4, n_actions=2, hidden_sizes=(1,) * 4_097
        )


def test_action_config_constructor_accepts_boundary_hidden_sizes() -> None:
    config = ActionConditionedWorldModelConfig(
        observation_dim=4, n_actions=2, hidden_sizes=(1,) * 4_096
    )
    assert config.hidden_sizes == (1,) * 4_096


def test_action_config_constructor_length_preflight_precedes_element_walk() -> None:
    """An oversized payload is rejected by length even when elements are invalid.

    The first element is deliberately not an integer: if the per-element
    ``_require_int32`` walk ran before the length preflight, the error would be
    about the element type instead of the length.
    """
    oversized: tuple[Any, ...] = ("not-an-int", *(1,) * 4_096)
    with pytest.raises(ValueError, match="hidden_sizes length"):
        ActionConditionedWorldModelConfig(
            observation_dim=4, n_actions=2, hidden_sizes=oversized
        )


def test_action_config_constructor_rejects_hostile_list_subclass() -> None:
    with pytest.raises(ValueError, match="hidden_sizes must be an actual tuple"):
        ActionConditionedWorldModelConfig(
            observation_dim=4,
            n_actions=2,
            hidden_sizes=_HostileHiddenSizes([1]),  # type: ignore[arg-type]
        )


# -----------------------------------------------------------------------------
# ActionConditionedWorldModelConfig.from_config
# -----------------------------------------------------------------------------


def _action_payload(hidden_sizes: object) -> dict[str, object]:
    return {
        "type": "ActionConditionedWorldModelConfig",
        "observation_dim": 4,
        "n_actions": 2,
        "hidden_sizes": hidden_sizes,
    }


def test_action_from_config_rejects_oversized_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        ActionConditionedWorldModelConfig.from_config(_action_payload([1] * 4_097))


def test_action_from_config_length_preflight_precedes_element_walk() -> None:
    """The ceiling fires before the payload is copied and its elements are walked.

    An invalid first element again discriminates: without the length preflight
    the payload is silently copied into a tuple and the constructor fails later
    with an element-type error instead of the length error.
    """
    oversized: list[Any] = ["not-an-int", *([1] * 4_096)]
    with pytest.raises(ValueError, match="hidden_sizes length"):
        ActionConditionedWorldModelConfig.from_config(_action_payload(oversized))


def test_action_from_config_rejects_hostile_list_subclass() -> None:
    with pytest.raises(ValueError, match="must be an actual list or tuple"):
        ActionConditionedWorldModelConfig.from_config(
            _action_payload(_HostileHiddenSizes([1]))
        )


def test_action_from_config_roundtrip_preserves_small_hidden_sizes() -> None:
    config = ActionConditionedWorldModelConfig(
        observation_dim=4, n_actions=2, hidden_sizes=(8, 16)
    )
    restored = ActionConditionedWorldModelConfig.from_config(config.to_config())
    assert restored == config


# -----------------------------------------------------------------------------
# WorldModelConfig constructor
# -----------------------------------------------------------------------------


def test_world_model_config_constructor_rejects_oversized_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        WorldModelConfig(observation_dim=4, hidden_sizes=(1,) * 4_097)


def test_world_model_config_constructor_accepts_boundary_hidden_sizes() -> None:
    config = WorldModelConfig(observation_dim=4, hidden_sizes=(1,) * 4_096)
    assert config.hidden_sizes == (1,) * 4_096


def test_world_model_config_constructor_length_preflight_precedes_element_walk() -> None:
    oversized: tuple[Any, ...] = ("not-an-int", *(1,) * 4_096)
    with pytest.raises(ValueError, match="hidden_sizes length"):
        WorldModelConfig(observation_dim=4, hidden_sizes=oversized)


def test_world_model_config_constructor_rejects_hostile_list_subclass() -> None:
    with pytest.raises(ValueError, match="hidden_sizes must be an actual tuple"):
        WorldModelConfig(
            observation_dim=4,
            hidden_sizes=_HostileHiddenSizes([1]),  # type: ignore[arg-type]
        )


# -----------------------------------------------------------------------------
# WorldModelConfig.from_config
# -----------------------------------------------------------------------------


def _world_model_payload(hidden_sizes: object) -> dict[str, object]:
    return {
        "type": "WorldModelConfig",
        "observation_dim": 4,
        "hidden_sizes": hidden_sizes,
    }


def test_world_model_from_config_rejects_oversized_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        WorldModelConfig.from_config(_world_model_payload([1] * 4_097))


def test_world_model_from_config_length_preflight_precedes_element_walk() -> None:
    oversized: list[Any] = ["not-an-int", *([1] * 4_096)]
    with pytest.raises(ValueError, match="hidden_sizes length"):
        WorldModelConfig.from_config(_world_model_payload(oversized))


def test_world_model_from_config_rejects_hostile_list_subclass() -> None:
    with pytest.raises(ValueError, match="must be an actual list or tuple"):
        WorldModelConfig.from_config(_world_model_payload(_HostileHiddenSizes([1])))


def test_world_model_from_config_roundtrip_preserves_small_hidden_sizes() -> None:
    config = WorldModelConfig(observation_dim=4, hidden_sizes=(8, 16))
    restored = WorldModelConfig.from_config(config.to_config())
    assert restored == config
