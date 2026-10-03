"""Serialized ``hidden_sizes`` cardinality bounds for the UPGD memory learner.

``UPGDMemoryConfig`` validation and ``UPGDMemoryConfig.from_config`` walked a
payload's ``hidden_sizes`` element by element (``from_config`` also copies it
into a tuple) with no ceiling on the number of layers, so a hostile or
mistaken multi-million-entry list cost minutes of pure-Python walking and a
multi-gigabyte copy inside the deserialization boundary before any rejection.
The config now enforces the house serialized-sequence ceiling — the 4096-item
bound already merged for working-memory decay lists (#2220), stacked-horde
demons, and ``types._MAX_HORDE_DEMONS`` — before any element is read, and it
rejects non-exact list/tuple payloads without touching their
``__len__``/``__iter__`` hooks.
"""

from __future__ import annotations

from typing import Any

import pytest

from alberta_framework.core.upgd_memory import (
    _MAX_HIDDEN_SIZES as _MAX_UPGD_MEMORY_HIDDEN_SIZES,
)
from alberta_framework.core.upgd_memory import (
    UPGDMemoryConfig,
    UPGDMemoryLearner,
)


class _HostileHiddenSizes(list):
    """A ``list`` subclass whose length and iteration hooks are booby-trapped."""

    def __len__(self) -> int:
        raise RuntimeError("len hook must not be reached")

    def __iter__(self):  # type: ignore[override]
        raise RuntimeError("iter hook must not be reached")


def test_house_ceiling_value() -> None:
    """The module pins the same 4096-item serialized-sequence ceiling."""
    assert _MAX_UPGD_MEMORY_HIDDEN_SIZES == 4_096


# -----------------------------------------------------------------------------
# UPGDMemoryConfig constructor
# -----------------------------------------------------------------------------


def test_config_constructor_rejects_oversized_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        UPGDMemoryConfig(feature_dim=4, n_heads=2, hidden_sizes=(1,) * 4_097)


def test_config_constructor_accepts_boundary_hidden_sizes() -> None:
    config = UPGDMemoryConfig(feature_dim=4, n_heads=2, hidden_sizes=(1,) * 4_096)
    assert config.hidden_sizes == (1,) * 4_096


def test_config_constructor_length_preflight_precedes_element_walk() -> None:
    """An oversized payload is rejected by length even when elements are invalid.

    The first element is deliberately not an integer: if the per-element
    ``_require_int`` walk ran before the length preflight, the error would be
    about the element type instead of the length.
    """
    oversized: tuple[Any, ...] = ("not-an-int", *(1,) * 4_096)
    with pytest.raises(ValueError, match="hidden_sizes length"):
        UPGDMemoryConfig(feature_dim=4, n_heads=2, hidden_sizes=oversized)


def test_config_constructor_rejects_hostile_list_subclass_without_hooks() -> None:
    with pytest.raises(TypeError, match="hidden_sizes must be an actual tuple"):
        UPGDMemoryConfig(
            feature_dim=4,
            n_heads=2,
            hidden_sizes=_HostileHiddenSizes([1]),  # type: ignore[arg-type]
        )


# -----------------------------------------------------------------------------
# UPGDMemoryConfig.from_config
# -----------------------------------------------------------------------------


def _config_payload(hidden_sizes: object) -> dict[str, object]:
    return {
        "feature_dim": 4,
        "n_heads": 2,
        "hidden_sizes": hidden_sizes,
    }


def test_from_config_rejects_oversized_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        UPGDMemoryConfig.from_config(_config_payload([1] * 4_097))


def test_from_config_length_preflight_precedes_tuple_copy() -> None:
    """The ceiling fires before the payload list is copied into a tuple.

    An invalid first element again discriminates: copying the list succeeds
    silently, so only a true length preflight can produce the length error.
    """
    oversized: list[Any] = ["not-an-int", *([1] * 4_096)]
    with pytest.raises(ValueError, match="hidden_sizes length"):
        UPGDMemoryConfig.from_config(_config_payload(oversized))


def test_from_config_rejects_hostile_list_subclass_without_hooks() -> None:
    with pytest.raises(ValueError, match="hidden_sizes must be a list or tuple"):
        UPGDMemoryConfig.from_config(_config_payload(_HostileHiddenSizes([1])))


def test_from_config_roundtrip_preserves_small_hidden_sizes() -> None:
    config = UPGDMemoryConfig(feature_dim=4, n_heads=2, hidden_sizes=(8, 16))
    restored = UPGDMemoryConfig.from_config(config.to_config())
    assert restored == config


# -----------------------------------------------------------------------------
# UPGDMemoryLearner.from_config
# -----------------------------------------------------------------------------


def _learner_payload(hidden_sizes: object) -> dict[str, object]:
    config = UPGDMemoryLearner(
        UPGDMemoryConfig(feature_dim=4, n_heads=2, hidden_sizes=(1,))
    ).to_config()
    inner = config["config"]
    assert isinstance(inner, dict)
    inner["hidden_sizes"] = hidden_sizes
    return config


def test_learner_from_config_rejects_oversized_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        UPGDMemoryLearner.from_config(_learner_payload([1] * 4_097))
