"""Serialized ``hidden_sizes`` cardinality bounds for UPGD and the multi-head MLP.

``UPGDLearner.__init__``, ``UPGDLearner.from_config``, and
``MultiHeadMLPLearner.from_config`` walked a payload's ``hidden_sizes`` element
by element (``from_config`` also copies it into a tuple) with no ceiling on the
number of layers, so a hostile or mistaken multi-million-entry list cost
minutes of pure-Python walking and multi-gigabyte copies inside the
deserialization boundary before any rejection.  Both modules now enforce the
house serialized-sequence ceiling — the 4096-item bound already merged for
working-memory decay lists (#2220), stacked-horde demons, and
``types._MAX_HORDE_DEMONS`` — before any element is read, and they reject list
subclasses without touching their ``__len__``/``__iter__`` hooks.
"""

from __future__ import annotations

import pytest

from alberta_framework.core.multi_head_learner import (
    _MAX_HIDDEN_SIZES as _MAX_MULTI_HEAD_HIDDEN_SIZES,
)
from alberta_framework.core.multi_head_learner import (
    MultiHeadMLPLearner,
)
from alberta_framework.core.upgd import (
    _MAX_HIDDEN_SIZES as _MAX_UPGD_HIDDEN_SIZES,
)
from alberta_framework.core.upgd import (
    UPGDLearner,
)


class _HostileHiddenSizes(list):
    """A ``list`` subclass whose length and iteration hooks are booby-trapped."""

    def __len__(self) -> int:
        raise RuntimeError("len hook must not be reached")

    def __iter__(self):  # type: ignore[override]
        raise RuntimeError("iter hook must not be reached")


def test_house_ceiling_value() -> None:
    """Both modules pin the same 4096-item serialized-sequence ceiling."""
    assert _MAX_UPGD_HIDDEN_SIZES == 4_096
    assert _MAX_MULTI_HEAD_HIDDEN_SIZES == 4_096


# -----------------------------------------------------------------------------
# UPGDLearner constructor
# -----------------------------------------------------------------------------


def test_upgd_constructor_rejects_oversized_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        UPGDLearner(n_heads=2, hidden_sizes=(1,) * 4_097, step_size=0.01)


def test_upgd_constructor_accepts_boundary_hidden_sizes() -> None:
    learner = UPGDLearner(n_heads=2, hidden_sizes=(1,) * 4_096, step_size=0.01)
    assert learner._hidden_sizes == (1,) * 4_096


def test_upgd_constructor_rejects_hostile_list_subclass_without_hooks() -> None:
    with pytest.raises(ValueError, match="hidden_sizes must be a tuple of integers"):
        UPGDLearner(
            n_heads=2,
            hidden_sizes=_HostileHiddenSizes([1]),  # type: ignore[arg-type]
            step_size=0.01,
        )


# -----------------------------------------------------------------------------
# UPGDLearner.from_config
# -----------------------------------------------------------------------------


def _upgd_payload(hidden_sizes: object) -> dict[str, object]:
    return {
        "type": "UPGDLearner",
        "hidden_sizes": hidden_sizes,
        "n_heads": 2,
        "step_size": 0.01,
    }


def test_upgd_from_config_rejects_oversized_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        UPGDLearner.from_config(_upgd_payload([1] * 4_097))


def test_upgd_from_config_accepts_boundary_hidden_sizes() -> None:
    learner = UPGDLearner.from_config(_upgd_payload([1] * 4_096))
    assert isinstance(learner, UPGDLearner)


def test_upgd_from_config_rejects_hostile_list_subclass_without_hooks() -> None:
    with pytest.raises(ValueError, match="hidden_sizes must be a tuple of integers"):
        UPGDLearner.from_config(_upgd_payload(_HostileHiddenSizes([1])))


# -----------------------------------------------------------------------------
# MultiHeadMLPLearner
# -----------------------------------------------------------------------------


def test_multi_head_constructor_rejects_oversized_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        MultiHeadMLPLearner(n_heads=2, hidden_sizes=(1,) * 4_097)


def test_multi_head_constructor_accepts_boundary_hidden_sizes() -> None:
    MultiHeadMLPLearner(n_heads=2, hidden_sizes=(1,) * 4_096)


def _multi_head_payload(hidden_sizes: object) -> dict[str, object]:
    config = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(1,)).to_config()
    config["hidden_sizes"] = hidden_sizes
    return config


def test_multi_head_from_config_rejects_oversized_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        MultiHeadMLPLearner.from_config(_multi_head_payload([1] * 4_097))


def test_multi_head_from_config_accepts_boundary_hidden_sizes() -> None:
    learner = MultiHeadMLPLearner.from_config(_multi_head_payload([1] * 4_096))
    assert isinstance(learner, MultiHeadMLPLearner)


def test_multi_head_from_config_rejects_hostile_list_subclass_without_hooks() -> None:
    with pytest.raises(ValueError, match="hidden_sizes must be a list"):
        MultiHeadMLPLearner.from_config(_multi_head_payload(_HostileHiddenSizes([1])))
