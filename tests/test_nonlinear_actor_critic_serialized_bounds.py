"""Serialized ``hidden_sizes`` cardinality bounds for the nonlinear actor-critic configs.

``NonlinearHordeActorCriticConfig`` and ``NonlinearQHordeActorCriticConfig`` walked a
payload's ``hidden_sizes`` element by element with no ceiling on the number of layers:
each ``from_config`` copies the payload into a tuple before ``__post_init__`` re-walks
every entry and ``_nonlinear_actor_resources`` walks them a third time, so a hostile or
mistaken multi-million-entry list cost a multi-hundred-megabyte copy plus repeated
pure-Python walks inside the deserialization boundary before any rejection — or, when
every element was valid, was silently accepted as a multi-million-layer actor
configuration. Both deserialization paths now enforce the house serialized-sequence
ceiling — the 4096-item bound already merged for working-memory decay lists (#2220),
stacked-horde demons, UPGD-memory and world-model hidden sizes, and
``types._MAX_HORDE_DEMONS`` — before any element is read and before the payload is
copied, and they reject non-exact list payloads without touching their
``__len__``/``__iter__`` hooks.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from alberta_framework.core.horde_actor_critic import (
    _MAX_HIDDEN_SIZES as _NONLINEAR_ACTOR_HIDDEN_SIZES_CEILING,
)
from alberta_framework.core.horde_actor_critic import (
    NonlinearHordeActorCriticConfig,
    NonlinearQHordeActorCriticConfig,
    _validate_hidden_sizes,
)


class _HookedList(list):  # type: ignore[type-arg]
    """A ``list`` subclass whose length and iteration hooks are booby-trapped."""

    def __len__(self) -> int:  # pragma: no cover - must not run
        raise AssertionError("len hook executed")

    def __iter__(self) -> Iterator[int]:  # pragma: no cover - must not run
        raise AssertionError("iter hook executed")


def test_house_ceiling_value() -> None:
    """The module pins the same 4096-item serialized-sequence ceiling."""
    assert _NONLINEAR_ACTOR_HIDDEN_SIZES_CEILING == 4_096


def test_hidden_sizes_walker_rejects_oversized_tuple() -> None:
    """The shared walker enforces the cardinality ceiling before its element walk."""
    with pytest.raises(
        ValueError,
        match=r"^hidden_sizes length must be an integer in \[0, 4096\]$",
    ):
        _validate_hidden_sizes((64,) * (_NONLINEAR_ACTOR_HIDDEN_SIZES_CEILING + 1))


def test_hidden_sizes_at_ceiling_is_accepted() -> None:
    """The ceiling sits six orders of magnitude above the (64,) last-fit default."""
    horde = NonlinearHordeActorCriticConfig(
        n_actions=2, hidden_sizes=(64,) * _NONLINEAR_ACTOR_HIDDEN_SIZES_CEILING
    )
    q_horde = NonlinearQHordeActorCriticConfig(
        n_actions=2, hidden_sizes=(64,) * _NONLINEAR_ACTOR_HIDDEN_SIZES_CEILING
    )
    assert len(horde.hidden_sizes) == _NONLINEAR_ACTOR_HIDDEN_SIZES_CEILING
    assert len(q_horde.hidden_sizes) == _NONLINEAR_ACTOR_HIDDEN_SIZES_CEILING


def test_nonlinear_horde_config_constructor_rejects_oversized_hidden_sizes() -> None:
    with pytest.raises(
        ValueError,
        match=r"^hidden_sizes length must be an integer in \[0, 4096\]$",
    ):
        NonlinearHordeActorCriticConfig(
            n_actions=2, hidden_sizes=(64,) * (_NONLINEAR_ACTOR_HIDDEN_SIZES_CEILING + 1)
        )


def test_nonlinear_q_horde_config_constructor_rejects_oversized_hidden_sizes() -> None:
    with pytest.raises(
        ValueError,
        match=r"^hidden_sizes length must be an integer in \[0, 4096\]$",
    ):
        NonlinearQHordeActorCriticConfig(
            n_actions=2, hidden_sizes=(64,) * (_NONLINEAR_ACTOR_HIDDEN_SIZES_CEILING + 1)
        )


def test_nonlinear_horde_from_config_rejects_oversized_before_copy_and_walk() -> None:
    """The deserialization boundary supplies the ceiling before the tuple copy.

    The ``^serialized`` anchor pins the error to the pre-copy boundary check: if
    ``from_config`` stopped supplying the maximum, the payload would be copied and
    the rejection would come from the ``__post_init__`` walker instead, whose
    message does not start with ``serialized``.
    """
    payload = NonlinearHordeActorCriticConfig(n_actions=2).to_config()
    payload["hidden_sizes"] = [64] * (_NONLINEAR_ACTOR_HIDDEN_SIZES_CEILING + 1)
    with pytest.raises(
        ValueError,
        match=r"^serialized hidden_sizes length must be an integer in \[0, 4096\]$",
    ):
        NonlinearHordeActorCriticConfig.from_config(payload)


def test_nonlinear_q_horde_from_config_rejects_oversized_before_copy_and_walk() -> None:
    """The deserialization boundary supplies the ceiling before the tuple copy."""
    payload = NonlinearQHordeActorCriticConfig(n_actions=2).to_config()
    payload["hidden_sizes"] = [64] * (_NONLINEAR_ACTOR_HIDDEN_SIZES_CEILING + 1)
    with pytest.raises(
        ValueError,
        match=r"^serialized hidden_sizes length must be an integer in \[0, 4096\]$",
    ):
        NonlinearQHordeActorCriticConfig.from_config(payload)


def test_nonlinear_horde_from_config_rejects_list_subclass_without_hooks() -> None:
    """Exact-type rejection must win before any ``__len__``/``__iter__`` hook runs."""
    payload = NonlinearHordeActorCriticConfig(n_actions=2).to_config()
    payload["hidden_sizes"] = _HookedList([64, 64])
    with pytest.raises(
        ValueError, match="serialized hidden_sizes must be an exact built-in list"
    ):
        NonlinearHordeActorCriticConfig.from_config(payload)


def test_nonlinear_q_horde_from_config_rejects_list_subclass_without_hooks() -> None:
    """Exact-type rejection must win before any ``__len__``/``__iter__`` hook runs."""
    payload = NonlinearQHordeActorCriticConfig(n_actions=2).to_config()
    payload["hidden_sizes"] = _HookedList([64, 64])
    with pytest.raises(
        ValueError, match="serialized hidden_sizes must be an exact built-in list"
    ):
        NonlinearQHordeActorCriticConfig.from_config(payload)
