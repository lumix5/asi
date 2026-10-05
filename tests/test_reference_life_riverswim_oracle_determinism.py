"""Bitwise CPU-independence contracts for the reference-life RiverSwim oracle.

The reference-life RiverSwim manifest embeds ``oracle_average_reward`` and
every downstream gate compares it bitwise: the scorecard's exact ``$.resolved``
comparison, the aggregate's canonical recomputation of the resolved
components, the execution oracle gate, and the manifest identity hashes. The
inherited exact enumeration measures each candidate kernel with
``numpy.linalg.lstsq``, whose OpenBLAS ``dgelsd`` kernel dispatch depends on
the host microarchitecture. Across the GitHub runner fleet this moved the
float64 gain by one ulp and failed the development scorecard aggregate
(SlopDotCash/asi run 34950746882: 26 of 144 shard jobs recorded
``0.8571501418872561`` while the aggregate job recorded
``0.857150141887256`` for the same scheduled protocol). These tests pin the
reference-life oracle to fixed-order float64 arithmetic that produces
identical bits under any OpenBLAS kernel dispatch.

The regressions are built to be mutation-sensitive against a revert to the
inherited solver:

- the approved oracle is pinned to its literal approved hexadecimal bit
  pattern (``_APPROVED_ORACLE_BITS``), which is also the correctly rounded
  float64 of the exact rational stationary gain, so a self-referential
  "default equals default" pass is impossible;
- the cross-dispatch probe reports the *inherited* ``lstsq`` bits next to the
  production oracle bits and requires a verified kernel dispatch that
  demonstrably changed the inherited bits (on Linux x86-64, where the
  scorecard fleet runs, absence of such a dispatch is a failure, not a skip);
  a reverted oracle would then track the inherited bits and disagree across
  that verified pair;
- a deterministic in-process guard asserts the production path never calls
  ``numpy.linalg.lstsq`` at all.
"""

from __future__ import annotations

import math
import os
import platform
import struct
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from alberta_framework.reference_life import (
    _deterministic_stationary_average_reward,
    _ImmutableRiverSwimMDP,
)
from alberta_framework.streams.closed_loop import RiverSwimConfig

pytestmark = pytest.mark.unit

# The exact scheduled development-scorecard RiverSwim protocol.
_CONFIG = RiverSwimConfig(  # type: ignore[call-arg]
    n_states=6,
    p_right_up=0.35,
    p_right_down=0.05,
    reward_left=0.005,
    reward_right=1.0,
)

# The approved ``oracle_average_reward`` bit pattern for the scheduled
# protocol: the correctly rounded float64 of the exact rational stationary
# gain of the gain-optimal all-right policy (independently asserted against
# exact rational arithmetic in
# ``test_deterministic_stationary_solve_matches_exact_arithmetic``). The two
# legacy OpenBLAS dispatches observed in run 34950746882 were
# ``3feb6dc622655c5c`` and ``3feb6dc622655c5c1``; neither matches this pin,
# so any revert to the inherited solver fails on every measured host.
_APPROVED_ORACLE_BITS = "3feb6dc622655c52"

_CHILD_PROGRAM = """
import struct

from alberta_framework.reference_life import _ImmutableRiverSwimMDP
from alberta_framework.streams.closed_loop import RiverSwimConfig, RiverSwimMDP

config = RiverSwimConfig(
    n_states=6,
    p_right_up=0.35,
    p_right_down=0.05,
    reward_left=0.005,
    reward_right=1.0,
)
legacy = RiverSwimMDP(config).optimal_average_reward()
oracle = _ImmutableRiverSwimMDP(config).optimal_average_reward()
print(
    struct.pack(">d", legacy).hex(),
    struct.pack(">d", oracle).hex(),
)
"""

# x86-64 OpenBLAS kernel dispatch targets, ordered from the universally
# executable SSE2 baseline upward. A target whose kernels need CPU features
# the host lacks dies with a nonzero exit and is dropped from the probe set,
# so the sweep degrades gracefully on older or non-x86 hosts.
_CORE_TYPES = ("Prescott", "Haswell", "SandyBridge", "Bulldozer", "SkylakeX")


def _probe(core_type: str | None) -> tuple[str, str] | None:
    """Run one child probe under one OpenBLAS kernel dispatch (or default).

    Returns ``(inherited_lstsq_bits, production_oracle_bits)`` or ``None``
    when the requested kernel dispatch could not execute on this host.
    """

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    if core_type is None:
        env.pop("OPENBLAS_CORETYPE", None)
    else:
        env["OPENBLAS_CORETYPE"] = core_type
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        check=False,
        timeout=600,
    )
    if proc.returncode != 0:
        return None
    fields = proc.stdout.split()
    if len(fields) != 2:
        return None
    return fields[0], fields[1]


def test_reference_life_oracle_matches_approved_bits() -> None:
    value = _ImmutableRiverSwimMDP(_CONFIG).optimal_average_reward()
    assert struct.pack(">d", value).hex() == _APPROVED_ORACLE_BITS


def test_reference_life_oracle_bits_survive_blas_kernel_dispatch() -> None:
    default = _probe(None)
    assert default is not None, "default-dispatch reference-life oracle probe failed"
    default_legacy, default_oracle = default
    observed_oracle = {default_oracle}
    divergent: list[tuple[str, str]] = []
    for core_type in _CORE_TYPES:
        probe = _probe(core_type)
        if probe is None:
            continue
        legacy, oracle = probe
        observed_oracle.add(oracle)
        if legacy != default_legacy:
            divergent.append((core_type, legacy))
    on_linux_x86_64 = sys.platform == "linux" and platform.machine() == "x86_64"
    if not divergent:
        if on_linux_x86_64:
            pytest.fail(
                "no alternate OpenBLAS kernel dispatch changed the inherited "
                "lstsq bits on this Linux x86-64 host, so this regression "
                "could not detect a revert to the inherited solver here"
            )
        pytest.skip(
            "no alternate OpenBLAS kernel dispatch executed with a distinct "
            "inherited result on this host (non-x86-64, non-OpenBLAS, or a "
            "build that ignores OPENBLAS_CORETYPE); the cross-dispatch "
            "bitwise contract is verified on the Linux x86-64 fleet"
        )
    assert observed_oracle == {_APPROVED_ORACLE_BITS}, (
        "reference-life oracle moved across OpenBLAS kernel dispatch "
        f"(divergent dispatches: {sorted(divergent)}): {sorted(observed_oracle)}"
    )


def test_reference_life_oracle_never_calls_inherited_lstsq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    real_lstsq = np.linalg.lstsq

    def _guard(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return real_lstsq(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(np.linalg, "lstsq", _guard)
    transitions = _ImmutableRiverSwimMDP(_CONFIG)
    assert math.isfinite(transitions.optimal_average_reward())
    assert len(transitions.optimal_policy()) == _CONFIG.n_states
    assert calls == []


def _exact_stationary_gain(kernel: np.ndarray, rewards: np.ndarray) -> Fraction:
    """Exact rational stationary gain, independent of float64 solving.

    Kernel entries are converted through ``float`` before ``Fraction``:
    NumPy float32 scalars are not ``numbers.Rational`` and are not accepted
    by ``fractions.Fraction`` on CPython < 3.14 (only a 3.14 interpreter
    routes them through ``as_integer_ratio``), so the conversion must be
    explicit to stay exact and portable across the supported 3.12+ floor.
    The ``float`` cast of a float32 scalar is exact, so the rational
    arithmetic is unchanged.
    """

    n = int(kernel.shape[0])
    normalized = [
        [
            Fraction(float(value)) / sum(Fraction(float(cell)) for cell in row)
            for value in row
        ]
        for row in kernel
    ]
    system = [
        [
            normalized[column][row]
            - (Fraction(1) if row == column else Fraction(0))
            for column in range(n)
        ]
        for row in range(n)
    ]
    system.append([Fraction(1)] * n)
    target = [Fraction(0)] * n + [Fraction(1)]
    for dropped in range(n + 1):
        rows = [index for index in range(n + 1) if index != dropped]
        matrix = [system[index][:] for index in rows]
        rhs = [target[index] for index in rows]
        solution = _fraction_solve(matrix, rhs, n)
        if solution is not None:
            total = sum(solution, Fraction(0))
            return sum(
                (solution[index] * Fraction(float(rewards[index])) for index in range(n)),
                Fraction(0),
            ) / total
    raise AssertionError("exact stationary system never became nonsingular")


def _fraction_solve(
    matrix: list[list[Fraction]], rhs: list[Fraction], n: int
) -> list[Fraction] | None:
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(matrix[row][column]))
        if matrix[pivot][column] == 0:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        rhs[column], rhs[pivot] = rhs[pivot], rhs[column]
        for row in range(column + 1, n):
            factor = matrix[row][column] / matrix[column][column]
            if factor != 0:
                for deeper in range(column, n):
                    matrix[row][deeper] -= factor * matrix[column][deeper]
                rhs[row] -= factor * rhs[column]
    solution = [Fraction(0)] * n
    for row in range(n - 1, -1, -1):
        tail = sum(
            (matrix[row][deeper] * solution[deeper] for deeper in range(row + 1, n)),
            Fraction(0),
        )
        solution[row] = (rhs[row] - tail) / matrix[row][row]
    return solution


def test_deterministic_stationary_solve_matches_exact_arithmetic() -> None:
    kernel = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.11, 0.09, 0.8, 0.0],
            [0.0, 0.21, 0.19, 0.6],
            [0.0, 0.0, 0.23, 0.77],
        ],
        dtype=np.float32,
    )
    rewards = np.array([0.0, 0.01, 0.0, 1.0], dtype=np.float32)
    gain = _deterministic_stationary_average_reward(kernel, rewards)
    exact = _exact_stationary_gain(kernel, rewards)
    assert math.isfinite(gain)
    # Fixed-order elimination is exactly rounded per operation but is not a
    # correctly rounded solver in general; two float64 ulps bound the
    # measured agreement with the exact rational gain. The former 1e-12
    # relative tolerance was also satisfied by the inherited lstsq solver and
    # therefore detected nothing.
    exact_as_float = float(exact)
    assert abs(gain - exact_as_float) <= 2.0 * math.ulp(exact_as_float)
    # The scheduled reference protocol must equal the exact rational gain
    # bit for bit, which is exactly the approved manifest pin above.
    transitions = _ImmutableRiverSwimMDP(_CONFIG)
    scheduled_gain = transitions.optimal_average_reward()
    scheduled_exact = _exact_stationary_gain(
        np.asarray(transitions.transition_tensor[1], dtype=np.float32),
        np.asarray(transitions.reward_tensor[:, 1], dtype=np.float32),
    )
    assert struct.pack(">d", scheduled_gain).hex() == _APPROVED_ORACLE_BITS
    assert scheduled_gain == float(scheduled_exact)
