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
"""

from __future__ import annotations

import math
import os
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

_CHILD_PROGRAM = """
import struct

from alberta_framework.reference_life import _ImmutableRiverSwimMDP
from alberta_framework.streams.closed_loop import RiverSwimConfig

config = RiverSwimConfig(
    n_states=6,
    p_right_up=0.35,
    p_right_down=0.05,
    reward_left=0.005,
    reward_right=1.0,
)
value = _ImmutableRiverSwimMDP(config).optimal_average_reward()
print(struct.pack(">d", value).hex())
"""

# x86-64 OpenBLAS kernel dispatch targets, ordered from the universally
# executable SSE2 baseline upward. A target whose kernels need CPU features
# the host lacks dies with a nonzero exit and is skipped, so the probe set
# degrades gracefully on older or non-x86 hosts.
_CORE_TYPES = ("Prescott", "Haswell", "SandyBridge", "Bulldozer", "SkylakeX")


def _oracle_bits(core_type: str | None) -> str:
    """Run the child probe under one OpenBLAS kernel dispatch (or default)."""

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
        return ""
    return proc.stdout.strip()


def test_reference_life_oracle_bits_survive_blas_kernel_dispatch() -> None:
    default_bits = _oracle_bits(None)
    assert default_bits, "default-dispatch reference-life oracle probe failed"
    observed = {default_bits}
    successful_probes = 1
    for core_type in _CORE_TYPES:
        bits = _oracle_bits(core_type)
        if bits:
            observed.add(bits)
            successful_probes += 1
    if successful_probes < 2:
        pytest.skip(
            "this host executed only one OpenBLAS kernel dispatch; the "
            "cross-kernel bitwise contract is untestable here"
        )
    assert observed == {default_bits}, (
        "reference-life oracle moved across OpenBLAS kernel dispatch: "
        f"{sorted(observed)}"
    )


def _exact_stationary_gain(kernel: np.ndarray, rewards: np.ndarray) -> float:
    """Exact rational stationary gain, independent of float64 solving."""

    n = int(kernel.shape[0])
    normalized = [
        [Fraction(value) / sum(Fraction(value) for value in row) for value in row]
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
            return float(
                sum(
                    (solution[index] * Fraction(float(rewards[index])) for index in range(n)),
                    Fraction(0),
                )
                / total
            )
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
    assert abs(gain - exact) <= 1e-12 * max(1.0, abs(exact))
    # The scheduled reference protocol must also agree with exact arithmetic.
    transitions = _ImmutableRiverSwimMDP(_CONFIG)
    scheduled_gain = transitions.optimal_average_reward()
    scheduled_exact = _exact_stationary_gain(
        np.asarray(transitions.transition_tensor[1], dtype=np.float32),
        np.asarray(transitions.reward_tensor[:, 1], dtype=np.float32),
    )
    assert abs(scheduled_gain - scheduled_exact) <= 1e-12 * max(1.0, abs(scheduled_exact))
