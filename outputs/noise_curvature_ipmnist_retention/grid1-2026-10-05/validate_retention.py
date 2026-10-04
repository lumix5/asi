#!/usr/bin/env python3
"""Retained validator for the noise-curvature IPMNIST retention grid1-2026-10-05.

Re-validates every retained shard byte-for-byte through the lane's own strict
receipt machinery, rebuilds and validates one nonpromoting development receipt
per (arm, seed), validates one matched four-arm panel per seed, computes the
declared paired decision rule against the mechanism-off arm, and writes
``analysis_summary.json``.

Usage (from the repository root, project venv)::

    .venv/bin/python outputs/noise_curvature_ipmnist_retention/grid1-2026-10-05/validate_retention.py

Exit 0 requires every check to pass.  This driver is development-only and
permanently nonpromoting; it retains outcomes, it does not promote anything.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

GRID = Path(__file__).resolve().parent
REPO = GRID.parents[2]
SHARDS = GRID / "shards"
RECEIPTS = GRID / "receipts"

ARMS = (
    "noise_curvature_fixed_adam_l2",
    "noise_curvature_gradient_only",
    "noise_curvature_volatility_only",
    "noise_curvature_combined",
)
CONTROL = "noise_curvature_fixed_adam_l2"
SEEDS = (0, 1, 2, 3, 4)
# Two-sided Student-t 95% critical value, df = 4.
T_CRIT_DF4 = 2.7764451051977987

from alberta_framework.benchmarks.ipmnist_screening import (  # noqa: E402
    IPMNISTConfig,
    ScreeningRunResult,
    screening_spec,
)
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (  # noqa: E402
    validate_matched_noise_curvature_results,
    validate_noise_curvature_development_result,
)
from alberta_framework.benchmarks.ipmnist_screening import (  # noqa: E402
    noise_curvature_development_result_payload,
)


def paired_t_interval(deltas: list[float]) -> tuple[float, float]:
    """Two-sided paired t interval at 95%, df = n - 1 = 4."""
    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    half = T_CRIT_DF4 * math.sqrt(var / n)
    return mean - half, mean + half


def decide(deltas: list[float]) -> str:
    """Declared rule: supported needs every pair positive and the interval above
    zero; rejected needs every pair negative and the interval below zero."""
    low, high = paired_t_interval(deltas)
    if all(d > 0.0 for d in deltas) and low > 0.0:
        return "supported"
    if all(d < 0.0 for d in deltas) and high < 0.0:
        return "rejected"
    return "inconclusive"


def main() -> int:
    import alberta_framework

    module_root = Path(alberta_framework.__file__).resolve().parents[1]
    if module_root != REPO:
        print(f"FAIL: resolved module root {module_root} != repository root {REPO}")
        return 2

    shards: dict[tuple[str, int], dict] = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = SHARDS / f"shard-{arm}-seed{seed}.json"
            shards[(arm, seed)] = json.loads(path.read_bytes())

    # Rebuild each shard into a ScreeningRunResult and a strict receipt.
    receipts: dict[tuple[str, int], dict] = {}
    outcomes: dict[str, str] = {}
    curves: dict[str, list[float]] = {}
    for arm in ARMS:
        control_acc = [shards[(CONTROL, s)]["per_task_accuracy"] for s in SEEDS]
        arm_acc = [shards[(arm, s)]["per_task_accuracy"] for s in SEEDS]
        deltas = [
            float(np.mean(a) - np.mean(c)) for a, c in zip(arm_acc, control_acc)
        ]
        # The mechanism-off arm is the comparator, not a candidate: its receipt
        # carries the neutral outcome; only candidates get the paired rule.
        outcome = "inconclusive" if arm == CONTROL else decide(deltas)
        outcomes[arm] = outcome
        curves[arm] = deltas

    for (arm, seed), shard in sorted(shards.items()):
        spec = screening_spec(arm)
        result = ScreeningRunResult(
            config_name=shard["config_name"],
            base_learner=shard["base_learner"],
            hyperparameters=dict(shard["hyperparameters"]),
            seed=shard["seed"],
            config=IPMNISTConfig(
                n_tasks=shard["config"]["n_tasks"],
                task_length=shard["config"]["task_length"],
            ),
            per_task_accuracy=np.asarray(shard["per_task_accuracy"], dtype=np.float64),
            per_task_loss=np.asarray(shard["per_task_loss"], dtype=np.float64),
            per_task_plasticity=np.asarray(shard["per_task_plasticity"], dtype=np.float64),
            wall_clock_seconds=shard["wall_clock_seconds"],
            noise_mode=shard["noise_mode"],
            noise_pool_steps=shard["noise_pool_steps"],
        )
        if result.config_name != arm or result.seed != seed:
            print(f"FAIL: shard identity mismatch {arm} seed {seed}")
            return 2
        if result.hyperparameters != dict(spec.hyperparameters):
            print(f"FAIL: hyperparameter drift {arm}")
            return 2
        payload = noise_curvature_development_result_payload(
            result, outcome=outcomes[arm]
        )
        validate_noise_curvature_development_result(payload)
        receipts[(arm, seed)] = payload

    # One matched four-arm panel per seed, through the lane's own validator.
    for seed in SEEDS:
        panel = [receipts[(arm, seed)] for arm in ARMS]
        validate_matched_noise_curvature_results(panel)

    # Bitwise combined == gradient_only check recorded by this grid.
    identical = []
    for seed in SEEDS:
        g = shards[("noise_curvature_gradient_only", seed)]
        c = shards[("noise_curvature_combined", seed)]
        identical.append(
            g["per_task_accuracy"] == c["per_task_accuracy"]
            and g["per_task_loss"] == c["per_task_loss"]
            and g["per_task_plasticity"] == c["per_task_plasticity"]
        )

    RECEIPTS.mkdir(exist_ok=True)
    for (arm, seed), payload in sorted(receipts.items()):
        (RECEIPTS / f"receipt-{arm}-seed{seed}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )

    summary = {
        "grid": "grid1-2026-10-05",
        "repository_head": shards[(ARMS[0], 0)]["source_provenance"]["git_commit"],
        "cell": {"n_tasks": 10, "task_length": 5000, "observations_per_run": 50_000},
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "control": CONTROL,
        "decision_rule": (
            "paired per-seed mean-online-accuracy delta vs "
            "noise_curvature_fixed_adam_l2; supported = all five deltas > 0 and "
            "two-sided paired-t 95% interval (df=4) entirely > 0; rejected = all "
            "five deltas < 0 and interval entirely < 0; otherwise inconclusive"
        ),
        "paired_mean_online_accuracy": {
            arm: [float(np.mean(shards[(arm, s)]["per_task_accuracy"])) for s in SEEDS]
            for arm in ARMS
        },
        "paired_deltas_vs_control": {
            arm: curves[arm] for arm in ARMS if arm != CONTROL
        },
        "paired_t95_intervals_vs_control": {
            arm: list(paired_t_interval(curves[arm]))
            for arm in ARMS
            if arm != CONTROL
        },
        "outcomes": outcomes,
        "combined_bitwise_equals_gradient_only_per_seed": identical,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
    (GRID / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(f"PASS: {len(receipts)} receipts validated, 5 matched panels validated")
    for arm in ARMS:
        mean_delta = (
            "control" if arm == CONTROL else f"{np.mean(curves[arm]):+.6f}"
        )
        print(f"  {arm}: outcome={outcomes[arm]} paired_mean_delta={mean_delta}")
    print(f"  combined == gradient_only bitwise per seed: {all(identical)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
