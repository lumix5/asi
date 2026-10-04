"""Re-validate the retained TeLAPA qualification grid and re-derive the analysis.

Run from the repository root:

    .venv/bin/python outputs/telapa_qualification_retention/grid1-2026-10-04/validate_retention.py

The script is retained beside the receipts so the pass is reproducible from the
committed bytes alone. It re-validates every receipt with the lane's own strict
validator (which additionally re-executes and replays every arm/seed record
from the bound configuration), checks that the six receipts cover exactly the
announced {steps} x {phase_length} grid at the frozen seeds and otherwise
default hyperparameters, binds one shared lane/runtime identity across the
grid, re-asserts the mechanism-off parity and matched resource axes, and
re-derives the paired reward deltas in SUMMARY.md. It writes nothing except the
analysis summary passed via --output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from alberta_framework.benchmarks.telapa_qualification import (
    FROZEN_DEVELOPMENT_SEEDS,
    SCHEMA,
    validate_result,
)

RECEIPTS = {
    "receipt-run1.json": (32, 4),
    "receipt-run2.json": (32, 8),
    "receipt-run3.json": (32, 16),
    "receipt-run4.json": (64, 4),
    "receipt-run5.json": (64, 8),
    "receipt-run6.json": (64, 16),
}
ARCHIVE_ARMS = ("diverse_archive", "one_model")
CONTROL_ARM = "fixed_snapshot"
REDUCTION_ARM = "mechanism_off"
ARMS = ("diverse_archive", "one_model", "fixed_snapshot", "mechanism_off")
DEFAULT_HYPERPARAMETERS = {
    "archive_byte_budget": 4096,
    "min_latent_distance": 0.05,
    "learning_rate": 0.125,
}
# The merged #2257 retention recorded, at the smoke's default (32, 4) config,
# per-seed reward sums of diverse_archive=13, one_model=12, fixed_snapshot=15,
# mechanism_off=15. This grid's default-config receipt must reproduce them.
HISTORICAL_DEFAULT_REWARD_SUMS = {
    "diverse_archive": 13,
    "one_model": 12,
    "fixed_snapshot": 15,
    "mechanism_off": 15,
}
PARITY_FIELDS = (
    "observation_sha256",
    "action_sha256",
    "reward_sha256",
    "initial_policy_sha256",
    "final_policy_sha256",
)
MATCHED_RESOURCE_FIELDS = (
    "environment_steps",
    "observations_consumed",
    "policy_updates",
    "policy_queries",
    "descriptor_model_queries",
    "task_boundary_disclosures",
)
IDENTITY_SOURCE_FIELDS = (
    "lane_source_sha256",
    "dependency_source_sha256",
    "runtime_identity",
    "dependency_versions",
    "paper_registry_sha256",
)
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)
    print(("PASS " if condition else "FAIL ") + message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    receipts = {
        name: json.loads((here / "receipts" / name).read_text())
        for name in RECEIPTS
    }

    # 1. The lane's own strict validator accepts every receipt. validate_result
    #    also re-executes every record from the bound configuration, so each
    #    acceptance doubles as an in-process replay check.
    for name, payload in receipts.items():
        try:
            validate_result(payload)
            check(True, f"{name}: validate_result accepted (12 records replayed)")
        except (ValueError, AssertionError) as error:
            check(False, f"{name}: validate_result rejected the receipt ({error})")

    # 2. The grid is exactly the announced {32, 64} x {4, 8, 16} product, every
    #    config distinct, at the frozen seeds and otherwise default settings.
    observed = set(RECEIPTS.values())
    expected = {(steps, phase) for steps in (32, 64) for phase in (4, 8, 16)}
    check(
        observed == expected and len(observed) == len(RECEIPTS),
        f"grid covers exactly {sorted(expected)}",
    )
    for name, payload in receipts.items():
        config = payload["config"]
        hyper = {key: config[key] for key in DEFAULT_HYPERPARAMETERS}
        check(
            hyper == DEFAULT_HYPERPARAMETERS,
            f"{name}: frozen hyperparameters unchanged ({hyper})",
        )
        check(
            tuple(config["seeds"]) == FROZEN_DEVELOPMENT_SEEDS,
            f"{name}: seeds are the frozen development roster {list(FROZEN_DEVELOPMENT_SEEDS)}",
        )
        check(
            payload["schema"] == SCHEMA,
            f"{name}: schema is the current {SCHEMA}",
        )
        for field in ("scientific_promotion_allowed", "paper_parity_claimed", "performance_claimed"):
            check(
                payload[field] is False,
                f"{name}: {field} remains false",
            )

    # 3. One shared lane/runtime/dependency/paper identity across the grid; the
    #    workload digest is the only identity component allowed to differ.
    reference = receipts["receipt-run1.json"]["identity"]
    for name, payload in receipts.items():
        identity = payload["identity"]
        for field in IDENTITY_SOURCE_FIELDS:
            check(
                identity[field] == reference[field],
                f"{name}: identity.{field} matches receipt-run1",
            )
    workload_digests = [payload["identity"]["workload_registry_sha256"] for payload in receipts.values()]
    check(
        len(set(workload_digests)) == len(RECEIPTS),
        "the six distinct configs bind six distinct workload digests",
    )

    tables: dict[str, dict[str, Any]] = {}
    for name, (steps, phase) in RECEIPTS.items():
        payload = receipts[name]
        by_pair = {(rec["seed"], rec["arm"]): rec for rec in payload["records"]}
        label = f"steps={steps},phase={phase}"

        # 4. Mechanism-off parity and matched resource axes (re-derived here so
        #    the retained tables do not lean on validator internals alone).
        for seed in FROZEN_DEVELOPMENT_SEEDS:
            fixed = by_pair[(seed, CONTROL_ARM)]
            off = by_pair[(seed, REDUCTION_ARM)]
            check(
                all(fixed[field] == off[field] for field in PARITY_FIELDS),
                f"{label}, seed {seed}: mechanism-off parity exact on all five transcript fields",
            )
        for field in MATCHED_RESOURCE_FIELDS:
            values = {
                by_pair[(seed, arm)]["resource_receipt"][field]
                for seed in FROZEN_DEVELOPMENT_SEEDS
                for arm in ARMS
            }
            check(
                len(values) == 1,
                f"{label}: {field} matched across all arms and seeds ({values})",
            )

        # 5. Archive byte accounting under the frozen budget.
        diverse_bytes = [
            by_pair[(seed, "diverse_archive")]["resource_receipt"]["archive_persistent_bytes"]
            for seed in FROZEN_DEVELOPMENT_SEEDS
        ]
        budget = payload["config"]["archive_byte_budget"]
        check(
            all(0 < value <= budget for value in diverse_bytes),
            f"{label}: diverse archive retains {max(diverse_bytes)} of the {budget}-byte budget",
        )
        check(
            all(
                by_pair[(seed, REDUCTION_ARM)]["resource_receipt"]["archive_persistent_bytes"] == 0
                for seed in FROZEN_DEVELOPMENT_SEEDS
            ),
            f"{label}: mechanism-off retains zero archive bytes",
        )

        # 6. Paired reward deltas of the archive arms against the controls.
        paired: dict[str, dict[str, Any]] = {}
        for arm in ARCHIVE_ARMS:
            vs_off = [
                by_pair[(seed, arm)]["reward_sum"]
                - by_pair[(seed, REDUCTION_ARM)]["reward_sum"]
                for seed in FROZEN_DEVELOPMENT_SEEDS
            ]
            vs_fixed = [
                by_pair[(seed, arm)]["reward_sum"]
                - by_pair[(seed, CONTROL_ARM)]["reward_sum"]
                for seed in FROZEN_DEVELOPMENT_SEEDS
            ]
            paired[arm] = {
                "reward_sum_deltas_vs_mechanism_off": vs_off,
                "reward_sum_deltas_vs_fixed_snapshot": vs_fixed,
                "mean_delta_vs_mechanism_off": sum(vs_off) / len(vs_off),
            }
            print(
                f"INFO {label} {arm}: paired deltas vs mechanism-off {vs_off} "
                f"(mean {paired[arm]['mean_delta_vs_mechanism_off']:+.2f})"
            )
        tables[label] = {
            "steps": steps,
            "phase_length": phase,
            "reward_sums": {
                arm: [by_pair[(seed, arm)]["reward_sum"] for seed in FROZEN_DEVELOPMENT_SEEDS]
                for arm in ARMS
            },
            "mean_rewards": {
                arm: [by_pair[(seed, arm)]["mean_reward"] for seed in FROZEN_DEVELOPMENT_SEEDS]
                for arm in ARMS
            },
            "archive_persistent_bytes": {
                arm: [
                    by_pair[(seed, arm)]["resource_receipt"]["archive_persistent_bytes"]
                    for seed in FROZEN_DEVELOPMENT_SEEDS
                ]
                for arm in ARMS + (REDUCTION_ARM,)
            },
            "paired": paired,
        }

    # 7. The default-config receipt reproduces the merged #2257 record exactly.
    default = tables["steps=32,phase=4"]["reward_sums"]
    check(
        all(
            set(values) == {HISTORICAL_DEFAULT_REWARD_SUMS[arm]}
            for arm, values in default.items()
        ),
        "default config (32, 4) reproduces the #2257 retained reward sums "
        f"({HISTORICAL_DEFAULT_REWARD_SUMS}) on every seed",
    )

    # 8. Grid-level negative outcome: no archive arm beats mechanism-off on any
    #    (config, seed) pair, and the tie only appears at the longest phases.
    all_deltas = [
        delta
        for table in tables.values()
        for arm in ARCHIVE_ARMS
        for delta in table["paired"][arm]["reward_sum_deltas_vs_mechanism_off"]
    ]
    check(
        all(delta <= 0 for delta in all_deltas),
        f"all {len(all_deltas)} paired archive-vs-mechanism-off deltas are <= 0 (negative outcome retained)",
    )

    summary = {
        "validator": "alberta_framework.benchmarks.telapa_qualification.validate_result",
        "schema": SCHEMA,
        "grid": sorted(expected),
        "seeds": list(FROZEN_DEVELOPMENT_SEEDS),
        "arms": list(ARMS),
        "tables": tables,
        "negative_outcome": {
            "archive_arms_never_beat_mechanism_off": all(delta <= 0 for delta in all_deltas),
            "paired_delta_count": len(all_deltas),
            "tie_only_at_phase_length_16": all(
                all(
                    delta == 0
                    for arm in ARCHIVE_ARMS
                    for delta in table["paired"][arm]["reward_sum_deltas_vs_mechanism_off"]
                )
                for label, table in tables.items()
                if table["phase_length"] == 16
            ),
        },
        "failures": FAILURES,
        "verdict": "all checks passed" if not FAILURES else f"{len(FAILURES)} checks failed",
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"WROTE {args.output}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
