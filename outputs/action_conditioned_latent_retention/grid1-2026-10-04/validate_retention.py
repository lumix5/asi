"""Re-validate the retained action-conditioned latent grid and re-derive the analysis.

Run from the repository root:

    .venv/bin/python outputs/action_conditioned_latent_retention/grid1-2026-10-04/validate_retention.py

The script is retained beside the receipts so the pass is reproducible from the
committed bytes alone. It re-validates every receipt with the lane's own strict
validator, checks cross-process determinism, the decision-off/mechanism-off
transcript-hash identity required by docs/research/action-conditioned-latent-protocol.md,
matched resource counters across arms, and re-derives the paired deltas in
SUMMARY.md. It writes nothing except the analysis summary passed via --output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from alberta_framework.benchmarks.action_conditioned_latent import (
    FROZEN_ARM_IDS,
    FROZEN_DEVELOPMENT_SEEDS,
    validate_action_latent_payload,
)

RECEIPTS = ("receipt-run1.json", "receipt-run2.json", "receipt-run3.json")
MODEL_ARMS = (
    "latent_action_interactions",
    "latent_no_interactions",
    "latent_action_masked",
    "latent_decision_off",
)
CONTROL_ARM = "mechanism_off"
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)
    print(("PASS " if condition else "FAIL ") + message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    receipts = {name: json.loads((here / "receipts" / name).read_text()) for name in RECEIPTS}

    for name, payload in receipts.items():
        validate_action_latent_payload(payload)
        print(f"PASS {name}: validate_action_latent_payload accepted the receipt")

    digests = {name: sha256_file(here / "receipts" / name) for name in RECEIPTS}
    check(
        len(set(digests.values())) == 1,
        f"all {len(RECEIPTS)} fresh-process receipts are bitwise identical (sha256 {digests[RECEIPTS[0]][:16]}...)",
    )

    base = receipts[RECEIPTS[0]]
    check(base["development_only"] is True, "receipt marks development_only=true")
    check(
        base["scientific_promotion_allowed"] is False,
        "receipt marks scientific_promotion_allowed=false",
    )
    check(
        [(a["seed"], a["arm_id"]) for a in base["arms"]]
        == [(s, arm) for s in FROZEN_DEVELOPMENT_SEEDS for arm in FROZEN_ARM_IDS],
        "receipt covers exactly the six frozen arms x four frozen development seeds",
    )

    rows: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in base["arms"]:
        rows.setdefault(arm["arm_id"], {})[arm["seed"]] = arm
    check(sorted(rows) == sorted(FROZEN_ARM_IDS), "arm roster matches FROZEN_ARM_IDS")

    # Transcript identity required by the protocol doc: the disabled-decision
    # arm must reproduce the mechanism-off action and reward hashes exactly.
    for seed in FROZEN_DEVELOPMENT_SEEDS:
        off = rows[CONTROL_ARM][seed]
        dec = rows["latent_decision_off"][seed]
        check(
            dec["action_sha256"] == off["action_sha256"]
            and dec["reward_sha256"] == off["reward_sha256"],
            f"seed {seed}: decision-off action/reward hashes equal mechanism-off",
        )

    # Matched environment exposure for every arm; the model-owning arms must
    # additionally share their update and prequential-training budgets. Query
    # counts legitimately differ for arms without a model or a decision
    # interface; they are reported per arm instead of asserted equal.
    for seed in FROZEN_DEVELOPMENT_SEEDS:
        steps = {rows[arm_id][seed]["environment_steps"] for arm_id in FROZEN_ARM_IDS}
        check(len(steps) == 1, f"seed {seed}: environment_steps matched across all arms ({steps})")
        model_budgets = {
            arm_id: (rows[arm_id][seed]["model_updates"], rows[arm_id][seed]["training_queries"])
            for arm_id in MODEL_ARMS
        }
        check(
            len(set(model_budgets.values())) == 1,
            f"seed {seed}: model_updates/training_queries matched across the four model-owning arms",
        )

    # Paired deltas of the model arms against the mechanism-off control.
    paired: dict[str, dict[str, Any]] = {}
    for arm_id in MODEL_ARMS:
        late = [rows[arm_id][s]["late_return_sum"] - rows[CONTROL_ARM][s]["late_return_sum"] for s in FROZEN_DEVELOPMENT_SEEDS]
        total = [rows[arm_id][s]["return_sum"] - rows[CONTROL_ARM][s]["return_sum"] for s in FROZEN_DEVELOPMENT_SEEDS]
        paired[arm_id] = {
            "late_return_deltas": late,
            "late_return_mean_delta": sum(late) / len(late),
            "late_return_all_nonpositive": all(d <= 0 for d in late),
            "late_return_all_positive": all(d > 0 for d in late),
            "return_sum_deltas": total,
            "return_sum_mean_delta": sum(total) / len(total),
        }
        print(
            f"INFO {arm_id}: late-return paired deltas {late} "
            f"(mean {paired[arm_id]['late_return_mean_delta']:+.2f}), "
            f"return-sum paired deltas {total}"
        )

    decision_off_match = paired["latent_decision_off"]["late_return_deltas"] == [0] * len(
        FROZEN_DEVELOPMENT_SEEDS
    ) and paired["latent_decision_off"]["return_sum_deltas"] == [0] * len(FROZEN_DEVELOPMENT_SEEDS)
    check(
        decision_off_match,
        "decision-off totals equal mechanism-off totals on every seed (learning-without-decisions is neutral)",
    )

    summary = {
        "receipt_sha256": digests[RECEIPTS[0]],
        "fresh_process_receipts_bitwise_identical": len(set(digests.values())) == 1,
        "validator": "alberta_framework.benchmarks.action_conditioned_latent.validate_action_latent_payload",
        "arms": FROZEN_ARM_IDS,
        "seeds": list(FROZEN_DEVELOPMENT_SEEDS),
        "paired_vs_mechanism_off": paired,
        "failures": FAILURES,
        "verdict": "all checks passed" if not FAILURES else f"{len(FAILURES)} checks failed",
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"WROTE {args.output}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
