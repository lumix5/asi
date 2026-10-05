"""End-to-end and hostile contracts for the optimization-readiness lane.

The lane consumes :mod:`alberta_framework.evaluation.optimization_readiness`
and must keep every pinned Appendix C.1 axis exact.  Tests run the
``contract-smoke`` profile on synthetic bounded pools only; real benchmark
execution happens through the CLI, never inside pytest.
"""

from __future__ import annotations

import copy
import json

import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks import optimization_readiness_development as lane
from alberta_framework.benchmarks.optimization_readiness_development import (
    ARM_GRADIENT_STRENGTH_ONLY,
    ARM_READINESS_FULL,
    FROZEN_SEEDS,
    INPUT_DIM,
    N_VALIDATION_OBSERVATIONS,
    PAPER_REVISION,
    PROFILES,
    SAMPLING_PROVENANCE,
    main,
    run_readiness_development,
    validate_result,
)
from alberta_framework.evaluation import optimization_readiness as oracle

pytestmark = pytest.mark.integration


def _fixture_pools() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = jr.key(20261005)
    image_key, label_key, valid_key, valid_label_key = jr.split(key, 4)
    train_images = np.asarray(jr.normal(image_key, (96, INPUT_DIM)), dtype=np.float32)
    train_labels = np.asarray(jr.randint(label_key, (96,), 0, 10), dtype=np.int32)
    validation_images = np.asarray(
        jr.normal(valid_key, (N_VALIDATION_OBSERVATIONS, INPUT_DIM)), dtype=np.float32
    )
    validation_labels = np.asarray(
        jr.randint(valid_label_key, (N_VALIDATION_OBSERVATIONS,), 0, 10), dtype=np.int32
    )
    return train_images, train_labels, validation_images, validation_labels


def _without_timing(result: dict) -> dict:
    stripped = copy.deepcopy(json.loads(json.dumps(result)))
    for block in stripped["comparisons"]:
        for receipt in block["receipts"]:
            receipt["resources"]["timing_seconds"] = 0.0
    return stripped


def test_sampling_provenance_matches_the_pinned_evaluation_constant() -> None:
    assert SAMPLING_PROVENANCE == (
        "caller_reported_independent_with_replacement_not_verified_from_gradients"
    )
    assert PAPER_REVISION == oracle.OPTIMIZATION_READINESS_PROTOCOL["paper_revision"]
    assert N_VALIDATION_OBSERVATIONS == 10_000


@pytest.mark.slow
def test_lane_runs_end_to_end_and_receipts_pass_matched_validation() -> None:
    pools = _fixture_pools()
    result = run_readiness_development(*pools, seed=FROZEN_SEEDS[0])
    assert result["schema"] == lane.DEVELOPMENT_SCHEMA
    assert result["paper_revision"] == PAPER_REVISION
    assert result["development_only"] is True
    assert result["scientific_promotion_allowed"] is False
    assert result["negative_results_must_be_retained"] is True
    assert result["labels_permuted"] is False
    assert result["task_protocol"] == "cumulative-input-permutation"
    profile = PROFILES[result["profile_id"]]
    assert profile.profile_id == "contract-smoke"
    comparisons = result["comparisons"]
    assert len(comparisons) == profile.n_tasks
    seen_tasks = []
    seen_ids = set()
    for index, block in enumerate(comparisons):
        receipts = block["receipts"]
        assert [receipt["arm_id"] for receipt in receipts] == [
            ARM_READINESS_FULL,
            ARM_GRADIENT_STRENGTH_ONLY,
        ]
        matched = oracle.validate_matched_development_results(receipts)
        full, off = matched
        # Checkpoint properties are shared; the arms differ only in their
        # predictor value, and the mechanism-off arm is the predeclared
        # gradient-strength-only reduction of the full estimator.
        assert full.optimization_readiness >= 0.0 and off.optimization_readiness >= 0.0
        assert full.gradient_norm == off.gradient_norm
        assert full.future_relative_loss_reduction == off.future_relative_loss_reduction
        assert full.reported_outcome == "inconclusive"
        assert full.protocol.sampling_provenance == SAMPLING_PROVENANCE
        assert full.protocol.seed == FROZEN_SEEDS[0]
        seen_tasks.append(block["task"])
        seen_ids.add(block["comparison_id"])
    assert seen_tasks == [f"task_{index:02d}" for index in range(len(comparisons))]
    assert len(seen_ids) == len(comparisons)
    # Stream accounting must match the frozen profile exactly.
    assert result["stream"]["parameter_updates"] == profile.stream_parameter_updates
    assert result["stream"]["observations"] == profile.stream_observations


@pytest.mark.slow
def test_validate_result_rejects_tampering() -> None:
    pools = _fixture_pools()
    result = run_readiness_development(*pools, seed=FROZEN_SEEDS[1])
    validate_result(result)

    promoted = copy.deepcopy(result)
    promoted["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="scientific_promotion_allowed"):
        validate_result(promoted)

    drifted = copy.deepcopy(result)
    drifted["comparisons"][0]["receipts"][0]["protocol"]["mini_batch_size"] = 8
    with pytest.raises(ValueError, match="Appendix C.1"):
        validate_result(drifted)

    dropped = copy.deepcopy(result)
    dropped["comparisons"] = dropped["comparisons"][:-1]
    with pytest.raises(ValueError, match="comparison count"):
        validate_result(dropped)

    reduced = copy.deepcopy(result)
    for block in reduced["comparisons"]:
        for receipt in block["receipts"]:
            receipt["metrics"]["future_relative_loss_reduction"] = 1.5
    with pytest.raises(ValueError, match="cannot exceed one"):
        validate_result(reduced)

    negative = copy.deepcopy(result)
    for block in negative["comparisons"]:
        for receipt in block["receipts"]:
            receipt["metrics"]["future_relative_loss_reduction"] = -0.25
    # Self-contained lower bound: the C.1 oracle alone accepts negatives.
    with pytest.raises(ValueError, match="cannot exceed one or be negative"):
        validate_result(negative)


def test_atomic_write_publishes_without_replacing(tmp_path) -> None:
    fresh = tmp_path / "envelope.json"
    lane._atomic_write(fresh, "first")
    assert fresh.read_text(encoding="utf-8") == "first"

    with pytest.raises(FileExistsError, match="no-replace publication"):
        lane._atomic_write(fresh, "replacement")
    # The existing artifact is untouched and no temporary files leak.
    assert fresh.read_text(encoding="utf-8") == "first"
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.slow
def test_schedule_is_deterministic_and_seed_separated() -> None:
    pools = _fixture_pools()
    first = run_readiness_development(*pools, seed=FROZEN_SEEDS[2])
    second = run_readiness_development(*pools, seed=FROZEN_SEEDS[2])
    assert first["task_permutation_digest"] == second["task_permutation_digest"]
    assert _without_timing(first) == _without_timing(second)
    other = run_readiness_development(*pools, seed=FROZEN_SEEDS[3])
    assert other["task_permutation_digest"] != first["task_permutation_digest"]


def test_seed_and_profile_gates_reject_unknowns() -> None:
    pools = _fixture_pools()
    with pytest.raises(ValueError, match="seed is outside the frozen development schedule"):
        run_readiness_development(*pools, seed=1)
    with pytest.raises(ValueError, match="unknown readiness profile"):
        run_readiness_development(*pools, seed=FROZEN_SEEDS[0], profile_id="nope")
    with pytest.raises(ValueError, match="validation pool must hold exactly"):
        run_readiness_development(
            pools[0],
            pools[1],
            pools[2][: N_VALIDATION_OBSERVATIONS - 1],
            pools[3][: N_VALIDATION_OBSERVATIONS - 1],
            seed=FROZEN_SEEDS[0],
        )


def test_cli_catalog_and_rejected_dataset(tmp_path, capsys) -> None:
    assert main(["--catalog"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == lane.DEVELOPMENT_SCHEMA
    assert payload["frozen_seeds"] == list(FROZEN_SEEDS)
    assert payload["development_only"] is True
    missing = tmp_path / "absent.npz"
    with pytest.raises(FileNotFoundError):
        main(
            [
                "--train-npz",
                str(missing),
                "--validation-npz",
                str(missing),
            ]
        )
