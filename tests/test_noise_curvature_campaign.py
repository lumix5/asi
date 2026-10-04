"""Contract tests for the hard-disabled noise-curvature campaign (#1567)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import scipy.stats

import alberta_framework.evaluation.noise_curvature_campaign as campaign
from alberta_framework.benchmarks.upgd_ipmnist import build_schedule, init_mlp_params
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    DEVELOPMENT_SEEDS,
    LIVE_CONTROL,
    registered_arms,
    registered_hyperparameters,
)


def _resign(payload: dict[str, Any], field: str) -> None:
    payload[field] = campaign.digest_without(payload, field)


def _shard(
    arm: str,
    seed: int,
    accuracy: float,
    *,
    runtime: dict[str, object] | None = None,
) -> dict[str, Any]:
    resources = cast(
        dict[str, object], campaign.frozen_plan_payload()["expected_resources_per_shard"]
    )
    init_sha, schedule_sha = campaign.campaign_schedule_init_identities(seed)
    payload: dict[str, Any] = {
        "schema": campaign.SHARD_SCHEMA,
        "status": "complete",
        "plan_sha256": campaign._digest(campaign.frozen_plan_payload()),
        "arm": arm,
        "seed": seed,
        "hyperparameters": registered_hyperparameters(arm),
        "metrics": {
            "mean_online_accuracy": accuracy,
            "mean_loss": 0.5,
            "mean_plasticity": 0.25,
        },
        "resources": {
            **resources,
            "timing_seconds": 10.0,
            "timing_is_telemetry_only": True,
        },
        "identity": {
            "source_sha256": campaign._source_identity(),
            "runtime": runtime if runtime is not None else campaign._runtime_identity(),
            "dependencies": campaign._dependency_identity(),
            "authorization": campaign._authorization_identity(),
            "dataset_sha256": campaign.DATASET_DIGEST,
            "schedule_sha256": schedule_sha,
            "init_sha256": init_sha,
            "process": campaign._process_identity(),
            "consistency_not_attestation": True,
        },
        "reexecution": {
            "required": True,
            "dispatches": 2,
            "dataset_digest_equal": True,
            "metrics_exact_equal": True,
            "counters_exact_equal": True,
            "timing_retained": False,
        },
        "outcome": "inconclusive",
        "outcome_retained": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
    _resign(payload, "shard_sha256")
    return payload


def _control_row(seed: int, accuracy: float) -> dict[str, Any]:
    return {
        "schema": campaign.CONTROL_ROW_SCHEMA,
        "arm": LIVE_CONTROL,
        "seed": seed,
        "mean_online_accuracy": accuracy,
        "dataset_sha256": campaign.DATASET_DIGEST,
        "receipt_sha256": "a" * 64,
        "separately_protocolled": True,
    }


def _panel_shards(
    accuracies: dict[tuple[str, int], float],
) -> list[dict[str, Any]]:
    return [
        _shard(arm, seed, accuracies[(arm, seed)])
        for seed in campaign.FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN.seeds
        for arm in registered_arms()
    ]


@pytest.fixture
def authorized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(campaign, "EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(campaign, "AUTHORIZATION_TRANSITION_APPROVED", True)
    monkeypatch.setattr(
        campaign,
        "FROZEN_PLAN_SHA256",
        hashlib.sha256(
            campaign._canonical_bytes(
                campaign._plan_payload(campaign.FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN)
            )
        ).hexdigest(),
    )


def test_frozen_plan_uses_fresh_seeds_and_published_geometry() -> None:
    plan = campaign.FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN
    assert plan.seeds == (1_567_001, 1_567_002, 1_567_003)
    assert set(plan.seeds).isdisjoint(DEVELOPMENT_SEEDS)
    assert len(set(plan.seeds)) == len(plan.seeds)
    assert plan.config.to_config() == {
        "n_tasks": 200,
        "task_length": 5000,
        "input_dim": 784,
        "hidden1": 300,
        "hidden2": 150,
        "n_classes": 10,
    }
    assert plan.live_control == LIVE_CONTROL


def test_plan_payload_is_pinned_with_exact_accounting() -> None:
    payload = campaign.frozen_plan_payload()
    assert campaign._digest(payload) == campaign.FROZEN_PLAN_SHA256
    resources = cast(dict[str, object], payload["expected_resources_per_shard"])
    assert resources == {
        "observations": 1_000_000,
        "updates": 1_000_000,
        "data_steps": 1_000_000,
        "environment_steps": 0,
        "model_queries": 3_075_000,
        "first_order_gradient_queries": 2_000_000,
        "loss_only_queries": 1_000_000,
        "hessian_vector_product_queries": 75_000,
        "controller_events": 25_000,
        "persistent_bytes": 4_640_356,
    }
    dispatch = cast(dict[str, object], payload["dispatch_accounting"])
    assert dispatch["scheduler_shards"] == 12
    assert dispatch["separately_protocolled_control_shards"] == 3
    assert dispatch["execution_dispatches_per_shard"] == 2
    assert dispatch["reservation_required_before_dataset_or_rng_work"] is True
    dataset = cast(dict[str, object], payload["dataset"])
    assert dataset["sha256"] == campaign.DATASET_DIGEST
    assert dataset["rows"] == 60_000
    assert dataset["columns"] == 784
    live_control = cast(dict[str, object], payload["live_control"])
    assert live_control["separately_protocolled"] is True


def test_gate_rule_constants_are_the_frozen_statistics() -> None:
    assert campaign.PAIRED_T_CRITICAL == float(scipy.stats.t.ppf(0.975, 2))
    assert campaign.HILLCLIMB_MIN_MEAN_DELTA == 0.005
    rules = cast(dict[str, object], campaign.frozen_plan_payload()["gate_rules"])
    assert rules["t_critical"] == campaign.PAIRED_T_CRITICAL
    assert rules["t_critical_degrees_of_freedom"] == 2
    assert rules["hillclimb_min_mean_delta"] == 0.005


def test_schedule_init_identities_reproduce_the_runner_derivation() -> None:
    seed = 1_567_001
    config = campaign.FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN.config
    root = jr.key(jnp.uint32(seed), impl="threefry2x32")
    key_init, key_schedule, _ = jr.split(root, 3)
    params = init_mlp_params(key_init, config)
    init_digest = hashlib.sha256(b"asi.noise-curvature-ipmnist.campaign-init.v1\0")
    for name in sorted(params):
        raw = np.asarray(jax.device_get(params[name]))
        init_digest.update(name.encode("ascii"))
        init_digest.update(raw.dtype.str.encode("ascii"))
        init_digest.update(str(raw.shape).encode("ascii"))
        init_digest.update(raw.tobytes(order="C"))
    schedule = build_schedule(key_schedule, config, 60_000)
    schedule_digest = hashlib.sha256(b"asi.noise-curvature-ipmnist.campaign-schedule.v1\0")
    for name in ("permutations", "example_indices"):
        raw = np.asarray(jax.device_get(getattr(schedule, name)))
        schedule_digest.update(name.encode("ascii"))
        schedule_digest.update(raw.dtype.str.encode("ascii"))
        schedule_digest.update(str(raw.shape).encode("ascii"))
        schedule_digest.update(raw.tobytes(order="C"))
    identities = cast(
        dict[str, object],
        campaign.frozen_plan_payload()["schedule_init_identities"],
    )
    pinned = cast(dict[str, object], identities[str(seed)])
    assert pinned["init_sha256"] == init_digest.hexdigest()
    assert pinned["schedule_sha256"] == schedule_digest.hexdigest()
    recomputed = campaign.campaign_schedule_init_identities(seed)
    assert recomputed == (init_digest.hexdigest(), schedule_digest.hexdigest())


def test_execution_is_hard_disabled() -> None:
    assert campaign.EXECUTION_AUTHORIZED is False
    assert campaign.AUTHORIZATION_TRANSITION_APPROVED is False
    with pytest.raises(PermissionError):
        campaign._require_execution_authorized()


def test_plan_document_roundtrip_and_drift_rejection() -> None:
    document = campaign.build_campaign_plan_document()
    campaign.validate_campaign_plan_document(document)
    for field, replacement in (
        ("schema", "other.v1"),
        ("plan_sha256", "0" * 64),
        ("document_sha256", "0" * 64),
    ):
        drifted = copy.deepcopy(document)
        drifted[field] = replacement
        with pytest.raises(ValueError):
            campaign.validate_campaign_plan_document(drifted)
    drifted = copy.deepcopy(document)
    cast(dict[str, object], drifted["identity"])["consistency_not_attestation"] = False
    with pytest.raises(ValueError):
        campaign.validate_campaign_plan_document(drifted)
    drifted = copy.deepcopy(document)
    cast(dict[str, object], drifted["policy"])["status"] = "promoting"
    with pytest.raises(ValueError):
        campaign.validate_campaign_plan_document(drifted)


def test_shard_validator_accepts_exact_receipt_and_rejects_drift() -> None:
    shard = _shard("noise_curvature_combined", 1_567_002, 0.83)
    campaign.validate_noise_curvature_campaign_shard(shard)
    hostile: list[dict[str, object]] = [
        {"arm": "noise_curvature_unknown"},
        {"seed": 0},
        {"seed": 1_567_004},
    ]
    for mutation in hostile:
        drifted = copy.deepcopy(shard)
        drifted.update(mutation)
        _resign(drifted, "shard_sha256")
        with pytest.raises(ValueError):
            campaign.validate_noise_curvature_campaign_shard(drifted)
    drifted = copy.deepcopy(shard)
    drifted["hyperparameters"] = {
        **cast(dict[str, object], drifted["hyperparameters"]),
        "step_size": 0.5,
    }
    _resign(drifted, "shard_sha256")
    with pytest.raises(ValueError):
        campaign.validate_noise_curvature_campaign_shard(drifted)
    drifted = copy.deepcopy(shard)
    cast(dict[str, object], drifted["resources"])["persistent_bytes"] = 1
    _resign(drifted, "shard_sha256")
    with pytest.raises(ValueError):
        campaign.validate_noise_curvature_campaign_shard(drifted)
    drifted = copy.deepcopy(shard)
    cast(dict[str, object], drifted["identity"])["dataset_sha256"] = "b" * 64
    _resign(drifted, "shard_sha256")
    with pytest.raises(ValueError):
        campaign.validate_noise_curvature_campaign_shard(drifted)
    drifted = copy.deepcopy(shard)
    cast(dict[str, object], drifted["identity"])["init_sha256"] = "c" * 64
    _resign(drifted, "shard_sha256")
    with pytest.raises(ValueError):
        campaign.validate_noise_curvature_campaign_shard(drifted)
    drifted = copy.deepcopy(shard)
    cast(dict[str, object], drifted["reexecution"])["dispatches"] = 1
    _resign(drifted, "shard_sha256")
    with pytest.raises(ValueError):
        campaign.validate_noise_curvature_campaign_shard(drifted)
    drifted = copy.deepcopy(shard)
    cast(dict[str, object], drifted["metrics"])["mean_online_accuracy"] = 1.5
    _resign(drifted, "shard_sha256")
    with pytest.raises(ValueError):
        campaign.validate_noise_curvature_campaign_shard(drifted)
    drifted = copy.deepcopy(shard)
    drifted["shard_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        campaign.validate_noise_curvature_campaign_shard(drifted)


def test_control_row_validation() -> None:
    row = _control_row(1_567_003, 0.81)
    campaign.validate_noise_curvature_control_row(row)
    for mutation in (
        {"arm": "noise_curvature_combined"},
        {"seed": 4},
        {"dataset_sha256": "d" * 64},
        {"separately_protocolled": False},
        {"mean_online_accuracy": -0.1},
    ):
        drifted = dict(row)
        drifted.update(mutation)
        with pytest.raises(ValueError):
            campaign.validate_noise_curvature_control_row(drifted)


def test_aggregate_computes_frozen_gates_exactly(
    authorized: None,  # noqa: ARG001
) -> None:
    seeds = campaign.FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN.seeds
    combined = {1_567_001: 0.83, 1_567_002: 0.84, 1_567_003: 0.85}
    fixed = {1_567_001: 0.80, 1_567_002: 0.81, 1_567_003: 0.82}
    accuracies: dict[tuple[str, int], float] = {}
    for seed in seeds:
        accuracies[("noise_curvature_combined", seed)] = combined[seed]
        accuracies[("noise_curvature_fixed_adam_l2", seed)] = fixed[seed]
        accuracies[("noise_curvature_gradient_only", seed)] = fixed[seed] + 0.002
        accuracies[("noise_curvature_volatility_only", seed)] = fixed[seed] + 0.003
    control = {seed: fixed[seed] - 0.001 for seed in seeds}
    aggregate = campaign.summarize_noise_curvature_campaign(
        _panel_shards(accuracies),
        [_control_row(seed, control[seed]) for seed in seeds],
    )
    campaign.validate_noise_curvature_campaign_aggregate(aggregate)
    gates = cast(dict[str, object], aggregate["gates"])
    mechanism = cast(dict[str, object], gates["mechanism"])
    deltas = [combined[seed] - fixed[seed] for seed in seeds]
    values = np.asarray(deltas, dtype=np.float64)
    expected_lower = float(np.mean(values)) - campaign.PAIRED_T_CRITICAL * float(
        np.std(values, ddof=1) / np.sqrt(len(deltas))
    )
    assert mechanism["delta_mean"] == pytest.approx(float(np.mean(values)))
    assert mechanism["interval_lower"] == pytest.approx(expected_lower)
    assert mechanism["outcome"] == "supported"
    assert cast(dict[str, object], gates["causal_gradient"])["outcome"] == "supported"
    assert cast(dict[str, object], gates["causal_volatility"])["outcome"] == "supported"
    hillclimb = cast(dict[str, object], gates["hillclimb"])
    assert hillclimb["outcome"] == "supported"
    assert hillclimb["min_mean_delta"] == 0.005


def test_aggregate_rejects_inconclusive_and_hostile_panels(
    authorized: None,  # noqa: ARG001
) -> None:
    seeds = campaign.FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN.seeds
    straddling: dict[tuple[str, int], float] = {}
    for seed in seeds:
        straddling[("noise_curvature_combined", seed)] = 0.8
        straddling[("noise_curvature_fixed_adam_l2", seed)] = 0.8
        straddling[("noise_curvature_gradient_only", seed)] = 0.8
        straddling[("noise_curvature_volatility_only", seed)] = 0.8
    control = [_control_row(seed, 0.8) for seed in seeds]
    aggregate = campaign.summarize_noise_curvature_campaign(
        _panel_shards(straddling), control
    )
    gates = cast(dict[str, object], aggregate["gates"])
    assert cast(dict[str, object], gates["mechanism"])["outcome"] == "inconclusive"
    assert cast(dict[str, object], gates["hillclimb"])["outcome"] == "rejected"
    below: dict[tuple[str, int], float] = {}
    for seed in seeds:
        below[("noise_curvature_combined", seed)] = 0.79
        below[("noise_curvature_fixed_adam_l2", seed)] = 0.80
        below[("noise_curvature_gradient_only", seed)] = 0.80
        below[("noise_curvature_volatility_only", seed)] = 0.80
    aggregate = campaign.summarize_noise_curvature_campaign(_panel_shards(below), control)
    gates = cast(dict[str, object], aggregate["gates"])
    assert cast(dict[str, object], gates["mechanism"])["outcome"] == "rejected"

    with pytest.raises(ValueError):
        campaign.summarize_noise_curvature_campaign(
            _panel_shards(straddling)[:-1], control
        )
    with pytest.raises(ValueError):
        campaign.summarize_noise_curvature_campaign(
            _panel_shards(straddling), control[:-1]
        )
    duplicate = _panel_shards(straddling)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(ValueError):
        campaign.summarize_noise_curvature_campaign(duplicate, control)
    mismatched_runtime = _panel_shards(straddling)
    drifted_runtime = copy.deepcopy(mismatched_runtime[-1])
    identity = cast(dict[str, object], drifted_runtime["identity"])
    cast(dict[str, object], identity["runtime"])["backend"] = "gpu"
    _resign(drifted_runtime, "shard_sha256")
    mismatched_runtime[-1] = drifted_runtime
    with pytest.raises(ValueError):
        campaign.summarize_noise_curvature_campaign(mismatched_runtime, control)


def test_failed_dispatch_builder_and_validator() -> None:
    record = campaign.build_failed_campaign_dispatch(
        "noise_curvature_combined", 1_567_001, "before_dataset_load"
    )
    campaign.validate_failed_campaign_dispatch(record)
    failure = cast(dict[str, object], record["failure"])
    assert failure["stage_precedes_dataset_work"] is True
    assert failure["exception_message_retained"] is False
    assert cast(dict[str, object], record["policy"])["used_for_outcome"] is False
    drifted = copy.deepcopy(record)
    cast(dict[str, object], drifted["failure"])["exception_message_retained"] = True
    with pytest.raises(ValueError):
        campaign.validate_failed_campaign_dispatch(drifted)
    drifted = copy.deepcopy(record)
    drifted["failure_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        campaign.validate_failed_campaign_dispatch(drifted)
    with pytest.raises(ValueError):
        campaign.build_failed_campaign_dispatch("unknown_arm", 1_567_001, "first_dispatch")
    with pytest.raises(ValueError):
        campaign.build_failed_campaign_dispatch(
            "noise_curvature_combined", 1_567_009, "first_dispatch"
        )
    with pytest.raises(ValueError):
        campaign.build_failed_campaign_dispatch(
            "noise_curvature_combined", 1_567_001, "after_everything"
        )


def test_reservation_publishes_immutable_plan(tmp_path: Path) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    destination = campaign.reserve_campaign_plan_document(repository_root=root)
    assert destination.is_file()
    assert destination.stat().st_mode & 0o777 == 0o444
    document = campaign.build_campaign_plan_document()
    assert destination.read_bytes() == campaign._canonical_bytes(document)
    reloaded = json.loads(destination.read_bytes())
    campaign.validate_campaign_plan_document(reloaded)
    with pytest.raises(OSError):
        campaign.reserve_campaign_plan_document(repository_root=root)


def test_cli_emits_and_validates_without_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan_path = tmp_path / "plan.json"
    document = campaign.build_campaign_plan_document()
    plan_path.write_bytes(campaign._canonical_bytes(document))
    assert campaign.main(["validate-plan", str(plan_path)]) == 0
    assert "plan_sha256" in capsys.readouterr().out
    assert (
        campaign.main(
            [
                "failed-dispatch",
                "--arm",
                "noise_curvature_combined",
                "--seed",
                "1567001",
                "--stage",
                "first_dispatch",
            ]
        )
        == 0
    )
    record = json.loads(capsys.readouterr().out)
    campaign.validate_failed_campaign_dispatch(record)
    root = tmp_path / "repo"
    root.mkdir()
    assert campaign.main(["reserve-plan", "--root", str(root)]) == 0
    reserved = Path(capsys.readouterr().out.strip())
    assert reserved.is_file()
    shard = _shard("noise_curvature_combined", 1_567_002, 0.83)
    shard_path = tmp_path / "shard.json"
    shard_path.write_bytes(campaign._canonical_bytes(shard))
    assert campaign.main(["validate-shard", str(shard_path)]) == 0
    assert "shard_sha256" in capsys.readouterr().out
