"""Prospective, hard-disabled noise-curvature campaign contract (#1567).

This module freezes the next development campaign around the existing
noise-curvature IPMNIST screening runner: a fresh-seed roster, the canonical
OpenML MNIST materialization digest, explicit per-seed schedule/initialization
identities, exact resource accounting, a reservation-first publication gate,
a durable failed-disposition record, and the frozen mechanism/causal/hillclimb
confidence rules.  It never loads a dataset, derives RNG, or executes a runner
step, and campaign execution is explicitly unauthorized in this revision.

The contract is permanently nonpromoting: an executed campaign result could
never establish promotion, paper parity, or scientific evidence by itself.
Every digest here binds consistency, not authenticated execution attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import secrets
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PosixPath
from typing import Final, NoReturn, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.benchmarks.noise_curvature_ipmnist import (
    noise_curvature_persistent_bytes,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    build_schedule,
    init_mlp_params,
)
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    DEVELOPMENT_SEEDS,
    LIVE_CONTROL,
    registered_arms,
    registered_hyperparameters,
)

PLAN_SCHEMA: Final[str] = "asi.noise-curvature-ipmnist.campaign-plan.v1"
SHARD_SCHEMA: Final[str] = "asi.noise-curvature-ipmnist.campaign-shard.v1"
CONTROL_ROW_SCHEMA: Final[str] = "asi.noise-curvature-ipmnist.campaign-control-row.v1"
AGGREGATE_SCHEMA: Final[str] = "asi.noise-curvature-ipmnist.campaign-aggregate.v1"
FAILED_DISPATCH_SCHEMA: Final[str] = "asi.noise-curvature-ipmnist.campaign-failed-dispatch.v1"
PROCESS_SCHEMA: Final[str] = "asi.noise-curvature-ipmnist.campaign-process.v1"

EXECUTION_AUTHORIZED: Final[bool] = False
AUTHORIZATION_TRANSITION_APPROVED: Final[bool] = False

OUTPUT_NAMESPACE: Final[str] = "outputs/noise_curvature_matched/development.v1"
DATASET_DIGEST: Final[str] = (
    "5d1d22587aab6cdd53963acb777ed790d26abb25155fde62f50c48c4130c6eaf"
)
DATASET_SELECTION: Final[str] = (
    "OpenML mnist_784 version=1 canonical 60,000-row train materialization "
    "scaled to [-1,1] by load_mnist_train"
)
N_TRAIN_ROWS: Final[int] = 60_000

# Two-sided 95% Student-t critical value for dof = len(CAMPAIGN_SEEDS) - 1 = 2.
PAIRED_T_CRITICAL: Final[float] = 4.302652729749462
HILLCLIMB_MIN_MEAN_DELTA: Final[float] = 0.005

_MAX_RESULT_BYTES: Final[int] = 1 << 20
_MAX_TEXT_BYTES: Final[int] = 4096
_MAX_JSON_NODES: Final[int] = 20_000
_INT64_MAX: Final[int] = 2**63 - 1
_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdef")

_DISPATCHES_PER_SHARD: Final[int] = 2
_FAILURE_STAGES: Final[tuple[str, ...]] = (
    "before_dataset_load",
    "dataset_loaded",
    "first_dispatch",
)


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _require_execution_authorized() -> None:
    """Refuse campaign execution until a maintainer-approved transition lands."""

    if EXECUTION_AUTHORIZED is not True or AUTHORIZATION_TRANSITION_APPROVED is not True:
        raise PermissionError(
            "noise-curvature campaign execution is not authorized in this source revision"
        )


def _authorization_identity() -> dict[str, bool]:
    return {
        "execution_authorized": EXECUTION_AUTHORIZED,
        "authorization_transition_approved": AUTHORIZATION_TRANSITION_APPROVED,
    }


@dataclass(frozen=True)
class NoiseCurvatureCampaignPlan:
    """Host-validated frozen configuration for the prospective campaign."""

    seeds: tuple[int, ...]
    config: IPMNISTConfig
    live_control: str

    def __post_init__(self) -> None:
        if type(self.seeds) is not tuple or not 3 <= len(self.seeds) <= 128:
            raise ValueError("seeds must be a bounded exact tuple of at least three seeds")
        seeds = tuple(require_jax_seed(seed, name="seed") for seed in self.seeds)
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        if set(seeds) & set(DEVELOPMENT_SEEDS):
            raise ValueError("campaign seeds must not reuse the consumed public roots 0-4")
        if type(self.config) is not IPMNISTConfig:
            raise ValueError("config must be an exact IPMNISTConfig")
        config = IPMNISTConfig(**self.config.to_config())
        if config.to_config() != IPMNISTConfig().to_config():
            raise ValueError("campaign geometry must remain the published protocol defaults")
        interval = int(registered_hyperparameters(registered_arms()[0])["control_interval"])
        if config.task_length % interval:
            raise ValueError("task_length must be divisible by the registered control interval")
        if type(self.live_control) is not str or self.live_control != LIVE_CONTROL:
            raise ValueError("live control must remain the registered RLS incumbent name")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "config", config)


FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN: Final[NoiseCurvatureCampaignPlan] = (
    NoiseCurvatureCampaignPlan(
        seeds=(1_567_001, 1_567_002, 1_567_003),
        config=IPMNISTConfig(n_tasks=200, task_length=5000),
        live_control=LIVE_CONTROL,
    )
)


def _canonical_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("value must be finite canonical JSON") from error
    if len(encoded) > _MAX_RESULT_BYTES:
        raise ValueError("value exceeds the byte ceiling")
    return encoded


def _json_preflight(value: object) -> None:
    nodes = 0
    stack: list[object] = [value]
    while stack:
        current = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ValueError("value exceeds the exact JSON node ceiling")
        current_type = type(current)
        if current_type is dict:
            mapping = cast(dict[object, object], current)
            if len(mapping) > 64:
                raise ValueError("value object exceeds the field ceiling")
            keys = list(mapping.keys())
            if any(type(key) is not str for key in keys):
                raise ValueError("value must be an exact JSON tree with exact string keys")
            stack.extend(mapping.values())
            stack.extend(keys)
        elif current_type is list:
            values = cast(list[object], current)
            if len(values) > 4096:
                raise ValueError("value list exceeds the item ceiling")
            stack.extend(values)
        elif current_type is str:
            try:
                text_bytes = len(cast(str, current).encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError("value text must be UTF-8") from error
            if text_bytes > _MAX_TEXT_BYTES:
                raise ValueError("value text exceeds the byte ceiling")
        elif current_type is int:
            if not -_INT64_MAX <= cast(int, current) <= _INT64_MAX:
                raise ValueError("value integer exceeds signed int64")
        elif current_type is float:
            if not math.isfinite(cast(float, current)):
                raise ValueError("value float must be finite")
        elif current_type is not bool and current is not None:
            raise ValueError("value must be an exact JSON tree")


def _fields(value: object, expected: tuple[str, ...], *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be an exact object")
    result = cast(dict[str, object], value)
    if len(result) != len(expected) or set(result) != set(expected):
        raise ValueError(f"{name} fields do not match the exact schema")
    return result


def _same(actual: object, expected: object) -> bool:
    """Type-exact equality for already-preflighted JSON subtrees."""

    return _canonical_bytes(actual) == _canonical_bytes(expected)


def _digest(value: object) -> str:
    _json_preflight(value)
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def digest_without(payload: Mapping[str, object], field: str) -> str:
    """Hash one JSON object after removing its self-digest field."""

    if type(payload) is not dict or type(field) is not str:
        raise TypeError("payload and field must be exact object/string values")
    return _digest({key: value for key, value in payload.items() if key != field})


def _is_digest(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _HEX_DIGITS


def _exact_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    return value


def _unit_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite float")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return value


def _nonnegative_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite float")
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_paths() -> tuple[Path, ...]:
    return (
        Path("alberta_framework/_scan_resources.py"),
        Path("alberta_framework/_seed_validation.py"),
        Path("alberta_framework/benchmarks/ipmnist_screening.py"),
        Path("alberta_framework/benchmarks/noise_curvature_ipmnist.py"),
        Path("alberta_framework/benchmarks/upgd_ipmnist.py"),
        Path("alberta_framework/core/baseline_optimizers.py"),
        Path("alberta_framework/core/update_safety.py"),
        Path("alberta_framework/evaluation/noise_curvature_campaign.py"),
        Path("alberta_framework/evaluation/noise_curvature_ipmnist_nonpromoting.py"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    )


def _source_identity() -> dict[str, str]:
    root = _repository_root()
    return {
        path.as_posix(): hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in _source_paths()
    }


def _text(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > _MAX_TEXT_BYTES
        or len(value.encode("utf-8")) > _MAX_TEXT_BYTES
    ):
        raise ValueError(f"{name} must be bounded non-empty text")
    return value


def _runtime_identity() -> dict[str, object]:
    devices = sorted(
        [
            {
                "id": int(device.id),
                "process_index": int(device.process_index),
                "platform": _text(str(device.platform), name="device platform"),
                "kind": _text(str(device.device_kind), name="device kind"),
            }
            for device in jax.devices()
        ],
        key=lambda item: (
            item["process_index"],
            item["id"],
            item["platform"],
            item["kind"],
        ),
    )
    if not 1 <= len(devices) <= 128:
        raise ValueError("runtime device inventory exceeds its bound")
    environment: dict[str, object] = {}
    for name in (
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_ENABLE_X64",
        "JAX_DEFAULT_PRNG_IMPL",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_RANDOM_SEED_OFFSET",
        "JAX_NUM_CPU_DEVICES",
        "XLA_FLAGS",
    ):
        raw = os.environ.get(name)
        environment[name] = None if raw is None else _text(raw, name=name)
    return {
        "schema": "asi.noise-curvature-ipmnist.campaign-runtime.v1",
        "python_implementation": _text(platform.python_implementation(), name="python"),
        "python_version": list(sys.version_info[:3]),
        "byteorder": _text(sys.byteorder, name="byteorder"),
        "platform": _text(sys.platform, name="platform"),
        "machine": _text(platform.machine(), name="machine"),
        "packages": {
            "chex": _text(importlib.metadata.version("chex"), name="chex"),
            "jax": _text(jax.__version__, name="jax"),
            "jaxlib": _text(importlib.metadata.version("jaxlib"), name="jaxlib"),
            "numpy": _text(np.__version__, name="numpy"),
        },
        "backend": _text(jax.default_backend(), name="backend"),
        "devices": devices,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_prng_impl": _text(str(jax.config.jax_default_prng_impl), name="default PRNG"),
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "jax_numpy_dtype_promotion": _text(
            str(jax.config.jax_numpy_dtype_promotion), name="dtype promotion"
        ),
        "jax_numpy_rank_promotion": _text(
            str(jax.config.jax_numpy_rank_promotion), name="rank promotion"
        ),
        "environment": environment,
    }


def _dependency_identity() -> dict[str, object]:
    root = _repository_root()
    return {
        "schema": "asi.noise-curvature-ipmnist.campaign-dependencies.v1",
        "packages": {
            name: _text(importlib.metadata.version(name), name=name)
            for name in ("chex", "jax", "jaxlib", "numpy")
        },
        "uv_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
    }


def _process_identity() -> dict[str, object]:
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    stat_path = Path("/proc/self/stat")
    if not boot_path.is_file() or not stat_path.is_file():
        raise RuntimeError("the campaign contract requires Linux /proc identity")
    boot_id = boot_path.read_text(encoding="ascii").strip()
    stat_text = stat_path.read_text(encoding="ascii")
    comm_end = stat_text.rfind(")")
    stat_fields = stat_text[comm_end + 1 :].split() if comm_end >= 0 else []
    if len(stat_fields) < 20 or not stat_fields[19].isdigit():
        raise RuntimeError("cannot resolve Linux process start identity")
    process: dict[str, object] = {
        "schema": PROCESS_SCHEMA,
        "pid": os.getpid(),
        "proc_start_ticks": int(stat_fields[19]),
        "boot_id_sha256": hashlib.sha256(boot_id.encode("ascii")).hexdigest(),
        "invocation_nonce": secrets.token_hex(16),
        "fresh_process_required": True,
        "identity_is_not_attestation": True,
    }
    process["execution_instance_id"] = _digest(process)
    return process


def _validate_process(value: object) -> None:
    process = _fields(
        value,
        (
            "schema",
            "pid",
            "proc_start_ticks",
            "boot_id_sha256",
            "invocation_nonce",
            "fresh_process_required",
            "identity_is_not_attestation",
            "execution_instance_id",
        ),
        name="process identity",
    )
    if process["schema"] != PROCESS_SCHEMA:
        _fail("process identity schema drifted")
    for field in ("pid", "proc_start_ticks"):
        if type(process[field]) is not int or not 0 <= cast(int, process[field]) <= _INT64_MAX:
            _fail(f"process {field} must be a bounded nonnegative integer")
    if not _is_digest(process["boot_id_sha256"]) or not _is_digest(
        process["execution_instance_id"]
    ):
        _fail("process identity digests are invalid")
    _text(process["invocation_nonce"], name="invocation nonce")
    if process["fresh_process_required"] is not True:
        _fail("fresh process requirement is not asserted")
    if process["identity_is_not_attestation"] is not True:
        _fail("process identity must not claim attestation")
    unsigned = {key: value for key, value in process.items() if key != "execution_instance_id"}
    if process["execution_instance_id"] != _digest(unsigned):
        _fail("process execution instance digest drifted")


@cache
def campaign_schedule_init_identities(seed: int) -> tuple[str, str]:
    """Re-derive the exact per-seed schedule/init identities, dataset-free.

    The derivation mirrors ``run_screening_config`` exactly: one explicit
    ``threefry2x32`` root per seed, split into the runner's init/schedule/noise
    streams.  Only the geometry and the frozen train-row count enter, so the
    digests are pure functions of the frozen plan and never touch data.
    """

    seed = require_jax_seed(seed, name="seed")
    if seed not in FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN.seeds:
        raise ValueError("seed is outside the frozen campaign roster")
    config = FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN.config
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
    schedule = build_schedule(key_schedule, config, N_TRAIN_ROWS)
    schedule_digest = hashlib.sha256(b"asi.noise-curvature-ipmnist.campaign-schedule.v1\0")
    for name in ("permutations", "example_indices"):
        raw = np.asarray(jax.device_get(getattr(schedule, name)))
        schedule_digest.update(name.encode("ascii"))
        schedule_digest.update(raw.dtype.str.encode("ascii"))
        schedule_digest.update(str(raw.shape).encode("ascii"))
        schedule_digest.update(raw.tobytes(order="C"))
    return init_digest.hexdigest(), schedule_digest.hexdigest()


def _expected_shard_accounting() -> dict[str, int]:
    """Exact per-shard counters and persistent bytes derived from the plan."""

    config = FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN.config
    hyperparameters = registered_hyperparameters(registered_arms()[0])
    interval = int(hyperparameters["control_interval"])
    power_iterations = int(hyperparameters["power_iterations"])
    observations = config.n_steps
    controller_events = observations // interval
    gradient_queries = observations + controller_events * interval
    loss_queries = observations
    hvp_queries = controller_events * 3 * power_iterations
    return {
        "observations": observations,
        "updates": observations,
        "data_steps": observations,
        "environment_steps": 0,
        "model_queries": gradient_queries + loss_queries + hvp_queries,
        "first_order_gradient_queries": gradient_queries,
        "loss_only_queries": loss_queries,
        "hessian_vector_product_queries": hvp_queries,
        "controller_events": controller_events,
        "persistent_bytes": noise_curvature_persistent_bytes(
            parameter_count=config.parameter_count,
            input_dim=config.input_dim,
            control_interval=interval,
        ),
    }


def _gate_rules() -> dict[str, object]:
    return {
        "metric": "mean_online_accuracy",
        "confidence_level": 0.95,
        "interval_rule": "mean +/- t_critical * sample_sd / sqrt(n)",
        "t_critical": PAIRED_T_CRITICAL,
        "t_critical_degrees_of_freedom": len(FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN.seeds) - 1,
        "mechanism": "combined minus fixed-Adam+L2 paired interval strictly above zero",
        "causal": (
            "combined minus gradient-only and combined minus volatility-only paired "
            "intervals both strictly above zero"
        ),
        "hillclimb": (
            "combined minus live-control paired mean at least +0.005 and its paired "
            "interval strictly above zero; the live control is separately protocolled"
        ),
        "hillclimb_min_mean_delta": HILLCLIMB_MIN_MEAN_DELTA,
        "outcome_vocabulary": ["supported", "rejected", "inconclusive"],
    }


def _plan_payload(plan: NoiseCurvatureCampaignPlan) -> dict[str, object]:
    checked = NoiseCurvatureCampaignPlan(
        seeds=plan.seeds, config=plan.config, live_control=plan.live_control
    )
    arms = registered_arms()
    accounting = _expected_shard_accounting()
    return {
        "schema": PLAN_SCHEMA,
        "issue": 1567,
        "seeds": list(checked.seeds),
        "arms": list(arms),
        "hyperparameters": {arm: registered_hyperparameters(arm) for arm in arms},
        "geometry": checked.config.to_config(),
        "live_control": {
            "name": checked.live_control,
            "separately_protocolled": True,
            "validated_by": "the screening campaign's own control receipt protocol",
            "required_for": "the frozen hillclimb gate only",
        },
        "matched_axes": [
            "seed",
            "dataset",
            "schedule",
            "initialization",
            "observations",
            "hyperparameters",
            "resource accounting",
        ],
        "expected_resources_per_shard": accounting,
        "schedule_init_identities": {
            str(seed): {
                "init_sha256": campaign_schedule_init_identities(seed)[0],
                "schedule_sha256": campaign_schedule_init_identities(seed)[1],
            }
            for seed in checked.seeds
        },
        "dataset": {
            "selection": DATASET_SELECTION,
            "rows": N_TRAIN_ROWS,
            "columns": checked.config.input_dim,
            "features_dtype": "float32",
            "labels_dtype": "int32",
            "sha256_domain": "asi.noise-curvature-ipmnist.campaign-dataset.v1",
            "sha256": DATASET_DIGEST,
            "digest_binding": (
                "sha256 over the domain tag then dtype, shape, and C-order bytes of the "
                "scaled features and int32 labels; every shard and both execution "
                "dispatches must bind this exact digest"
            ),
        },
        "dispatch_accounting": {
            "scheduler_shards": len(checked.seeds) * len(arms),
            "separately_protocolled_control_shards": len(checked.seeds),
            "execution_dispatches_per_shard": _DISPATCHES_PER_SHARD,
            "dispatches": (
                "initial execution followed by one independent dataset-bound strict "
                "reexecution in a fresh process"
            ),
            "dataset_loads_per_shard_process": 1,
            "reservation_required_before_dataset_or_rng_work": True,
        },
        "gate_rules": _gate_rules(),
        "output_namespace": OUTPUT_NAMESPACE,
        "authorization": _authorization_identity(),
        "policy": {
            "status": "development-only-nonpromoting",
            "development_only": True,
            "scientific_promotion_allowed": False,
            "negative_outcomes_retained": True,
            "timing_is_telemetry_only": True,
        },
    }


FROZEN_PLAN_SHA256: Final[str] = "0dfbb580f29a6651bce702cf48ca088cfdd8001cb50b324931463cf00bad9064"


def frozen_plan_payload() -> dict[str, object]:
    """Return the literal campaign plan only while its preregistered digest holds."""

    payload = _plan_payload(FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN)
    observed = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if observed != FROZEN_PLAN_SHA256:
        raise RuntimeError("frozen noise-curvature campaign plan drifted from its digest")
    return payload


def _current_identity() -> dict[str, object]:
    return {
        "source_sha256": _source_identity(),
        "runtime": _runtime_identity(),
        "dependencies": _dependency_identity(),
        "authorization": _authorization_identity(),
        "consistency_not_attestation": True,
    }


def build_campaign_plan_document() -> dict[str, object]:
    """Build the source-bound literal plan document without dataset or RNG work."""

    payload = frozen_plan_payload()
    document: dict[str, object] = {
        "schema": PLAN_SCHEMA,
        "plan": payload,
        "plan_sha256": _digest(payload),
        "identity": _current_identity(),
        "policy": dict(cast(Mapping[str, object], payload["policy"])),
    }
    document["document_sha256"] = _digest(document)
    validate_campaign_plan_document(document)
    return document


def validate_campaign_plan_document(value: object) -> dict[str, object]:
    """Validate exact structure, current identities, and the pinned plan literal."""

    _json_preflight(value)
    root = _fields(
        value,
        ("schema", "plan", "plan_sha256", "identity", "policy", "document_sha256"),
        name="campaign plan document",
    )
    expected_plan = frozen_plan_payload()
    if root["schema"] != PLAN_SCHEMA or not _same(root["plan"], expected_plan):
        _fail("campaign plan document does not carry the current literal plan")
    if root["plan_sha256"] != _digest(expected_plan):
        _fail("campaign plan document plan digest drifted")
    if not _same(root["identity"], _current_identity()):
        _fail("campaign plan document identity drifted from the current source or runtime")
    if not _same(root["policy"], expected_plan["policy"]):
        _fail("campaign plan document policy drifted from the frozen plan")
    if root["document_sha256"] != digest_without(root, "document_sha256"):
        _fail("campaign plan document digest drifted")
    if len(_canonical_bytes(root)) > _MAX_RESULT_BYTES:
        _fail("campaign plan document exceeds the byte ceiling")
    return root


_METRIC_FIELDS: Final[tuple[str, ...]] = (
    "mean_online_accuracy",
    "mean_loss",
    "mean_plasticity",
)
_RESOURCE_FIELDS: Final[tuple[str, ...]] = (
    "observations",
    "updates",
    "data_steps",
    "environment_steps",
    "model_queries",
    "first_order_gradient_queries",
    "loss_only_queries",
    "hessian_vector_product_queries",
    "controller_events",
    "persistent_bytes",
    "timing_seconds",
    "timing_is_telemetry_only",
)
_REEXECUTION_FIELDS: Final[tuple[str, ...]] = (
    "required",
    "dispatches",
    "dataset_digest_equal",
    "metrics_exact_equal",
    "counters_exact_equal",
    "timing_retained",
)


def validate_noise_curvature_campaign_shard(value: object) -> dict[str, object]:
    """Strictly validate one future campaign shard receipt.

    No shard can exist while execution stays unauthorized: this validator binds
    structure, the frozen plan, the canonical dataset digest, the per-seed
    schedule/init identities, exact resource accounting, and the required
    independent dataset-bound strict-reexecution binding.  It is a consistency
    check, never authenticated execution attestation.
    """

    _json_preflight(value)
    root = _fields(
        value,
        (
            "schema",
            "status",
            "plan_sha256",
            "arm",
            "seed",
            "hyperparameters",
            "metrics",
            "resources",
            "identity",
            "reexecution",
            "outcome",
            "outcome_retained",
            "development_only",
            "scientific_promotion_allowed",
            "shard_sha256",
        ),
        name="campaign shard",
    )
    if root["schema"] != SHARD_SCHEMA or root["status"] != "complete":
        _fail("campaign shard schema or status drifted")
    if root["plan_sha256"] != _digest(frozen_plan_payload()):
        _fail("campaign shard plan digest drifted")
    arm = root["arm"]
    if type(arm) is not str or arm not in registered_arms():
        _fail("campaign shard arm is outside the frozen roster")
    seed = _exact_int(root["seed"], name="seed")
    if seed not in FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN.seeds:
        _fail("campaign shard seed is outside the frozen fresh-seed roster")
    if not _same(root["hyperparameters"], registered_hyperparameters(arm)):
        _fail("campaign shard hyperparameters do not match the registered arm")
    metrics = _fields(root["metrics"], _METRIC_FIELDS, name="shard metrics")
    _unit_float(metrics["mean_online_accuracy"], name="mean_online_accuracy")
    _nonnegative_float(metrics["mean_loss"], name="mean_loss")
    _unit_float(metrics["mean_plasticity"], name="mean_plasticity")
    expected_resources = cast(
        dict[str, object], frozen_plan_payload()["expected_resources_per_shard"]
    )
    resources = _fields(root["resources"], _RESOURCE_FIELDS, name="shard resources")
    for field in _RESOURCE_FIELDS:
        if field in ("timing_seconds", "timing_is_telemetry_only"):
            continue
        if type(resources[field]) is not int or resources[field] != expected_resources[field]:
            _fail(f"campaign shard resource counter {field} does not match the frozen plan")
    timing = _nonnegative_float(resources["timing_seconds"], name="timing_seconds")
    if timing > 604_800.0:
        _fail("timing_seconds exceeds the seven-day development bound")
    if resources["timing_is_telemetry_only"] is not True:
        _fail("shard timing must permanently remain telemetry-only")
    identity = _fields(
        root["identity"],
        (
            "source_sha256",
            "runtime",
            "dependencies",
            "authorization",
            "dataset_sha256",
            "schedule_sha256",
            "init_sha256",
            "process",
            "consistency_not_attestation",
        ),
        name="shard identity",
    )
    if not _same(identity["source_sha256"], _source_identity()):
        _fail("campaign shard source identity drifted from the current tree")
    if identity["dataset_sha256"] != DATASET_DIGEST:
        _fail("campaign shard dataset digest is not the canonical frozen digest")
    init_sha, schedule_sha = campaign_schedule_init_identities(seed)
    if identity["init_sha256"] != init_sha or identity["schedule_sha256"] != schedule_sha:
        _fail("campaign shard schedule/init identity does not derive from the seed")
    _json_preflight(identity["runtime"])
    _json_preflight(identity["dependencies"])
    if not _same(identity["authorization"], _authorization_identity()):
        _fail("campaign shard authorization identity does not match this revision")
    _validate_process(identity["process"])
    if identity["consistency_not_attestation"] is not True:
        _fail("campaign shard must not claim execution attestation")
    reexecution = _fields(root["reexecution"], _REEXECUTION_FIELDS, name="shard reexecution")
    if not _same(
        reexecution,
        {
            "required": True,
            "dispatches": _DISPATCHES_PER_SHARD,
            "dataset_digest_equal": True,
            "metrics_exact_equal": True,
            "counters_exact_equal": True,
            "timing_retained": False,
        },
    ):
        _fail("campaign shard does not bind the independent dataset-bound reexecution")
    if type(root["outcome"]) is not str or root["outcome"] not in (
        "supported",
        "rejected",
        "inconclusive",
    ):
        _fail("campaign shard outcome is outside the frozen vocabulary")
    if root["outcome_retained"] is not True:
        _fail("campaign shard outcome must be retained")
    if root["development_only"] is not True:
        _fail("campaign shard must remain development-only")
    if root["scientific_promotion_allowed"] is not False:
        _fail("campaign shard must remain permanently nonpromoting")
    if root["shard_sha256"] != digest_without(root, "shard_sha256"):
        _fail("campaign shard digest drifted")
    if len(_canonical_bytes(root)) > _MAX_RESULT_BYTES:
        _fail("campaign shard exceeds the byte ceiling")
    return root


def validate_noise_curvature_control_row(value: object) -> dict[str, object]:
    """Validate one minimal separately-protocolled live-control receipt row."""

    _json_preflight(value)
    root = _fields(
        value,
        (
            "schema",
            "arm",
            "seed",
            "mean_online_accuracy",
            "dataset_sha256",
            "receipt_sha256",
            "separately_protocolled",
        ),
        name="control row",
    )
    if root["schema"] != CONTROL_ROW_SCHEMA:
        _fail("control row schema drifted")
    if root["arm"] != LIVE_CONTROL:
        _fail("control row arm is not the registered RLS incumbent")
    seed = _exact_int(root["seed"], name="seed")
    if seed not in FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN.seeds:
        _fail("control row seed is outside the frozen fresh-seed roster")
    _unit_float(root["mean_online_accuracy"], name="mean_online_accuracy")
    if root["dataset_sha256"] != DATASET_DIGEST:
        _fail("control row dataset digest is not the canonical frozen digest")
    if not _is_digest(root["receipt_sha256"]):
        _fail("control row receipt digest is invalid")
    if root["separately_protocolled"] is not True:
        _fail("control row must declare separate protocol binding")
    return root


def _paired_interval(deltas: list[float]) -> tuple[float, float]:
    values = np.asarray(deltas, dtype=np.float64)
    mean = float(np.mean(values))
    if len(deltas) == 1:
        return mean, mean
    stderr = float(np.std(values, ddof=1) / math.sqrt(len(deltas)))
    return mean - PAIRED_T_CRITICAL * stderr, mean + PAIRED_T_CRITICAL * stderr


def _gate_outcome(lower: float, upper: float, *, min_mean_delta: float, mean: float) -> str:
    if lower > 0.0 and mean >= min_mean_delta:
        return "supported"
    if upper < min_mean_delta:
        return "rejected"
    return "inconclusive"


def _gate_record(name: str, deltas: list[float], *, min_mean_delta: float) -> dict[str, object]:
    lower, upper = _paired_interval(deltas)
    mean = float(np.mean(np.asarray(deltas, dtype=np.float64)))
    return {
        "gate": name,
        "metric": "mean_online_accuracy",
        "n": len(deltas),
        "paired_deltas": deltas,
        "delta_mean": mean,
        "interval_lower": lower,
        "interval_upper": upper,
        "min_mean_delta": min_mean_delta,
        "outcome": _gate_outcome(lower, upper, min_mean_delta=min_mean_delta, mean=mean),
    }


def summarize_noise_curvature_campaign(
    shards: object, control_rows: object
) -> dict[str, object]:
    """Build one strictly validated frozen-gate aggregate from shard receipts.

    Takes every scheduler shard receipt (validated by
    :func:`validate_noise_curvature_campaign_shard`) plus the minimal
    separately-protocolled live-control rows, checks the matched axes, and
    computes the frozen mechanism, causal, and hillclimb confidence gates.
    """

    _require_execution_authorized()
    plan = FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN
    arms = registered_arms()
    if type(shards) is not list or len(shards) != len(plan.seeds) * len(arms):
        _fail("aggregate requires one shard per arm and seed")
    if type(control_rows) is not list or len(control_rows) != len(plan.seeds):
        _fail("aggregate requires one control row per seed")
    validated = [validate_noise_curvature_campaign_shard(shard) for shard in shards]
    controls = [validate_noise_curvature_control_row(row) for row in control_rows]
    by_key: dict[tuple[str, int], dict[str, object]] = {}
    for shard in validated:
        key = (cast(str, shard["arm"]), cast(int, shard["seed"]))
        if key in by_key:
            _fail("aggregate contains a duplicate arm/seed shard")
        by_key[key] = shard
    if set(by_key) != {(arm, seed) for arm in arms for seed in plan.seeds}:
        _fail("aggregate does not cover the exact arm/seed roster")
    if {cast(int, row["seed"]) for row in controls} != set(plan.seeds):
        _fail("aggregate control rows do not cover the exact seed roster")
    first = validated[0]
    first_identity = cast(dict[str, object], first["identity"])
    for shard in validated[1:]:
        identity = cast(dict[str, object], shard["identity"])
        for field in ("runtime", "dependencies", "dataset_sha256"):
            if not _same(identity[field], first_identity[field]):
                _fail(f"aggregate shards differ on identity axis {field}")
    if any(
        not _same(row["dataset_sha256"], first_identity["dataset_sha256"])
        for row in controls
    ):
        _fail("aggregate control rows bind a different dataset digest")
    accuracy: dict[tuple[str, int], float] = {}
    for shard in validated:
        metrics = cast(dict[str, object], shard["metrics"])
        accuracy[(cast(str, shard["arm"]), cast(int, shard["seed"]))] = cast(
            float, metrics["mean_online_accuracy"]
        )
    control_accuracy: dict[int, float] = {}
    for row in controls:
        control_accuracy[cast(int, row["seed"])] = cast(
            float, row["mean_online_accuracy"]
        )
    panels = []
    mechanism_deltas: list[float] = []
    causal_gradient_deltas: list[float] = []
    causal_volatility_deltas: list[float] = []
    hillclimb_deltas: list[float] = []
    for seed in plan.seeds:
        panel: dict[str, object] = {
            "seed": seed,
            "arms": {arm: accuracy[(arm, seed)] for arm in arms},
            "live_control": control_accuracy[seed],
        }
        panels.append(panel)
        mechanism_deltas.append(accuracy[("noise_curvature_combined", seed)] - accuracy[
            ("noise_curvature_fixed_adam_l2", seed)
        ])
        causal_gradient_deltas.append(
            accuracy[("noise_curvature_combined", seed)]
            - accuracy[("noise_curvature_gradient_only", seed)]
        )
        causal_volatility_deltas.append(
            accuracy[("noise_curvature_combined", seed)]
            - accuracy[("noise_curvature_volatility_only", seed)]
        )
        hillclimb_deltas.append(
            accuracy[("noise_curvature_combined", seed)] - control_accuracy[seed]
        )
    gates = {
        "mechanism": _gate_record("mechanism", mechanism_deltas, min_mean_delta=0.0),
        "causal_gradient": _gate_record(
            "causal_gradient", causal_gradient_deltas, min_mean_delta=0.0
        ),
        "causal_volatility": _gate_record(
            "causal_volatility", causal_volatility_deltas, min_mean_delta=0.0
        ),
        "hillclimb": _gate_record(
            "hillclimb", hillclimb_deltas, min_mean_delta=HILLCLIMB_MIN_MEAN_DELTA
        ),
    }
    aggregate: dict[str, object] = {
        "schema": AGGREGATE_SCHEMA,
        "plan_sha256": _digest(frozen_plan_payload()),
        "identity": _current_identity(),
        "shards": [cast(str, shard["shard_sha256"]) for shard in validated],
        "control_rows": [_digest(row) for row in controls],
        "panels": panels,
        "gates": gates,
        "policy": dict(cast(Mapping[str, object], frozen_plan_payload()["policy"])),
    }
    aggregate["aggregate_sha256"] = digest_without(aggregate, "aggregate_sha256")
    validate_noise_curvature_campaign_aggregate(aggregate)
    return aggregate


def validate_noise_curvature_campaign_aggregate(value: object) -> dict[str, object]:
    """Validate one aggregate receipt and re-derive every frozen gate."""

    _json_preflight(value)
    root = _fields(
        value,
        (
            "schema",
            "plan_sha256",
            "identity",
            "shards",
            "control_rows",
            "panels",
            "gates",
            "policy",
            "aggregate_sha256",
        ),
        name="campaign aggregate",
    )
    if root["schema"] != AGGREGATE_SCHEMA:
        _fail("campaign aggregate schema drifted")
    if root["plan_sha256"] != _digest(frozen_plan_payload()):
        _fail("campaign aggregate plan digest drifted")
    if not _same(root["identity"], _current_identity()):
        _fail("campaign aggregate identity drifted from the current source or runtime")
    plan = FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN
    arms = registered_arms()
    if type(root["shards"]) is not list or len(root["shards"]) != len(plan.seeds) * len(arms):
        _fail("campaign aggregate shard roster is incomplete")
    for digest_value in root["shards"]:
        if not _is_digest(digest_value):
            _fail("campaign aggregate shard digest is invalid")
    if len(set(cast(list[object], root["shards"]))) != len(cast(list[object], root["shards"])):
        _fail("campaign aggregate binds a duplicate shard")
    if type(root["control_rows"]) is not list or len(root["control_rows"]) != len(plan.seeds):
        _fail("campaign aggregate control roster is incomplete")
    for digest_value in root["control_rows"]:
        if not _is_digest(digest_value):
            _fail("campaign aggregate control digest is invalid")
    panels = root["panels"]
    if type(panels) is not list or len(panels) != len(plan.seeds):
        _fail("campaign aggregate panels are incomplete")
    mechanism_deltas: list[float] = []
    causal_gradient_deltas: list[float] = []
    causal_volatility_deltas: list[float] = []
    hillclimb_deltas: list[float] = []
    for index, (panel_raw, seed) in enumerate(zip(cast(list[object], panels), plan.seeds)):
        panel = _fields(
            panel_raw, ("seed", "arms", "live_control"), name=f"panels[{index}]"
        )
        if panel["seed"] != seed:
            _fail("campaign aggregate panel ordering does not match the plan")
        arm_values = _fields(panel["arms"], tuple(arms), name="panel arms")
        values = {
            arm: _unit_float(arm_values[arm], name=f"panel arm {arm}") for arm in arms
        }
        control = _unit_float(panel["live_control"], name="panel live control")
        mechanism_deltas.append(
            values["noise_curvature_combined"] - values["noise_curvature_fixed_adam_l2"]
        )
        causal_gradient_deltas.append(
            values["noise_curvature_combined"] - values["noise_curvature_gradient_only"]
        )
        causal_volatility_deltas.append(
            values["noise_curvature_combined"] - values["noise_curvature_volatility_only"]
        )
        hillclimb_deltas.append(values["noise_curvature_combined"] - control)
    expected_gates = {
        "mechanism": _gate_record("mechanism", mechanism_deltas, min_mean_delta=0.0),
        "causal_gradient": _gate_record(
            "causal_gradient", causal_gradient_deltas, min_mean_delta=0.0
        ),
        "causal_volatility": _gate_record(
            "causal_volatility", causal_volatility_deltas, min_mean_delta=0.0
        ),
        "hillclimb": _gate_record(
            "hillclimb", hillclimb_deltas, min_mean_delta=HILLCLIMB_MIN_MEAN_DELTA
        ),
    }
    if not _same(root["gates"], expected_gates):
        _fail("campaign aggregate gates are not derived from the panels")
    if not _same(root["policy"], frozen_plan_payload()["policy"]):
        _fail("campaign aggregate policy drifted from the frozen plan")
    if root["aggregate_sha256"] != digest_without(root, "aggregate_sha256"):
        _fail("campaign aggregate digest drifted")
    if len(_canonical_bytes(root)) > _MAX_RESULT_BYTES:
        _fail("campaign aggregate exceeds the byte ceiling")
    return root


def build_failed_campaign_dispatch(
    arm: str, seed: int, stage: str
) -> dict[str, object]:
    """Build one durable record for a failed first dispatch attempt.

    This is host-side bookkeeping only: it never executes a runner step and
    retains no exception bytes, so a failed attempt cannot leak environment
    detail while still leaving a durable, digest-bound disposition.
    """

    roster = (*registered_arms(), LIVE_CONTROL)
    if type(arm) is not str or arm not in roster:
        _fail("failed dispatch arm is outside the frozen roster")
    if type(seed) is not int or seed not in FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN.seeds:
        _fail("failed dispatch seed is outside the frozen fresh-seed roster")
    if type(stage) is not str or stage not in _FAILURE_STAGES:
        _fail("failed dispatch stage is outside the frozen stages")
    payload: dict[str, object] = {
        "schema": FAILED_DISPATCH_SCHEMA,
        "status": "failed",
        "plan_sha256": _digest(frozen_plan_payload()),
        "spec": {"arm": arm, "seed": seed},
        "stage": stage,
        "identity": {**_current_identity(), "process": _process_identity()},
        "policy": {
            **_authorization_identity(),
            "development_only": True,
            "scientific_promotion_allowed": False,
            "used_for_outcome": False,
            "retry_authorized": False,
        },
        "failure": {
            "classification": "first_dispatch_failure",
            "stage_precedes_dataset_work": stage == "before_dataset_load",
            "exception_type_retained": False,
            "exception_message_retained": False,
            "exception_repr_retained": False,
            "base_exception_retention_guaranteed": False,
        },
    }
    payload["failure_sha256"] = digest_without(payload, "failure_sha256")
    validate_failed_campaign_dispatch(payload)
    return payload


def validate_failed_campaign_dispatch(value: object) -> dict[str, object]:
    """Strictly validate one durable failed-dispatch disposition record."""

    _json_preflight(value)
    root = _fields(
        value,
        (
            "schema",
            "status",
            "plan_sha256",
            "spec",
            "stage",
            "identity",
            "policy",
            "failure",
            "failure_sha256",
        ),
        name="failed dispatch",
    )
    if root["schema"] != FAILED_DISPATCH_SCHEMA or root["status"] != "failed":
        _fail("failed dispatch schema or status drifted")
    if root["plan_sha256"] != _digest(frozen_plan_payload()):
        _fail("failed dispatch plan digest drifted")
    spec = _fields(root["spec"], ("arm", "seed"), name="failed dispatch spec")
    roster = (*registered_arms(), LIVE_CONTROL)
    if type(spec["arm"]) is not str or spec["arm"] not in roster:
        _fail("failed dispatch arm is outside the frozen roster")
    if type(spec["seed"]) is not int or spec["seed"] not in (
        FROZEN_NOISE_CURVATURE_CAMPAIGN_PLAN.seeds
    ):
        _fail("failed dispatch seed is outside the frozen fresh-seed roster")
    if type(root["stage"]) is not str or root["stage"] not in _FAILURE_STAGES:
        _fail("failed dispatch stage is outside the frozen stages")
    identity = _fields(
        root["identity"],
        (
            "source_sha256",
            "runtime",
            "dependencies",
            "authorization",
            "consistency_not_attestation",
            "process",
        ),
        name="failed dispatch identity",
    )
    if not _same(
        {key: value for key, value in identity.items() if key != "process"},
        _current_identity(),
    ):
        _fail("failed dispatch source or runtime identity drifted")
    _validate_process(identity["process"])
    if not _same(
        root["policy"],
        {
            **_authorization_identity(),
            "development_only": True,
            "scientific_promotion_allowed": False,
            "used_for_outcome": False,
            "retry_authorized": False,
        },
    ):
        _fail("failed dispatch policy drifted")
    if not _same(
        root["failure"],
        {
            "classification": "first_dispatch_failure",
            "stage_precedes_dataset_work": root["stage"] == "before_dataset_load",
            "exception_type_retained": False,
            "exception_message_retained": False,
            "exception_repr_retained": False,
            "base_exception_retention_guaranteed": False,
        },
    ):
        _fail("failed dispatch disclosure boundary drifted")
    if root["failure_sha256"] != digest_without(root, "failure_sha256"):
        _fail("failed dispatch digest drifted")
    if len(_canonical_bytes(root)) > _MAX_RESULT_BYTES:
        _fail("failed dispatch exceeds the byte ceiling")
    return root


def reserve_campaign_plan_document(*, repository_root: Path) -> Path:
    """Publish one immutable plan reservation through no-follow directory FDs.

    Reservation precedes any dataset, RNG, or runner work by construction: this
    function only hashes the current source tree and writes canonical bytes.
    The publication is no-replace (``O_EXCL`` plus hard-link), read-only, and
    strict-reloads through the validator before returning.
    """

    if type(repository_root) is not PosixPath or not repository_root.is_absolute():
        raise ValueError("repository_root must be an exact absolute POSIX Path")
    document = build_campaign_plan_document()
    encoded = _canonical_bytes(document)
    digest_value = cast(str, document["document_sha256"])
    segments = tuple(Path(OUTPUT_NAMESPACE).parts)
    destination = repository_root.joinpath(*segments) / f"plan.{digest_value}.json"
    temporary_name = f".plan.{digest_value}.tmp"
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_descriptor = os.open(repository_root, directory_flags)
    try:
        for segment in segments:
            try:
                os.mkdir(segment, mode=0o755, dir_fd=directory_descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(segment, directory_flags, dir_fd=directory_descriptor)
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o444,
            dir_fd=directory_descriptor,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        finally:
            os.close(descriptor)
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        read_descriptor = os.open(
            destination.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_descriptor
        )
        try:
            with os.fdopen(read_descriptor, "rb", closefd=False) as stream:
                retained = stream.read(_MAX_RESULT_BYTES + 1)
        finally:
            os.close(read_descriptor)
        if retained != encoded:
            raise RuntimeError("reserved plan bytes changed during publication")
        loaded = json.loads(retained)
        if not _same(validate_campaign_plan_document(loaded), document):
            raise RuntimeError("reserved plan failed strict reload validation")
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return destination


def _load_strict_document(path: Path) -> dict[str, object]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot open regular non-symlink JSON input: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_RESULT_BYTES:
            raise ValueError("JSON input must be one bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(_MAX_RESULT_BYTES + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        len(raw) > _MAX_RESULT_BYTES
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError("JSON input changed while being read")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=lambda pairs: _reject_duplicate_pairs(pairs),
            parse_constant=lambda token: _fail(f"invalid JSON constant: {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("JSON input is not one strict document") from exc
    _json_preflight(value)
    if type(value) is not dict:
        _fail("JSON document root must be an exact object")
    return cast(dict[str, object], value)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect the hard-disabled campaign contract without executing anything."""

    parser = argparse.ArgumentParser(
        prog="asi-noise-curvature-campaign",
        description=(
            "Emit, validate, or reserve the permanently unauthorized noise-curvature "
            "development campaign contract (#1567). No command loads a dataset, "
            "derives RNG, or executes a runner step."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="print the current plan document")
    validate_plan = subparsers.add_parser("validate-plan", help="validate one plan document")
    validate_plan.add_argument("path", type=Path)
    validate_shard = subparsers.add_parser("validate-shard", help="validate one shard receipt")
    validate_shard.add_argument("path", type=Path)
    validate_aggregate = subparsers.add_parser(
        "validate-aggregate", help="validate one aggregate receipt"
    )
    validate_aggregate.add_argument("path", type=Path)
    failed_dispatch = subparsers.add_parser(
        "failed-dispatch", help="emit one durable failed-dispatch disposition"
    )
    failed_dispatch.add_argument("--arm", required=True)
    failed_dispatch.add_argument("--seed", type=int, required=True)
    failed_dispatch.add_argument("--stage", required=True)
    reserve_plan = subparsers.add_parser(
        "reserve-plan", help="publish the immutable plan reservation"
    )
    reserve_plan.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "plan":
        document = build_campaign_plan_document()
        print(_canonical_bytes(document).decode("utf-8"))
        return 0
    if args.command == "validate-plan":
        document = validate_campaign_plan_document(_load_strict_document(args.path))
        print(
            json.dumps(
                {
                    "plan_sha256": document["plan_sha256"],
                    "document_sha256": document["document_sha256"],
                }
            )
        )
        return 0
    if args.command == "validate-shard":
        shard = validate_noise_curvature_campaign_shard(_load_strict_document(args.path))
        print(json.dumps({"shard_sha256": shard["shard_sha256"]}))
        return 0
    if args.command == "validate-aggregate":
        aggregate = validate_noise_curvature_campaign_aggregate(
            _load_strict_document(args.path)
        )
        print(json.dumps({"aggregate_sha256": aggregate["aggregate_sha256"]}))
        return 0
    if args.command == "failed-dispatch":
        record = build_failed_campaign_dispatch(args.arm, args.seed, args.stage)
        print(_canonical_bytes(record).decode("utf-8"))
        return 0
    if args.command == "reserve-plan":
        root = args.root if args.root.is_absolute() else args.root.resolve()
        destination = reserve_campaign_plan_document(repository_root=root)
        print(str(destination))
        return 0
    raise AssertionError("argparse returned an unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
