"""Prospective optimization-readiness development lane (nonpromoting).

First consumer of :mod:`alberta_framework.evaluation.optimization_readiness`.
The lane freezes a bounded supervised stream (cumulative input-permuted
MNIST), takes a diagnostic measurement at every task start, and scores each
measurement against the realized future relative validation-loss reduction of
the paper's matched future-gain rollouts (Wang et al., arXiv:2605.09044v1,
Appendix C.1 sampling contract).

Two matched predictor arms are emitted per checkpoint: the full readiness
estimator (gradient strength times reliability) and its predeclared
gradient-strength-only mechanism-off reduction.  The rank, norm, gradient,
and curvature baselines ride along in every receipt.

This is a development-only diagnostic.  It is not the paper's experiment (the
paper scores RL/LLM checkpoints; this lane scores a small supervised MLP
stream), it claims no benchmark improvement, and every receipt is permanently
nonpromoting.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import tempfile
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.evaluation.optimization_readiness import (
    PROTOCOL_SCHEMA as RECEIPT_PROTOCOL_SCHEMA,
)
from alberta_framework.evaluation.optimization_readiness import (
    RESOURCE_SCHEMA as RECEIPT_RESOURCE_SCHEMA,
)
from alberta_framework.evaluation.optimization_readiness import (
    RESULT_SCHEMA as RECEIPT_RESULT_SCHEMA,
)
from alberta_framework.evaluation.optimization_readiness import (
    OptimizationReadiness,
    energy_rank,
    estimate_appendix_c1_optimization_readiness,
    validate_matched_development_results,
)

DEVELOPMENT_SCHEMA = "asi.optimization-readiness.development.v1"
PAPER_REVISION = "arXiv:2605.09044v1"
TASK_PROTOCOL = "cumulative-input-permutation"
FROZEN_SEEDS = (15840, 15841, 15842, 15843)
INPUT_DIM = 784
N_CLASSES = 10
N_VALIDATION_OBSERVATIONS = 10_000
DIAGNOSTIC_BATCH_COUNT = 128
DIAGNOSTIC_BATCH_SIZE = 4
FUTURE_GAIN_ROLLOUT_COUNT = 128
FUTURE_GAIN_STEPS = 1
FUTURE_GAIN_BATCH_SIZE = 4
FUTURE_GAIN_STEP_SIZE = 1e-3
ARM_READINESS_FULL = "readiness_full_strength_x_reliability"
ARM_GRADIENT_STRENGTH_ONLY = "gradient_strength_mechanism_off"
SAMPLING_PROVENANCE = (
    "caller_reported_independent_with_replacement_not_verified_from_gradients"
)
ALLOWED_BOUNDARY_INFORMATION: tuple[str, ...] = ("task_start",)
ALLOWED_TASK_INFORMATION: tuple[str, ...] = (
    "current_validation_inputs",
    "current_validation_labels",
)
_MAX_TRAIN_POOL = 200_000
_MAX_NPZ_MEMBER_BYTES = _MAX_TRAIN_POOL * (INPUT_DIM * 4 + 4) + 8192
_NPZ_REQUIRED_MEMBERS = frozenset({"images.npy", "labels.npy"})
_PARAM_ORDER = ("w1", "b1", "w2", "b2", "w3", "b3")
_ENVELOPE_KEYS = frozenset({
    "schema",
    "paper_revision",
    "lane",
    "profile_id",
    "seed",
    "dataset_sha256",
    "task_permutation_digest",
    "task_protocol",
    "labels_permuted",
    "runtime_identity",
    "implementation_sha256",
    "protocol_differences",
    "stream",
    "development_only",
    "scientific_promotion_allowed",
    "negative_results_must_be_retained",
    "comparison_prefix",
    "comparisons",
})
_STREAM_KEYS = frozenset({
    "tasks",
    "examples_per_task",
    "passes_per_task",
    "train_batch_size",
    "learning_rate",
    "hidden_width",
    "curvature_probe_rows",
    "parameter_updates",
    "observations",
})
_BLOCK_KEYS = frozenset({
    "comparison_id",
    "checkpoint",
    "task",
    "campaign_observations",
    "receipts",
})

PROTOCOL_DIFFERENCES: Mapping[str, object] = MappingProxyType({
    "workload": "supervised_cumulative_input_permutation_mnist_small_mlp",
    "not_paper_experiment": (
        "the paper scores RL/LLM checkpoints; this lane scores a small "
        "supervised MLP stream and claims no cross-domain transfer"
    ),
    "validation_population": (
        "caller-supplied exactly-10,000-row normalized validation pool used "
        "as the current validation population for gradients, losses, and "
        "future gain"
    ),
    "curvature_probe": (
        "per-sample-gradient energy rank on the first curvature_probe_rows "
        "validation rows, a deterministic prefix already covered by the "
        "pinned full-validation gradient charge"
    ),
    "future_gain_measurement": (
        "relative full-validation loss reduction of the pinned 128 one-step "
        "rollouts on the same validation population as the diagnostics"
    ),
    "matched_predictor_arms": (
        "readiness_full_strength_x_reliability versus the predeclared "
        "gradient-strength-only mechanism-off reduction"
    ),
    "per_receipt_outcome": (
        "inconclusive: one checkpoint observation cannot support or reject; "
        "across-checkpoint comparisons live in the retained summary only"
    ),
})


def _runtime_identity() -> tuple[str, str, str, str]:
    return (platform.python_version(), jax.__version__, np.__version__, jax.default_backend())


def _exact_int(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return value


def _exact_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty exact string")
    return value


def _hex_digest(value: object, name: str) -> str:
    text = _exact_string(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    return text


@dataclasses.dataclass(frozen=True, slots=True)
class ReadinessProfile:
    """Bounded stream shape for one lane run.

    The diagnostic and rollout sampling axes are not free: they are pinned to
    the Appendix C.1 contract by
    :func:`alberta_framework.evaluation.optimization_readiness.validate_development_result`
    and are therefore module constants here.
    """

    profile_id: str
    n_tasks: int
    examples_per_task: int
    passes_per_task: int
    train_batch_size: int
    hidden_width: int
    curvature_probe_rows: int
    learning_rate: float

    def __post_init__(self) -> None:
        if self.profile_id not in ("contract-smoke", "bounded-development"):
            raise ValueError("unknown readiness profile")
        _exact_int(self.n_tasks, "n_tasks", 2, 16)
        _exact_int(self.examples_per_task, "examples_per_task", 4, 4096)
        _exact_int(self.passes_per_task, "passes_per_task", 1, 8)
        _exact_int(self.train_batch_size, "train_batch_size", 1, 256)
        _exact_int(self.hidden_width, "hidden_width", 4, 128)
        _exact_int(self.curvature_probe_rows, "curvature_probe_rows", 8, 256)
        if self.examples_per_task % self.train_batch_size != 0:
            raise ValueError("examples_per_task must divide by train_batch_size")
        if (
            type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError("learning_rate must be a positive finite exact float")

    @property
    def stream_parameter_updates(self) -> int:
        return (
            self.n_tasks
            * (self.examples_per_task // self.train_batch_size)
            * self.passes_per_task
        )

    @property
    def stream_observations(self) -> int:
        return self.n_tasks * self.examples_per_task * self.passes_per_task


PROFILES: Mapping[str, ReadinessProfile] = MappingProxyType({
    "contract-smoke": ReadinessProfile(
        profile_id="contract-smoke",
        n_tasks=2,
        examples_per_task=16,
        passes_per_task=1,
        train_batch_size=8,
        hidden_width=8,
        curvature_probe_rows=32,
        learning_rate=0.05,
    ),
    "bounded-development": ReadinessProfile(
        profile_id="bounded-development",
        n_tasks=6,
        examples_per_task=512,
        passes_per_task=2,
        train_batch_size=16,
        hidden_width=64,
        curvature_probe_rows=128,
        learning_rate=0.05,
    ),
})


def _init_params(key: Array, width: int) -> dict[str, Array]:
    k1, k2, k3 = jr.split(key, 3)
    return {
        "w1": jr.normal(k1, (INPUT_DIM, width), dtype=jnp.float32)
        * math.sqrt(2.0 / INPUT_DIM),
        "b1": jnp.zeros((width,), dtype=jnp.float32),
        "w2": jr.normal(k2, (width, width), dtype=jnp.float32) * math.sqrt(2.0 / width),
        "b2": jnp.zeros((width,), dtype=jnp.float32),
        "w3": jr.normal(k3, (width, N_CLASSES), dtype=jnp.float32)
        * math.sqrt(2.0 / width),
        "b3": jnp.zeros((N_CLASSES,), dtype=jnp.float32),
    }


def _hidden_features(params: Mapping[str, Array], inputs: Array) -> Array:
    """Second ReLU layer: the representation whose rank the lane reports."""
    hidden1 = jax.nn.relu(inputs @ params["w1"] + params["b1"])
    return jax.nn.relu(hidden1 @ params["w2"] + params["b2"])


def _batch_logits(params: Mapping[str, Array], inputs: Array) -> Array:
    return _hidden_features(params, inputs) @ params["w3"] + params["b3"]


def _batch_loss(params: Mapping[str, Array], inputs: Array, labels: Array) -> Array:
    logits = _batch_logits(params, inputs)
    return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(inputs.shape[0]), labels])


def _sgd_step(
    params: Mapping[str, Array], inputs: Array, labels: Array, step_size: Array
) -> dict[str, Array]:
    gradient = jax.grad(_batch_loss)(params, inputs, labels)
    return {name: params[name] - step_size * gradient[name] for name in _PARAM_ORDER}


def _single_row_loss(params: Mapping[str, Array], row: Array, label: Array) -> Array:
    logits = _batch_logits(params, row[None, :])[0]
    return -jax.nn.log_softmax(logits)[label]


def _host_row_losses(
    params: Mapping[str, Array],
    images: np.ndarray,
    labels: np.ndarray,
    *,
    chunk_rows: int = 1_000,
) -> float:
    """Mean per-row loss over a pool with exact float64 host accumulation."""
    per_row = jax.jit(jax.vmap(_single_row_loss, in_axes=(None, 0, 0)))
    total = 0.0
    for start in range(0, images.shape[0], chunk_rows):
        rows = jnp.asarray(images[start : start + chunk_rows])
        targets = jnp.asarray(labels[start : start + chunk_rows])
        total += float(np.sum(np.asarray(per_row(params, rows, targets), dtype=np.float64)))
    return total / float(images.shape[0])


def _stacked_host_losses(
    stepped: dict[str, Array],
    images: np.ndarray,
    labels: np.ndarray,
    *,
    chunk_rows: int = 1_000,
) -> np.ndarray:
    """Mean loss of every stacked rollout parameter set over a whole pool."""
    per_chunk = jax.jit(
        jax.vmap(jax.vmap(_single_row_loss, in_axes=(None, 0, 0)), in_axes=(0, None, None))
    )
    totals = np.zeros(int(stepped["w1"].shape[0]), dtype=np.float64)
    for start in range(0, images.shape[0], chunk_rows):
        rows = jnp.asarray(images[start : start + chunk_rows])
        targets = jnp.asarray(labels[start : start + chunk_rows])
        totals += np.asarray(per_chunk(stepped, rows, targets), dtype=np.float64).sum(axis=1)
    return totals / float(images.shape[0])


def _full_gradient(
    params: Mapping[str, Array],
    images: np.ndarray,
    labels: np.ndarray,
    *,
    chunk_rows: int = 1_000,
) -> np.ndarray:
    """Mean gradient over a whole pool, accumulated exactly on the host."""
    accumulated = {
        name: np.zeros(int(params[name].size), dtype=np.float64) for name in _PARAM_ORDER
    }
    grad_fn = jax.jit(jax.grad(_batch_loss))
    for start in range(0, images.shape[0], chunk_rows):
        rows = jnp.asarray(images[start : start + chunk_rows])
        targets = jnp.asarray(labels[start : start + chunk_rows])
        weight = rows.shape[0] / float(images.shape[0])
        piece = grad_fn(params, rows, targets)
        for name in _PARAM_ORDER:
            accumulated[name] += np.asarray(piece[name], dtype=np.float64).ravel() * weight
    return np.concatenate([accumulated[name] for name in _PARAM_ORDER])


def _parameter_count(params: Mapping[str, Array]) -> int:
    return sum(int(params[name].size) for name in _PARAM_ORDER)


def _persistent_bytes(params: Mapping[str, Array]) -> int:
    return sum(int(np.asarray(params[name]).nbytes) for name in _PARAM_ORDER)


def _batch_gradients(
    params: Mapping[str, Array],
    images: np.ndarray,
    labels: np.ndarray,
    *,
    key: Array,
    batch_count: int,
    batch_size: int,
) -> np.ndarray:
    """Sample ``batch_count`` size-``batch_size`` mini-batch gradients.

    Rows are drawn independently with replacement from the validation pool by
    explicit Threefry keys.  The receipt's sampling provenance records that
    gradient arrays cannot prove sample independence.
    """
    grad_fn = jax.jit(jax.grad(_batch_loss))
    gradients = np.empty((batch_count, _parameter_count(params)), dtype=np.float64)
    for index in range(batch_count):
        key, draw_key = jr.split(key)
        draw = np.asarray(jr.randint(draw_key, (batch_size,), 0, images.shape[0]))
        piece = grad_fn(params, jnp.asarray(images[draw]), jnp.asarray(labels[draw]))
        gradients[index] = np.concatenate([
            np.asarray(piece[name], dtype=np.float64).ravel() for name in _PARAM_ORDER
        ])
    return gradients


def _per_sample_gradients(
    params: Mapping[str, Array], images: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Per-row gradients for a bounded validation prefix (curvature probe)."""
    per_row_grad = jax.jit(
        jax.vmap(jax.grad(_single_row_loss), in_axes=(None, 0, 0))
    )
    pieces = per_row_grad(params, jnp.asarray(images), jnp.asarray(labels))
    return np.concatenate(
        [
            np.asarray(pieces[name], dtype=np.float64).reshape(images.shape[0], -1)
            for name in _PARAM_ORDER
        ],
        axis=1,
    )


def _schedule(
    images: np.ndarray, labels: np.ndarray, profile: ReadinessProfile, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    key = jr.key(seed)
    permutation = np.arange(INPUT_DIM, dtype=np.int32)
    tasks = []
    for _ in range(profile.n_tasks):
        key, pixel_key, data_key = jr.split(key, 3)
        next_permutation = np.asarray(jr.permutation(pixel_key, INPUT_DIM), dtype=np.int32)
        permutation = permutation[next_permutation]
        order = np.asarray(jr.permutation(data_key, images.shape[0]), dtype=np.int32)
        indices = order[: profile.examples_per_task]
        tasks.append(
            (
                np.ascontiguousarray(images[indices][:, permutation]),
                labels[indices].copy(),
            )
        )
    return tuple(tasks)


def _permutation_digest(tasks: tuple[tuple[np.ndarray, np.ndarray], ...]) -> str:
    digest = hashlib.sha256()
    digest.update(str(len(tasks)).encode())
    for inputs, labels in tasks:
        digest.update(str(inputs.shape).encode())
        digest.update(inputs.tobytes(order="C"))
        digest.update(labels.tobytes(order="C"))
    return digest.hexdigest()


def _pool_digest(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    validation_images: np.ndarray,
    validation_labels: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"optimization-readiness-development-pools-v1\0")
    for array in (
        train_images.astype("<f4", copy=False),
        train_labels.astype("<i4", copy=False),
        validation_images.astype("<f4", copy=False),
        validation_labels.astype("<i4", copy=False),
    ):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _train_stream(
    params: dict[str, Array],
    tasks: tuple[tuple[np.ndarray, np.ndarray], ...],
    profile: ReadinessProfile,
) -> dict[str, Array]:
    """Plain small-batch SGD over each task pool, ``passes_per_task`` passes."""
    step_size = jnp.asarray(profile.learning_rate, dtype=jnp.float32)
    for inputs, labels in tasks:
        for _ in range(profile.passes_per_task):
            for start in range(0, inputs.shape[0], profile.train_batch_size):
                rows = jnp.asarray(inputs[start : start + profile.train_batch_size])
                targets = jnp.asarray(labels[start : start + profile.train_batch_size])
                params = _sgd_step(params, rows, targets, step_size)
    return params


def _pool_arrays(
    images: object, labels: object, *, name: str
) -> tuple[np.ndarray, np.ndarray]:
    if type(images) is not np.ndarray or images.dtype != np.float32:
        raise ValueError(f"{name} images must be an exact float32 ndarray")
    if type(labels) is not np.ndarray or labels.dtype != np.int32:
        raise ValueError(f"{name} labels must be an exact int32 ndarray")
    if images.ndim != 2 or images.shape[1] != INPUT_DIM or labels.shape != (images.shape[0],):
        raise ValueError(f"{name} arrays must have shapes [N,784] and [N]")
    if not np.all(np.isfinite(images)):
        raise ValueError(f"{name} images must be finite")
    if np.any(labels < 0) or np.any(labels >= N_CLASSES):
        raise ValueError(f"{name} labels must lie in [0,{N_CLASSES - 1}]")
    return images, labels


def _measurement(
    params: Mapping[str, Array],
    validation_images: np.ndarray,
    validation_labels: np.ndarray,
    *,
    key: Array,
    profile: ReadinessProfile,
    seed: int,
    comparison_id: str,
    task: str,
) -> dict[str, object]:
    """Run the pinned C.1 campaign at one checkpoint for both predictor arms."""
    started_ns = time.perf_counter_ns()
    loss = _host_row_losses(params, validation_images, validation_labels)
    full_gradient = _full_gradient(params, validation_images, validation_labels)
    diagnostic_key, rollout_key = jr.split(key)
    batch_gradients = _batch_gradients(
        params,
        validation_images,
        validation_labels,
        key=diagnostic_key,
        batch_count=DIAGNOSTIC_BATCH_COUNT,
        batch_size=DIAGNOSTIC_BATCH_SIZE,
    )
    full = estimate_appendix_c1_optimization_readiness(
        loss=float(loss),
        full_validation_gradient=full_gradient,
        batch_gradients=batch_gradients,
        full_validation_observations=N_VALIDATION_OBSERVATIONS,
        mini_batch_size=DIAGNOSTIC_BATCH_SIZE,
        sampling_provenance=SAMPLING_PROVENANCE,
        include_reliability=True,
    )
    off = estimate_appendix_c1_optimization_readiness(
        loss=float(loss),
        full_validation_gradient=full_gradient,
        batch_gradients=batch_gradients,
        full_validation_observations=N_VALIDATION_OBSERVATIONS,
        mini_batch_size=DIAGNOSTIC_BATCH_SIZE,
        sampling_provenance=SAMPLING_PROVENANCE,
        include_reliability=False,
    )

    # Rank the representation (second ReLU layer), not the logits: the
    # logits-projected matrix is capped at N_CLASSES=10 nonzero directions,
    # which would pin the reported rank at 10 for every hidden width.
    features = jax.jit(_hidden_features)(params, jnp.asarray(validation_images))
    representation_rank = energy_rank(
        np.asarray(features, dtype=np.float64), threshold=0.99
    )
    probe = _per_sample_gradients(
        params,
        validation_images[: profile.curvature_probe_rows],
        validation_labels[: profile.curvature_probe_rows],
    )
    curvature_rank = energy_rank(probe, threshold=0.99)

    stepped_rows: list[dict[str, Array]] = []
    for _ in range(FUTURE_GAIN_ROLLOUT_COUNT):
        rollout_key, draw_key = jr.split(rollout_key)
        draw = np.asarray(
            jr.randint(draw_key, (FUTURE_GAIN_BATCH_SIZE,), 0, validation_images.shape[0])
        )
        stepped = _sgd_step(
            params,
            jnp.asarray(validation_images[draw]),
            jnp.asarray(validation_labels[draw]),
            jnp.asarray(FUTURE_GAIN_STEP_SIZE, dtype=jnp.float32),
        )
        stepped_rows.append({name: stepped[name][None] for name in _PARAM_ORDER})
    stepped_stack = {
        name: jnp.concatenate([row[name] for row in stepped_rows])
        for name in _PARAM_ORDER
    }
    terminal = _stacked_host_losses(stepped_stack, validation_images, validation_labels)
    reductions = (float(loss) - terminal) / float(loss) if loss > 0.0 else np.zeros_like(terminal)
    future_gain = float(np.mean(reductions))
    if future_gain > 1.0:
        raise ValueError("future_relative_loss_reduction cannot exceed one")

    observations = (
        N_VALIDATION_OBSERVATIONS
        + DIAGNOSTIC_BATCH_COUNT * DIAGNOSTIC_BATCH_SIZE
        + FUTURE_GAIN_ROLLOUT_COUNT * FUTURE_GAIN_STEPS * FUTURE_GAIN_BATCH_SIZE
        + FUTURE_GAIN_ROLLOUT_COUNT * N_VALIDATION_OBSERVATIONS
    )
    parameter_updates = FUTURE_GAIN_ROLLOUT_COUNT * FUTURE_GAIN_STEPS
    parameter_count = _parameter_count(params)
    peak_working_set_bytes = (
        (DIAGNOSTIC_BATCH_COUNT + 1) * parameter_count * 8
        + profile.curvature_probe_rows * parameter_count * 8
        + int(validation_images.nbytes)
        + FUTURE_GAIN_ROLLOUT_COUNT * N_VALIDATION_OBSERVATIONS * 8
    )
    timing_seconds = (time.perf_counter_ns() - started_ns) / 1e9

    protocol = {
        "schema": RECEIPT_PROTOCOL_SCHEMA,
        "seed": int(seed),
        "checkpoint": "task_start",
        "task": task,
        "updates": parameter_updates,
        "observations": observations,
        "full_validation_observations": N_VALIDATION_OBSERVATIONS,
        "mini_batch_size": DIAGNOSTIC_BATCH_SIZE,
        "diagnostic_batch_count": DIAGNOSTIC_BATCH_COUNT,
        "future_gain_steps": FUTURE_GAIN_STEPS,
        "future_gain_rollout_count": FUTURE_GAIN_ROLLOUT_COUNT,
        "future_gain_batch_size": FUTURE_GAIN_BATCH_SIZE,
        "future_gain_step_size": FUTURE_GAIN_STEP_SIZE,
        "parameter_count": parameter_count,
        "sampling_provenance": SAMPLING_PROVENANCE,
        "allowed_boundary_information": list(ALLOWED_BOUNDARY_INFORMATION),
        "allowed_task_information": list(ALLOWED_TASK_INFORMATION),
    }
    resources = {
        "schema": RECEIPT_RESOURCE_SCHEMA,
        "persistent_bytes": _persistent_bytes(params),
        "peak_working_set_bytes": peak_working_set_bytes,
        "environment_steps": 0,
        "data_steps": observations,
        "model_queries": observations,
        "parameter_updates": parameter_updates,
        "timing_seconds": float(timing_seconds),
        "timing_is_telemetry_only": True,
    }

    def receipt(arm_id: str, estimator: OptimizationReadiness) -> dict[str, object]:
        return {
            "schema": RECEIPT_RESULT_SCHEMA,
            "comparison_id": comparison_id,
            "arm_id": arm_id,
            "protocol": dict(protocol),
            "resources": dict(resources),
            "metrics": {
                "optimization_readiness": float(estimator.optimization_readiness),
                "gradient_norm": float(estimator.gradient_norm),
                "representation_energy_rank_0_99": int(representation_rank),
                "curvature_energy_rank_0_99": int(curvature_rank),
                "parameter_norm": float(np.linalg.norm(
                    np.concatenate([
                        np.asarray(params[name], dtype=np.float64).ravel()
                        for name in _PARAM_ORDER
                    ])
                )),
                "future_relative_loss_reduction": float(future_gain),
            },
            "reported_outcome": "inconclusive",
            "reported_outcome_retained": True,
            "development_only": True,
            "scientific_promotion_allowed": False,
        }

    receipts = [receipt(ARM_READINESS_FULL, full), receipt(ARM_GRADIENT_STRENGTH_ONLY, off)]
    if off.optimization_readiness != full.gradient_strength:
        raise ValueError(
            "the mechanism-off estimator must reduce exactly to the full arm's strength"
        )
    validate_matched_development_results(receipts)
    return {
        "comparison_id": comparison_id,
        "checkpoint": "task_start",
        "task": task,
        "campaign_observations": observations,
        "receipts": receipts,
    }


def run_readiness_development(
    train_images: object,
    train_labels: object,
    validation_images: object,
    validation_labels: object,
    *,
    seed: object,
    profile_id: str = "contract-smoke",
) -> dict[str, object]:
    """Run one frozen-seed lane pass and return the validated envelope."""
    train_pool = _pool_arrays(train_images, train_labels, name="train")
    validation_pool = _pool_arrays(validation_images, validation_labels, name="validation")
    if type(seed) is not int or seed not in FROZEN_SEEDS:
        raise ValueError("seed is outside the frozen development schedule")
    if type(profile_id) is not str or profile_id not in PROFILES:
        raise ValueError("unknown readiness profile")
    profile = PROFILES[profile_id]
    if train_pool[0].shape[0] < profile.examples_per_task:
        raise ValueError("train pool has too few examples for the frozen profile")
    if validation_pool[0].shape[0] != N_VALIDATION_OBSERVATIONS:
        raise ValueError(
            "validation pool must hold exactly "
            f"{N_VALIDATION_OBSERVATIONS} normalized rows"
        )

    train_images_array, train_labels_array = train_pool
    validation_images_array, validation_labels_array = validation_pool
    tasks = _schedule(train_images_array, train_labels_array, profile, seed)
    key = jr.key(seed)
    key, init_key = jr.split(key)
    params = _init_params(init_key, profile.hidden_width)

    prefix = f"readiness-{profile.profile_id}-seed{seed}"
    comparisons = []
    for index, task_data in enumerate(tasks):
        # Advance the root key at every task start so each checkpoint draws
        # independent diagnostic indices and rollout batches.
        key, measurement_key = jr.split(key)
        comparisons.append(
            _measurement(
                params,
                validation_images_array,
                validation_labels_array,
                key=measurement_key,
                profile=profile,
                seed=seed,
                comparison_id=f"{prefix}-{index:02d}",
                task=f"task_{index:02d}",
            )
        )
        params = _train_stream(params, (task_data,), profile)

    result = {
        "schema": DEVELOPMENT_SCHEMA,
        "paper_revision": PAPER_REVISION,
        "lane": "optimization_readiness_development",
        "profile_id": profile.profile_id,
        "seed": seed,
        "dataset_sha256": _pool_digest(
            train_images_array,
            train_labels_array,
            validation_images_array,
            validation_labels_array,
        ),
        "task_permutation_digest": _permutation_digest(tasks),
        "task_protocol": TASK_PROTOCOL,
        "labels_permuted": False,
        "runtime_identity": list(_runtime_identity()),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_differences": dict(PROTOCOL_DIFFERENCES),
        "stream": {
            "tasks": profile.n_tasks,
            "examples_per_task": profile.examples_per_task,
            "passes_per_task": profile.passes_per_task,
            "train_batch_size": profile.train_batch_size,
            "learning_rate": profile.learning_rate,
            "hidden_width": profile.hidden_width,
            "curvature_probe_rows": profile.curvature_probe_rows,
            "parameter_updates": profile.stream_parameter_updates,
            "observations": profile.stream_observations,
        },
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_results_must_be_retained": True,
        "comparison_prefix": prefix,
        "comparisons": tuple(comparisons),
    }
    return validate_result(result)


def validate_result(value: object) -> dict[str, object]:
    """Strictly validate one lane envelope and every embedded receipt."""
    if type(value) is not dict:
        raise ValueError("envelope must be an exact dict")
    if set(value.keys()) != set(_ENVELOPE_KEYS):
        raise ValueError(f"envelope keys must be exactly {sorted(_ENVELOPE_KEYS)}")
    if value["schema"] != DEVELOPMENT_SCHEMA:
        raise ValueError("envelope schema is not supported")
    if value["paper_revision"] != PAPER_REVISION:
        raise ValueError("envelope paper revision is not pinned")
    if value["lane"] != "optimization_readiness_development":
        raise ValueError("envelope lane identity is not supported")
    if value["task_protocol"] != TASK_PROTOCOL or value["labels_permuted"] is not False:
        raise ValueError("envelope task protocol must be the frozen input-permutation stream")
    if value["development_only"] is not True:
        raise ValueError("envelope development_only must permanently remain True")
    if value["scientific_promotion_allowed"] is not False:
        raise ValueError(
            "envelope scientific_promotion_allowed must permanently remain False"
        )
    if value["negative_results_must_be_retained"] is not True:
        raise ValueError("envelope negative_results_must_be_retained must remain True")
    profile_id = _exact_string(value["profile_id"], "profile_id")
    if profile_id not in PROFILES:
        raise ValueError("unknown readiness profile")
    profile = PROFILES[profile_id]
    seed = value["seed"]
    if type(seed) is not int or seed not in FROZEN_SEEDS:
        raise ValueError("seed is outside the frozen development schedule")
    _hex_digest(value["dataset_sha256"], "dataset_sha256")
    _hex_digest(value["task_permutation_digest"], "task_permutation_digest")
    _hex_digest(value["implementation_sha256"], "implementation_sha256")
    runtime = value["runtime_identity"]
    if type(runtime) is not list or len(runtime) != 4 or any(
        type(item) is not str or not item for item in runtime
    ):
        raise ValueError("runtime_identity must be four non-empty exact strings")
    differences = value["protocol_differences"]
    if type(differences) is not dict or "not_paper_experiment" not in differences:
        raise ValueError("protocol_differences must record the declared lane differences")

    stream = value["stream"]
    if type(stream) is not dict or set(stream.keys()) != set(_STREAM_KEYS):
        raise ValueError(f"stream keys must be exactly {sorted(_STREAM_KEYS)}")
    expected_stream = {
        "tasks": profile.n_tasks,
        "examples_per_task": profile.examples_per_task,
        "passes_per_task": profile.passes_per_task,
        "train_batch_size": profile.train_batch_size,
        "learning_rate": profile.learning_rate,
        "hidden_width": profile.hidden_width,
        "curvature_probe_rows": profile.curvature_probe_rows,
        "parameter_updates": profile.stream_parameter_updates,
        "observations": profile.stream_observations,
    }
    for field, expected in expected_stream.items():
        actual = stream[field]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"stream.{field} disagrees with the frozen profile")

    comparisons = value["comparisons"]
    if type(comparisons) not in (list, tuple):
        raise ValueError("comparisons must be an exact list or tuple")
    if len(comparisons) != profile.n_tasks:
        raise ValueError("comparison count disagrees with the frozen profile")
    seen_ids: set[str] = set()
    for index, block in enumerate(comparisons):
        if type(block) is not dict or set(block.keys()) != set(_BLOCK_KEYS):
            raise ValueError(f"comparison {index} keys must be exactly {sorted(_BLOCK_KEYS)}")
        if block["task"] != f"task_{index:02d}" or block["checkpoint"] != "task_start":
            raise ValueError(f"comparison {index} must sit at the frozen task boundary")
        comparison_id = _exact_string(block["comparison_id"], "comparison_id")
        if comparison_id in seen_ids:
            raise ValueError("comparison_id values must be unique")
        seen_ids.add(comparison_id)
        receipts = block["receipts"]
        if type(receipts) not in (list, tuple) or len(receipts) != 2:
            raise ValueError("every comparison must carry exactly two matched receipts")
        for item in receipts:
            if type(item) is not dict or "arm_id" not in item:
                raise ValueError("every receipt must be an exact dict with an arm_id")
        if [item["arm_id"] for item in receipts] != [
            ARM_READINESS_FULL,
            ARM_GRADIENT_STRENGTH_ONLY,
        ]:
            raise ValueError("receipt arm order must be full then mechanism-off")
        matched = validate_matched_development_results(receipts)
        full, off = matched
        checkpoint_properties = (
            "gradient_norm",
            "representation_energy_rank_0_99",
            "curvature_energy_rank_0_99",
            "parameter_norm",
            "future_relative_loss_reduction",
        )
        for property_name in checkpoint_properties:
            if getattr(full, property_name) != getattr(off, property_name):
                raise ValueError(
                    "checkpoint properties must agree across matched predictor arms"
                )
        # Self-contained invariant (not borrowed from the C.1 oracle): the
        # future-gain metric is a relative loss reduction, so it must lie in
        # [0, 1] — the oracle alone only rejects values above one.
        gain = full.future_relative_loss_reduction
        if gain < 0.0 or gain > 1.0:
            raise ValueError(
                "future_relative_loss_reduction cannot exceed one or be negative"
            )
        if full.reported_outcome != "inconclusive" or off.reported_outcome != "inconclusive":
            raise ValueError("per-checkpoint receipts must report an inconclusive outcome")
        if full.protocol.seed != seed or off.protocol.seed != seed:
            raise ValueError("receipt seeds must match the envelope seed")
        expected_campaign = (
            N_VALIDATION_OBSERVATIONS
            + DIAGNOSTIC_BATCH_COUNT * DIAGNOSTIC_BATCH_SIZE
            + FUTURE_GAIN_ROLLOUT_COUNT * FUTURE_GAIN_STEPS * FUTURE_GAIN_BATCH_SIZE
            + FUTURE_GAIN_ROLLOUT_COUNT * N_VALIDATION_OBSERVATIONS
        )
        if block["campaign_observations"] != expected_campaign:
            raise ValueError("campaign_observations must equal the pinned C.1 charge")
        if full.protocol.observations != expected_campaign:
            raise ValueError("receipt observations must equal the pinned C.1 charge")
    return value


def _load_pool_npz(path: Path, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with zipfile.ZipFile(path) as archive:
        members = {info.filename for info in archive.infolist()}
        if members != _NPZ_REQUIRED_MEMBERS:
            raise ValueError(f"{name} npz members must be exactly {sorted(_NPZ_REQUIRED_MEMBERS)}")
        for info in archive.infolist():
            if info.file_size > _MAX_NPZ_MEMBER_BYTES:
                raise ValueError(f"{name} npz member exceeds its bounded size")
    with np.load(path, allow_pickle=False) as loaded:
        images = np.ascontiguousarray(loaded["images.npy"], dtype=np.float32)
        labels = np.ascontiguousarray(loaded["labels.npy"], dtype=np.int32)
    return _pool_arrays(images, labels, name=name)


def _json_envelope(result: Mapping[str, object]) -> str:
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def _atomic_write(path: Path, text: str) -> None:
    """Publish one artifact atomically without ever replacing a destination.

    link(2) fails with EEXIST when the destination is taken, so an existing
    development artifact can never be silently rewritten (append, don't
    rewrite). The temporary file is removed on every exit path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite existing output (no-replace publication): {path}"
        ) from None
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--train-npz", type=Path)
    parser.add_argument("--validation-npz", type=Path)
    parser.add_argument("--seed", type=int, default=FROZEN_SEEDS[0])
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default="contract-smoke"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.catalog:
        print(
            _json_envelope({
                "schema": DEVELOPMENT_SCHEMA,
                "paper_revision": PAPER_REVISION,
                "frozen_seeds": list(FROZEN_SEEDS),
                "profiles": {
                    name: dataclasses.asdict(profile) for name, profile in PROFILES.items()
                },
                "protocol_differences": dict(PROTOCOL_DIFFERENCES),
                "development_only": True,
                "scientific_promotion_allowed": False,
                "negative_results_must_be_retained": True,
            })
        )
        return 0
    if args.train_npz is None or args.validation_npz is None:
        parser.error("--train-npz and --validation-npz are required without --catalog")
    train_images, train_labels = _load_pool_npz(args.train_npz, name="train")
    validation_images, validation_labels = _load_pool_npz(
        args.validation_npz, name="validation"
    )
    result = run_readiness_development(
        train_images,
        train_labels,
        validation_images,
        validation_labels,
        seed=args.seed,
        profile_id=args.profile,
    )
    text = _json_envelope(result)
    if args.out is not None:
        _atomic_write(args.out, text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
