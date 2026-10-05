#!/usr/bin/env python3
"""Execute the first retained native supervised CL development grid (nonpromoting).

Runs the frozen four-benchmark x four-frozen-seed grid of
``alberta_framework.benchmarks.native_supervised_suite.run_native_suite``
against canonical caller-supplied dataset bytes and writes one shard record
per cell, plus a grid index. Every written shard is reloaded from disk,
reconstructed into the exact ``SuiteResult`` type, and revalidated through
the module's own ``validate_result`` before the run is allowed to exit zero.

This script never mutates the library, never promotes, and never claims
scientific evidence. Timing in the receipts is telemetry-only.

Usage (paths are the prepared canonical arrays; see dataset_provenance.json):

    python run_grid.py \
        --mnist-images /tmp/nsuite/mnist784_train60k_images_f32.npy \
        --mnist-labels /tmp/nsuite/mnist784_train60k_labels_i32.npy \
        --cifar-images /tmp/nsuite/cifar100_train_images_f32.npy \
        --cifar-labels /tmp/nsuite/cifar100_train_labels_i32.npy \
        --head <exact git commit> \
        --mnist-raw-sha256 <sha256 of mnist_784.arff.gz> \
        --output-dir .

"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

from alberta_framework.benchmarks.native_supervised_suite import (
    ARM_IDS,
    BENCHMARK_IDS,
    FROZEN_SEEDS,
    SCHEMA,
    ArmResult,
    ResourceReceipt,
    SuiteResult,
    _dataset_sha256,
    run_native_suite,
    validate_result,
)

EXAMPLES_PER_TASK = 8
REPLAY_CAPACITY = 16
CALLER_TRANSFORM = (
    "uint8 pixel bytes [0,255] scaled to float32 [0,1] (images / 255.0); no other transform"
)

# Raw-artifact digests bound by dataset_provenance.json, measured at
# acquisition time and re-asserted here so the record cannot silently drift.
CIFAR100_TAR_MD5 = "eb9058c3a382ffc7106e4002c42a8d852f12"
CIFAR100_TAR_SHA256 = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"


def _load(path: str) -> np.ndarray:
    array = np.load(path)
    assert isinstance(array, np.ndarray)
    return array


def _sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _result_from_json(payload: dict) -> SuiteResult:
    """Rebuild the exact frozen dataclasses from a shard JSON payload."""
    arms = []
    for arm in payload["arms"]:
        raw = arm["receipt"]
        receipt = ResourceReceipt(
            data_steps=int(raw["data_steps"]),
            data_bytes_read=int(raw["data_bytes_read"]),
            model_queries=int(raw["model_queries"]),
            parameter_updates=int(raw["parameter_updates"]),
            replay_inserts=int(raw["replay_inserts"]),
            replay_samples=int(raw["replay_samples"]),
            logical_compute_units=int(raw["logical_compute_units"]),
            persistent_bytes=int(raw["persistent_bytes"]),
            peak_replay_bytes=int(raw["peak_replay_bytes"]),
            elapsed_ns=int(raw["elapsed_ns"]),
        )
        arms.append(
            ArmResult(
                arm_id=str(arm["arm_id"]),
                online_accuracy=float(arm["online_accuracy"]),
                task_accuracies=tuple(float(value) for value in arm["task_accuracies"]),
                receipt=receipt,
            )
        )
    identity = payload["runtime_identity"]
    assert isinstance(identity, list) and len(identity) == 4
    return SuiteResult(
        schema=str(payload["schema"]),
        benchmark_id=str(payload["benchmark_id"]),
        seed=int(payload["seed"]),
        examples_per_task=int(payload["examples_per_task"]),
        replay_capacity=int(payload["replay_capacity"]),
        input_dim=int(payload["input_dim"]),
        n_classes=int(payload["n_classes"]),
        dataset_sha256=str(payload["dataset_sha256"]),
        schedule_sha256=str(payload["schedule_sha256"]),
        source_sha256=str(payload["source_sha256"]),
        runtime_identity=tuple(str(part) for part in identity),
        arms=tuple(arms),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the retained native-suite grid")
    parser.add_argument("--mnist-images", required=True)
    parser.add_argument("--mnist-labels", required=True)
    parser.add_argument("--cifar-images", required=True)
    parser.add_argument("--cifar-labels", required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--head", required=True, help="exact git commit of the execution")
    parser.add_argument("--mnist-raw-sha256", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "shards"
    stderr_dir = output_dir / "shards.stderr"
    shard_dir.mkdir(parents=True, exist_ok=True)
    stderr_dir.mkdir(parents=True, exist_ok=True)

    mnist_images = _load(args.mnist_images).reshape(-1, 28, 28).astype(np.float32) / 255.0
    mnist_labels = _load(args.mnist_labels).astype(np.int32)
    cifar_images = _load(args.cifar_images).astype(np.float32) / 255.0
    cifar_labels = _load(args.cifar_labels).astype(np.int32)
    assert mnist_images.shape[0] == 60000 and cifar_images.shape[0] == 50000
    assert int(mnist_labels.min()) == 0 and int(mnist_labels.max()) == 9
    assert int(cifar_labels.min()) == 0 and int(cifar_labels.max()) == 99
    assert float(mnist_images.min()) >= 0.0 and float(mnist_images.max()) <= 1.0
    assert float(cifar_images.min()) >= 0.0 and float(cifar_images.max()) <= 1.0

    datasets = {
        "split_mnist": (mnist_images, mnist_labels),
        "rotated_mnist": (mnist_images, mnist_labels),
        "ipmnist": (mnist_images, mnist_labels),
        "split_cifar100": (cifar_images, cifar_labels),
    }

    suite_module = sys.modules["alberta_framework.benchmarks.native_supervised_suite"]
    provenance = {
        "schema": "asi.native_supervised_grid_provenance.v1",
        "head": args.head,
        "suite_module_path": str(Path(suite_module.__file__).resolve()),
        "suite_module_sha256": _sha256_file(str(suite_module.__file__)),
        "runner_sha256": _sha256_file(__file__),
        "mnist_openml_arff_gz_sha256": args.mnist_raw_sha256,
        "cifar100_tar_md5": CIFAR100_TAR_MD5,
        "cifar100_tar_sha256": CIFAR100_TAR_SHA256,
        "caller_transform": CALLER_TRANSFORM,
        "examples_per_task": EXAMPLES_PER_TASK,
        "replay_capacity": REPLAY_CAPACITY,
        "dataset_sha256_by_benchmark": {
            benchmark: _dataset_sha256(images, labels)
            for benchmark, (images, labels) in datasets.items()
        },
        "array_shapes": {
            "mnist": list(mnist_images.shape),
            "cifar100": list(cifar_images.shape),
        },
    }
    (output_dir / "dataset_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    grid = {
        "schema": "asi.native_supervised_grid.v1",
        "suite_schema": SCHEMA,
        "head": args.head,
        "benchmarks": list(BENCHMARK_IDS),
        "frozen_seeds": list(FROZEN_SEEDS),
        "arm_ids": list(ARM_IDS),
        "development_only": True,
        "scientific_promotion_allowed": False,
        "shards": {},
    }
    failures = 0
    for benchmark in BENCHMARK_IDS:
        images, labels = datasets[benchmark]
        for seed in FROZEN_SEEDS:
            name = f"{benchmark}__seed{seed}"
            result = run_native_suite(
                benchmark,
                images,
                labels,
                seed=seed,
                examples_per_task=EXAMPLES_PER_TASK,
                replay_capacity=REPLAY_CAPACITY,
            )
            payload = dataclasses.asdict(result)
            shard_path = shard_dir / f"{name}.json"
            shard_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            # Revalidate the retained bytes: strict reload -> exact types ->
            # the module's own validator. The run fails closed if the record
            # it just wrote does not revalidate.
            reloaded = _result_from_json(json.loads(shard_path.read_text()))
            validate_result(reloaded)
            if dataclasses.asdict(reloaded) != payload:
                failures += 1
                print(f"{name}: retained JSON does not round-trip exactly", file=sys.stderr)
            (stderr_dir / f"{name}.stderr").write_text(
                f"{name}: exit 0\n"
                f"source_sha256={result.source_sha256}\n"
                f"runtime_identity={result.runtime_identity}\n"
            )
            grid["shards"][name] = {
                arm["arm_id"]: arm["online_accuracy"] for arm in payload["arms"]
            }
            print(f"{name}: ok")
    (output_dir / "grid.json").write_text(json.dumps(grid, indent=2, sort_keys=True) + "\n")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
