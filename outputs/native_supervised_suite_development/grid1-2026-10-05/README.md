# Native supervised CL suite — retained development grid 1 (nonpromoting)

First executed and retained outcome grid for the native supervised
continual-learning development suite
(`alberta_framework/benchmarks/native_supervised_suite.py`), advancing
[SlopDotCash/asi#1578](https://github.com/SlopDotCash/asi/issues/1578).
The module already pinned the catalog, frozen seeds, arms, receipts, and
strict validator; nothing had ever executed it end-to-end on canonical
real dataset bytes and retained the result.

- **Execution head:** `f3d32c451ed1c1715e477ad56b782fc7ea89b206` (current
  `main` at execution time).
- **Cells:** 4 benchmarks (`split_mnist`, `rotated_mnist`, `split_cifar100`,
  `ipmnist`) x 4 frozen development seeds (15780–15783) = 16 shards.
- **Arms per cell:** `online_sgd`, `replay_sgd`, `running_centroid`,
  `frozen_no_learning` (the module's frozen roster), 4 seeds x 4 arms x
  4 benchmarks = 64 arm outcomes.
- **Protocol:** `examples_per_task=8`, `replay_capacity=16` (module
  defaults), task-agnostic learners (`task_information_used_by_learner=False`
  is enforced by the module), predict-before-update semantics enforced and
  receipt-checked by `validate_result`.
- **Data:** canonical OpenML `mnist_784` v1 first 60,000 rows (raw ARFF
  gzip sha256 `fe4410d8dbb50f6db6482b187557c5cb8bccfbcec74eeb6abc47c858f4ffab78`)
  and canonical `cifar-100-python.tar.gz` (md5
  `eb9058c3a382ffc7106e4002c42a8d852f12`, sha256
  `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`).
  Caller transform: uint8 pixels scaled to float32 `[0,1]`; nothing else.
  Per-benchmark caller-array digests are pinned in
  `dataset_provenance.json` and are bound into every shard's
  `dataset_sha256`.
- **Runtime:** CPython 3.14.4, JAX 0.11.0, NumPy 2.5.1, CPU backend
  (recorded per shard as `runtime_identity`; execution provenance, not a
  revalidation gate).

## Determinism witness

The whole grid was executed twice in fresh processes from the same arrays.
All 64 per-arm online accuracies are bit-identical between the two runs;
only the telemetry-only `elapsed_ns` receipt field differs.

## What was run

`run_grid.py` (sha256
`9478da6d62f8ee42e91aaeac286ac4b7c2ffdc8feda382d93ccbcbf69b09546e`, pinned
in `dataset_provenance.json`) loads the prepared arrays, executes
`run_native_suite` for every cell, writes `shards/<benchmark>__seed<seed>.json`,
then reloads every written shard from disk, reconstructs the exact frozen
dataclasses, and revalidates through the module's own `validate_result`
before exiting zero. Every shard also passed that revalidation.

## Reproduce

From a checkout at the exact execution head, with the two canonical raw
datasets acquired and prepared to float32/int32 NumPy arrays (MNIST pixels
`60000x28x28`, CIFAR-100 pixels `50000x3072`, labels `int32`):

```bash
python run_grid.py \
  --mnist-images <mnist_f32.npy> --mnist-labels <mnist_i32.npy> \
  --cifar-images <cifar_f32.npy> --cifar-labels <cifar_i32.npy> \
  --head f3d32c451ed1c1715e477ad56b782fc7ea89b206 \
  --mnist-raw-sha256 fe4410d8dbb50f6db6482b187557c5cb8bccfbcec74eeb6abc47c858f4ffab78 \
  --output-dir <fresh-dir>
```

The learner arithmetic, schedules, and receipt counters reproduce bitwise
on the same runtime; `elapsed_ns` is host timing telemetry.

## Status and limits

- Development-only, permanently nonpromoting. `SuiteResult` enforces
  `development_only=True`, `scientific_promotion_allowed=False`,
  `negative_results_must_be_retained=True`; this grid creates no
  `reference-dev` population, no promotion, and no scientific evidence.
- 8 examples per task is a deliberately tiny budget: absolute accuracies
  are far from the canonical protocol's operating point, so only the
  within-grid, matched-budget comparisons are meaningful, and even those
  carry wide seed noise (4 seeds).
- The linear/centroid arms are the suite's built-in controls, not ASI's
  continual-learning mechanisms; this record is the suite's first
  executed baseline floor, not a mechanism comparison.
- `source_sha256` in each shard pins the suite module bytes
  (`978104a27a6dc4a68fababcd0970a01c3930198f4f91aa782b2463300caefc5e`).
  After any change to that module the grid must be re-executed to refresh
  the record; stale records stay valid as historical development history.
