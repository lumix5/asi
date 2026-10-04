# AdaLin retention grid 1 (2026-10-04) — first retained outcomes for the lane

Advances #1571 (does not close it): the AdaLin lane had a pinned protocol, a
validated runner (`alberta_framework/benchmarks/adalin.py`), and unit tests,
but **zero retained development outcomes** anywhere. This directory retains
the first one: both registered arms (`adalin`, `relu_alpha_zero_mechanism_off`)
over four paired development seeds in each of two bounded PMNIST cells, one
fresh process per run.

Artifacts only — no library, doc, test, or registry bytes are touched, so no
pinned source identity drifts. Every receipt is
`development_only: true` and `scientific_promotion_allowed: false`; this is a
permanently nonpromoting development measurement and makes no performance,
parity, promotion, or SOTA claim.

## Provenance

| field | value |
| --- | --- |
| base revision | `f3d32c451ed1c1715e477ad56b782fc7ea89b206` (`main` at run time) |
| `provenance.implementation_sha256` in all 16 receipts | `64df8ddcdc6fef2fe80fde47c2785019d2e8326a9223e58e98f1ea79a90a8704` |
| receipt schema | `asi.adalin.pmnist-development-result.v1` (all identical) |
| environment | CPython 3.14.4, JAX 0.11.0, NumPy 2.5.1, scikit-learn 1.9.0, Linux x86_64 |
| execution | serial fresh processes, `OMP_NUM_THREADS=1`, imports resolved from a clean worktree at the base revision (`PYTHONPATH` override; the editable install was bypassed and the resolved module directory asserted before the grid) |

Every receipt passed `validate_adalin_result` at write time inside the runner
and was re-validated afterwards by a separate process reading the retained
bytes. All 16 `.stderr` files are empty (0 bytes); every run exited 0.

## Dataset (derived, not stored here)

The runner input is caller-supplied by design (the receipt's
`comparison.residual_gaps` records exactly this), so the pool is pinned here:

- `mnist_784` version 1 from the cached OpenML source
  (`data_home=outputs/upgd_ipmnist/openml_cache`, one download if absent),
  divided by 255 into float32.
- **Cell A** train pool: first 1024 train rows; **Cell B** train pool: first
  2048 train rows. Test split for both cells: first 1000 test rows (bounded by
  the runner's 1,000,000-element per-call array cap: 1000 × 784).
- Derived NPZ digest: `9102d8f16ba6b99bfdaa69443012659946d573ccb481190de53b3a06818b899d`.
- Receipt `dataset.sha256` (the load-bearing identity, over the exact arrays
  passed): cell A `7a36c86e7bce22bbc0cad4ad8458a5faef6630bae3d3031d4bdc300115949b94`,
  cell B `783414df00ccaba64a5c7ed216c1ac9b9adaee64e398d8294ada62db628735c1`.

## Configuration

Both cells use the paper's `hidden_widths=(100, 100)`, `learning_rate=1e-2`,
batch 16, one epoch per task — and deliberately bounded task/example counts
(the paper protocol is 400 × 10000; the declared ASI campaign target is
200 × 5000 batch 1, which is not what is measured here):

| cell | tasks | examples_per_task | batch | updates/run | seeds |
| --- | --- | --- | --- | --- | --- |
| A | 20 | 1024 | 16 | 1280 | 15710–15713 |
| B | 40 | 2048 | 16 | 5120 | 15714–15717 |

Seeds are fresh development seeds consumed by this retention grid; they are
not calibration seeds of any frozen protocol and can never back a promotion.
Each seed runs paired arms sharing identical weight-init keys; the off arm is
the lane's registered exact reduction (`alpha_zero_exact_base_activation`):
zero-initialized alphas that never update (`final_alpha_l2` exactly `0.0` in
all eight off receipts, ~8.0–8.6 in all eight on receipts).

## Driver (exact bytes, sha256
`6ba183fc5b73358115c9b2271a50323d30103e31e886f2c4c15622c5ce7b662d`)

The lane has no CLI yet, so receipts are the exact stdout of this driver; its
bytes are pinned here verbatim and its four assertions (schema, nonpromoting
policy, development-only policy, arm name) run inside every process:

```python
"""AdaLin retained-outcome driver (development-only, nonpromoting).

Prints exactly one validated `run_adalin_development` receipt as JSON on
stdout. Any failure exits nonzero with the error on stderr. Deterministic:
the caller supplies one fixed MNIST pool and the runner owns the schedule.
"""

import json
import sys

import numpy as np

from alberta_framework.benchmarks.adalin import (
    ADALIN_RESULT_SCHEMA,
    AdaLinConfig,
    run_adalin_development,
)


def main() -> int:
    payload = np.load(sys.argv[1], allow_pickle=False)
    images = np.ascontiguousarray(payload["images"], dtype=np.float32)
    labels = np.ascontiguousarray(payload["labels"], dtype=np.int32)

    cell = sys.argv[2]
    if cell == "A":
        config = AdaLinConfig(
            tasks=20, examples_per_task=1024, batch_size=16, hidden_widths=(100, 100)
        )
        seed = int(sys.argv[3])
    elif cell == "B":
        config = AdaLinConfig(
            tasks=40, examples_per_task=2048, batch_size=16, hidden_widths=(100, 100)
        )
        seed = int(sys.argv[3])
    else:
        raise ValueError("cell must be A or B")

    mechanism_enabled = sys.argv[4] == "on"
    train_x = np.ascontiguousarray(images[: config.examples_per_task])
    train_y = np.ascontiguousarray(labels[: config.examples_per_task])
    test_x = np.ascontiguousarray(images[60_000:61_000])
    test_y = np.ascontiguousarray(labels[60_000:61_000])

    result = run_adalin_development(
        train_x,
        train_y,
        test_x,
        test_y,
        config=config,
        seed=seed,
        mechanism_enabled=mechanism_enabled,
    )
    if result["schema"] != ADALIN_RESULT_SCHEMA:
        raise ValueError("receipt schema mismatch")
    if result["policy"]["scientific_promotion_allowed"] is not False:
        raise ValueError("receipt must be nonpromoting")
    if result["policy"]["development_only"] is not True:
        raise ValueError("receipt must be development-only")
    expected_arm = "adalin" if mechanism_enabled else "relu_alpha_zero_mechanism_off"
    if result["arm"] != expected_arm:
        raise ValueError("receipt arm mismatch")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## Reproduction

```bash
# from a clean worktree at f3d32c451ed1c1715e477ad56b782fc7ea89b206
PYTHONPATH=<worktree> .venv/bin/python - <<'EOF'
import numpy as np
from sklearn.datasets import fetch_openml

raw = fetch_openml(
    "mnist_784", version=1, as_frame=False,
    data_home="<repo>/outputs/upgd_ipmnist/openml_cache",
    n_retries=3, delay=2.0,
)
x = np.ascontiguousarray(raw.data, dtype=np.float32) / np.float32(255.0)
y = np.ascontiguousarray(raw.target, dtype=np.int32)
np.savez("mnist.npz", images=x, labels=y)
EOF

# save the driver above as run_grid.py, then:
for cell in A B; do
  if [ "$cell" = "A" ]; then seeds="15710 15711 15712 15713"; else seeds="15714 15715 15716 15717"; fi
  for seed in $seeds; do
    for arm in on off; do
      PYTHONPATH=<worktree> OMP_NUM_THREADS=1 .venv/bin/python run_grid.py \
        mnist.npz "$cell" "$seed" "$arm" \
        > "receipt-cell${cell}-seed${seed}-${arm}.json" \
        2> "receipt-cell${cell}-seed${seed}-${arm}.stderr"
    done
  done
done
```

Receipt files are the exact driver stdout bytes; `.stderr` files are the exact
stderr bytes (all empty). Any nonzero exit or stderr content would have been
retained as-is instead.

## Cost

| cell | runs | wall clock each (fresh process, telemetry) | total |
| --- | --- | --- | --- |
| A | 8 | ~30 s | ~4 min |
| B | 8 | ~64–130 s | ~11 min |

Resource accounting is inside each receipt (`environment_data_steps`,
`label_queries`, `optimizer_updates`, `model_queries`,
`model_forward_calls`, persistent byte counts). Timing is telemetry-only and
selects nothing.

## Reading

See `SUMMARY.md` for the paired outcome table and the honest interpretation,
including the scale caveats that keep this grid from ordering the lane beyond
its own bounds.
