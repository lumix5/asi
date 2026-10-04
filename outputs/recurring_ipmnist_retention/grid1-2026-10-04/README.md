# Recurring-IPMNIST A/B/A retention grid 1 (2026-10-04) — first retained outcomes for the lane

Advances the roadmap's "test survivors on recurrence" priority
(`docs/research/asi-roadmap.md`, current program priorities item 2). The lane
had a validated, threshold-free A/B/A adapter
(`run_recurring_ipmnist_retention_development` in
`alberta_framework/benchmarks/ipmnist_screening.py`), unit tests
(`tests/test_ipmnist_recurring_adapter.py`,
`tests/test_recurring_ipmnist_retention.py`), and — deliberately — no default
protocol, artifact writer, or evidence path. The standard input-permuted-MNIST
stream never repeats a permutation (negative-results ledger entry 17), so
recurrence behavior had never been measured on this campaign at all. This
directory retains the first executed outcomes: three probe-compatible
registered arms over three paired development seeds, one fresh process per
run, on one fixed, explicitly recorded A/B/A protocol.

Artifacts only — no library, doc, test, or registry bytes are touched, so no
pinned source identity drifts. Every receipt is
`development_status: development-only-not-assessed`,
`scientific_promotion_allowed: false`,
`performance_thresholds_applied: false`, `retention_claimed: false`, and
`catastrophic_forgetting_absence_claimed: false`. This is a permanently
nonpromoting development measurement and makes no retention, performance,
promotion, or SOTA claim.

## Provenance

| field | value |
| --- | --- |
| base revision | `f3d32c451ed1c1715e477ad56b782fc7ea89b206` (`main` at run time) |
| environment | CPython 3.14.4, JAX 0.11.0, NumPy 2.5.1, scikit-learn 1.9.0, Linux x86_64, CPU backend |
| execution | serial fresh processes, `OMP_NUM_THREADS=1` (plus `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`), imports resolved from the clean worktree at the base revision (`PYTHONPATH` override; the editable install was bypassed and the resolved module directory verified: `alberta_framework.benchmarks.ipmnist_screening` resolved to `/home/lumix/asi/alberta_framework/benchmarks/ipmnist_screening.py`) |
| driver | `run_grid.py`, sha256 `8dd22ea4b15ffde545e614c2bf05bb2ea103844a451267a76dca948b825f6552` (exact bytes below) |
| receipts | 9 files `receipt-<arm>-seed<seed>.json` (exact driver stdout), sha256 in `receipts.sha256` |
| stderr | all 9 `receipt-*.stderr` files are 0 bytes |
| statuses | `statuses.txt` (verbatim below) records every process exit status and wall-clock seconds; all exits 0 |

Every receipt passed the driver's in-process policy assertions and was
re-validated afterwards by the separate validation pass below, which
independently re-derives the dataset, permutations, sentinel indices,
online-example order, and each receipt's adapter protocol commitment from the
recorded derivation seeds.

## Dataset (derived, not stored here)

The adapter input is caller-supplied by design, so the pool is pinned here:

- `mnist_784` version 1 from the cached OpenML source
  (`data_home=outputs/upgd_ipmnist/openml_cache`, already populated on this
  host), first 60,000 rows, float32, scaled by
  `load_mnist_train` to the campaign's frozen `[-1, 1]` domain.
- Array-identity digest over the exact passed arrays
  (`sha256` over `dtype`+`shape`+C-order bytes of `x` then `y`):
  `ade9da013ea4cfda2e0ef68dfc3222cf0075b77b207b92d82f6f87f2014b222e`
  (recomputed once from an independent reload of the same cached source via
  the default data home; the receipts themselves bind the data through the
  sentinel-set commitments below, which the validation pass re-derives).

## Fixed protocol

Identical for every run; only the per-run seed varies (MLP init key chain and
online example order).

| constant | value |
| --- | --- |
| config | `IPMNISTConfig(n_tasks=3, task_length=5000)` — campaign geometry 784-300-150-10 |
| phase lengths | `(5000, 5000, 5000)` — permutation A, then B, then A again |
| permutation A | `np.random.default_rng(910).permutation(784).astype(np.int32)`; sha256 `7c314e5a3c885d1c904681e3177d3f13e7e57e43b9d29e9392880022a8a84e77` |
| permutation B | `np.random.default_rng(911).permutation(784).astype(np.int32)`; sha256 `a6a6c8ccace663789c6d71d2b5132246e5b56a9d0c0460b67627caf48ad03de2` |
| sentinel rows | 256 held-out rows, `np.random.default_rng(20261004).choice(60000, 256, replace=False)` (draw order kept); index-order commitment recorded in every receipt |
| sentinel-set commitments | A `4d2f874d52a0f6226798e6520d04e581f8ea387c1e8724c11dfd8df2ad175706`, B `978d31e575d899ad2a9f0ec872f05ad692470b57fdcda9f9392f50b44eebc65f` (over `x[indices]`, `y[indices]`, and the permutation, so they also bind the dataset bytes) |
| relearning window | 250 steps |
| probe schedule | evaluator-fixed: after every completed phase, every permutation seen so far is probed (A after phase 0; A and B after phases 1 and 2) — five probes per run |
| arms | `sigma0_shiftnorm_d099` (stored 200-task confirmed incumbent), `upgd_w_control` (published-configuration UPGD-W reference), `upgd_ema_norm` (EMA-normalized UPGD) |
| seeds | 0, 1, 2 — the campaign's consumed development screen seeds; reused here for a paired, permanently nonpromoting diagnostic only |

The stored overall champion family (`rls_head_resid_*`) cannot run through
this adapter: its sentinel-probe hook fails closed because the deployed model
is the champion body + RLS readout rather than the protocol MLP head the probe
harness scores. That fail-closed behavior is kept, and this grid therefore
covers the probe-compatible incumbent and controls, not the RLS champion.

## Driver (exact bytes, sha256
`8dd22ea4b15ffde545e614c2bf05bb2ea103844a451267a76dca948b825f6552`)

The lane has no CLI, so receipts are the exact stdout of this driver; its
policy assertions (development-only status, nonpromoting policy, thresholds
not applied) run inside every process:

```python
"""Recurring-IPMNIST A/B/A retained-outcome driver (development-only, nonpromoting).

Loads the canonical 60,000-example MNIST train split with the campaign's frozen
[-1, 1] scaling, then runs exactly one
``run_recurring_ipmnist_retention_development`` A/B/A retention report for the
arm named on the command line and prints its ``to_config()`` payload as JSON on
stdout. Any failure exits nonzero with the error on stderr.

The protocol is fixed and explicit: three 5,000-step phases (permutation A,
permutation B, permutation A again), 256 held-out sentinel rows, and a
250-step relearning window. The two permutations and the sentinel index draw
are derived from recorded one-off generator seeds and are identical for every
run; only the per-run seed varies (MLP init key chain and online example
order). The report binds permutation, sentinel-set, and online-order SHA-256
commitments itself; it applies no threshold and claims no retention.
"""

import json
import sys
from pathlib import Path

import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import (
    run_recurring_ipmnist_retention_development,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, load_mnist_train

ARMS = (
    "sigma0_shiftnorm_d099",
    "upgd_w_control",
    "upgd_ema_norm",
)
RUN_SEEDS = (0, 1, 2)
CONFIG = IPMNISTConfig(n_tasks=3, task_length=5000)
PHASE_LENGTHS = (5000, 5000, 5000)
RELEARNING_WINDOW = 250
PERMUTATION_A_SEED = 910
PERMUTATION_B_SEED = 911
SENTINEL_DRAW_SEED = 20261004
SENTINEL_COUNT = 256


def main() -> int:
    arm = sys.argv[1]
    seed = int(sys.argv[2])
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if seed not in RUN_SEEDS:
        raise ValueError(f"seed must be one of {RUN_SEEDS}")

    data_x, data_y = load_mnist_train(
        data_home=Path("outputs/upgd_ipmnist/openml_cache")
    )
    data_x = np.ascontiguousarray(data_x, dtype=np.float32)
    data_y = np.ascontiguousarray(data_y, dtype=np.int32)

    permutation_a = np.random.default_rng(PERMUTATION_A_SEED).permutation(
        CONFIG.input_dim
    ).astype(np.int32)
    permutation_b = np.random.default_rng(PERMUTATION_B_SEED).permutation(
        CONFIG.input_dim
    ).astype(np.int32)
    sentinel_indices = np.random.default_rng(SENTINEL_DRAW_SEED).choice(
        int(data_x.shape[0]), size=SENTINEL_COUNT, replace=False
    ).astype(np.int64)

    report = run_recurring_ipmnist_retention_development(
        data_x,
        data_y,
        screening_spec(arm),
        seed=seed,
        config=CONFIG,
        phase_lengths=PHASE_LENGTHS,
        permutations=(permutation_a, permutation_b, permutation_a.copy()),
        sentinel_indices=sentinel_indices,
        relearning_window=RELEARNING_WINDOW,
    )
    payload = report.to_config()
    if payload["development_status"] != "development-only-not-assessed":
        raise ValueError("report must be development-only and not assessed")
    if payload["scientific_promotion_allowed"] is not False:
        raise ValueError("report must be permanently nonpromoting")
    if payload["performance_thresholds_applied"] is not False:
        raise ValueError("report must apply no performance thresholds")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## Reproduction

```bash
# from a clean worktree at f3d32c451ed1c1715e477ad56b782fc7ea89b206
# assert the backend this environment record describes before any run:
PYTHONPATH=<worktree> .venv/bin/python -c "import jax; assert jax.default_backend() == 'cpu'; print(jax.devices())"

# save the driver above as run_grid.py, then run the grid with per-run exit
# statuses and wall-clock captured, aborting on the first nonzero exit:
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=<worktree>
: > statuses.txt
for arm in sigma0_shiftnorm_d099 upgd_w_control upgd_ema_norm; do
  for seed in 0 1 2; do
    start=$(date +%s)
    .venv/bin/python run_grid.py "$arm" "$seed" \
      > "receipt-$arm-seed$seed.json" \
      2> "receipt-$arm-seed$seed.stderr"
    status=$?
    end=$(date +%s)
    printf '%s seed%s exit=%s seconds=%s\n' "$arm" "$seed" "$status" "$((end-start))" >> statuses.txt
    [ "$status" -eq 0 ] || { echo "run failed; aborting" >&2; exit "$status"; }
  done
done
```

`statuses.txt` (retained verbatim):

```text
sigma0_shiftnorm_d099 seed0 exit=0 seconds=14
sigma0_shiftnorm_d099 seed1 exit=0 seconds=13
sigma0_shiftnorm_d099 seed2 exit=0 seconds=13
upgd_w_control seed0 exit=0 seconds=23
upgd_w_control seed1 exit=0 seconds=23
upgd_w_control seed2 exit=0 seconds=23
upgd_ema_norm seed0 exit=0 seconds=23
upgd_ema_norm seed1 exit=0 seconds=23
upgd_ema_norm seed2 exit=0 seconds=23
```

Receipt sha256 values are retained in `receipts.sha256`.

### Separate validation pass over the retained bytes

Run as its own process after the grid; it re-derives the permutations,
sentinel draw, per-seed online-example order, and each receipt's adapter
protocol commitment (`ipmnist-screening-aba-<sha256>.v1`, which binds arm
name, base learner, hyperparameters, seed, config, phase lengths, permutation
digests, sentinel-index digest, online-index digests, and relearning window)
and fails fast on any mismatch. sha256
`28043a57eef39b8a94b24d4d14180351e6ef0f92fa45b00a33e3129ba47fd3dd`:

```python
"""Independent validation pass over the retained recurring-IPMNIST grid bytes.

Re-derives the dataset, permutations, sentinel indices, online-index order,
and each receipt's adapter protocol commitment from the recorded derivation
seeds, then checks every retained receipt against them. Fails fast on any
mismatch. This script reads retained bytes only; it executes no learner.
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import (
    _array_bundle_sha256,
    _recurring_protocol_id,
    build_recurring_ipmnist_online_indices,
    ipmnist_permutation_sha256,
    ipmnist_sentinel_set_sha256,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, load_mnist_train

OUT = Path(sys.argv[1])
ARMS = ("sigma0_shiftnorm_d099", "upgd_w_control", "upgd_ema_norm")
RUN_SEEDS = (0, 1, 2)
CONFIG = IPMNISTConfig(n_tasks=3, task_length=5000)
PHASE_LENGTHS = (5000, 5000, 5000)
RELEARNING_WINDOW = 250
PERMUTATION_A_SEED = 910
PERMUTATION_B_SEED = 911
SENTINEL_DRAW_SEED = 20261004
SENTINEL_COUNT = 256

data_x, data_y = load_mnist_train(data_home=Path("outputs/upgd_ipmnist/openml_cache"))
data_x = np.ascontiguousarray(data_x, dtype=np.float32)
data_y = np.ascontiguousarray(data_y, dtype=np.int32)

permutation_a = (
    np.random.default_rng(PERMUTATION_A_SEED).permutation(CONFIG.input_dim).astype(np.int32)
)
permutation_b = (
    np.random.default_rng(PERMUTATION_B_SEED).permutation(CONFIG.input_dim).astype(np.int32)
)
sentinel_indices = (
    np.random.default_rng(SENTINEL_DRAW_SEED)
    .choice(int(data_x.shape[0]), size=SENTINEL_COUNT, replace=False)
    .astype(np.int64)
)

expected_permutation = (
    ipmnist_permutation_sha256(permutation_a),
    ipmnist_permutation_sha256(permutation_b),
)
expected_sentinel_sets = (
    ipmnist_sentinel_set_sha256(data_x, data_y, permutation_a, sentinel_indices),
    ipmnist_sentinel_set_sha256(data_x, data_y, permutation_b, sentinel_indices),
)

paths = sorted(glob.glob(str(OUT / "receipt-*.json")))
assert len(paths) == 9, paths
for path in paths:
    stem = Path(path).stem
    arm, seed_text = stem[len("receipt-") :].rsplit("-seed", 1)
    seed = int(seed_text)
    assert arm in ARMS and seed in RUN_SEEDS, path
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["schema"] == "alberta.recurring-ipmnist-retention.report.v1", path
    assert payload["development_status"] == "development-only-not-assessed", path
    assert payload["assessment_status"] == "not-assessed", path
    assert payload["scientific_promotion_allowed"] is False, path
    assert payload["performance_thresholds_applied"] is False, path
    assert payload["retention_claimed"] is False, path
    assert payload["catastrophic_forgetting_absence_claimed"] is False, path

    protocol = payload["protocol"]
    phases = protocol["phases"]
    assert [(p["phase_index"], p["start_step"], p["length"], p["exposure_index"]) for p in phases] == [
        (0, 0, 5000, 0),
        (1, 5000, 5000, 0),
        (2, 10000, 5000, 1),
    ], path
    assert protocol["relearning_window"] == RELEARNING_WINDOW, path
    bindings = protocol["sentinel_bindings"]
    assert bindings[0]["permutation_sha256"] == expected_permutation[0], path
    assert bindings[1]["permutation_sha256"] == expected_permutation[1], path
    assert bindings[0]["sentinel_set_sha256"] == expected_sentinel_sets[0], path
    assert bindings[1]["sentinel_set_sha256"] == expected_sentinel_sets[1], path
    assert all(b["sentinel_case_count"] == SENTINEL_COUNT for b in bindings), path

    resolved_seed = seed
    online = build_recurring_ipmnist_online_indices(
        seed=resolved_seed,
        n_examples=int(data_x.shape[0]),
        phase_lengths=PHASE_LENGTHS,
        sentinel_indices=sentinel_indices,
    )
    expected_online = tuple(
        _array_bundle_sha256(
            "alberta.ipmnist-screening.online-example-order.v1",
            {"example_indices": phase_indices},
        )
        for phase_indices in online
    )
    expected_id = _recurring_protocol_id(
        spec=screening_spec(arm),
        seed=resolved_seed,
        config=CONFIG,
        phase_lengths=PHASE_LENGTHS,
        permutation_sha256=(
            expected_permutation[0],
            expected_permutation[1],
            expected_permutation[0],
        ),
        sentinel_indices_sha256=_array_bundle_sha256(
            "alberta.ipmnist-screening.sentinel-index-order.v1",
            {"sentinel_indices": sentinel_indices},
        ),
        online_indices_sha256=expected_online,
        relearning_window=RELEARNING_WINDOW,
    )
    assert protocol["protocol_id"] == expected_id, (path, protocol["protocol_id"], expected_id)

    assert len(payload["phase_summaries"]) == 3, path
    assert len(payload["sentinel_scores"]) == 5, path
    assert isinstance(payload["trace_sha256"], str) and payload["trace_sha256"], path
    assert (
        isinstance(payload["sentinel_snapshots_sha256"], str)
        and payload["sentinel_snapshots_sha256"]
    ), path

print(f"validated {len(paths)} receipts against independently re-derived bindings")
```

Executed output (verbatim):

```text
validated 9 receipts against independently re-derived bindings
```

## Outcome grid (descriptive only — no threshold is applied or implied)

Per-phase pre-update online accuracy (5,000 observations per phase; mean over
seeds 0–2, per-seed values in the receipts):

| arm | phase 0 (A, first exposure) | phase 1 (B) | phase 2 (A, revisit) |
| --- | ---: | ---: | ---: |
| `sigma0_shiftnorm_d099` | 0.7803 | 0.8609 | 0.8960 |
| `upgd_ema_norm` | 0.7831 | 0.8514 | 0.8757 |
| `upgd_w_control` | 0.6879 | 0.7729 | 0.8123 |

Sentinel accuracy on the same 256 held-out rows per probe point (mean over
seeds 0–2; A rows under permutation A, B rows under permutation B):

| arm | A after phase 0 | A after phase 1 (post-B recall) | B after phase 1 | A after phase 2 | B after phase 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sigma0_shiftnorm_d099` | 0.8594 | 0.5911 | 0.8646 | 0.8685 | 0.7031 |
| `upgd_ema_norm` | 0.8672 | 0.5312 | 0.8724 | 0.8685 | 0.6758 |
| `upgd_w_control` | 0.8372 | 0.6159 | 0.8568 | 0.8398 | 0.6133 |

Plain descriptive reading, at n=3, with no statistical claim:

- Every arm, on every seed, reaches higher online accuracy during the A
  revisit than during the first A exposure (for example
  `sigma0_shiftnorm_d099`: 0.8960 vs 0.7803) — relearning is faster than
  first learning in this A/B/A structure.
- Sentinel accuracy for the revisited A permutation drops sharply during the
  B phase in every arm (to 0.53–0.62 from 0.84–0.87) and recovers after the
  5,000-step revisit to 0.84–0.87 — at or slightly above each arm's own
  post-first-exposure level for the two conditioned arms.
- Interference is roughly symmetric: B recall after the A revisit
  (0.61–0.70) sits in the same band as A recall after B.
- `upgd_w_control` seed 1 is the widest per-seed outlier
  (A-after-phase-2 sentinel accuracy 0.4297 vs 0.6797/0.7305 on its sibling
  seeds).

The report schema's own `metric_definitions` (for example
`peak_to_revisit_forgetting`, `relearning_savings_accuracy`,
`retention_change_from_acquisition`) define how a future assessor could
quantify these observations; computing and gating them against thresholds is
exactly what this threshold-free retention deliberately does not do.

## Limits

- One fixed A/B/A recurrence, one geometry, three seeds: a development
  diagnostic, not external validation; the receipts' own `limitations` field
  carries the schema-level limits (evaluator-supplied uncalibrated sentinel
  cases, state hashes detect declared probe mutation only, relearning savings
  can coexist with poor direct retention, one-step plasticity is not retained
  knowledge).
- The stored champion family (`rls_head_resid_*`), `naive_bayes`,
  `nb_ensemble_*`, `rff_rls`, hidden-RMS-normalized, and bounded-structure
  arms fail closed on sentinel probes and are out of scope for this grid.
- Wall-clock values are telemetry from an unqualified timing protocol; they
  carry no resource or latency claim.
- These records bind no registered evidence source, populate no
  `reference-dev` configuration, and are permanently nonpromoting.
