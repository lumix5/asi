# Noise-curvature IPMNIST retention — grid1-2026-10-05

First retained outcome grid for the four-arm layerwise noise-curvature
scheduling lane (`alberta_framework/benchmarks/noise_curvature_ipmnist.py`,
strict receipts in
`alberta_framework/evaluation/noise_curvature_ipmnist_nonpromoting.py`).
The lane had a registered runner, a strict validator, and 17 passing unit
tests, but **zero retained outcomes anywhere in the repository** before this
grid. This directory retains those first outcomes, including the negative
ones, exactly as the lane's receipts are designed to do.

Development-only and permanently nonpromoting: every receipt carries
`development_only: true` and `scientific_promotion_allowed: false`. Nothing
here promotes a claim, populates `reference-dev`, or constitutes performance
or scientific evidence.

## Provenance

- Repository head: `f3d32c451ed1c1715e477ad56b782fc7ea89b206` (tree
  `602620ab20bd149ba3bca258f2b0cc94ce398527`).
- Source identity: `relevant_source_sha256`
  `5c908bb81e548f355276825232fde23d2d089862ccaa7eedd7b77257fd4e0aea` over
  `tracked:alberta_framework/**,pyproject.toml,uv.lock` (230 files), bound in
  every shard. No library, doc, test, or registry bytes are touched by this
  retention, so the bound identity stays exact at this head.
- Receipt schema `asi.noise-curvature-ipmnist.development-result.v1`,
  comparison `asi.noise-curvature-ipmnist.current-runner.v1`, paper pin
  `arXiv:2509.19698v3` (official code status: none identified), live control
  id `rls_head_resid_l1_preset005` (not executed here; see "Not measured").
- Dataset: OpenML `mnist_784` version 1, rows `[0, 60000)`, scaled to
  `[-1, 1]`; `x` sha256 `b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313`,
  `y` sha256 `4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a`.
  Reconstruct with `load_mnist_train(data_home)`; the cache directory was
  passed as `--data-home outputs/upgd_ipmnist/openml_cache`.
- Runtime: CPython 3.14.4, jax 0.11.0, jaxlib 0.11.0, numpy 2.5.1,
  scikit-learn 1.9.0, CPU backend, Linux x86_64, `OMP_NUM_THREADS=1` (full
  per-shard environment in each shard's `environment` block).

## Grid

Four registered arms x five frozen development seeds (`0`–`4`, the lane's
declared `DEVELOPMENT_SEEDS`) x one bounded cell (10 tasks x 5000 examples =
50,000 online updates per run; `task_length` divisible by the registered
`control_interval` 40). Twenty fresh-process runs, serial, `OMP_NUM_THREADS=1`,
all exit 0. Each stderr file contains only the CLI's own INFO progress log —
no warnings and no tracebacks. Total wall clock 1338.3 s (telemetry only).

Every run shares the control's task/example schedule for the same seed (the
screening runner derives schedules identically across arms), so arm-vs-control
comparisons are paired by seed.

Resource accounting per run (validated by the lane's own strict validator):
50,000 data steps, 1,250 controller events, 153,750 model queries
(100,000 first-order gradient + 50,000 loss-only + 3,750 HVP), 4,640,356
persistent bytes. Timing is telemetry-only.

## Files

- `shards/shard-<arm>-seed<s>.json` — raw screening CLI outputs
  (`schema alberta.ipmnist_screening.shard.v2`), one per run.
- `shards/stderr/shard-<arm>-seed<s>.stderr` — captured stderr per run.
- `receipts/receipt-<arm>-seed<s>.json` — strict nonpromoting development
  receipts rebuilt from the shard bytes by the retained driver and validated
  with `validate_noise_curvature_development_result`; the five per-seed
  four-arm panels also pass `validate_matched_noise_curvature_results`.
- `validate_retention.py` — retained driver; re-validates every retained
  shard byte, rebuilds and validates the receipts, applies the declared
  decision rule, and writes `analysis_summary.json`.
- `analysis_summary.json` — machine-readable paired results and outcomes.
- `SUMMARY.md` — tables and honest reading.

## Reproduction

From the repository root at the head above, with the project venv and the
OpenML cache:

```bash
export OMP_NUM_THREADS=1
for arm in noise_curvature_fixed_adam_l2 noise_curvature_gradient_only \
           noise_curvature_volatility_only noise_curvature_combined; do
  for seed in 0 1 2 3 4; do
    .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
      --config-name "$arm" --seed "$seed" --n-tasks 10 --task-length 5000 \
      --data-home outputs/upgd_ipmnist/openml_cache \
      --out "shards/shard-${arm}-seed${seed}.json" --progress-every 5
  done
done
.venv/bin/python outputs/noise_curvature_ipmnist_retention/grid1-2026-10-05/validate_retention.py
```

The validator is deterministic: re-running it reproduces
`analysis_summary.json` byte-for-byte (checked twice on this host).

## Decision rule (declared)

Per scheduler arm A vs the mechanism-off arm `noise_curvature_fixed_adam_l2`:
paired per-seed deltas of mean online accuracy, d_s = mean(A, s) − mean(control, s),
five seeds. `supported` requires every d_s > 0 and the two-sided paired-t 95%
interval (df = 4, t = 2.776445) entirely above zero; `rejected` requires every
d_s < 0 and the interval entirely below zero; otherwise `inconclusive`. The
control arm is the comparator, not a candidate, so its receipt carries the
neutral `inconclusive` outcome.

The hillclimb gate against the separately-protocolled live RLS control and the
200-task development confirmation are **not** measured or claimed by this grid.
