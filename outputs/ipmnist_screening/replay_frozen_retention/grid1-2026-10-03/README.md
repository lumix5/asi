# grid1-2026-10-03 — first retained replay/frozen-feature ceiling outcome grid

First executed and retained development comparison of the eight registered
`replay_in_context` / `frozen_feature_ceiling` arms of
`alberta_framework/benchmarks/ipmnist_screening.py` (issue #1573 slice;
"executed screening over new campaign infrastructure" per maintainer guidance
on #2280). Permanently nonpromoting: every receipt carries
`development_only: true`, `scientific_promotion_allowed: false`, and
`negative_outcome_retained: true`. No claim here supports promotion, an
external-paper comparison, or SOTA.

## Provenance

| field | value |
| --- | --- |
| base revision | `8b544f8ce79c2cb393f6d45f9fd50d0eb8b06686` (branch `benchmarks/replay-frozen-retained-outcome-v1`, stacked on PR #3057 `fix/frozen-extractor-threefry-roots` at the same commit; `git_tree de8eb35e2c9f…`, worktree clean at run time) |
| `alberta_framework/benchmarks/ipmnist_screening.py` | sha256 `644e5d365e84d07d17cafb27ebe4aad83ec6a3e4966f06e53da1886700e7c457` |
| `alberta_framework/benchmarks/replay_frozen_ipmnist.py` | sha256 `6a802a0bf7c7dd61a3d3891df4f02f31873ec78eb65d12553d1d65a944816a44` |
| `alberta_framework/evaluation/replay_frozen_ipmnist_nonpromoting.py` | sha256 `cf9142e15fb198d0e82edf28af6ed7dcde5dfa9f28cf7d05673ff1c482e998c0` |
| dataset | OpenML `mnist_784` v1, rows `[0, 60000)`; materialized float32 X sha256 `b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313`, int32 y sha256 `4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a` (recorded identically in all 24 shards) |
| runtime | CPython 3.14.4, JAX CPU backend, `jax_default_prng_impl=threefry2x32`, `jax_enable_x64=False`, Linux x86_64 `7.0.0-34-generic`; per-shard full bindings in each shard's `environment` block |
| protocol | 60 tasks × 5,000 examples, `--noise-mode step`, registered arm hyperparameters, frozen seeds {0, 1, 2}, 8 arms = 24 fresh-process shards |
| merged summary | `summary.json` (schema `alberta.ipmnist_screening.summary.v2`, control `replay_context_mechanism_off`, `slope_window 15`) |

The comparison receipt payload (`asi.replay-frozen-ipmnist.development-result.v1`)
is embedded as `mechanism_receipt` in every shard and strict-revalidated by
`load_shard` at merge time; `merge` also re-derives source/runtime/dataset
bindings and rejects any drift (it rejected an early merge attempt from a
process without `OMP_NUM_THREADS=1`, which is the behavior working as
designed).

## Reproduction

From the repository root at the base revision, with the project venv:

```bash
R=outputs/ipmnist_screening/replay_frozen_retention/grid1-2026-10-03
for arm in replay_context_mechanism_off replay_gradient_only \
           replay_context_only replay_context_full randumb_random_features \
           ranpac_random_projection prol_prompt_mechanism_off prol_prompt_proxy; do
  for seed in 0 1 2; do
    OMP_NUM_THREADS=1 .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
      --config-name "$arm" --seed "$seed" --n-tasks 60 --task-length 5000 \
      --noise-mode step --data-home <openml-cache> --out "$R/shards/${arm}_seed${seed}.json"
  done
done
OMP_NUM_THREADS=1 .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards "$R"/shards/*.json --control-name replay_context_mechanism_off \
  --output "$R/summary.json"
```

`<openml-cache>` was
`/home/lumix/asi/outputs/upgd_ipmnist/openml_cache` (the lane's standard
`default_openml_data_home()` fallback location, already populated).

Execution honesty: shards ran as one fresh process each, but three arms were
executed concurrently (three single-thread lanes, `OMP_NUM_THREADS=1`), so
`wall_clock_seconds` telemetry is inflated by contention for the replay-family
arms that were co-scheduled; frozen-extractor arms ran after the trained arms
finished and show near-bare-metal times. Timing is telemetry-only
(`timing_is_telemetry_only: true`) and no decision here uses it. The
`replay_gradient_only` seed-2 shard was interrupted once by an operator process
cleanup and rerun from scratch in a fresh process; its retained stdout/stderr
are from the successful rerun.

## Files

- `shards/<arm>_seed<seed>.json` — 24 strict v2 screening shards, each with its
  embedded validated `mechanism_receipt`.
- `shards/<arm>_seed<seed>.stdout.txt` / `.stderr.txt` — exact fresh-process
  byte streams (stderr carries the lane's progress logger lines only; no
  tracebacks).
- `summary.json` — strict merged ranking with paired-vs-control blocks.

## Outcome

See `SUMMARY.md`. Development-only, permanently nonpromoting, retained
including the negative and mixed findings.
