# Intentional Updates IPMNIST — retention grid 1 (2026-10-04)

First retained development outcomes for the registered `intentional_updates_*`
screening family. Advances
[issue #1561](https://github.com/SlopDotCash/asi/issues/1561) (does not close
it). `docs/research/sota-landscape.md` states for this lane: "The
implementation and its tests are development infrastructure; no screening
result exists until a matched campaign is run." This grid is that first
matched campaign, at a deliberately bounded scale.

Everything here is development screening evidence:
`development_only: true`, `scientific_promotion_allowed: false` in every
receipt and both merged summaries. Nothing here promotes, confirms, or
refutes the paper.

## Identity

- Source: `SlopDotCash/asi` at `f3d32c451ed1c1715e477ad56b782fc7ea89b206`
  (git tree `602620ab20bd149ba3bca258f2b0cc94ce398527`, worktree clean).
- Bound source identity (recorded inside every receipt and summary):
  `relevant_source_sha256 5c908bb81e548f355276825232fde23d2d089862ccaa7eedd7b77257fd4e0aea`
  over `tracked:alberta_framework/**,pyproject.toml,uv.lock`
  (`uv_lock_sha256 0fc52a642ba70bd64bfa80049b1efc99c35829d01e44705370994ad818ee82d7`).
- Dataset: OpenML `mnist_784` v1, rows 0–60000, materialization
  `alberta.ipmnist.float32-neg1-pos1-int32-labels.v1`;
  `x.sha256 b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313`,
  `y.sha256 4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a`.
- Environment: CPython 3.14.4, JAX 0.11.0 / jaxlib 0.11.0 (CPU, threefry),
  NumPy 2.5.1, chex 0.1.92, `OMP_NUM_THREADS=1`.
- Artifacts only: no library, test, doc, or registry bytes are touched, so no
  pinned source identity drifts.

## Design

Five registered arms × four paired development seeds (18600–18603; fresh
seeds consumed by this grid, backing no promotion now or later) × two bounded
cells of the screening protocol, `noise_mode=step`:

- **cell A** — 8 tasks × 5,000 steps (`--n-tasks 8 --task-length 5000`)
- **cell B** — 20 tasks × 5,000 steps (`--n-tasks 20 --task-length 5000`)

Because `build_schedule` folds the task index into per-seed keys, each cell-B
run is an exact extension of the same-seed cell-A run's schedule. Cells use
the lane's default architecture (hidden 300×150). The lane's 60-task proxy
protocol and its `validate-proxy` gates were deliberately **not** used: this
grid is a bounded retention cell, not a proxy-validated wave, and the
intentional arms are not proxy controls.

Arms (from `SCREENING_REGISTRY`):

| arm | mechanism state |
| --- | --- |
| `intentional_updates_ipmnist` | full supervised Eq. 5 extension |
| `intentional_updates_no_diag` | − RMSProp diagonal normalization |
| `intentional_updates_no_clip` | − adaptive delta clip |
| `intentional_updates_head_only` | feature updates frozen (head only) |
| `intentional_updates_off` | mechanism-off control (fixed-step normalized SGD) |

## Execution

One fresh process per (arm, seed, cell), `OMP_NUM_THREADS=1`, up to 3-way
parallel on a shared workstation:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name <arm> --seed <seed> --n-tasks {8|20} --task-length 5000 \
  --noise-mode step --progress-every {4|10} \
  --out outputs/intentional_updates_retention/grid1-2026-10-04/receipt-<cell>-<arm>-seed<seed>.json

OMP_NUM_THREADS=1 .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards outputs/intentional_updates_retention/grid1-2026-10-04/receipt-cellA-*.json \
  --control-name intentional_updates_off \
  --output outputs/intentional_updates_retention/grid1-2026-10-04-summary.json

OMP_NUM_THREADS=1 .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards outputs/intentional_updates_retention/grid1-2026-10-04/receipt-cellB-*.json \
  --control-name intentional_updates_off \
  --output outputs/intentional_updates_retention/grid1-2026-10-04-summary-cellB.json
```

All 40 runs exited 0 and were written atomically by the CLI's immutable-output
preflight (`_preflight_new_output` refuses occupied paths), so no receipt was
ever overwritten. The merge CLI validated every shard's protocol config,
noise mode, seeds, and arm identity at merge time.

Per-run worker stdout/stderr logs are **not** retained: the shared-machine
run needed idempotent re-attempts after externally terminated workers, so
surviving log files would ambiguously mix attempts. Receipts are
self-contained (config, hyperparameters, per-task vectors, wall clock,
dataset and source provenance, evidence policy), and all 40 receipts were
re-loaded and re-validated from the retained bytes by separate fresh
processes after the runs. For the record, the immutability guard was observed
working during the campaign: one duplicate re-attempt
(`receipt-cellA-intentional_updates_off-seed18603.json`) was refused with
`FileExistsError` before any bytes were touched.

## Validation

- 40/40 receipts re-validated from retained bytes via `load_shard` in fresh
  processes; `development_only: true` and
  `scientific_promotion_allowed: false` asserted on every one.
- Both merged summaries re-load and pin the same source/dataset identities.
- `tests/test_intentional_updates_ipmnist.py`: 16 passed at this revision.
- Timing is telemetry-only (no qualified timing protocol); wall-clock fields
  in receipts reflect shared-machine load and must not be read as arm cost.

See `SUMMARY.md` for the measured outcome and its honest reading.
