# Summary — NaP IPMNIST retention grid 1 (2026-10-03)

Development-only, permanently nonpromoting. Five matched arms —
`sgd_current_control`, `nap_mechanism_off`, `normalization_only`,
`projection_only`, `nap` — receive the same frozen seed, schedule, labels,
update count, observation count, and task-hidden learner inputs
(`task_ids_visible_to_learner: false`,
`task_boundaries_visible_to_learner: false`). "Accuracy" below is the lane's
own mean per-task accuracy over its 64-example scheduled tasks under
per-example SGD — a plasticity-lane signal, **not** a held-out generalization
metric.

## Parity gate (all eight runs)

`sgd_current_control` and `nap_mechanism_off` match exactly in every curve,
final-state digest, hidden norm, and non-timing receipt field in all eight
runs; the CLI fails closed otherwise and all eight exited 0 with empty stderr.
`observations_matched_before_causal_divergence: true` in every receipt.

## bounded-development (8 tasks x 64 examples, width 64, lr 3e-3)

Mean per-task accuracy per seed:

| seed | control | mechanism_off | norm_only | proj_only | nap | d(norm) | d(proj) | d(nap) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15640 | 0.177734 | 0.177734 | **0.242188** | 0.177734 | 0.232422 | +0.064453 | +0.000000 | +0.054688 |
| 15641 | 0.136719 | 0.136719 | **0.199219** | 0.136719 | 0.187500 | +0.062500 | +0.000000 | +0.050781 |
| 15642 | 0.148438 | 0.148438 | **0.205078** | 0.148438 | 0.205078 | +0.056641 | +0.000000 | +0.056641 |
| 15643 | 0.152344 | 0.152344 | 0.222656 | 0.152344 | **0.226562** | +0.070312 | +0.000000 | +0.074219 |

Four-seed means (per-task effective rank and dead-unit fraction in
parentheses, mean over tasks and seeds):

| arm | accuracy | effective_rank | dead_unit_fraction |
| --- | --- | --- | --- |
| sgd_current_control | 0.153809 | 28.757 | 0.0205 |
| nap_mechanism_off | 0.153809 | 28.757 | 0.0205 |
| normalization_only | **0.217285** | 26.429 | 0.0378 |
| projection_only | 0.153809 | 28.723 | 0.0203 |
| nap | 0.212891 | 26.265 | 0.0374 |

## contract-smoke (2 tasks x 4 examples, width 8)

All five arms sit at 0.031–0.0625 mean accuracy — pure plumbing scale. It is
retained because the qualification doc's canonical command runs exactly this
profile, not because it can order arms. `normalization_only`/`nap` already
show the slightly higher effective rank (2.886 vs 2.651) they show at
development scale.

## Honest reading (development-only, nonpromoting)

- **The normalization half carries the whole effect.** `normalization_only` is
  ahead of the control in all four seeds (+0.057 to +0.070). `nap` is also
  ahead of the control in all four seeds but trails `normalization_only` in
  two seeds, ties it in one, and leads in one. Nothing here suggests the
  projection mechanism adds accuracy at this scale.
- **`projection_only` changed no seed's mean accuracy at all** (+0.000000
  every seed) — its final-state and effective-rank digests do differ slightly
  from the control's, so the trajectory moves, but not enough to move any
  64-example task's predictions. This is a valid (near-)tie and is retained on
  purpose.
- **The gain is not free in representation terms.** Both normalization arms
  show *lower* final effective rank (26.3 vs 28.8) and a *higher* dead-unit
  fraction (0.037 vs 0.021) than the control. If the lane's hidden-state
  health metrics are the point, normalization's accuracy gain at this scale
  comes with more concentrated hidden representations — worth knowing before
  anyone reads `nap` as a strictly better arm.
- **Scale caveat is decisive.** These are 512-update, width-64, 8-task runs on
  the diagnostics lane's own schedule with near-chance absolute accuracies
  (~0.15–0.24 vs 0.1 chance). The differences order the lane's ablation
  structure for development; they say nothing about NaP's paper claims
  (random-label CIFAR-10, sequential ALE/Rainbow — none implemented here), do
  not establish paper parity (`official_nap_code_available: false`,
  paper-parity gates closed), and cannot promote anything
  (`scientific_promotion_allowed: false` in every receipt).

## What would make the next retention informative

The same five-arm matched protocol at more tasks, more examples per task, more
seeds, and a real held-out readout would let the normalization-vs-projection
question separate from schedule noise — still development-only. A
strengthening control would add an `adam`-side normalization arm: the paper's
continual experiment is Adam-based, while this lane's control is plain SGD, so
any paper-adjacent reading of the current numbers is blocked by design.
