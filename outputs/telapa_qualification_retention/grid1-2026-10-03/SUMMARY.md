# Summary — grid1 (2026-10-03)

Per-arm mean reward over each smoke run (three frozen seeds each; every seed
returned the same per-arm mean inside a setting, so one column per arm is
shown; per-seed values are in the receipts). `fixed_snapshot` and
`mechanism_off` matched on every hash axis in every run, as the smoke's
internal parity gate requires; their columns are equal by construction and
both are shown for completeness.

| steps | phase | diverse_archive | one_model | fixed_snapshot | mechanism_off | archive entries |
|------:|------:|----------------:|----------:|---------------:|--------------:|----------------:|
| 32 | 4  | 0.40625 | 0.37500 | **0.46875** | **0.46875** | 4 / 1 / 1 / 0 |
| 32 | 8  | 0.46875 | 0.43750 | **0.46875** | **0.46875** | 3 / 1 / 1 / 0 |
| 32 | 16 | **0.46875** | **0.46875** | **0.46875** | **0.46875** | 2 / 1 / 1 / 0 |
| 64 | 4  | 0.39062 | 0.37500 | **0.48438** | **0.48438** | 4 / 1 / 1 / 0 |
| 64 | 8  | 0.45312 | 0.43750 | **0.48438** | **0.48438** | 4 / 1 / 1 / 0 |
| 64 | 16 | **0.48438** | 0.46875 | **0.48438** | **0.48438** | 3 / 1 / 1 / 0 |

Archive-entry counts are per seed for `diverse_archive / one_model /
fixed_snapshot / mechanism_off` (identical across the three seeds in every
setting).

## Honest reading (development-only, nonpromoting)

- **The archive never beat the fixed snapshot in this grid.** Five of six
  settings have `fixed_snapshot`/`mechanism_off` strictly ahead of
  `diverse_archive`; (32, 16) and (64, 16) are ties. Against `one_model` the
  archive is ahead in four settings and tied in two. Under the lane's own
  retention policy these are valid outcomes, ties and regressions included,
  and they are retained here rather than dropped.
- **Scale caveat is the whole story.** The smoke caps at 64 steps / 16
  boundaries with a 2x2 tabular policy: this qualifies the byte-bounded
  archive primitive and the retention path, and says nothing about archive
  benefit at task scale, about TeLAPA (paper parity is rejected), or about ASI
  progress. No promotion, performance, or paper-parity claim is made or
  licensable from these receipts (`scientific_promotion_allowed: false`,
  `paper_parity_allowed: false` in every record).
- **Parity gate held everywhere**: `fixed_snapshot` and `mechanism_off` have
  identical observation/action/reward and initial/final policy hashes in all
  six runs (exit-0 receipts require it).
- **Resources are bounded and recorded per run** in each receipt
  (`environment_steps`, `observations_consumed`, `policy_updates`,
  `policy_queries`, `descriptor_model_queries`, `task_boundary_disclosures`,
  `archive_persistent_bytes` at most 318 B, `active_policy_persistent_bytes`
  16 B). Six runs cost seconds of wall clock; timing stays telemetry-only.

## What would make the next retention informative

The frozen seeds and the 64-step ceiling are the smoke's design; a comparison
that can detect an archive effect needs the same matched four-arm protocol at
many more steps and boundaries with more seeds, which is a separate,
still-nonpromoting development run outside this smoke's ceiling — and, per the
runbook, paper-parity gates stay closed until the license and source-identity
gates close.
