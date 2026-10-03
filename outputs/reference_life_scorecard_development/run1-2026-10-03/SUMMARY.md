# SUMMARY — reference-life development scorecard run 1 (2026-10-03)

**Permanently nonpromoting development selection output.** No `reference-dev`
selection, no performance claim, no scientific evidence. Numbers below are
development diagnostics from one single-host execution of the literal frozen
plan (`plan_sha256 4c4396848da3abebc53fb61c7b241866566ad2b44fa83ee3395a5a389a869b86`; see
`artifact.json` for the authoritative digest); they are expected to move and
must be re-measured before any comparison use.

## Execution

- **144/144 fresh-process shards completed**; strict shard validator passed on
  every record; `summarize` accepted the complete matrix and the aggregate
  validator accepted `artifact.json` (`valid: true`).
- Aggregate `status = valid_baseline_failure`, `failure_count = 0`,
  `parameter_change_failure_count = 0`: every life ran to completion; the
  status is a **development-gate outcome** (below), not an execution or
  validation failure.
- Source: exact `main` `f3d32c45`, executed from an isolated worktree. The
  first launch attempt ran in the shared checkout and **failed closed exactly
  as designed**: a concurrent editor mutated `core/upgd_memory.py` and
  `core/world_model.py` mid-run, and every affected shard was rejected with
  `source identity differs from the current source tree`. Nothing from that
  attempt was retained; the retained run's source identity
  (`1b34e65ecdc2bd7a` over 226 files) was verified identical before shard
  execution, during the run, and at aggregation.
- Runtime: Python 3.12.12, jax/jaxlib 0.11.0, NumPy 2.5.1, CPU backend,
  single Linux host. Wall clock ≈ 9.5 h (switching ≈ 1 h at 6 workers;
  riverswim ≈ 8.5 h at 10 workers; riverswim shards are ~5× the horizon).
  Timing is telemetry-only.

## Headline development numbers

`reward_sum` mean ± stderr over the 12 seeds; `norm` = within-environment
normalized score `(arm − random) / mean_seed(oracle − random)`; `lcb` =
paired-t 95% lower confidence bound of the normalized score.

### switching_two_state (scale = 1993.08; calibration qualified)

| arm | reward_sum | norm | lcb95 |
|---|---|---|---|
| privileged_oracle | 4000.00 ± 0.00 | 1.0000 | 0.9882 |
| differential_sarsa | 3277.08 ± 9.23 | 0.6373 | 0.6161 |
| prototype | 3006.92 ± 21.10 | 0.5017 | 0.4722 |
| sarsa | 2932.75 ± 12.32 | 0.4645 | 0.4424 |
| random | 2006.92 ± 10.72 | 0.0000 | 0.0000 |
| prototype_frozen | 1996.67 ± 3.36 | −0.0051 | −0.0190 |

Control ordering is sane (oracle > learned arms > random ≈ frozen). Both
SARSA-family arms qualify (`lcb > 0.1`); `differential_sarsa` is the best
qualifying control. Prototype clears the frozen/no-learning delta
(`prototype_vs_frozen` paired LCB = +0.4841, gate ≥ 0.05 → **passed**) but
trails the best qualifying SARSA control (paired LCB = −0.1582).

### riverswim (scale = 17034.18; calibration NOT qualified)

| arm | reward_sum | norm | lcb95 |
|---|---|---|---|
| privileged_oracle | 17094.67 ± 29.53 | 1.0000 | 0.9961 |
| sarsa | 6304.94 ± 2214.73 | 0.3666 | 0.0805 |
| prototype | 5463.07 ± 1708.22 | 0.3172 | 0.0963 |
| prototype_frozen | 1080.11 ± 1021.31 | 0.0599 | −0.0722 |
| differential_sarsa | 96.61 ± 0.04 | 0.0021 | 0.0017 |
| random | 60.48 ± 3.20 | 0.0000 | 0.0000 |

No arm qualifies (`minimum_lcb_exclusive = 0.1`): `sarsa` reaches LCB 0.0805
and `prototype` 0.0963 — both below the gate — and `differential_sarsa`
collapses to ≈random at this horizon (norm 0.0021). The prototype-vs-frozen
gate **fails by a hair**: paired LCB = +0.0468 against the ≥ 0.05 inclusive
threshold.

## Reading the numbers

- The aggregate therefore reports `candidate_selection_status =
  "not_evaluated_baseline_failure"` and
  `control_calibration_gate_passed = false` **driven by RiverSwim**: the
  frozen control-calibration gates fail-closed rather than letting a
  miscalibrated environment feed candidate selection. That is the machinery
  working, not a defect.
- Honest negative outcomes retained: (1) on RiverSwim at horizon 20000,
  neither SARSA-family arm qualifies and differential SARSA ≈ random;
  (2) prototype's learning delta over its own frozen ablation misses the
  0.05 LCB floor on RiverSwim by 0.0032 while passing decisively on
  SwitchingTwoState (+0.4841).
- `cross_environment_pooling_forbidden = true`: the two environments must not
  be pooled; each table above stands alone.
- `pareto_resource_decision`: latency remains telemetry-only; no resource
  Pareto decision is taken (`reason: cold/warmed latency is telemetry-only
  and cannot be used in selection`).

## Cost

- 144 fresh processes, ≈ 9.5 h wall clock single-host, aggregate shard-record
  bytes ≈ 15 MiB (see `receipt.json` for the exact per-file inventory).
- Persistent-state bytes, environment/data steps, and model-query accounting
  are recorded per shard inside each record (canonical initial/final numeric
  payload accounting, static oracle policy bytes included).

## What this does not establish

- Single host, single runtime, one source revision: no cross-machine
  source-identity or RiverSwim oracle ULP variance is exercised (see #2886,
  #3047, #3052); no CI-authorized attestation exists for this run.
- No promotion of any kind: the scorecard is permanently nonpromoting; these
  numbers select nothing and validate nothing beyond internal consistency.
- Development seeds 70000–70011 are consumed by this run; they must not be
  reused for any promotion protocol.
- Timing values are telemetry-only.
