# reference-life development scorecard — retained run 1 (2026-10-03)

**Permanently nonpromoting.** This directory retains the first executed and
strictly validated run of the literal frozen 144-shard reference-life
development scorecard (run schema `asi.reference_life_scorecard.run.v1`,
aggregate schema `asi.reference_life_scorecard.artifact.v1`, plan
`4c4396848da3abebc53fb61c7b241866566ad2b44fa83ee3395a5a389a869b86`).

It is development selection output only. It does not select `reference-dev`,
attest execution, or constitute performance or scientific evidence. Consistency
hashes bind bytes; they are not authenticated execution proof
(`identity_scope_note: "consistency binding only; not authenticated execution
attestation"` in `artifact.json`).

## What was run

- Source: `SlopDotCash/asi` commit `f3d32c451ed1c1715e477ad56b782fc7ea89b206`
  (exact `origin/main` head at run time), executed from a dedicated isolated
  worktree of that revision. See `SUMMARY.md` for why the run does not (and
  must not) execute in a concurrently edited checkout.
- Schedule: 12 consumed development seeds (70000–70011) × 2 environments
  (`switching_two_state` horizon 4000, `riverswim` horizon 20000) × 6 arms
  (`prototype`, `prototype_frozen`, `random`, `privileged_oracle`,
  `differential_sarsa`, `sarsa`) = 144 fresh-process shards, one canonical
  shard per process, exactly the frozen matrix.
- Runtime: environment locked by `uv.lock` — Python 3.12.12, jax/jaxlib 0.11.0,
  NumPy 2.5.1, CPU backend; single Linux host; the switching half ran 6
  concurrent fresh processes and the riverswim half 10 (each shard is still
  one fresh process; concurrency only overlaps processes). Wall clock ≈ 9.5 h.
- Launcher: `run_all.sh` (retained here), mirroring the per-shard semantics of
  `.github/workflows/reference-life-scorecard-dev.yml`: shard exit 0/1 both
  produce a strictly validated record; exit >1 is fatal. The final log is
  `run_all.log`.

## How to re-validate

```bash
.venv/bin/python -m alberta_framework.benchmarks.reference_life_scorecard \
  validate outputs/reference_life_scorecard_development/run1-2026-10-03/artifact.json
```

Every shard record passed the strict shard validator inside `run-shard` at
production time and again inside `summarize`; `summarize` built `artifact.json`
over exactly these 144 files and the aggregate validator accepted it
(`valid: true`, `status: valid_baseline_failure` — a development-gate outcome,
not a validation failure). `receipt.json` holds the full byte inventory
(plan, aggregate, all 144 shards) plus the honest source/runtime identity.

## What this run is not

- Not the CI-authorized workflow execution
  (`asi.reference_life_scorecard.github_run.v1`): there is no maintainer
  dispatch, no GitHub run identity, and no cross-machine shard spread. It was
  executed on one host, so it exercises neither cross-machine source-identity
  variance nor the RiverSwim stationary-oracle ULP variance observed across CI
  runners (see #3047 / #3052 for that failure mode).
- Not a `reference-dev` selection, promotion, benchmark claim, or evidence
  artifact. The scorecard is permanently nonpromoting by construction
  (`status_is_promotion: false`).
- Timing fields are telemetry-only.

See `SUMMARY.md` for the headline numbers and gates.
