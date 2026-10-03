# TeLAPA qualification-smoke retention — grid1 (2026-10-03)

First retained development outcomes for the lane's own retention contract
(`docs/runbooks/telapa-qualification.md`): "Every valid development outcome,
including a tie or regression, must be retained outside this smoke under a new
nonpromoting path; this CLI does not write `outputs/`." No outcome had been
retained anywhere before this directory.

## What is here

Six `asi-telapa-qualification-smoke` receipts over the lane's frozen
three-seed protocol (`FROZEN_DEVELOPMENT_SEEDS = (1586000, 1586001, 1586002)`,
validated as immutable by `TeLAPASmokeConfig`), one per
`steps x phase_length` in `{32, 64} x {4, 8, 16}`:

    receipt-steps<S>-phase<P>.json   full receipt, schema
                                     asi.telapa_qualification_smoke.development.v2
    receipt-steps<S>-phase<P>.stderr empty (clean run)

Each receipt carries the complete catalog (paper identity, fail-closed license
state, `paper_parity_allowed: false`), the lane source and dependency hashes at
the generating commit, per-seed/per-arm records with observation/action/reward
hashes, initial/final policy hashes, resource receipts, and
`scientific_promotion_allowed: false`. The CLI's own validators accepted every
receipt (exit 0); a valid receipt is a required, not sufficient, condition for
anything downstream.

## Reproduce

From a clean checkout at the generating commit:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m \
  alberta_framework.benchmarks.telapa_qualification \
  --steps 32 --phase-length 4 > receipt-steps32-phase4.json
```

Runs are deterministic per `(steps, phase_length)`: the receipts here should
reproduce byte-for-byte on the same runtime identity recorded inside them
(CPython on x86_64 Linux, JAX 0.11.0 CPU backend, numpy 2.5.1).

## Classification

Permanently nonpromoting development smoke outcomes only. This is not TeLAPA,
not a paper reproduction, not a benchmark result, not an ASI performance
claim, and not scientific evidence. The catalog rejects paper parity and the
license review is incomplete (`license_file_present: false`). Timing is absent
from the records and stays telemetry-only; it is not a decision axis here.
See `SUMMARY.md` for the tabulated outcomes and their honest reading.
