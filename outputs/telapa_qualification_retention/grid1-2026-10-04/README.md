# TeLAPA qualification retention — grid1 (2026-10-04)

Retained development outcomes for the `#1586` TeLAPA policy-archive lane
(`alberta_framework/benchmarks/telapa_qualification.py`, CLI
`asi-telapa-qualification-smoke`, schema
`asi.telapa_qualification_smoke.development.v2`). Until this record, the only
retained artifacts for this lane were the two `#2257` files
(`outputs/telapa_qualification/development_result_pr2257*.v2.json`); the
runbook's retention requirement ("every valid development outcome, including
ties and regressions, must be retained") had no grid-level record anywhere.

This directory is **artifacts only**: no library source, doc, test, or registry
byte is touched. Everything here is permanently nonpromoting development
evidence for a bounded synthetic smoke that is not a TeLAPA reproduction.

## Contents

- `receipts/receipt-run{1..6}.json` — six exit-0 fresh-process CLI receipts,
  `OMP_NUM_THREADS=1`, empty stderr, over the full
  `{steps} x {phase_length}` grid: (32,4), (32,8), (32,16), (64,4), (64,8),
  (64,16). Each receipt carries the lane's frozen three-seed roster
  (1586000–1586002) and all four arms. Every receipt was additionally
  re-executed in a second fresh process and compared bitwise-identical
  (`cmp` exit 0) — a determinism check, not independent replication. Receipt
  SHA-256:
  - run1 `ee7cf33ecbed9b181edc4ce1c47e1289fe268a34b1af288217bc68ef9fe7f7b4`
  - run2 `8f65b00c238c14a8240c2886c961ebc25bca565f42f30b85d9a6fcdbca316f96`
  - run3 `ccf5bc2ef3dae18e13f2b675b5cafa63cc4bf55d2fc79d4f54b6ea3ddf4a3bae`
  - run4 `c24c50b9e08901acd123f6a93c14da79fdb27cd8716f9549a9d4ce7517688dcc`
  - run5 `fbe26ffb11dd530e83df88e0c61001be7a9b5388043f3b547dcb4c6fd6b1bcd9`
  - run6 `4c92adaf00384c66258ced4890edcd0557e99401d09665618a12ff017a53e6fa`
- `validate_retention.py` — retained driver that re-validates every receipt
  with the lane's own `validate_result` (which also replays every arm/seed
  record from the bound configuration), checks grid coverage, frozen
  hyperparameters and seeds, one shared lane/runtime/dependency/paper identity
  across the grid with per-config workload digests, mechanism-off parity,
  matched resource axes, archive byte budgets, and re-derives the paired
  deltas in `SUMMARY.md`.
- `validation_pass.txt` — the recorded validation pass (exit 0, 142/142 checks
  PASS) and `analysis_summary.json` — its machine-readable result.
- `SUMMARY.md` — paired tables and honest reading.

## Identity (identical across all six receipts)

- lane source `721370a55a94fa4d…` (`benchmarks/telapa_qualification.py` at
  repository state `f3d32c45` = current `main`)
- dependencies: `development_provenance` `0e6a12d0fe2f…`,
  `core/policy_archive` `dba1dd9fc19a…`, `streams/closed_loop` `9a308dfebd6e…`
- runtime: CPython 3.14.4, Linux x86_64, JAX 0.11.0 (cpu), NumPy 2.5.1
- catalog: issue 1586, paper `arXiv:2604.15414v1`, pinned public revision
  `a4dc16ed0ea015b1b8efb271e4d664931adccd3e`, license review incomplete,
  paper parity disallowed — bound verbatim in every receipt

## Reproduction

From the repository root:

```bash
# regenerate any receipt and compare against the retained bytes
OMP_NUM_THREADS=1 .venv/bin/asi-telapa-qualification-smoke --steps 32 --phase-length 4 \
  | cmp - outputs/telapa_qualification_retention/grid1-2026-10-04/receipts/receipt-run1.json

# re-run the full retained validation pass (exit 0, 142 PASS)
.venv/bin/python outputs/telapa_qualification_retention/grid1-2026-10-04/validate_retention.py \
  --output /tmp/analysis_summary.json
```

Development-only: these receipts cannot authorize a scientific, paper-parity,
or performance claim, and no registered evidence source is touched
(`alberta-evidence-status` behavior is byte-identical to `main`).
