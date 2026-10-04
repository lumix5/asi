# Action-conditioned latent development lane — first retained outcome grid

This directory retains the first executed outcome receipts for the bounded,
permanently nonpromoting action-conditioned latent comparison lane of
`docs/research/action-conditioned-latent-protocol.md` (the `#1575` lane). The
lane's CLI had a frozen protocol, a strict validator, and a frozen
six-arm x four-seed development roster, but no retained outcome existed
anywhere in the repository before this record.

Artifacts only: no library source, doc, test, or registry bytes are touched.
Development-only and permanently nonpromoting; these receipts cannot authorize
a scientific claim, and the lane's own doc reserves all external (Crafter,
Atari100k, physical-planning) comparisons behind separately frozen protocols.

## What was run

One command, three times, three fresh Python processes:

```bash
OMP_NUM_THREADS=1 .venv/bin/asi-action-conditioned-latent \
  > outputs/action_conditioned_latent_retention/grid1-2026-10-04/receipts/receipt-runN.json \
  2> receipts/receipt-runN.stderr
```

No protocol knobs were passed, so every run uses the frozen default protocol
recorded in each receipt: `steps=256`, `phase_length=32`, `warmup_steps=16`,
`exploration_period=4`, seeds `1575000..1575003` (`FROZEN_DEVELOPMENT_SEEDS`).
All three stderr captures were empty and all three runs exited 0.

The lane needs no external dataset: the environment is the repository's own
`SwitchingTwoStateMDP` closed-loop stream with an A/B/A reward-phase schedule,
so the receipts' `persistent_environment_bytes` (8) cover the whole workload.

The three receipts are **bitwise identical** across processes
(sha256 `18c79e63252b8cdb…`, full digest in `analysis_summary.json`); the lane
derives all randomness from the frozen seeds, so cross-process identity is the
expected behavior and is retained as the determinism check, not as independent
replication.

## Provenance (copied from the receipts' `identity` block)

- lane module: `alberta_framework/benchmarks/action_conditioned_latent.py`,
  `lane_source_sha256` `9c2d9ef094a230b2406574daaeaae7461b88a420ba29f8377c008c536897c45a`
- dependency sources: `alberta_framework.benchmarks.development_provenance`
  `0e6a12d0…`, `alberta_framework.core.latent_world_model` `90f5ed88…`,
  `alberta_framework.core.sarsa` `3932ea7f…`,
  `alberta_framework.streams.closed_loop` `9a308dfe…` (full 64-hex digests in
  every receipt)
- workload registry `9de5121d…`, paper registry `7cadbbf9…`
- research pins: Dreamer-CDP `arXiv:2603.07083v2` / code `a851fa3e…`,
  JEDI `arXiv:2605.13013v1` (no code), JEPA-WM `arXiv:2512.24497v3` / code
  `13cf1d9c…` — provenance only, not implementation parity
- runtime: CPython 3.14.4, Linux x86_64, JAX 0.11.0 (CPU backend), NumPy 2.5.1,
  `OMP_NUM_THREADS=1`
- repository state: branch point `f3d32c451ed1c1715e477ad56b782fc7ea89b206`
  (current `origin/main` at execution time)

## Validation pass

`validate_retention.py` re-validates the retained bytes from the repository
root:

```bash
.venv/bin/python \
  outputs/action_conditioned_latent_retention/grid1-2026-10-04/validate_retention.py \
  --output outputs/action_conditioned_latent_retention/grid1-2026-10-04/analysis_summary.json
```

It runs the lane's own strict `validate_action_latent_payload` on every
receipt, checks cross-process bitwise identity, the decision-off vs
mechanism-off action/reward transcript-hash identity required by the protocol
doc, matched environment exposure and matched model-arm update/training
budgets, and re-derives the paired deltas tabulated in `SUMMARY.md`. The
committed `validation_pass.txt` is the recorded output of that pass (exit 0,
20 of 20 checks PASS); `analysis_summary.json` is its machine-readable result.

## Reproduction

1. Check out a tree whose `alberta_framework/benchmarks/action_conditioned_latent.py`
   matches `lane_source_sha256` above (plus the four dependency sources).
2. Run the command under "What was run" with a fresh `N`.
3. `cmp` the new receipt against `receipts/receipt-run1.json`; expect bitwise
   identity on the same runtime family (x86_64 CPU, float32 JAX defaults).
4. Re-run the validation pass; expect the recorded `validation_pass.txt` output.
