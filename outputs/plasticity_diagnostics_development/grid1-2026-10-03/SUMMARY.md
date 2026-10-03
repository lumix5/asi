# SUMMARY — grid1 (2026-10-03)

Two profiles x four frozen seeds (15830–15833) x three arms
(`sgd_control`, `cbp_mechanism_off`, `cbp_bounded`), one fresh process per
receipt. Every number below is transcribed from the retained receipt JSON
bytes in this directory; nothing is re-measured or averaged beyond the stated
mean.

## 1. Mechanism-off reduction is bitwise exact

In all 8 runs, `cbp_mechanism_off` and `sgd_control` produced identical
`task_accuracy`, `task_loss`, `dead_unit_fraction`, `effective_rank`, and
identical `final_state_sha256` values. The only receipt field that differs is
`elapsed_ns` (timing telemetry). Setting the replacement rate to zero reduces
the CBP arm to the SGD control bit-for-bit, which is the property the three
roster arms exist to pin.

## 2. `bounded-development` (8 tasks x 64 examples, width 64)

Final-model mean accuracy over the eight task permutations:

| Seed | `sgd_control` | `cbp_bounded` | delta |
| --- | --- | --- | --- |
| 15830 | 0.130859 | 0.142578 | **+0.011719** |
| 15831 | 0.132812 | 0.134766 | **+0.001953** |
| 15832 | 0.121094 | 0.125000 | **+0.003906** |
| 15833 | 0.169922 | 0.164062 | **−0.005859** |

- Three deltas positive, one negative — a **mixed, inconclusive outcome**,
  retained as required.
- The arm is nearly inert at this scale: `replacement_rate = 1e-4` over 512
  steps produced exactly **4 replacements per run** (accrual starts once units
  pass the maturity threshold of 100 steps).
- Per-task differences are one or two flipped examples (each task is 64
  examples, so one flipped example moves a task accuracy by 0.015625): seeds
  15830/15831/15832 improve on tasks 4–6, seed 15833 regresses on tasks 6–7.
- `dead_unit_fraction` stays small for every arm (≤ 0.04 at width 64 after 64
  examples per task); this schedule does not produce the paper's dense-layer
  collapse, so there is little plasticity loss for CBP to recover here.

## 3. `contract-smoke` (2 tasks x 4 examples, width 8)

Accuracy is 0.0 for every arm in every seed (the smoke is a contract
exercise, not a learning run). The mechanism is nonetheless visibly active:
`cbp_bounded` performed 14 replacements per run and held lower
`dead_unit_fraction` than the control in 8 of 8 task measurements (e.g. seed
15833: 0.25/0.25 vs 0.375/0.375). This pins the replacement path executing
end-to-end without claiming any accuracy effect.

## 4. Honest reading

- No preregistered primary-outcome rule exists in this lane; per the issue's
  acceptance criteria these are retained development observations, not a
  comparison verdict.
- At the frozen bounded scale, bounded CBP is within ±1 task-flip noise of
  the SGD control on three seeds and one seed regresses. Treating this as a
  mechanism win would be dishonest; treating it as a mechanism refutation
  would be equally dishonest — the schedule gives the mechanism ~4
  replacement events to matter.
- Negative/mixed outcome retained under a new nonpromoting path per the
  lane's `negative_results_must_be_retained` contract and the issue's
  acceptance criteria. The costly ImageNet / RL lanes of #1583 remain gated
  (`execution_authorized: false`) and were not executed.
