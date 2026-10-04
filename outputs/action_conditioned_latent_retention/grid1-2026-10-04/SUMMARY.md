# SUMMARY — action-conditioned latent grid1 (2026-10-04)

First retained outcome record for the `#1575` action-conditioned latent lane
(`docs/research/action-conditioned-latent-protocol.md`). Frozen default
protocol: 256 steps, 32-step phases, 16 warm-up steps, exploration period 4,
seeds `1575000..1575003`, six arms. Development-only and permanently
nonpromoting; nothing here can promote a claim, and the lane's own doc reserves
every external comparison behind separately frozen protocols.

## Late-window return (`late_return_sum`), per seed

| arm | 1575000 | 1575001 | 1575002 | 1575003 | paired Δ vs mechanism_off | mean Δ |
|---|---|---|---|---|---|---|
| latent_action_interactions | 3 | 9 | 17 | 12 | −16, −9, 0, −4 | **−7.25** |
| latent_no_interactions | 30 | 26 | 26 | 22 | +11, +8, +9, +6 | **+8.50** |
| latent_action_masked | 30 | 22 | 22 | 22 | +11, +4, +5, +6 | **+6.50** |
| latent_decision_off | 19 | 18 | 17 | 16 | 0, 0, 0, 0 | +0.00 |
| mechanism_off | 19 | 18 | 17 | 16 | — | — |
| sarsa_control | 32 | 19 | 32 | 19 | +13, +1, +15, +3 | +8.00 |

## Whole-stream return (`return_sum`) and resources

| arm | return Δ vs mechanism_off (per seed) | mean prequential loss | mechanism bytes | decision / training / model-update counts |
|---|---|---|---|---|
| latent_action_interactions | +4, −8, +24, −7 | 0.29798 | 892 | 360 / 256 / 256 |
| latent_no_interactions | +11, +3, −6, −3 | 0.31287 | 508 | 360 / 256 / 256 |
| latent_action_masked | +15, +8, −11, −5 | 0.31785 | 892 | 360 / 256 / 256 |
| latent_decision_off | 0, 0, 0, 0 | 0.32556 | 892 | 0 / 256 / 256 |
| mechanism_off | — | n/a (no model) | 0 | 0 / 0 / 0 |
| sarsa_control | −7, +31, +7, −20 | n/a (no model) | 120 | 257 / 0 / 0 |

Environment exposure is matched everywhere (256 steps per arm per seed); the
four model-owning arms share identical update and prequential-training budgets.
`persistent_environment_bytes` is 8 for every arm. Receipts carry
`negative_outcome_retained: true` — the schema's marker that negative outcomes
are permanently retained — not per-arm verdicts.

## Honest reading

1. **The primary arm is the negative outcome of this grid.** The
   action×latent-interaction arm — the lane's headline mechanism — has all four
   late-return paired deltas ≤ 0 against mechanism-off (mean −7.25) and the
   worst late window of the three decision-using model arms. Under this
   protocol, adding action×latent interaction features hurt recurrence-driven
   control.
2. **The two simpler model arms beat the no-model control at all four seeds on
   the late window** (+8.50 and +6.50 mean paired Δ for no-interactions and
   action-masked). This is a screening-level signal only: four frozen
   development seeds, a 96-step late window inside a 256-step horizon, and
   mixed-sign whole-stream deltas (+1.25 and +1.75 means) in the same
   direction-families. It justifies further bounded development of the simpler
   arms; it is not a supported claim, and the frozen roster cannot be extended
   without a separately frozen protocol.
3. **The causal sanity invariant holds exactly.** Decision-off reproduces
   mechanism-off action and reward transcript hashes, and its totals, at every
   seed — the identity the protocol doc requires. Learning without a decision
   interface is exactly neutral here.
4. **Prediction loss is not the decision objective.** The interactions arm has
   the lowest mean prequential loss (0.29798) and the worst late return; the
   arms select actions from predicted immediate rewards, so better next-latent
   prediction does not imply better action ranking. This decoupling is the
   lane's main diagnostic lesson and matches why the doc requires reward-facing
   decision heads rather than next-observation FTL scoring.
5. **The live SARSA control is competitive at similar or smaller mechanism
   bytes** (120 vs 508–892), with high seed variance (+8.00 mean late Δ,
   range +1..+15). Any future world-model arm should be compared against it as
   the strong live control, as the doc intends.

## Confounds and boundaries

One protocol cell (256 steps, 32-step phases); four seeds bound by the frozen
roster; the environment is the two-state nonvisual stream, so none of this
bears on visual JEPA-style transfer; consistency hashes are not execution
proof; wall-clock timing is deliberately absent (no qualified timing protocol).
Readings 1–2 are development-selection observations for the lane's own next
steps — e.g. the doc's open gates (longer horizons, delayed rewards,
stochasticity, retention) — and are permanently nonpromoting.
