# AdaLin retention grid 1 — outcome summary (development-only, nonpromoting)

Sixteen fresh-process runs at `f3d32c45`, implementation
`64df8ddc…`, paired seeds, both registered arms. `adalin` = learned per-unit
alphas on both hidden layers; `relu_alpha_zero_mechanism_off` = the exact
zero-alpha reduction (alphas never update; `final_alpha_l2` exactly 0).

## Whole-stream pre-update online accuracy (paired per seed)

| cell | seed | adalin | off | Δ |
| --- | --- | --- | --- | --- |
| A | 15710 | 0.501660 | 0.456104 | +0.045557 |
| A | 15711 | 0.502441 | 0.452100 | +0.050342 |
| A | 15712 | 0.499268 | 0.469434 | +0.029834 |
| A | 15713 | 0.482080 | 0.445557 | +0.036523 |
| A | **mean** | **0.496362** | **0.455798** | **+0.040564** |
| B | 15714 | 0.641174 | 0.630542 | +0.010632 |
| B | 15715 | 0.644543 | 0.630920 | +0.013623 |
| B | 15716 | 0.644067 | 0.634338 | +0.009729 |
| B | 15717 | 0.644177 | 0.639185 | +0.004993 |
| B | **mean** | **0.643491** | **0.633746** | **+0.009744** |

## Mean per-task post-update test accuracy (paired per seed)

| cell | seed | adalin | off | Δ |
| --- | --- | --- | --- | --- |
| A | 15710 | 0.6251 | 0.5782 | +0.0469 |
| A | 15711 | 0.6248 | 0.5731 | +0.0517 |
| A | 15712 | 0.6262 | 0.5932 | +0.0330 |
| A | 15713 | 0.5998 | 0.5536 | +0.0462 |
| A | **mean** | **0.6190** | **0.5745** | **+0.0444** |
| B | 15714 | 0.7414 | 0.7311 | +0.0104 |
| B | 15715 | 0.7418 | 0.7311 | +0.0107 |
| B | 15716 | 0.7439 | 0.7340 | +0.0099 |
| B | 15717 | 0.7451 | 0.7380 | +0.0072 |
| B | **mean** | **0.7431** | **0.7335** | **+0.0095** |

`adalin` is ahead of its paired off arm in **8/8 seed pairs on both metrics**.
Learned alphas converge to similar total magnitudes in every on run
(`final_alpha_l2` 7.98–8.56).

## Honest reading

- **The mechanism helps at both measured scales, and the advantage shrinks
  with more data:** cell A (1024-example pool, 20 tasks) shows a large
  +0.041 online / +0.044 test mean advantage; cell B (2048-example pool,
  40 tasks, 4× the updates) retains the sign in every pair but the mean
  advantage drops to +0.010 on both metrics. This grid cannot say where the
  curve goes beyond its own bounds — the campaign-scale configuration
  (200 × 5000, batch 1) remains unmeasured.
- **4 paired seeds is screening-grade,** not confirmation. No significance
  claim is made; the grid exists so the lane's first behavior is retained
  rather than unknown.
- **Not paper-comparable, by the receipt's own record:** the pinned official
  AdaLin commit contains no runnable code, the pool is a caller-supplied MNIST
  prefix rather than the paper's unspecified sampling, and these cells are far
  below both the paper protocol and the declared ASI campaign protocol.
  `comparison.paper_comparable` is `false` in all 16 receipts.
- **Boundary handling is as declared:** the learner never sees task identity;
  the runner uses boundaries only to permute pixels and evaluate. The
  pre-update online metric is whole-stream.
- **Costs are bounded and recorded:** ~15 minutes total wall clock, receipts
  carry the full resource accounting, timing is telemetry-only.

Permanently nonpromoting: `development_only: true` and
`scientific_promotion_allowed: false` in every receipt. These outcomes
order nothing outside this grid and can never back a promotion claim.
