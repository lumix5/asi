# Summary — noise-curvature retention grid1-2026-10-05

First retained outcomes for the noise-curvature IPMNIST lane. Cell: 10 tasks x
5000 examples, five frozen development seeds (0–4), four registered arms,
paired by seed against the mechanism-off arm. All 20 runs exit 0 and every
receipt passes the lane's strict validator.

## Mean online accuracy (per-seed and mean)

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_adam_l2 (mechanism-off control) | 0.7203 | 0.7050 | 0.7118 | 0.7204 | 0.7245 | **0.7164** | comparator |
| volatility_only | 0.5701 | 0.5719 | 0.5838 | 0.5762 | 0.5425 | 0.5689 | rejected |
| gradient_only | 0.3525 | 0.3450 | 0.3240 | 0.3683 | 0.3574 | 0.3495 | rejected |
| combined | 0.3525 | 0.3450 | 0.3240 | 0.3683 | 0.3574 | 0.3495 | rejected |

## Paired deltas vs the mechanism-off control

| arm | per-seed deltas | mean | paired-t 95% interval (df=4) |
| --- | --- | --- | --- |
| combined | −0.3678 −0.3599 −0.3878 −0.3521 −0.3670 | −0.3669 | [−0.3834, −0.3504] |
| gradient_only | −0.3678 −0.3599 −0.3878 −0.3521 −0.3670 | −0.3669 | [−0.3834, −0.3504] |
| volatility_only | −0.1502 −0.1331 −0.1280 −0.1442 −0.1819 | −0.1475 | [−0.1738, −0.1212] |

Every scheduler arm loses in every seed pair; all three arms are `rejected`
under the declared rule. The lane's mechanism gate ("paired 95% interval for
joint minus fixed-Adam+L2 strictly above zero") fails by a wide margin.

## Findings recorded (development-selection only)

- **The scheduler is a large negative at this workload.** All three scheduling
  arms are dominated by their own mechanism-off reduction in 5/5 seed pairs.
  The lane's registered controller (warm-up `warm_fraction` 0.3, then
  layerwise cooling toward `effective_step_floor` 0.12) progressively destroys
  later-task adaptation: e.g. combined, seed 0, per-task accuracy runs
  0.782, 0.782, 0.686, 0.452, 0.191, 0.156, 0.188, 0.086, 0.126, 0.076 while
  the control holds 0.71–0.76 across all ten tasks. The early warm-up phase is
  the only interval where the scheduler arms match or beat the control.
- **The curvature-volatility signal contributes nothing measurable here.**
  `combined` and `gradient_only` produce bitwise-identical accuracy, loss, and
  plasticity trajectories in all five seeds. Whatever the volatility term adds
  to the layerwise schedule, it does not move the executed trajectory at this
  cell; the gradient-noise signal alone carries the entire (negative) effect.
  The causal gate (joint minus each single-signal ablation strictly above
  zero) therefore also fails — by exact zero against `gradient_only`.
- **Volatility-only is less damaging than gradient-only** (−0.1475 vs
  −0.3669 mean paired delta): of the two signals the paper combines, the
  gradient-noise estimator is the harmful component at this workload.
- **Not free to find out:** 3,750 charged HVP queries per scheduler run bought
  no accuracy anywhere in the grid.

## Honest limits

- One bounded cell (50k updates), one host, one toolchain, five seeds —
  screening-grade. This grid cannot locate where the scheduler's damage
  curve bends, and it makes no claim about the paper's batch-256/250-epoch
  regime or about ASI's 200-task confirmation scale.
- The separately-protocolled live RLS control (`rls_head_resid_l1_preset005`)
  was not executed; the hillclimb and transfer gates are unmeasured, not
  passed or failed.
- Timing is telemetry-only. Development-only; permanently nonpromoting;
  negative outcomes retained on purpose.
