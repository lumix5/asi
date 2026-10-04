# Summary — intentional-updates retention grid 1

Metric: whole-stream average online accuracy (pre-update), 4 paired seeds per
arm, `paired_vs_control` against `intentional_updates_off` (the mechanism-off
fixed-step normalized-SGD control). Negative diff = worse than the
mechanism-off control. Development-only, permanently nonpromoting.

## Cell A — 8 tasks × 5,000 steps

| arm | online mean | ±se | plasticity | paired vs off (mean ± se) | all seeds improve |
| --- | ---: | ---: | ---: | ---: | :- |
| intentional_updates_off (control) | 0.8503 | 0.0010 | 0.3812 | — | — |
| intentional_updates_no_diag | 0.7886 | 0.0009 | 0.4139 | −0.0617 ± 0.0008 | no (0/4) |
| intentional_updates_ipmnist | 0.7832 | 0.0003 | 0.4295 | −0.0671 ± 0.0013 | no (0/4) |
| intentional_updates_no_clip | 0.7832 | 0.0003 | 0.4295 | −0.0671 ± 0.0013 | no (0/4) |
| intentional_updates_head_only | 0.6479 | 0.0017 | 0.4105 | −0.2024 ± 0.0023 | no (0/4) |

## Cell B — 20 tasks × 5,000 steps

| arm | online mean | ±se | plasticity | paired vs off (mean ± se) | all seeds improve |
| --- | ---: | ---: | ---: | ---: | :- |
| intentional_updates_off (control) | 0.8389 | 0.0007 | 0.3859 | — | — |
| intentional_updates_no_diag | 0.7737 | 0.0006 | 0.4198 | −0.0652 ± 0.0002 | no (0/4) |
| intentional_updates_ipmnist | 0.7677 | 0.0004 | 0.4339 | −0.0712 ± 0.0009 | no (0/4) |
| intentional_updates_no_clip | 0.7677 | 0.0004 | 0.4339 | −0.0712 ± 0.0009 | no (0/4) |
| intentional_updates_head_only | 0.6390 | 0.0016 | 0.4111 | −0.1999 ± 0.0017 | no (0/4) |

## Honest reading

1. **Retained negative outcome.** Every intentional-updates variant loses to
   its own mechanism-off control in **every seed pair in both cells** (8/8
   per variant for the full mechanism), and the deficit does **not** shrink
   with a 2.5× longer stream (−0.0671 → −0.0712 for the full arm; the control
   is ahead by roughly the same margin at both scales). Within this bounded
   protocol, the supervised Eq.-5 step-size rule is strictly dominated by the
   fixed-step normalized SGD it reduces to.
2. **The adaptive clip is provably inert at this scale.**
   `intentional_updates_no_clip` is **bit-identical** to
   `intentional_updates_ipmnist` in all 8 runs — `per_task_accuracy`,
   `per_task_loss`, and `per_task_plasticity` are exactly equal, not close.
   Mechanically: the cap is `clip_mult × RMS(EMA of loss²)` = 20 × running
   loss RMS ≈ 46 at this scale, while the loss itself is ≈ 2.3, so
   `min(loss, cap)` never binds. The registered arm pair therefore measures a
   mechanism that is switched off by its own constants in this regime; any
   future clip-relevant measurement needs a regime where the loss approaches
   its own running RMS bound (e.g. much larger intended fractions or
   step-size scales).
3. **Diagonal normalization is the harmful half, not the helpful half.**
   Removing the RMSProp diagonal (`no_diag`) *gains* +0.0054 (A) / +0.0060
   (B) over the full mechanism in every seed. The diagonal direction is
   currently a net cost in this lane.
4. **Freezing features collapses accuracy but not plasticity.**
   `head_only` trails by ≈ 0.20 — the largest deficit — yet its plasticity
   metric (0.41) is *higher* than the control's (0.38). Plasticity here
   tracks parameter movement, not retained accuracy; the two arms with the
   biggest parameter movement (`ipmnist`/`no_clip`, 0.43) have the worst
   online accuracy. The control wins by moving less and retaining more.
5. **What this cannot say.** Four seeds, one architecture (300×150), one
   step-size family (`fixed_step_size 0.01` with the paper's
   `intended_fraction 0.5`), bounded 8/20-task cells. It cannot order the
   family against anything outside its own control, cannot test the paper's
   RL setting, and makes no paper-comparability or significance claim. The
   60-task proxy protocol and the 200-task confirmation lane remain
   unmeasured for this family.

All numbers above come from the two merged summaries
(`../grid1-2026-10-04-summary.json`, `../grid1-2026-10-04-summary-cellB.json`);
per-seed values and vectors are in the receipts beside this file.
