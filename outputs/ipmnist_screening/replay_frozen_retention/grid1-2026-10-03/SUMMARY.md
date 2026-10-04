# SUMMARY — grid1-2026-10-03 (development-only, permanently nonpromoting)

Eight registered replay/frozen-feature ceiling arms × frozen seeds {0, 1, 2},
60 tasks × 5,000 examples, exact-step noise, IPMNIST online pre-update metric.
Control: `replay_context_mechanism_off`. Confirmation threshold: paired mean
delta > +0.005 with all seeds positive (the summary's own rule).

| arm | mean online acc (3 seeds) | ± stderr | paired Δ vs control | all seeds improve | summary verdict |
| --- | --- | --- | --- | --- | --- |
| `replay_gradient_only` | **0.790032** | 0.000661 | **+0.034452** | yes (3/3) | beats control, confirmation candidate |
| `replay_context_mechanism_off` (control) | 0.755580 | 0.000462 | — | — | — |
| `replay_context_full` | 0.749609 | 0.000696 | −0.005971 | no (0/3) | below control |
| `replay_context_only` | 0.710849 | 0.000304 | −0.044731 | no (0/3) | below control |
| `prol_prompt_proxy` | 0.519224 | 0.002998 | −0.236356 | no (0/3) | below control (beats its own off-arm) |
| `prol_prompt_mechanism_off` | 0.494560 | 0.004799 | −0.261020 | no (0/3) | below control |
| `randumb_random_features` | 0.238608 | 0.000708 | −0.516972 | no (0/3) | ceiling binds far below trained arms |
| `ranpac_random_projection` | 0.231723 | 0.000745 | −0.523857 | no (0/3) | ceiling binds far below trained arms |

Per-seed means (in seed order 0, 1, 2):

- `replay_gradient_only`: 0.789690, 0.791310, 0.789097
- `replay_context_mechanism_off`: 0.756223, 0.755833, 0.754683
- `replay_context_full`: 0.749007, 0.750997, 0.748823
- `replay_context_only`: 0.711217, 0.711083, 0.710247
- `prol_prompt_proxy`: 0.513233, 0.522450, 0.521990
- `prol_prompt_mechanism_off`: 0.485110, 0.500740, 0.497830
- `randumb_random_features`: 0.238097, 0.237720, 0.240007
- `ranpac_random_projection`: 0.231443, 0.233130, 0.230597

## Honest reading

1. **The replay half of the replay paper's mechanism transfers; the in-context
   half does not.** Prior-example replay gradients alone (`replay_gradient_only`)
   beat the charged AdamW control by +0.0345 with every seed positive, clearing
   the lane's +0.005 candidate bar. The bounded label-attention context proxy
   alone (`replay_context_only`) *hurts* by −0.0447 in every seed, and adding it
   on top of replay gradients (`replay_context_full`) leaves the combination
   −0.0060 below the control: the context term is a measured net negative at
   this scale, and it cancels most of the replay-gradient gain it is supposed to
   complement.
2. **The frozen-feature ceilings bind exactly as ceilings should.** Random
   Fourier features (RanDumb-style) and a random ReLU projection with recursive
   ridge (RanPAC-style) over raw pixels sit ~52 points below the trained
   backbone, with tiny across-seed spread. These are compatibility-ballast
   proxies (see the lane's `protocol_gaps`), so this retains the expected
   ordering rather than a paper comparison; it does bound what a frozen
   random-extractor arm could claim here.
3. **The PROL prompt proxy beats only its own mechanism-off arm** (+0.0247,
   0.5192 vs 0.4946) and remains ~23 points below the trained control. The
   prompt/affine machinery adds something over its charged off-arm, but nowhere
   near the trained-backbone regime.
4. **Development selection only.** This grid selects
   `replay_gradient_only` as the only arm worth a paired confirmation schedule
   against the same control, and records that the in-context attention proxy is
   a net negative — a useful negative result for #1573's replay/in-context
   lane. It populates no `reference-dev` configuration, promotes nothing, and
   supports no external-paper or SOTA claim; the lane's protocol gaps
   (`official` implementations not imported, proxies in place of pretrained
   models) remain open and are restated in every receipt.

Wall-clock telemetry (inflated by three-lane concurrency for the replay family;
telemetry-only, unused by any decision): replay-family arms 336–1269 s per
shard, frozen-extractor arms 8.4–11.1 s per shard.
