# Summary — TeLAPA qualification retention grid1

Six fresh-process receipts over the full `{steps} x {phase_length}` grid at the
lane's frozen three-seed roster, four arms per receipt. Every number below is
identical across the three seeds of a config: the primitive
`SwitchingTwoStateMDP` adapter produces seed-invariant reward totals in this
regime, which is itself recorded as a lane property, not hidden.

## Reward sums per config (three seeds each, identical values)

| config (steps, phase) | diverse_archive | one_model | fixed_snapshot | mechanism_off |
| --- | --- | --- | --- | --- |
| (32, 4)  | 13 | 12 | 15 | 15 |
| (32, 8)  | 15 | 14 | 15 | 15 |
| (32, 16) | 15 | 15 | 15 | 15 |
| (64, 4)  | 25 | 24 | 31 | 31 |
| (64, 8)  | 29 | 28 | 31 | 31 |
| (64, 16) | 31 | 30 | 31 | 31 |

Paired deltas vs mechanism-off (archive arm − control), summed over a seed:

| config | diverse_archive | one_model |
| --- | --- | --- |
| (32, 4)  | −2 | −3 |
| (32, 8)  |  0 | −1 |
| (32, 16) |  0 |  0 |
| (64, 4)  | −6 | −7 |
| (64, 8)  | −2 | −3 |
| (64, 16) |  0 | −1 |

## Honest reading (development-selection only)

- **The negative outcome is the result.** Across all 36 (config, arm, seed)
  pairs, neither archive arm ever beats the archive-off control: 30 strictly
  negative paired deltas and 6 exact ties. Retrieving archived policies at task
  boundaries never helped and often hurt this adapter.
- **The damage tracks boundary density, not bytes.** The gap is largest at the
  shortest phases ((64,4): −6/−7 of a 31-reward control) and disappears at
  `phase_length=16` for `diverse_archive` (one_model still loses 1 at
  (64,16)). More boundaries mean more disruptive retrievals of a two-state
  policy whose tabular update has already re-converged.
- **The diverse arm adds cost without return here.** It retains 160–319 archive
  bytes vs 79–80 (snapshot arms) and 0 (mechanism-off), always within the
  4096-byte budget, and never exceeds `one_model` on reward at any config.
- **The default-config receipt reproduces the #2257 record exactly**
  (diverse 13, one_model 12, fixed 15, off 15 on every seed), tying this grid
  to the lane's only prior retained artifact.
- **Mechanism-off parity holds exactly** — observation, action, reward, and
  both policy hashes match `fixed_snapshot` on every seed and config — so the
  reduction the validator demands is intact across the grid.

This advances the #1586 lane record without closing it: license review of the
pinned public source remains incomplete and paper parity remains disallowed in
the catalog. Development-only and permanently nonpromoting — none of these
receipts can authorize a scientific claim, and every paper-scale comparison
(PPO/MAP-Elites/learned embedders) remains behind separately frozen gates.
