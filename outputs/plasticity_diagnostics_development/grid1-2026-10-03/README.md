# Loss-of-plasticity diagnostics — first retained development outcome grid

First retained outcome of the bounded hidden-network loss-of-plasticity
diagnostic (`alberta_framework/benchmarks/plasticity_diagnostics.py`, the
authorized MNIST slice of issue **#1583**). Before this retention the lane had
a receipts-capable CLI but **zero retained outcomes**, while every receipt it
prints carries `negative_results_must_be_retained: true`. This directory
retains the first grid — including the regressing seed — under a new
permanently nonpromoting path.

Everything here is **development-only and permanently nonpromoting**
(`development_only: true`, `scientific_promotion_allowed: false` in every
receipt). No paper-parity, performance, promotion, or SOTA claim is made or
licensable from these bytes. The lane's own validator rejects describing it as
random-label MNIST: the implemented workload is cumulative input-permuted
MNIST with fixed labels (`task_protocol: "cumulative-input-permutation"`,
`labels_permuted: false`), a deliberately short development diagnostic rather
than a reproduction of the paper's 800-task experiment. The costly ImageNet
and reinforcement-learning lanes of #1583 remain
`execution_authorized: false` in `costly_lane_gates()` and were not run.

## Provenance

| Field | Value |
| --- | --- |
| Base revision | `SlopDotCash/asi@f3d32c451ed1c1715e477ad56b782fc7ea89b206` (`origin/main`); retained on branch `benchmarks/plasticity-retained-outcome-v1`, whose only source change is the `python -m` guard below |
| Lane module sha256 (`source_sha256`) | `eec32a83c564369d35fecb757ced5439c0701087c8390d9ff59fb1741c58b703` |
| Lane schema | `asi.loss_of_plasticity_mnist_development.v1` |
| Paper revision / code commit | `arXiv:2306.13812v3` / `a6b79580d85f3025bdb601566d3627c5f489f13b` (pinned by the lane, not audited here) |
| Frozen seeds | 15830, 15831, 15832, 15833 (`FROZEN_SEEDS`) |
| Profiles | `contract-smoke`, `bounded-development` (both `PROFILES` members) |
| Runtime | Python 3.14.4, JAX 0.11.0, NumPy 2.5.1, CPU backend, `OMP_NUM_THREADS=1` |
| Dataset | OpenML `mnist_784` v1, first 60,000 rows (canonical train split), `images` float32 in [0, 1] (raw/255), `labels` int32 |
| `dataset_sha256` | `9aebd816fe1c082ed534408b26c76b5ed038403a3745bae158571d1d1615b10f` |
| Schedule | `bounded-development` = 8 tasks x 64 examples, width 64, lr 0.003, replacement rate 1e-4, maturity threshold 100 (immutable `PROFILES` registry) |

Every receipt binds its own `source_sha256`, `dataset_sha256`, and
`runtime_identity`; all sixteen receipts in this grid share one lane source
digest and one dataset digest (the same dataset derivation as the NaP
comparator grid), and each was produced by one fresh Linux process.

The `python -m` entry this grid ran through was itself broken on the base
revision: the module had no `__main__` guard, so `python -m` exited 0 printing
nothing. The commit retaining this grid adds the guard (mirroring
`nap_ipmnist.py`) plus a subprocess test that fails on the silent no-op.

## Reproduction

The dataset NPZ is a derived artifact and is **not** stored here. Rebuild it
from the cached OpenML source (one download if no cache exists), then run the
grid from the same revision:

```bash
/home/lumix/asi/.venv/bin/python - <<'EOF'
import numpy as np
from sklearn.datasets import fetch_openml

raw = fetch_openml(
    "mnist_784", version=1, as_frame=False,
    data_home="/home/lumix/asi/outputs/upgd_ipmnist/openml_cache",
    n_retries=3, delay=2.0,
)
x = np.ascontiguousarray(raw.data, dtype=np.float32)[:60_000] / np.float32(255.0)
y = np.ascontiguousarray(raw.target, dtype=np.int32)[:60_000]
np.savez("mnist.npz", images=x, labels=y)
EOF

for profile in contract-smoke bounded-development; do
  for seed in 15830 15831 15832 15833; do
    OMP_NUM_THREADS=1 /home/lumix/asi/.venv/bin/python -m \
      alberta_framework.benchmarks.plasticity_diagnostics \
      --dataset mnist.npz --seed "$seed" --profile "$profile" \
      > "receipt-${profile}-seed${seed}.json" \
      2> "receipt-${profile}-seed${seed}.stderr"
  done
done
```

Receipt files are the exact CLI stdout bytes; `.stderr` files are the exact
stderr bytes (all empty — every run exited 0). Any nonzero exit or stderr
content would have been retained as-is instead.

## Cost

| Profile | Runs | Wall clock each (fresh process) | Total |
| --- | --- | --- | --- |
| `contract-smoke` | 4 | ~3.7 s | ~15 s |
| `bounded-development` | 4 | ~4.9 s | ~20 s |

Per-arm resource receipts are inside each JSON (`data_steps` 512 and
`persistent_bytes` 221,232 per arm at `bounded-development`; 8 steps and
25,904 bytes at `contract-smoke`; logical forward/gradient multiply-accumulates
and model-query counters are recorded per arm). Timing is telemetry-only and
selects nothing.

## Reading

See `SUMMARY.md` for the outcome tables and the honest interpretation.
