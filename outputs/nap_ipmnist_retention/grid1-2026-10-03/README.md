# NaP IPMNIST comparator — first retained development outcome grid

First retained outcome of the bounded Normalize-and-Project comparator
(`alberta_framework/benchmarks/nap_ipmnist.py`, issue **#1564**). Before this
retention the lane had receipts-capable code and a qualification contract
(`docs/research/nap-qualification.md`) but **zero retained outcomes**, so its
own retention requirement — "Every valid negative, tie, and regression must be
retained in a new nonpromoting path" — was unmet. This directory retains the
first grid, ties and near-inert arms included, under a new nonpromoting path.

Everything here is **development-only and permanently nonpromoting**
(`development_only: true`, `scientific_promotion_allowed: false` in every
receipt). No paper-parity, performance, promotion, or SOTA claim is made or
licensable from these bytes.

## Provenance

| Field | Value |
| --- | --- |
| Source revision | `SlopDotCash/asi@f3d32c451ed1c1715e477ad56b782fc7ea89b206` (`origin/main`) |
| Lane module sha256 (`source_sha256`) | `a6663e8d9f0646b8335bea6325b7d72244e735c677688733fb8bce7dc5cacbf5` |
| Diagnostics dependency sha256 | `4a554afd4095df642a5852881ca99602632b1894211cb6d02abd1b3e35deca28` |
| `nap_project` dependency sha256 | `60642c868c57b9be5806e84bde5c8fd0e95fa4954fe39a90ddc103bedbf03199` |
| Lane schema | `asi.nap_ipmnist_comparator.development.v1` |
| Frozen seeds | 15640, 15641, 15642, 15643 (`FROZEN_SEEDS`) |
| Profiles | `contract-smoke`, `bounded-development` (both `PROFILES` members) |
| Runtime | Python 3.14.4, JAX 0.11.0, NumPy 2.5.1, CPU backend, `OMP_NUM_THREADS=1` |
| Dataset | OpenML `mnist_784` v1, first 60,000 rows (canonical train split), `images` float32 in [0, 1] (raw/255), `labels` int32 |
| `dataset_sha256` | `9aebd816fe1c082ed534408b26c76b5ed038403a3745bae158571d1d1615b10f` |
| `schedule_sha256` (bounded-development) | `581b890eb27aaad0a40dc64e1c4d4acd8e256bec8cd4347147b637bf19038c0c` |

Each receipt binds its own `source_sha256` and dependency hashes; all eight
receipts in this grid share one dataset digest and one lane source digest, and
each was produced by one fresh Linux process through the lane CLI.

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
  for seed in 15640 15641 15642 15643; do
    OMP_NUM_THREADS=1 /home/lumix/asi/.venv/bin/python -m \
      alberta_framework.benchmarks.nap_ipmnist \
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
| `contract-smoke` | 4 | ~4.3 s | ~17 s |
| `bounded-development` | 4 | ~6.8 s | ~27 s |

Per-arm resource receipts are inside each JSON (`data_steps` 512 and
`state_persistent_bytes` 221,232 per arm at `bounded-development`; logical
forward/gradient multiply-accumulates, normalization/projection element
counts, and model-query counters are recorded per arm). Timing is
telemetry-only and selects nothing.

## Reading

See `SUMMARY.md` for the outcome tables and the honest interpretation.
