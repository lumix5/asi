# Grid 1 summary — native supervised CL suite (nonpromoting)

Mean online accuracy over the four frozen development seeds (mean ± sample
standard deviation; per-seed values in `grid.json`):

| benchmark | online_sgd | replay_sgd | running_centroid | frozen_no_learning |
| --- | --- | --- | --- | --- |
| split_mnist | 0.412500 ± 0.047871 | 0.250000 ± 0.035355 | 0.425000 ± 0.020412 | 0.100000 ± 0.035355 |
| rotated_mnist | 0.175000 ± 0.040825 | 0.143750 ± 0.071807 | 0.206250 ± 0.059073 | 0.100000 ± 0.020412 |
| split_cifar100 | 0.176563 ± 0.013858 | 0.059375 ± 0.023105 | 0.046875 ± 0.019432 | 0.010938 ± 0.005984 |
| ipmnist | 0.155781 ± 0.007349 | 0.146719 ± 0.006948 | 0.233438 ± 0.010123 | 0.100156 ± 0.006503 |

Resource receipts (identical across seeds for each benchmark/arm; from the
shard records):

- `data_steps`: 40 (split_mnist, rotated_mnist), 160 (split_cifar100),
  1600 (ipmnist) per arm per seed; `data_bytes_read = steps * (4*input_dim + 4)`.
- `model_queries` / `parameter_updates`: SGD arms `3*steps-1` / `2*steps-1`
  (the predict-before-update query plus one optimizer call per example and
  one budget-matching replay or repeat call); centroid arms `steps` / `steps`;
  frozen arm `steps` / `0`.
- `persistent_bytes`: 31400 on every MNIST lane (784x10 float32 weights +
  10 bias, or the same-footprint centroid class sums) and 1229200 on
  split_cifar100 (3072x100 + 100).
- `peak_replay_bytes` (`replay_sgd` only; capacity 16 reached on every
  lane): 50240 on the MNIST lanes (16 x 3140 bytes/example), 196672 on
  split_cifar100 (16 x 12292); 0 elsewhere.
- `elapsed_ns`: host timing telemetry only; not a resource contract.

## Honest reading (development selection only)

1. **All learning arms beat the frozen/no-learning control on every
   benchmark** (e.g. ipmnist 0.147–0.233 vs 0.100; split_cifar100
   0.047–0.177 vs 0.011). The suite's predict-before-update accounting and
   task-agnostic construction behave as designed; this is a sanity floor,
   not a mechanism result.
2. **`replay_sgd` underperforms `online_sgd` on every benchmark at this
   budget** (split_mnist 0.250 vs 0.413; split_cifar100 0.059 vs 0.177).
   With capacity 16 and 8-example tasks, the replay window spans more than
   two full tasks, so replayed pairs are mostly stale-task data
   interfering with the current task under a shared linear model. This
   is a negative outcome for naive capacity-16 replay at this scale and is
   retained as such; it makes no claim about replay at canonical budgets.
3. **`running_centroid` is the nominal winner on the three MNIST-family
   lanes** (split 0.425, rotated 0.206, ipmnist 0.233) while `online_sgd`
   wins split_cifar100 (0.177 vs 0.047) — consistent with a linear
   classifier struggling on 100-way raw-pixel CIFAR where the
   class-conditional-mean rule at least ranks coarse structure, and with
   rotation drift hurting the fixed-weight SGD arm hardest. With 4 seeds
   and this noise these are nominal orderings, not established rankings.
4. **Rotated MNIST is the weakest lane for every learner** (best arm
   0.206), matching the expectation that a fixed linear model cannot
   compensate 180-degree input rotation within 40 updates.

No paired significance test is applied at 4 seeds; nothing here promotes,
populates `reference-dev`, or constitutes scientific evidence. The grid
and its receipts are retained development history for #1578.
