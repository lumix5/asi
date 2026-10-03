#!/usr/bin/env bash
# Execute the literal 144-shard reference-life development scorecard on the
# locked CI runtime (python 3.12.12, jax 0.11.0) from exact main f3d32c45.
# Mirrors .github/workflows/reference-life-scorecard-dev.yml shard semantics:
# exit 0 or 1 produces a validated shard record; exit >1 is fatal.
set -u
PY=/home/lumix/.tmp/opencode/asi-lock312/venv/bin/python
MOD=alberta_framework.benchmarks.reference_life_scorecard
OUT=/home/lumix/.tmp/opencode/scorecard-v1
cd /home/lumix/.tmp/opencode/asi-run-tree || exit 99

rm -f "$OUT/RUN_DONE_MARKER" "$OUT/RUN_FATAL"
for env in switching_two_state riverswim; do
  for arm in prototype prototype_frozen random privileged_oracle differential_sarsa sarsa; do
    for seed in $(seq 70000 70011); do
      out="$OUT/shards/${env}__${arm}__${seed}.json"
      if [ -f "$out" ]; then
        echo "skip existing ${env} ${arm} ${seed}" >&2
        continue
      fi
      printf '%s %s %s\n' "$env" "$arm" "$seed"
    done
  done
done | xargs -P 10 -L 1 bash -c '
  env=$1; arm=$2; seed=$3
  out="'"$OUT"'/shards/${env}__${arm}__${seed}.json"
  if [ -f "$out" ]; then exit 0; fi
  '"$PY"' -m '"$MOD"' run-shard --environment "$env" --arm "$arm" --seed "$seed" --output "$out"
  st=$?
  if [ "$st" -gt 1 ]; then
    echo "FATAL ${env} ${arm} ${seed} exit ${st}" | tee -a "'"$OUT"'/RUN_FATAL"
  fi
' _

if [ -s "$OUT/RUN_FATAL" ]; then
  echo "completed with fatal shard failures" > "$OUT/RUN_DONE_MARKER"
  exit 1
fi
echo "completed all shards" > "$OUT/RUN_DONE_MARKER"
