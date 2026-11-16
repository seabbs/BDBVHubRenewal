#!/bin/bash
# Run scripts/covariate_ablation.py in K single-thread shards under nohup.
# Usage: scripts/covariate_ablation.sh OUT_DIR K [extra args]
out=$1; k=$2; shift 2
mkdir -p "$out"
for i in $(seq 0 $((k - 1))); do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup nice -n 19 .venv/bin/python \
    scripts/covariate_ablation.py ~/bdbv-rounds/rounds/r1/sims 30140 30143 \
    "$DELAYS" ~/bdbv-rounds/checkpoints/n1000/finetuned-ckpt \
    "$out/shard-$i.csv" --shard "$i" "$k" --label "covab-shard$i" "$@" \
    > "$out/shard-$i.log" 2>&1 < /dev/null &
done
