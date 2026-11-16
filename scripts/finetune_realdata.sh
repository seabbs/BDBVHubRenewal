#!/usr/bin/env bash
# Cut the real-data blocks and train the real-data checkpoint on CPU.
#
#   scripts/finetune_realdata.sh OUT [--publish]
#
# Run from a clean checkout with the German triangle in data/raw/germany/
# (`bdbvhubrenewal germany-fetch`) and the BDBV data in data/
# (`bdbvhubrenewal data`). Both block sets are cut at this commit (skipped
# when OUT/blocks-commit already names it), then
# `scripts/finetune_realdata.py train` trains, resuming from OUT/run/state
# if it exists. Training uses at most 6 torch threads, and every batch
# recomputes activations in backward, as the CPU memory needs. Start it
# under nohup and systemd-run with a memory cap, and watch it with
# scripts/job_watch.sh.
set -euo pipefail
out=$1
shift
mkdir -p "$out"
commit=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "uncommitted changes: the checkpoint must match its commit" >&2
  exit 1
fi
py=.venv/bin/python
if [ "$(cat "$out/blocks-commit" 2> /dev/null)" != "$commit" ]; then
  OMP_NUM_THREADS=1 $py scripts/finetune_realdata.py blocks-germany \
    data/raw/germany/COVID-19_hospitalizations.csv "$out/germany.pkl"
  OMP_NUM_THREADS=1 $py scripts/finetune_realdata.py blocks-bdbv \
    data "$out/bdbv.pkl"
  echo "$commit" > "$out/blocks-commit"
fi
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 HF_HUB_OFFLINE=1 \
  $py scripts/finetune_realdata.py train "$out/run" \
  --blocks "$out/germany.pkl" "$out/bdbv.pkl" \
  --steps 1000 --threads 6 --checkpoint-above 0 \
  --note "German development half and public BDBV data, CPU" "$@"
