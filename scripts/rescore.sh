#!/usr/bin/env bash
# Rescore #192's Chronos-2 models on main's rolling complete truth: fit
# all six at every real origin into a scratch store (archie's GPU), load
# them, run the release backtest with the random walks, and summarise
# against randomwalknowcast on matched cells (summary.py).
# Usage: rescore.sh WORKTREE OUT
set -euo pipefail
W=$1 OUT=$2
here=$(cd "$(dirname "$0")" && pwd)
export PATH=$HOME/.local/bin:$PATH
models="fm_chronos_weekly_complete fm_chronos_weekly_ragged fm_chronos_weekly_ragged_ft fm_chronos_daily_complete fm_chronos_daily_ragged fm_chronos_daily_ragged_ft"
cd "$W"
git log --oneline -1
uv run --frozen bdbvhubrenewal data
mkdir -p "$OUT"
S=$OUT/store R=$OUT/results
# shellcheck disable=SC2086
uv run --frozen bdbvhubrenewal fit-store plan --store "$S" --models $models --out "$OUT/plan.json"
python3 -c 'import json,sys; [print(e["model"], e["origin"]) for e in json.load(open(sys.argv[1]))["include"]]' "$OUT/plan.json" \
  | while read -r m o; do
      uv run --frozen bdbvhubrenewal fit-store fit --model "$m" --origin "$o" --store "$S" 2>&1 | grep -E "Wrote|ERROR" || true
    done
rm -rf "$R" && mkdir -p "$R"
# shellcheck disable=SC2086
uv run --frozen bdbvhubrenewal fit-store load --store "$S" --results "$R" --models $models
uv run --frozen bdbvhubrenewal backtest --models randomwalk randomwalknowcast --results "$R" --output "$R" --no-recovery
uv run --frozen python "$here/summary.py" "$R"
echo RESCORED
