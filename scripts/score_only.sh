#!/usr/bin/env bash
# The scoring half of rescore.sh, on a store already fitted.
set -euo pipefail
W=$1 OUT=$2
here=$(cd "$(dirname "$0")" && pwd)
export PATH=$HOME/.local/bin:$PATH
models="fm_chronos_weekly_complete fm_chronos_weekly_ragged fm_chronos_weekly_ragged_ft fm_chronos_daily_complete fm_chronos_daily_ragged fm_chronos_daily_ragged_ft"
cd "$W"
S=$OUT/store R=$OUT/results
rm -rf "$R" && mkdir -p "$R"
# shellcheck disable=SC2086
uv run --frozen bdbvhubrenewal fit-store load --store "$S" --results "$R" --models $models
uv run --frozen bdbvhubrenewal backtest --models randomwalk randomwalknowcast --results "$R" --output "$R" --no-recovery
uv run --frozen python "$here/summary.py" "$R"
echo RESCORED
