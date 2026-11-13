#!/usr/bin/env bash
# Score one arm of #193's fits: score-ft.sh STORE RESULTS, from the
# fm-finetune worktree's root. Loads the ragged models' fits, zero-shot
# and fine-tuned, runs the release backtest with the random walks and
# summarises on matched cells (summary.py).
set -euo pipefail
store=$1 R=$2
here=$(dirname "$0")
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XLA_FLAGS="--xla_force_host_platform_device_count=2 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
rm -rf "$R" && mkdir -p "$R"
uv run bdbvhubrenewal fit-store load --store "$store" --results "$R" --models \
  fm_chronos_weekly_ragged fm_chronos_weekly_ragged_ft \
  fm_chronos_daily_ragged fm_chronos_daily_ragged_ft
uv run bdbvhubrenewal backtest --models randomwalk randomwalknowcast \
  --results "$R" --output "$R" --no-recovery
uv run python "$here/summary.py" "$R"
