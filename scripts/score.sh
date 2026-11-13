#!/usr/bin/env bash
# Load the fits of MODEL... from stores/base, drop their stored delays (the
# Chronos-2 models give none, so they are scored on the complete target
# only), run the release backtest with the random walks and summarise.
# Usage: score.sh RESULTS MODEL...
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
W=/Users/lshsa2/code/seabbs/bdbv-hub-baseline-forecasts/.worktrees/fm-chronos
R=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift
cd "$W"
export XLA_FLAGS="--xla_force_host_platform_device_count=2 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
rm -rf "$R" && mkdir -p "$R"
uv run bdbvhubrenewal fit-store load --store "$here/stores/base" --results "$R" --models "$@"
uv run python -c "
import sys, pandas as pd
p = sys.argv[1] + '/delay-summary.csv'
d = pd.read_csv(p)
d.loc[~d['model'].str.startswith('fm_chronos_')].to_csv(p, index=False)
" "$R"
uv run bdbvhubrenewal backtest --models randomwalk randomwalknowcast \
  --results "$R" --output "$R" --no-recovery
uv run python "$here/scripts/summary.py" "$R"
echo "SCORED $(date +%T)"
