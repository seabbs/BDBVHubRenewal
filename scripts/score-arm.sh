#!/usr/bin/env bash
# Score one arm's scratch store: score-arm.sh ARM MODEL...
set -euo pipefail
S=$HOME/fm-chronos-evidence
W=$HOME/code/seabbs/BDBVHubRenewal/.worktrees/fm-chronos
arm=$1; shift
cd "$W"
export XLA_FLAGS="--xla_force_host_platform_device_count=2 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
R=$S/results-$arm
rm -rf "$R" && mkdir -p "$R"
uv run bdbvhubrenewal fit-store load --store "$S/arm-$arm" --results "$R" --models "$@"
uv run bdbvhubrenewal backtest --models randomwalk randomwalknowcast \
  --results "$R" --output "$R" --no-recovery
uv run python "$S/summary.py" "$R"
echo "SCORED $arm $(date +%T)"
