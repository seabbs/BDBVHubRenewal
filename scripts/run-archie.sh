#!/usr/bin/env bash
# Fit the Chronos-2 models at every real origin into a scratch store,
# then score them with the random walks on the release backtest.
# Usage: run.sh MODEL... ; set GPU=1 to use the GPU.
set -euo pipefail
S=$HOME/fm-chronos-evidence
W=$HOME/code/seabbs/BDBVHubRenewal/.worktrees/fm-chronos
cd "$W"
export LD_LIBRARY_PATH=$HOME/.local/lib/nvidia-driver HF_HOME=$S/hf BDBV_CHRONOS_CACHE=$S/chronos-cache
export XLA_FLAGS="--xla_force_host_platform_device_count=2 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
[ "${GPU:-0}" = 1 ] || export CUDA_VISIBLE_DEVICES=""
origins=$(uv run python -c "
from bdbvhubrenewal.data import load_target, load_delays, load_daily
from bdbvhubrenewal.evaluation.backtest import load_config
from bdbvhubrenewal.evaluation.store import origin_tasks
print(' '.join(origin_tasks(load_target('data'), load_delays('data'), load_config(), load_daily('data'))))")
echo "origins: $origins"
for m in "$@"; do
  for o in $origins; do
    echo "== $m $o $(date +%T)"
    uv run bdbvhubrenewal fit-store fit --model "$m" --origin "$o" --store "$S/store"
  done
done
echo "DONE $(date +%T)"
