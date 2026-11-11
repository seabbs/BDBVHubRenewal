#!/usr/bin/env bash
# Fit the Chronos-2 models at every real origin on the laptop GPU (MPS),
# one at a time, into a scratch store.
set -euo pipefail
W=/Users/lshsa2/code/seabbs/bdbv-hub-baseline-forecasts/.worktrees/fm-chronos
S=/private/tmp/claude-501/-Users-lshsa2-code/a1882b20-4e64-4505-919c-312979ff9039/scratchpad/fmrun/mac
mkdir -p "$S"
cd "$W"
export UV_CACHE_DIR=$TMPDIR/uvcache HF_HOME=/private/tmp/claude-501/-Users-lshsa2-code/a1882b20-4e64-4505-919c-312979ff9039/scratchpad/fm-explore/.hf HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export XLA_FLAGS="--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
origins="2026-10-09 2026-10-17 2026-10-24 2026-11-01 2026-11-08"
for m in "$@"; do
  for o in $origins; do
    /usr/bin/time -l uv run bdbvhubrenewal fit-store fit --model "$m" --origin "$o" --store "$S/store" 2>&1 \
      | grep -E "Wrote|ERROR|real|maximum resident" | tr '\n' ' '; echo " [$m $o]"
  done
done
echo DONE
