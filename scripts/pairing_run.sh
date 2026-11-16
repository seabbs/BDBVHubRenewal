#!/usr/bin/env bash
# Run one pairing evidence script per model in the background, each in
# its own 8 GB scope at nice 19, and write a mailbox line when each ends.
# Usage, from a checkout of the copula branch (its root):
#   pairing_run.sh SCRIPT OUT MODEL...
set -u
script=$1 out=$2
shift 2
mailbox=~/bdbv-archie-mailbox.md
mkdir -p "$out"
export GH_TOKEN=${GH_TOKEN:-$(gh auth token)}
export BDBV_CHRONOS_FT=${BDBV_CHRONOS_FT:-6b7aaa8d4f55}
export HF_HOME=${HF_HOME:-~/fm-chronos-evidence/hf}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1 \
--xla_cpu_multi_thread_eigen=false"
for model in "$@"; do
  log="$out/$model.log"
  (
    systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
      nice -n 19 uv run --offline python "$script" "$out" "$model" \
      > "$log" 2>&1
    status=$?
    stamp=$(date '+%Y-%m-%d %H:%M')
    name=$(basename "$script" .py)
    if [ "$status" -eq 0 ]; then
      echo "$stamp archie (bdbv-copula): $name $model finished, $out" \
        >> "$mailbox"
    else
      echo "$stamp archie (bdbv-copula): $name $model FAILED ($status)," \
        "log $log: $(tail -n 1 "$log")" >> "$mailbox"
    fi
  ) &
done
wait
