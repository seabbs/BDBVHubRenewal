#!/usr/bin/env bash
# Checkpoint curve evidence for the paper: score every published
# checkpoint of grid GRID on the German development half and the BDBV
# release backtest from native quantiles, build
# evidence/checkpoint-scores.csv, and commit and push it on this branch.
# Resumes from its caches; reports each stage to the archie mailbox.
#
# Usage: scripts/paper_checkpoints.sh [GRID]
# Env: RUN_TREE (worktree with the venv and data, default ckpt-lora),
#      NATIVE (native output root, default ~/ckpt-curve), OUT (~/paper-ckpt).
set -euo pipefail
GRID=${1:-r3}
HERE=$(cd "$(dirname "$0")/.." && pwd)
RUN_TREE=${RUN_TREE:-$HOME/code/seabbs/BDBVHubRenewal/.worktrees/ckpt-lora}
NATIVE=${NATIVE:-$HOME/ckpt-curve}
OUT=${OUT:-$HOME/paper-ckpt}/$GRID
PY=$RUN_TREE/.venv/bin/python
MB=${MAILBOX:-$HOME/bdbv-archie-mailbox.md}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 MPLCONFIGDIR=$OUT/mpl
mkdir -p "$OUT/log" "$OUT/mpl"
say() { echo "$(date '+%F %H:%M') paper: checkpoints $GRID: $*" >> "$MB"; }
trap 'say "FAILED at line $LINENO (logs $OUT/log)"' ERR
scope() { systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
  nice -n 19 "$@"; }
cd "$RUN_TREE"

# 1. Published checkpoints of the grid.
$PY "$HERE/scripts/paper_checkpoint_inputs.py" releases "$GRID" \
  "$OUT/checkpoints.csv" > "$OUT/log/releases.log"
mapfile -t CKPTS < <(tail -n +2 "$OUT/checkpoints.csv" | cut -d, -f1)
say "${#CKPTS[@]} checkpoints: ${CKPTS[*]}"

# 2. randomwalknowcast on the German development half, 1,000 draws.
scope $PY "$HERE/scripts/germany_baseline.py" "$OUT/germany-baseline" 1000 \
  randomwalknowcast > "$OUT/log/germany-baseline.log" 2>&1
say "German randomwalknowcast forecasts done"

# 3. Native quantiles of each label, unless another job is on it.
ZS="fm_chronos_daily_ragged fm_chronos_weekly_ragged"
FT="fm_chronos_daily_ragged_ft fm_chronos_weekly_ragged_ft"
busy() { pgrep -f "native_run.py $1 $2 " > /dev/null; }
for label in zero-shot "${CKPTS[@]}"; do
  if [ "$label" = zero-shot ]; then m=$ZS; ft=${CKPTS[0]}; else m=$FT; ft=$label; fi
  while busy germany "$label" || busy bdbv "$label"; do sleep 600; done
  if [ ! -f "$NATIVE/native-bdbv/$label/quantiles.csv.gz" ]; then
    BDBV_CHRONOS_FT=$ft scope $PY "$HERE/scripts/native_run.py" bdbv "$label" \
      "$NATIVE/native-bdbv" "$NATIVE/cells-bdbv.csv" $m \
      > "$OUT/log/bdbv-$label.log" 2>&1
  fi
  BDBV_CHRONOS_FT=$ft scope $PY "$HERE/scripts/native_run.py" germany "$label" \
    "$NATIVE/native-germany" $m > "$OUT/log/germany-$label.log" 2>&1
  say "$label scored"
done

# 4. The table, committed on this branch.
$PY "$HERE/scripts/paper_checkpoint_inputs.py" germany \
  "$NATIVE/native-germany" "$OUT/germany-baseline" "$OUT/in-germany"
$PY "$HERE/scripts/paper_checkpoint_inputs.py" bdbv \
  "$NATIVE/native-bdbv" "$NATIVE/cells-bdbv.csv" "$OUT/in-bdbv" \
  > "$OUT/log/bdbv-inputs.log"
cp "$OUT/checkpoints.csv" "$HERE/evidence/checkpoints.csv"
PYTHONPATH="$HERE/scripts" $PY "$HERE/scripts/checkpoint_curve.py" \
  "$HERE/evidence/checkpoints.csv" "$OUT/in-germany" "$OUT/in-bdbv" \
  "$HERE/evidence/checkpoint-scores.csv" > "$OUT/log/table.log"
cd "$HERE"
git add evidence/checkpoints.csv evidence/checkpoint-scores.csv
if git diff --cached --quiet; then
  say "table unchanged"
else
  git commit -q -m "evidence: $GRID checkpoint scores from native quantiles, German baseline at 1,000 draws"
  git push -q origin HEAD
  say "table committed at $(git rev-parse --short=9 HEAD) on $(git branch --show-current); bump the checkpoint_scores pin"
fi
