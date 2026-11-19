#!/usr/bin/env bash
# The horizon-loss screen: each arm fine-tunes Chronos-2 on the same 500
# r6 simulations (round 21, round 20 held out) with the live r6 recipe
# and its own --horizon-loss, never published, then is scored on the
# held-out sims (heldout_quantiles.py) and the German development half at
# nowcast and forecast horizons (german_horizons.py).
#
# Scoring has the GPU first: while a checkpoint-scoring unit runs, the
# arm's trainer is stopped and then restarted, resuming from its state.
#
# Env: CUDALIB (libcuda symlink dir), ROOT (screen dir), ROUNDS (the r6
# rounds dir), MAILBOX, MEMORY_GB (the 2070's budget, 5), HELDOUT_BLOCKS
# (optional: a blocks.pkl from heldout_quantiles.py to score on).
# Args: the arms, from A (all), F (nowcast), G (nowcast+1w), H (distil).
set -uo pipefail
cd "$(dirname "$0")/../.."
note() {
  echo "$(date '+%Y-%m-%d %H:%M') [archie bdbv-ftpractice] $*" >> "$MAILBOX"
}
scoring() { pgrep -f bdbvhubrenewal.evaluation.checkpoint > /dev/null; }
export LD_LIBRARY_PATH=$CUDALIB DEVICE=cuda HF_HUB_OFFLINE=1

sims=$ROOT/sims
mkdir -p "$sims"
for r in 20 21; do
  for f in "$ROUNDS"/r$r/sims/counts-*.csv.gz "$ROUNDS"/r$r/sims/settings-*.csv; do
    [ -e "$sims/$(basename "$f")" ] || ln "$f" "$sims/" 2>/dev/null || cp "$f" "$sims/"
  done
done
{
  echo simulation
  for r in 20 21; do tail -n +2 "$ROUNDS/r$r/qc/excluded.csv" | cut -d, -f1; done
} > "$ROOT/excluded.csv"
seq 32100 32139 > "$ROOT/seeds.txt"
# Held-out blocks shared with the width check, so every arm is scored on
# the same blocks as zero-shot and the published checkpoints.
mkdir -p "$ROOT/heldout"
[ -n "${HELDOUT_BLOCKS:-}" ] && ln -sf "$HELDOUT_BLOCKS" "$ROOT/heldout/blocks.pkl"

declare -A MODE=([A]=all [F]=nowcast [G]=nowcast+1w [H]=distil)
for arm in "$@"; do
  out=$ROOT/$arm/train
  mkdir -p "$out"
  cmd=(.venv/bin/python scripts/finetune_stream.py "$sims" "$out"
    --grid r6 --seed-list "$ROOT/seeds.txt" --held-out-seeds 32000 32039
    --holdout 90 --outbreaks 500 --prefixes 250 --stop-from 501
    --blocks-per-simulation 32 --draws-per-simulation 1 --subset-seed 1
    --learning-rate 1e-5
    --sources daily weekly 3day daily+weekly daily+rolling7 daily+3day daily+rolling3
    --univariate-share 0.2 --note "horizon screen $arm" --chunk 250
    --eval-every 200 --by-source-every 200 --eval local
    --exclude "$ROOT/excluded.csv" --series-bias --bias-lr 1e-3
    --memory-gb "$MEMORY_GB" --horizon-loss "${MODE[$arm]}")
  note "screen arm $arm (${MODE[$arm]}) starting"
  until [ -f "$out/n500/settings.json" ]; do
    while scoring; do sleep 60; done
    setsid "${cmd[@]}" >> "$ROOT/$arm/train.log" 2>&1 &
    pid=$!
    paused=0
    while kill -0 "$pid" 2> /dev/null; do
      if scoring; then
        kill -- -"$pid" 2> /dev/null
        paused=1
        echo "$(date '+%H:%M:%S') paused for scoring" >> "$ROOT/$arm/train.log"
        break
      fi
      sleep 20
    done
    wait "$pid"
    rc=$?
    if [ "$paused" = 0 ] && [ ! -f "$out/n500/settings.json" ]; then
      note "screen arm $arm FAILED (exit $rc), see $ROOT/$arm/train.log"
      exit 1
    fi
  done
  note "screen arm $arm trained"
  adapter=$out/n500/finetuned-ckpt
  while scoring; do sleep 60; done
  .venv/bin/python scripts/ftpractice/heldout_quantiles.py \
    "$ROUNDS/r20/sims" "$ROOT/heldout" --seeds 32000 32039 \
    --per-simulation 8 --checkpoint "arm$arm=$adapter" \
    >> "$ROOT/score.log" 2>&1 || note "screen arm $arm held-out scoring FAILED"
  while scoring; do sleep 60; done
  .venv/bin/python scripts/ftpractice/german_horizons.py \
    "arm$arm=$adapter" "$ROOT/germany" \
    >> "$ROOT/score.log" 2>&1 || note "screen arm $arm German scoring FAILED"
  note "screen arm $arm scored"
done
note "horizon screen done: $*"
