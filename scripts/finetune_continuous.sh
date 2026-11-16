#!/usr/bin/env bash
# Fine-tune Chronos-2 continuously on the r4 rounds archie simulates.
#
#   nohup scripts/finetune_continuous.sh > output/finetune/continuous.out 2>&1 &
#
# Each round of 1,000 simulations lands on the remote (archie) in
# `$REMOTE_ROOT/rounds/r<N>/` (`seeds.txt`, `held-out.txt`, `sims/`). This
# copies each new round once into output/finetune/continuous/, appends its
# seeds to one seed list, and runs one trainer over the growing list from
# base Chronos-2 to OUTBREAKS with Chronos-2's recipe
# (scripts/finetune_stream.py): learning rate 1e-5 falling linearly to zero
# over LR_STEPS, each block's horizon as its prediction length, every draw
# cut at 16 snapshots. The trainer checks and publishes a checkpoint at each
# doubling from 1,000, whatever its scores, and waits for rounds as they
# land. Each checkpoint is copied to `$REMOTE_ROOT/checkpoints/` with a
# line in the remote's mailbox; a stopped trainer is an alert in the
# mailbox and the local INBOX. Run from the repo root of a clean worktree.
set -uo pipefail

REMOTE=${REMOTE:-archie}
REMOTE_ROOT=${REMOTE_ROOT:-bdbv-rounds}
REMOTE_MAILBOX=${REMOTE_MAILBOX:-bdbv-archie-mailbox.md}
MAILBOX=${MAILBOX:-}
HERE=${HERE:-output/finetune/continuous}
OUTBREAKS=${OUTBREAKS:-8000}
# Each round simulates 100 held-out outbreaks.
HOLDOUT=${HOLDOUT:-100}
PREFIXES=${PREFIXES:-1000 2000 4000 8000}
# Two draws at each of 16 snapshots, sampled across the 10: 32 blocks, about
# 2.2 h per 1,000 simulations on an M-series laptop (2026-11-16 probe:
# 22,820 steps per 1,000 at 160 blocks, 1.75 s a step). The learning rate
# reaches zero at OUTBREAKS: 8 x 4,564 steps.
DRAWS=${DRAWS:-2}
SUBSET_SEED=${SUBSET_SEED:-1}
LR_STEPS=${LR_STEPS:-36500}
# Zero-shot Chronos-2's loss on the BDBV pseudo-origins
# (scripts/finetune_real_validation.py), which n1000 must beat.
ZERO_SHOT_REAL=${ZERO_SHOT_REAL:-0.666}
SOURCES=${SOURCES:-weekly daily 3day}
UNIVARIATE_SHARE=${UNIVARIATE_SHARE:-0.2}
REAL_VALIDATION=${REAL_VALIDATION:-output/finetune/real-validation.pkl}

remote_ls() { ssh -o ConnectTimeout=20 "$REMOTE" "ls $REMOTE_ROOT/rounds" 2> /dev/null; }
pull() { rsync -a "$REMOTE:$REMOTE_ROOT/rounds/$1/" "$2/"; }

sync_rounds() {
  # Copy each new round, in order, once; its seeds join the seed list.
  mkdir -p "$HERE/sims" "$HERE/rounds"
  touch "$HERE/seeds.txt"
  local n
  for n in $(remote_ls | sed -n 's/^r\([0-9]*\)$/\1/p' | sort -n); do
    [ -e "$HERE/rounds/r$n.done" ] && continue
    pull "r$n" "$HERE/rounds/r$n" || return 1
    [ -s "$HERE/rounds/r$n/seeds.txt" ] || return 0
    cp -R "$HERE/rounds/r$n/sims/." "$HERE/sims/"
    cat "$HERE/rounds/r$n/seeds.txt" >> "$HERE/seeds.txt"
    touch "$HERE/rounds/r$n.done"
  done
}

beats_zero_shot() {
  # Whether checkpoint settings $1 improve on zero-shot on both the
  # held-out simulations and the BDBV pseudo-origins.
  python3 -c '
import json, sys
s = json.load(open(sys.argv[1]))
real = s.get("real_validation_loss", float("inf"))
ok = s["held_out_loss"] < s["held_out_loss_start"] and real < float(sys.argv[2])
sys.exit(0 if ok else 1)
' "$1" "$ZERO_SHOT_REAL"
}

[[ "${BASH_SOURCE[0]}" != "$0" ]] && return 0

alert() {
  local line
  line="$(date '+%Y-%m-%d %H:%M') fm-finetune3 continuous: $*"
  [ -n "$MAILBOX" ] && echo "$line" >> "$MAILBOX"
  ssh "$REMOTE" "echo \"$line\" >> $REMOTE_MAILBOX"
}

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "the worktree has uncommitted changes; commit them to train" >&2
  exit 1
fi
trainer=""
published=""
while true; do
  sync_rounds || echo "$(date) sync failed; trying again"
  if [ -z "$trainer" ] && [ -e "$HERE/rounds/r1.done" ]; then
    read -r -a held < "$HERE/rounds/r1/held-out.txt"
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=${MPS_HIGH_WATERMARK:-0.85} \
    PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 \
    HF_HOME=output/finetune/hf HF_HUB_OFFLINE=1 MPLCONFIGDIR="${TMPDIR:-/tmp}" \
      nohup .venv/bin/python scripts/finetune_stream.py \
      "$HERE/sims" "$HERE/train" --grid r4 --seed-list "$HERE/seeds.txt" \
      --held-out-seeds "${held[0]}" "${held[1]}" --holdout "$HOLDOUT" \
      --outbreaks "$OUTBREAKS" \
      --prefixes $PREFIXES --stop-from "$((OUTBREAKS + 1))" \
      --blocks-per-simulation $((16 * DRAWS)) --draws-per-simulation "$DRAWS" \
      --subset-seed "$SUBSET_SEED" --learning-rate 1e-5 --lr-steps "$LR_STEPS" \
      --sources $SOURCES --univariate-share "$UNIVARIATE_SHARE" \
      --real-validation "$REAL_VALIDATION" --note "continuous r4" --publish \
      >> "$HERE/train.log" 2>&1 < /dev/null &
    trainer=$!
    echo "$trainer" > "$HERE/trainer.pid"
    alert "started training from base Chronos-2 (pid $trainer)"
  fi
  for ckpt in "$HERE"/train/n*/settings.json; do
    [ -e "$ckpt" ] || continue
    dir=$(dirname "$ckpt")
    n=$(basename "$dir")
    [[ " $published " == *" $n "* ]] && continue
    rsync -a "$dir/" "$REMOTE:$REMOTE_ROOT/checkpoints/$n/" || continue
    key=$(python3 -c "import json;print(json.load(open('$ckpt'))['hash'])")
    alert "checkpoint $n chronos-ft-$key at $REMOTE_ROOT/checkpoints/$n"
    published="$published $n"
    if [ "$n" = n1000 ] && ! beats_zero_shot "$ckpt"; then
      alert "ALERT n1000 does not beat zero-shot on held-out sims and pseudo-origins; training goes on, check before n2000"
    fi
  done
  if [ -n "$trainer" ] && ! kill -0 "$trainer" 2> /dev/null; then
    if grep -q '"stage": "finished"' "$HERE/train/stream-status.json" 2> /dev/null; then
      alert "training finished"
    else
      alert "ALERT trainer stopped before finishing; see $HERE/train.log"
    fi
    break
  fi
  sleep 300
done
