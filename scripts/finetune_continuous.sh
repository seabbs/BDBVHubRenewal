#!/usr/bin/env bash
# Fine-tune Chronos-2 continuously on the r4 rounds as they are simulated.
#
#   nohup scripts/finetune_continuous.sh > output/finetune/continuous.out 2>&1 &
#
# Each round of 1,000 simulations lands on the rounds host (REMOTE, or this
# machine; scripts/local_config.sh) in `$REMOTE_ROOT/rounds/r<N>/` (`manifest.csv`, `held-out.txt`, `sims/`).
# This streams each round's finished batches into output/finetune/continuous/
# as its manifest lists them, appends their seeds to one seed list, and runs
# one trainer over the growing list from
# base Chronos-2 to OUTBREAKS with Chronos-2's recipe
# (scripts/finetune_stream.py): learning rate 1e-5 falling linearly to zero
# over LR_STEPS, each block's horizon as its prediction length, every draw
# cut at 16 snapshots. The trainer checks and publishes a checkpoint at each
# doubling from 1,000, whatever its scores, and waits for rounds as they
# land. Each checkpoint is copied to `$REMOTE_ROOT/checkpoints/` with a
# line in the shared mailbox; a stopped trainer is an alert in the
# mailbox and the local INBOX. Run from the repo root of a clean worktree.
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/local_config.sh"
HERE=${HERE:-output/finetune/continuous}
OUTBREAKS=${OUTBREAKS:-8000}
# Each round simulates 100 held-out outbreaks.
HOLDOUT=${HOLDOUT:-100}
# Early checkpoints at 125 to 750 compare runs as they train; each lands
# on a chunk's end.
PREFIXES=${PREFIXES:-125 250 500 750 1000 2000 4000 8000}
# Two draws at each of 16 snapshots, sampled across the 10: 32 blocks, about
# 2.2 h per 1,000 simulations on an M-series laptop (2026-11-16 probe:
# 22,820 steps per 1,000 at 160 blocks, 1.75 s a step). The learning rate
# reaches zero at OUTBREAKS: 8 x 4,564 steps.
DRAWS=${DRAWS:-2}
SUBSET_SEED=${SUBSET_SEED:-1}
LR_STEPS=${LR_STEPS:-36500}
# Zero-shot Chronos-2's loss on the BDBV pseudo-origins
# (scripts/finetune_real_validation.py, blocks with their own 16-step
# prediction length), which n1000 must beat.
ZERO_SHOT_REAL=${ZERO_SHOT_REAL:-4.147}
SOURCES=${SOURCES:-daily weekly 3day daily+weekly daily+3day}
UNIVARIATE_SHARE=${UNIVARIATE_SHARE:-0.2}
# Simulations per training chunk: training starts once this many have
# landed, in seed order.
CHUNK=${CHUNK:-125}
REAL_VALIDATION=${REAL_VALIDATION:-output/finetune/real-validation.pkl}
# The series bias (src/bdbvhubrenewal/forecasters/series_bias.py): 1 trains
# a learned group attention bias for same-place and parent-child series
# beside the adapter. Such a run keeps to its own HERE and checkpoint
# folder (checkpoints/slots/, the paper's finetune_run name), so the run
# without it is never touched, and its releases are chronos-slots-<hash>.
SERIES_BIAS=${SERIES_BIAS:-0}

remote_ls() { on_rounds "ls $REMOTE_ROOT/rounds" 2> /dev/null; }
# Copy a round's file or folder ($2, "sims/" for its simulations) to $3.
pull() {
  rsync -a --exclude '.tmp-*' --exclude 'tmp/' "$(at_rounds "$REMOTE_ROOT/rounds/$1/$2")" "$3/"
}

sync_rounds() {
  # Stream each round's finished batches as its manifest lists them, so
  # training need not wait for the round to be ready: the manifest first,
  # then the simulations, so every seed it lists has its files; its new
  # training seeds join the seed list once each, in manifest order.
  mkdir -p "$HERE/sims" "$HERE/rounds"
  touch "$HERE/seeds.txt"
  local n d
  for n in $(remote_ls | sed -n 's/^r\([0-9]*\)$/\1/p' | sort -n); do
    d=$HERE/rounds/r$n
    mkdir -p "$d"
    pull "r$n" manifest.csv "$d" || continue
    pull "r$n" held-out.txt "$d" 2> /dev/null || true
    pull "r$n" sims/ "$HERE/sims" || return 1
    python3 - "$d/manifest.csv" "$HERE/seeds.txt" <<'PY'
import csv, sys
from pathlib import Path
manifest, seeds = Path(sys.argv[1]), Path(sys.argv[2])
have = set(seeds.read_text().split())
new = [
    r["seed"]
    for r in csv.DictReader(manifest.open())
    if r["kind"] == "train" and r["seed"] not in have
]
with seeds.open("a") as f:
    f.writelines(f"{s}\n" for s in dict.fromkeys(new))
PY
  done
}

bias_args() {
  # The trainer's bias argument; a bias run must not use the default HERE.
  [ "$SERIES_BIAS" = 1 ] || return 0
  if [ "$HERE" = output/finetune/continuous ]; then
    echo "SERIES_BIAS needs its own HERE, not $HERE" >&2
    return 1
  fi
  echo "--series-bias"
}

remote_checkpoints() {
  # The folder under REMOTE_ROOT this run copies its checkpoints to.
  if [ "$SERIES_BIAS" = 1 ]; then
    echo checkpoints/slots
  else
    echo checkpoints
  fi
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

alert() { post "continuous: $*"; }

bias=$(bias_args) || exit 1
ckpts=$(remote_checkpoints)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "the worktree has uncommitted changes; commit them to train" >&2
  exit 1
fi
trainer=""
published=""
while true; do
  sync_rounds || echo "$(date) sync failed; trying again"
  if [ -z "$trainer" ] && [ -s "$HERE/rounds/r1/held-out.txt" ]; then
    read -r -a held < "$HERE/rounds/r1/held-out.txt"
    # $bias is empty or one word, so it is left unquoted.
    # shellcheck disable=SC2086
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=${MPS_HIGH_WATERMARK:-0.85} \
    PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 \
    HF_HOME=${HF_HOME:-output/finetune/hf} HF_HUB_OFFLINE=1 MPLCONFIGDIR="${TMPDIR:-/tmp}" \
      nohup .venv/bin/python scripts/finetune_stream.py \
      "$HERE/sims" "$HERE/train" --grid r4 --seed-list "$HERE/seeds.txt" \
      --held-out-seeds "${held[0]}" "${held[1]}" --holdout "$HOLDOUT" \
      --outbreaks "$OUTBREAKS" \
      --prefixes $PREFIXES --stop-from "$((OUTBREAKS + 1))" \
      --blocks-per-simulation $((16 * DRAWS)) --draws-per-simulation "$DRAWS" \
      --subset-seed "$SUBSET_SEED" --learning-rate 1e-5 --lr-steps "$LR_STEPS" \
      --sources $SOURCES --univariate-share "$UNIVARIATE_SHARE" \
      --real-validation "$REAL_VALIDATION" --note "continuous r4" --publish \
      --chunk "$CHUNK" --wait-for-seeds \
      --exclude "$HERE/excluded.txt" $bias \
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
    on_rounds "mkdir -p $REMOTE_ROOT/$ckpts" \
      && rsync -a "$dir/" "$(at_rounds "$REMOTE_ROOT/$ckpts/$n/")" || continue
    key=$(python3 -c "import json;print(json.load(open('$ckpt'))['hash'])")
    alert "checkpoint $n ($bias) hash $key at $REMOTE_ROOT/$ckpts/$n"
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
