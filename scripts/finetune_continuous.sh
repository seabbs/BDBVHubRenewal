#!/usr/bin/env bash
# Fine-tune Chronos-2 continuously on the grid r5 rounds as they are
# simulated, from round FIRST_ROUND.
#
#   nohup scripts/finetune_continuous.sh > output/finetune/continuous-r5.out 2>&1 &
#
# Each round, one grid block of 840 simulations, lands on the rounds host (REMOTE, or this
# machine; scripts/local_config.sh) in `$REMOTE_ROOT/rounds/r<N>/` (`manifest.csv`, `held-out.txt`, `sims/`).
# This streams each round's finished batches into HERE as its manifest
# lists them and runs one trainer, which waits for each batch in turn,
# on the held-out round FIRST_ROUND and the next TRAIN_ROUNDS rounds in
# seed order, from
# base Chronos-2 to OUTBREAKS with Chronos-2's recipe
# (scripts/finetune_stream.py): learning rate 1e-5 for the adapter and
# 1e-3 for the attention biases, falling linearly to zero over OUTBREAKS,
# each block's horizon as its prediction length, one draw of each
# simulation cut at 32 snapshots. The trainer checks and publishes a
# checkpoint at each of PREFIXES, whatever its scores, and waits for
# rounds as they land. Each checkpoint is copied to
# `$REMOTE_ROOT/checkpoints/slots/` with a
# line in the shared mailbox; a stopped trainer is an alert in the
# mailbox and the local INBOX. Run from the repo root of a clean worktree.
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/local_config.sh"
HERE=${HERE:-output/finetune/continuous-r5}
# Ten of grid r5's balanced blocks of 840 simulations.
OUTBREAKS=${OUTBREAKS:-8400}
# The trainer monitors the first 90 simulations of the held-out round, a
# random sample of its block's cells; the scoring workflow scores the
# whole round at each checkpoint.
HOLDOUT=${HOLDOUT:-90}
# Early checkpoints at 250 to 750 compare runs as they train; each lands
# on a chunk's end.
PREFIXES=${PREFIXES:-250 500 750 1000 2000 4000 8400}
# The first round of grid r5, the held-out block; earlier rounds are other
# grids. The next TRAIN_ROUNDS rounds are the blocks trained on.
FIRST_ROUND=${FIRST_ROUND:-9}
TRAIN_ROUNDS=${TRAIN_ROUNDS:-10}
# Round r's seeds are SEED_BASE + 100 r on, ROUND_BATCHES of them, grid
# block r (scripts/sim_rounds.sh).
SEED_BASE=${SEED_BASE:-30000}
ROUND_BATCHES=${ROUND_BATCHES:-40}
# One draw (grid r5 simulates one) at each of CUTS snapshots
# (finetune_chronos.CUTS): 32 blocks a simulation.
DRAWS=${DRAWS:-1}
CUTS=${CUTS:-32}
SUBSET_SEED=${SUBSET_SEED:-1}
# Zero-shot Chronos-2's loss on the BDBV pseudo-origins
# (scripts/finetune_real_validation.py, blocks with their own 16-step
# prediction length), which n1000 must beat.
ZERO_SHOT_REAL=${ZERO_SHOT_REAL:-4.147}
SOURCES=${SOURCES:-daily weekly 3day daily+weekly daily+3day}
UNIVARIATE_SHARE=${UNIVARIATE_SHARE:-0.2}
# Simulations per training chunk: training starts once this many have
# landed, in seed order.
CHUNK=${CHUNK:-250}
REAL_VALIDATION=${REAL_VALIDATION:-output/finetune/real-validation.pkl}
# The series bias (src/bdbvhubrenewal/forecasters/series_bias.py): learned
# group attention biases for same-place and parent-child series, trained
# beside the adapter at their own rate, into checkpoints/slots/ (the
# paper's finetune_run name), released as chronos-slots-<hash>. 0 is the
# no-bias control, which sets its own HERE and copies to checkpoints/.
SERIES_BIAS=${SERIES_BIAS:-1}

remote_ls() { on_rounds "ls $REMOTE_ROOT/rounds" 2> /dev/null; }
# Copy a round's file or folder ($2, "sims/" for its simulations) to $3.
pull() {
  rsync -a --exclude '.tmp-*' --exclude 'tmp/' "$(at_rounds "$REMOTE_ROOT/rounds/$1/$2")" "$3/"
}

# Seeds of round $1, in order.
round_seeds() {
  seq $((SEED_BASE + 100 * $1)) $((SEED_BASE + 100 * $1 + ROUND_BATCHES - 1))
}

# The first and last seed of the held-out round.
held_out_seeds() {
  local s
  s=$(round_seeds "$FIRST_ROUND")
  echo "$(head -n 1 <<< "$s") $(tail -n 1 <<< "$s")"
}

sync_rounds() {
  # Stream each round's finished batches as its manifest lists them, so
  # training need not wait for the round to be ready: the manifest first,
  # then the simulations, so every seed it lists has its files. The seed
  # list is every seed of the training rounds in seed order, finished or
  # not, so the trainer takes them in that order, as the workflow does.
  mkdir -p "$HERE/sims" "$HERE/rounds"
  local n d last=$((FIRST_ROUND + TRAIN_ROUNDS))
  for n in $(remote_ls | sed -n 's/^r\([0-9]*\)$/\1/p' | sort -n); do
    [ "$n" -ge "$FIRST_ROUND" ] && [ "$n" -le "$last" ] || continue
    d=$HERE/rounds/r$n
    mkdir -p "$d"
    pull "r$n" manifest.csv "$d" || continue
    pull "r$n" sims/ "$HERE/sims" || return 1
  done
  for n in $(seq $((FIRST_ROUND + 1)) "$last"); do
    round_seeds "$n"
  done > "$HERE/seeds.txt.tmp"
  mv "$HERE/seeds.txt.tmp" "$HERE/seeds.txt"
}

bias_args() {
  # The trainer's bias argument; the no-bias control must not use the
  # design run's HERE.
  if [ "$SERIES_BIAS" = 1 ]; then
    echo "--series-bias"
  elif [ "$HERE" = output/finetune/continuous-r5 ]; then
    echo "SERIES_BIAS=0 needs its own HERE, not $HERE" >&2
    return 1
  fi
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
  if [ -z "$trainer" ] && [ -s "$HERE/rounds/r$FIRST_ROUND/manifest.csv" ]; then
    # The trainer waits for the held-out simulations it monitors.
    read -r -a held <<< "$(held_out_seeds)"
    # $bias is empty or one word, so it is left unquoted.
    # shellcheck disable=SC2086
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=${MPS_HIGH_WATERMARK:-0.85} \
    PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 \
    HF_HOME=${HF_HOME:-output/finetune/hf} HF_HUB_OFFLINE=1 MPLCONFIGDIR="${TMPDIR:-/tmp}" \
      nohup .venv/bin/python scripts/finetune_stream.py \
      "$HERE/sims" "$HERE/train" --grid r5 --seed-list "$HERE/seeds.txt" \
      --held-out-seeds "${held[0]}" "${held[1]}" --holdout "$HOLDOUT" \
      --outbreaks "$OUTBREAKS" \
      --prefixes $PREFIXES --stop-from "$((OUTBREAKS + 1))" \
      --blocks-per-simulation $((CUTS * DRAWS)) --draws-per-simulation "$DRAWS" \
      --subset-seed "$SUBSET_SEED" --learning-rate 1e-5 \
      --sources $SOURCES --univariate-share "$UNIVARIATE_SHARE" \
      --real-validation "$REAL_VALIDATION" --note "continuous r5" --publish \
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
