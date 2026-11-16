#!/usr/bin/env bash
# Train one round of the fine-tuning loop on this machine's GPU.
#
#   scripts/finetune_round.sh ROUND INIT [REMOTE_ROUND]
#
# ROUND is the round number and INIT the adapter it continues from (a
# `finetuned-ckpt` directory), or "none" for round 0. The round's inputs
# are on the remote (archie) in REMOTE_ROUND, by default
# `rounds/r<ROUND>` under `$REMOTE_ROOT`:
#
#   seeds.txt          training seeds, one a line, in order
#   held-out.txt       first and last held-out seed
#   sims/              their counts-*.csv.gz and settings-*.csv
#   real-training.pkl  real-data training blocks, used only with
#                      USE_REAL_TRAINING=1 (a separate arm, not round 1)
#
# The round is copied to output/finetune/rounds/r<ROUND>, trained with
# Chronos-2's recipe (scripts/finetune_stream.py: learning rate 1e-5
# falling linearly to zero over LR_STEPS, each block's horizon as its
# prediction length), checked against its code_commit and published as
# a release, and its checkpoint is copied back to REMOTE_ROUND/checkpoint
# with a line in the remote's mailbox. Run from the repo root of a clean
# worktree.
set -uo pipefail

REMOTE=${REMOTE:-archie}
REMOTE_ROOT=${REMOTE_ROOT:-bdbv-rounds}
REMOTE_MAILBOX=${REMOTE_MAILBOX:-bdbv-archie-mailbox.md}
OUTBREAKS=${OUTBREAKS:-1000}
LR_STEPS=${LR_STEPS:-2750}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
GRID=${GRID:-r4}
SOURCES=${SOURCES:-weekly daily 3day}
REAL_VALIDATION=${REAL_VALIDATION:-output/finetune/real-validation.pkl}

# The trainer's arguments for round $1 from adapter $2 ("none" for none).
trainer_args() {
  local held
  read -r -a held < "$ROUND_LOCAL/held-out.txt"
  args=("$ROUND_LOCAL/sims" "$ROUND_LOCAL/train"
    --grid "$GRID" --seed-list "$ROUND_LOCAL/seeds.txt"
    --held-out-seeds "${held[0]}" "${held[1]}"
    --outbreaks "$OUTBREAKS" --prefixes "$OUTBREAKS"
    --stop-from "$((OUTBREAKS + 1))" --blocks-per-simulation 16
    --exclude "$ROUND_LOCAL/qc/excluded.csv"
    --learning-rate "$LEARNING_RATE" --lr-steps "$LR_STEPS"
    --sources $SOURCES --real-validation "$REAL_VALIDATION"
    --note "round $1" --publish)
  [ "$2" != none ] && args+=(--init-adapter "$2")
  [ "${USE_REAL_TRAINING:-0}" = 1 ] \
    && args+=(--real-training "$ROUND_LOCAL/real-training.pkl")
  return 0
}

[[ "${BASH_SOURCE[0]}" != "$0" ]] && return 0

round=$1
init=$2
remote_round=${3:-$REMOTE_ROOT/rounds/r$round}
ROUND_LOCAL=output/finetune/rounds/r$round
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "the worktree has uncommitted changes; commit them to train" >&2
  exit 1
fi
mkdir -p "$ROUND_LOCAL"
rsync -a "$REMOTE:$remote_round/seeds.txt" "$REMOTE:$remote_round/held-out.txt" \
  "$ROUND_LOCAL/" || { echo "round $round inputs not on $REMOTE" >&2; exit 1; }
# Only a round archie marked READY (simulated and reviewed) is trained.
ssh "$REMOTE" test -f "$remote_round/READY" \
  || { echo "round $round is not ready on $REMOTE" >&2; exit 1; }
rsync -a "$REMOTE:$remote_round/sims/" "$ROUND_LOCAL/sims/" || exit 1
rsync -a "$REMOTE:$remote_round/qc/" "$ROUND_LOCAL/qc/" || exit 1
if [ "${USE_REAL_TRAINING:-0}" = 1 ]; then
  rsync -a "$REMOTE:$remote_round/real-training.pkl" "$ROUND_LOCAL/" || exit 1
fi
trainer_args "$round" "$init"
PYTORCH_MPS_HIGH_WATERMARK_RATIO=${MPS_HIGH_WATERMARK:-0.85} \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 \
HF_HOME=output/finetune/hf HF_HUB_OFFLINE=1 MPLCONFIGDIR="${TMPDIR:-/tmp}" \
  .venv/bin/python scripts/finetune_stream.py "${args[@]}" \
  >> "$ROUND_LOCAL/train.log" 2>&1 || {
    echo "round $round training failed; see $ROUND_LOCAL/train.log" >&2
    exit 1
  }
ckpt=$ROUND_LOCAL/train/n$OUTBREAKS
.venv/bin/python scripts/finetune_verify.py "$ckpt" || {
  echo "round $round checkpoint does not match its code_commit" >&2
  exit 1
}
key=$(python3 -c "import json;print(json.load(open('$ckpt/settings.json'))['hash'])")
rsync -a "$ckpt/" "$REMOTE:$remote_round/checkpoint/" || exit 1
ssh "$REMOTE" "echo \"$(date '+%Y-%m-%d %H:%M') fm-finetune3: round $round trained, chronos-ft-$key at $remote_round/checkpoint\" >> $REMOTE_MAILBOX"
echo "round $round: chronos-ft-$key"
