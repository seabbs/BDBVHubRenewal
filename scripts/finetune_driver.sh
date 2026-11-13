#!/usr/bin/env bash
# Simulate, fine-tune and publish a round of Chronos-2 fine-tuning,
# unattended. Run from the repo root, detached:
#
#   nohup scripts/finetune_driver.sh output/finetune/round.env \
#     > /dev/null 2>&1 &
#
# The env file sets the round:
#
#   SIM_DIR       local directory of the simulations
#   TRAIN_FIRST, TRAIN_LAST       training seeds
#   HELD_FIRST, HELD_LAST         held-out seeds (simulated first)
#   LOCAL_FIRST, LOCAL_LAST       training seeds run on the laptop
#   LOCAL_SLOTS   laptop simulation processes (2)
#   REMOTE, REMOTE_DIR, REMOTE_SLOTS   ssh host, its repo worktree (with
#                 scripts/finetune_simulate.R of this commit and an
#                 epinowcast library in output/finetune/rlib) and its
#                 simulation processes; every seed not run locally runs
#                 there, held-out seeds first
#   OUT           checkpoint directory
#   OUTBREAKS     training outbreaks (8000)
#   PUBLISH       1 to publish each checkpoint release
#
# It keeps both hosts' simulation slots busy, copies the remote results
# back, and runs scripts/finetune_stream.py, which trains on the
# outbreaks as they arrive and publishes each checkpoint. Every step is
# logged to output/finetune/driver.log, and output/finetune/STATUS holds
# one line of state. It stops with an error in both on a failed
# simulation batch, a bad batch or a failed trainer. Pause by creating
# output/finetune/PAUSE (no new simulations, the trainer waits) and
# resume by removing it.
set -uo pipefail
source "$1"
LOG=output/finetune/driver.log
STATUS=output/finetune/STATUS
PAUSE=output/finetune/PAUSE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$SIM_DIR/logs" "$OUT"
last_error=""

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [driver] $*" >> "$LOG"; }

status() {
  local done total step ckpt stage
  total=$(( TRAIN_LAST - TRAIN_FIRST + 1 + HELD_LAST - HELD_FIRST + 1 ))
  done=$(ls "$SIM_DIR" | grep -c '^settings-')
  stage=$(python3 -c "import json;print(json.load(open('$OUT/stream-status.json')).get('stage',''))" 2>/dev/null)
  step=$(python3 -c "import json;print(json.load(open('$OUT/stream-status.json')).get('step',0))" 2>/dev/null)
  ckpt=$(python3 -c "import json;print(json.load(open('$OUT/stream-status.json')).get('last_checkpoint','none'))" 2>/dev/null)
  echo "$(date '+%Y-%m-%d %H:%M:%S') stage=${1:-${stage:-starting}} batches=$done/$total train_step=${step:-0} last_checkpoint=${ckpt:-none} last_error=${last_error:-none}" > "$STATUS"
}

fail() {
  last_error="$*"
  log "ERROR: $*"
  status "stopped"
  exit 1
}

seeds() {
  # Remote seeds: held-out first, then training seeds not run locally.
  seq "$HELD_FIRST" "$HELD_LAST"
  for s in $(seq "$TRAIN_FIRST" "$TRAIN_LAST"); do
    if (( s < LOCAL_FIRST || s > LOCAL_LAST )); then echo "$s"; fi
  done
}

remote_scheduler() {
  # Start the remote scheduler unless it runs; it skips started seeds.
  if ssh -o ConnectTimeout=20 "$REMOTE" "pgrep -f '[f]inetune_remote.sh'" \
      > /dev/null 2>&1; then
    return 0
  fi
  local list
  list=$(seeds | tr '\n' ' ')
  ssh -o ConnectTimeout=20 "$REMOTE" "cat > $REMOTE_DIR/output/finetune/finetune_remote.sh" <<EOF || return 1
#!/usr/bin/env bash
cd "$REMOTE_DIR"
export PATH=/usr/local/bin:/usr/bin:/bin:\$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $SIM_REMOTE/logs
for s in $list; do
  [ -e $SIM_REMOTE/logs/\$s.log ] && continue
  while [ "\$(pgrep -fc 'finetune_simulate.R --args')" -ge $REMOTE_SLOTS ]; do sleep 10; done
  Rscript scripts/finetune_simulate.R 25 \$s $SIM_REMOTE > $SIM_REMOTE/logs/\$s.log 2>&1 &
  sleep 3
done
wait
while pgrep -f 'finetune_simulate.R --args' > /dev/null; do sleep 30; done
echo all done > $SIM_REMOTE/remote-done
EOF
  ssh -o ConnectTimeout=20 "$REMOTE" "cd $REMOTE_DIR && chmod +x output/finetune/finetune_remote.sh && (nohup output/finetune/finetune_remote.sh > $SIM_REMOTE/remote.log 2>&1 < /dev/null &)" \
    && log "started the remote scheduler on $REMOTE"
}

local_slots() {
  # Fill the laptop's slots with its seeds not yet started.
  [ -e "$PAUSE" ] && return 0
  local running s
  for s in $(seq "$LOCAL_FIRST" "$LOCAL_LAST"); do
    [ -e "$SIM_DIR/logs/$s.log" ] && continue
    running=$(pgrep -f 'finetune_simulate.R --args|sim-real.R --args' | wc -l)
    (( running >= LOCAL_SLOTS )) && return 0
    nohup Rscript scripts/finetune_simulate.R 25 "$s" "$SIM_DIR" \
      > "$SIM_DIR/logs/$s.log" 2>&1 < /dev/null &
    log "started local batch $s"
    sleep 5
  done
}

sync_remote() {
  rsync -a --include='counts-*' --include='settings-*' \
    --include='failures-*' --include='logs/***' --exclude='*' \
    "$REMOTE:$REMOTE_DIR/$SIM_REMOTE/" "$SIM_DIR/" 2>> "$LOG" \
    || log "rsync from $REMOTE failed; retrying later"
}

check_batches() {
  # A batch whose R process stopped with an error fails the round.
  local f s
  for f in "$SIM_DIR"/logs/*.log; do
    [ -e "$f" ] || continue
    s=$(basename "$f" .log)
    if grep -q "Execution halted" "$f" && [ ! -e "$SIM_DIR/settings-$s.csv" ]; then
      fail "simulation batch $s failed, see $f"
    fi
  done
}

log "starting round from $1: training seeds $TRAIN_FIRST-$TRAIN_LAST, held-out $HELD_FIRST-$HELD_LAST, laptop $LOCAL_FIRST-$LOCAL_LAST"
remote_scheduler || log "could not reach $REMOTE; retrying later"
trainer_args=("$SIM_DIR" "$OUT" --train-seeds "$TRAIN_FIRST" "$TRAIN_LAST"
  --held-out-seeds "$HELD_FIRST" "$HELD_LAST" --outbreaks "$OUTBREAKS")
[ "${PUBLISH:-0}" = 1 ] && trainer_args+=(--publish)
HF_HOME=output/finetune/hf HF_HUB_OFFLINE=1 MPLCONFIGDIR="${TMPDIR:-/tmp}" \
  nohup .venv/bin/python scripts/finetune_stream.py "${trainer_args[@]}" \
  >> "$LOG" 2>&1 < /dev/null &
trainer=$!
echo "$trainer" > output/finetune/trainer.pid
log "started the trainer, pid $trainer"
while true; do
  local_slots
  sync_remote
  check_batches
  if ! kill -0 "$trainer" 2> /dev/null; then
    wait "$trainer"
    code=$?
    if [ "$code" -ne 0 ]; then
      fail "the trainer stopped with exit code $code, see $LOG"
    fi
    log "the trainer finished"
    status "finished"
    exit 0
  fi
  if [ ! -e "$PAUSE" ] && ! ssh -o ConnectTimeout=20 "$REMOTE" \
      "test -e $REMOTE_DIR/$SIM_REMOTE/remote-done" 2> /dev/null; then
    remote_scheduler || log "could not reach $REMOTE; retrying later"
  fi
  status
  sleep 180
done
