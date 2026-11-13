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
#   LOCAL_SLOTS, LOCAL_SLOTS_TRAINING   laptop simulation processes
#                 before and during training (2, 1)
#   REMOTE, REMOTE_DIR, SIM_REMOTE   ssh host, its repo worktree (with
#                 scripts/finetune_simulate.R of this commit and an
#                 epinowcast library in output/finetune/rlib) and its
#                 simulation directory; every seed not run locally runs
#                 there, held-out seeds first
#   REMOTE_SLOTS, REMOTE_MEM_KB   its simulation processes and the
#                 memory limit of each (ulimit -v), sized by memory
#   REMOTE_MIN_FREE_GB   free disk below which no new remote batch starts
#   REMOTE_RETRY, REMOTE_GRACE   seconds between tries of the remote,
#                 and without an answer before the laptop takes over the
#                 held-out seeds and FIRST_CHUNK_SEEDS training seeds
#   OUT           checkpoint directory
#   OUTBREAKS     training outbreaks (8000)
#   PUBLISH       1 to publish each checkpoint release
#   NOTE          text recorded in each checkpoint's settings
#
# If the training seeds' yield projects fewer than OUTBREAKS outbreaks,
# the driver adds seeds after TRAIN_LAST (`extend_seeds`).
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
  echo "$(date '+%Y-%m-%d %H:%M:%S') stage=${1:-${stage:-starting}} batches=$done/$total train_step=${step:-0} remote=$( (( $(date +%s) - remote_ok <= REMOTE_RETRY + 60 )) && echo up || echo down) last_checkpoint=${ckpt:-none} last_error=${last_error:-none}" > "$STATUS"
}

fail() {
  last_error="$*"
  log "ERROR: $*"
  status "stopped"
  exit 1
}

CLAIMED="$SIM_DIR/local-seeds.txt"
touch "$CLAIMED"
remote_ok=$(date +%s)
remote_tried=0

done_seed() { [ -e "$SIM_DIR/settings-$1.csv" ]; }

STARTED="$SIM_DIR/local-started.txt"
touch "$STARTED"

local_seeds() {
  # The laptop's seeds: seeds taken over from the remote first, as they
  # hold the held-out and first training outbreaks, then its own range.
  cat "$CLAIMED"
  seq "$LOCAL_FIRST" "$LOCAL_LAST"
}

seeds() {
  # Remote seeds: held-out first, then training seeds not run locally.
  local s
  for s in $(seq "$HELD_FIRST" "$HELD_LAST") $(seq "$TRAIN_FIRST" "$TRAIN_LAST"); do
    if (( s >= LOCAL_FIRST && s <= LOCAL_LAST )); then continue; fi
    grep -qx "$s" "$CLAIMED" && continue
    done_seed "$s" && continue
    echo "$s"
  done
}

remote() { ssh -o ConnectTimeout=20 -o BatchMode=yes "$REMOTE" "$@"; }

remote_scheduler() {
  # Start the remote scheduler unless it runs. It runs at most REMOTE_SLOTS simulations, each
  # limited to REMOTE_MEM_KB of memory, and skips seeds already done or
  # running there.
  if remote "pgrep -f '[f]inetune_remote.sh'" > /dev/null 2>&1; then
    return 0
  fi
  local list
  list=$(seeds | tr '\n' ' ')
  [ -z "$list" ] && return 0
  remote "cat > $REMOTE_DIR/output/finetune/finetune_remote.sh" <<EOF || return 1
#!/usr/bin/env bash
cd "$REMOTE_DIR"
export PATH=/usr/local/bin:/usr/bin:/bin:\$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $SIM_REMOTE/logs
free_gb() { df -BG --output=avail . | tail -1 | tr -dc 0-9; }
for s in $list; do
  [ -e $SIM_REMOTE/settings-\$s.csv ] && continue
  pgrep -f "finetune_simulate.R --args 25 \$s " > /dev/null && continue
  while [ "\$(pgrep -fc 'finetune_simulate.R --args')" -ge $REMOTE_SLOTS ]; do sleep 10; done
  # Wait while the disk is short; each batch's R temp files go in its own
  # directory, removed when the batch ends.
  while [ "\$(free_gb)" -lt $REMOTE_MIN_FREE_GB ]; do
    echo "\$(date) waiting: \$(free_gb) GB free"; sleep 60
  done
  (
    export TMPDIR="\$PWD/output/finetune/tmp/\$s"
    mkdir -p "\$TMPDIR"
    ulimit -v $REMOTE_MEM_KB
    Rscript scripts/finetune_simulate.R 25 \$s $SIM_REMOTE > $SIM_REMOTE/logs/\$s.log 2>&1
    rm -rf "\$TMPDIR"
  ) &
  sleep 3
done
wait
while pgrep -f 'finetune_simulate.R --args' > /dev/null; do sleep 30; done
echo all done > $SIM_REMOTE/remote-done
EOF
  remote "cd $REMOTE_DIR && chmod +x output/finetune/finetune_remote.sh && (nohup output/finetune/finetune_remote.sh > $SIM_REMOTE/remote.log 2>&1 < /dev/null &)" \
    && log "started the remote scheduler on $REMOTE ($REMOTE_SLOTS slots, $REMOTE_MEM_KB kB each)"
}

check_remote() {
  # Try the remote every REMOTE_RETRY seconds. If it has not answered, or
  # has had less than REMOTE_MIN_FREE_GB of disk free, for REMOTE_GRACE
  # seconds, the laptop takes over the held-out seeds and
  # enough training seeds for the first chunk.
  local now
  now=$(date +%s)
  (( now - remote_tried < REMOTE_RETRY )) && return 0
  remote_tried=$now
  local free
  free=$(remote "cd $REMOTE_DIR && df -BG --output=avail . | tail -1" 2> /dev/null | tr -dc 0-9)
  if [ -n "$free" ] && (( free < REMOTE_MIN_FREE_GB )); then
    log "$REMOTE has $free GB free, under $REMOTE_MIN_FREE_GB GB: counted as down"
  fi
  if [ -n "$free" ] && (( free >= REMOTE_MIN_FREE_GB )); then
    remote_ok=$now
    # A scheduler started by hand runs more processes than memory allows.
    if remote "pkill -f '[r]un4.sh'" > /dev/null 2>&1; then
      log "stopped run4.sh on $REMOTE; its running batches finish"
    fi
    [ -e "$PAUSE" ] && return 0
    remote "test -e $REMOTE_DIR/$SIM_REMOTE/remote-done" 2> /dev/null \
      || remote_scheduler || log "could not start the remote scheduler"
    return 0
  fi
  [ -z "$free" ] && log "$REMOTE does not answer"
  if (( now - remote_ok >= REMOTE_GRACE )) && [ ! -s "$CLAIMED" ]; then
    { seq "$HELD_FIRST" "$HELD_LAST"; seq "$TRAIN_FIRST" $(( TRAIN_FIRST + FIRST_CHUNK_SEEDS - 1 )); } \
      | while read -r s; do done_seed "$s" || echo "$s"; done > "$CLAIMED"
    log "$REMOTE down for $(( (now - remote_ok) / 60 )) min: the laptop takes seeds $(tr '\n' ' ' < "$CLAIMED")"
  fi
}

local_slots() {
  # Fill the laptop's slots: LOCAL_SLOTS while the trainer waits, and
  # LOCAL_SLOTS_TRAINING while it trains, to leave memory for MPS.
  [ -e "$PAUSE" ] && return 0
  local running s slots stage
  slots=$LOCAL_SLOTS
  stage=$(python3 -c "import json;print(json.load(open('$OUT/stream-status.json')).get('stage',''))" 2>/dev/null)
  [ "$stage" = training ] && slots=$LOCAL_SLOTS_TRAINING
  for s in $(local_seeds); do
    done_seed "$s" && continue
    pgrep -f "(finetune_simulate|sim-real).R --args 25 $s " > /dev/null && continue
    running=$(pgrep -f 'finetune_simulate.R --args|sim-real.R --args' | wc -l)
    (( running >= slots )) && return 0
    (
      export TMPDIR="$PWD/output/finetune/tmp/$s"
      mkdir -p "$TMPDIR"
      Rscript scripts/finetune_simulate.R 25 "$s" "$SIM_DIR" \
        > "$SIM_DIR/logs/$s.log" 2>&1 < /dev/null
      rm -rf "$TMPDIR"
    ) &
    echo "$s" >> "$STARTED"
    log "started local batch $s"
    sleep 5
  done
}

sync_remote() {
  # Copy the remote's batches, except those the laptop has started.
  local excludes=() s
  for s in $(sort -u "$STARTED"); do
    excludes+=(--exclude="counts-$s.csv.gz" --exclude="settings-$s.csv"
      --exclude="failures-$s.csv" --exclude="logs/$s.log")
  done
  rsync -a -e "ssh -o ConnectTimeout=20 -o BatchMode=yes" "${excludes[@]}" \
    --include='counts-*' --include='settings-*' \
    --include='failures-*' --include='logs/***' --exclude='*' \
    "$REMOTE:$REMOTE_DIR/$SIM_REMOTE/" "$SIM_DIR/" 2>> "$LOG" \
    || log "rsync from $REMOTE failed; retrying later"
}

extend_seeds() {
  # If the training seeds' yield so far projects fewer than OUTBREAKS
  # outbreaks with 3% to spare, add seeds after TRAIN_LAST for 5% to
  # spare, tell the trainer, and restart the remote scheduler with them.
  local extra
  extra=$(python3 - "$SIM_DIR" "$TRAIN_FIRST" "$TRAIN_LAST" "$OUTBREAKS" <<'EOF'
import math, sys
from pathlib import Path
sim, first, last, target = Path(sys.argv[1]), *map(int, sys.argv[2:])
done = [s for s in range(first, last + 1)
        if (sim / f"settings-{s}.csv").exists()]
if len(done) < 10:
    print(0)
    sys.exit()
n = sum(len((sim / f"settings-{s}.csv").read_text().splitlines()) - 1
        for s in done)
per = n / len(done)
projected = n + per * (last - first + 1 - len(done))
need = 0 if projected >= 1.03 * target else 1.05 * target - projected
print(math.ceil(need / per))
EOF
)
  [ -z "$extra" ] && return 0
  (( extra > 0 )) || return 0
  TRAIN_LAST=$(( TRAIN_LAST + extra ))
  echo "$TRAIN_LAST" > "$OUT/train-last.txt"
  log "projected short of $OUTBREAKS outbreaks: added $extra training seeds, now to $TRAIN_LAST"
  remote "pkill -f '[f]inetune_remote.sh'" > /dev/null 2>&1
  remote_tried=0
}

check_batches() {
  # A batch whose R process stopped with an error fails the round.
  local f s
  for f in "$SIM_DIR"/logs/*.log; do
    [ -e "$f" ] || continue
    s=$(basename "$f" .log)
    if grep -q "Execution halted" "$f" && ! done_seed "$s"; then
      fail "simulation batch $s failed, see $f"
    fi
  done
}

# Sourced (by the tests), the script only defines its functions.
[[ "${BASH_SOURCE[0]}" != "$0" ]] && return 0

# A restarted driver replaces the remote scheduler, whose running batches
# finish.
remote "pkill -f '[f]inetune_remote.sh'" > /dev/null 2>&1 \
  && log "stopped the previous remote scheduler"
log "starting round from $1: training seeds $TRAIN_FIRST-$TRAIN_LAST, held-out $HELD_FIRST-$HELD_LAST, laptop $LOCAL_FIRST-$LOCAL_LAST"
trainer_args=("$SIM_DIR" "$OUT" --train-seeds "$TRAIN_FIRST" "$TRAIN_LAST"
  --held-out-seeds "$HELD_FIRST" "$HELD_LAST" --outbreaks "$OUTBREAKS"
  --note "${NOTE:-}")
[ "${PUBLISH:-0}" = 1 ] && trainer_args+=(--publish)
HF_HOME=output/finetune/hf HF_HUB_OFFLINE=1 MPLCONFIGDIR="${TMPDIR:-/tmp}" \
  nohup .venv/bin/python scripts/finetune_stream.py "${trainer_args[@]}" \
  >> "$LOG" 2>&1 < /dev/null &
trainer=$!
echo "$trainer" > output/finetune/trainer.pid
log "started the trainer, pid $trainer"
[ -e "$OUT/train-last.txt" ] && TRAIN_LAST=$(cat "$OUT/train-last.txt")
while true; do
  extend_seeds
  check_remote
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
  status
  sleep 180
done
