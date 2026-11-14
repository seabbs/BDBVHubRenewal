#!/usr/bin/env bash
# Simulate, fine-tune and publish a round of Chronos-2 fine-tuning,
# unattended. Run from the repo root, detached:
#
#   nohup scripts/finetune_driver.sh config/finetune/r2-cal5.env \
#     > /dev/null 2>&1 &
#
# The env file sets the round:
#
#   SIM_DIR       local directory of the simulations
#   TRAIN_FIRST, TRAIN_LAST       training seeds
#   HELD_FIRST, HELD_LAST         held-out seeds (simulated first)
#   LOCAL_FIRST, LOCAL_LAST       training seeds run on the laptop
#   LOCAL_SLOTS, LOCAL_SLOTS_TRAINING, LOCAL_MIN_FREE_MEM_GB   most
#                 local simulation processes before and during training,
#                 and the free memory a new local batch needs
#   REMOTE, REMOTE_DIR, SIM_REMOTE   optional ssh host (none: every
#                 seed runs locally), its repo worktree (with
#                 scripts/finetune_simulate.R of this commit and an
#                 epinowcast library in output/finetune/rlib) and its
#                 simulation directory; every seed not run locally runs
#                 there, held-out seeds first
#   REMOTE_SLOTS, REMOTE_MEM_KB, REMOTE_MIN_FREE_MEM_GB, REMOTE_MAX_LOAD
#                 its most simulation processes, the memory limit of each
#                 (ulimit -v), and the free memory and highest load at
#                 which a new batch starts
#   REMOTE_MIN_FREE_GB   free disk below which no new remote batch starts
#   REMOTE_RETRY, REMOTE_GRACE   seconds between tries of the remote,
#                 and without an answer before the laptop takes over the
#                 held-out seeds and FIRST_CHUNK_SEEDS training seeds
#   GRID          the simulator's grid name; remote batches of another
#                 grid are moved aside
#   OUT           checkpoint directory
#   OUTBREAKS     most training simulations
#   PREFIXES      simulations at each saved checkpoint (doublings)
#   STOP_FROM, MIN_GAIN   from this checkpoint on, training stops once a
#                 doubling cuts held-out loss by less than this share
#   BLOCKS_PER_SIMULATION   blocks sampled from each simulation's draws
#                 and cut-offs
#   PUBLISH       1 to publish each checkpoint release
#   NOTE          text recorded in each checkpoint's settings
#   MPS_HIGH_WATERMARK   the trainer's cap on MPS memory, as a share of
#                 the GPU's recommended working set
#   MEMORY_GB     the memory budget of a training step, from which
#                 checkpointing and the largest block follow
#
# If the training seeds' yield projects fewer than OUTBREAKS outbreaks,
# the driver adds seeds after TRAIN_LAST (`extend_seeds`).
#
# It keeps both hosts' simulation slots busy, copies the remote results
# back, and runs scripts/finetune_stream.py (logging to
# output/finetune/trainer.log), which trains on the
# outbreaks as they arrive and publishes each checkpoint. Every step is
# logged to output/finetune/driver.log, and output/finetune/STATUS holds
# one line of state. It stops with an error in both on a failed
# simulation batch or a bad batch. A failed trainer restarts from its
# last saved state after a backoff while simulation carries on. Pause by creating
# output/finetune/PAUSE (no new simulations, the trainer waits) and
# resume by removing it.
set -uo pipefail
source "$1"
LOG=output/finetune/driver.log
TRAINER_LOG=output/finetune/trainer.log
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
  # With no remote, every seed.
  if [ -z "${REMOTE:-}" ]; then
    seq "$HELD_FIRST" "$HELD_LAST"
    seq "$TRAIN_FIRST" "$TRAIN_LAST"
    return 0
  fi
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
  # The remote runs this commit's simulator.
  rsync -a -e "ssh -o ConnectTimeout=20 -o BatchMode=yes" \
    scripts/finetune_simulate.R "$REMOTE:$REMOTE_DIR/scripts/" || return 1
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
  # Start a batch only with a free slot, REMOTE_MIN_FREE_MEM_GB of memory
  # free (one outbreak can peak at 8 GB) and a load under REMOTE_MAX_LOAD,
  # as other jobs share the host.
  while [ "\$(pgrep -fc 'finetune_simulate.R --args')" -ge $REMOTE_SLOTS ] \\
    || [ "\$(free -g | awk '/^Mem/ {print \$7}')" -lt $REMOTE_MIN_FREE_MEM_GB ] \\
    || [ "\$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge $REMOTE_MAX_LOAD ]; do sleep 30; done
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
  [ -z "${REMOTE:-}" ] && return 0
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
    # Batches of another grid are moved aside, never copied back.
    remote "cd $REMOTE_DIR && if [ \"\$(cat $SIM_REMOTE/grid 2> /dev/null)\" != $GRID ]; then pkill -f '[f]inetune_remote.sh'; pkill -f '[f]inetune_simulate.R --args'; [ -d $SIM_REMOTE ] && mv $SIM_REMOTE $SIM_REMOTE-old-\$(date +%s); mkdir -p $SIM_REMOTE/logs; echo $GRID > $SIM_REMOTE/grid; echo moved; fi" 2> /dev/null \
      | grep -q moved && log "moved $REMOTE's batches of another grid aside"
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

free_mem_gb() {
  # Memory free on this machine in GB, on Linux or macOS.
  if command -v free > /dev/null; then
    free -g | awk '/^Mem/ {print $7}'
  else
    local pct total
    pct=$(memory_pressure | awk '/free percentage/ {print $5}' | tr -d %)
    total=$(sysctl -n hw.memsize)
    echo $(( total / 1000000000 * pct / 100 ))
  fi
}

unstarted_seeds() {
  # Training seeds no machine is running, lowest first: never started, or
  # with no result and a log untouched for STALE_MIN minutes (a batch the
  # remote lost). The laptop takes these, as the trainer needs seeds in
  # order.
  local s f
  for s in $(seq "$TRAIN_FIRST" "$TRAIN_LAST"); do
    f="$SIM_DIR/logs/$s.log"
    done_seed "$s" && continue
    if [ ! -e "$f" ] || [ -n "$(find "$f" -mmin +"${STALE_MIN:-90}")" ]; then
      echo "$s"
    fi
  done
}

local_slots() {
  # Fill the laptop's slots: LOCAL_SLOTS while the trainer waits, and
  # LOCAL_SLOTS_TRAINING while it trains, to leave memory for MPS.
  [ -e "$PAUSE" ] && return 0
  local running s slots stage
  slots=$LOCAL_SLOTS
  stage=$(python3 -c "import json;print(json.load(open('$OUT/stream-status.json')).get('stage',''))" 2>/dev/null)
  [ "$stage" = training ] && slots=$LOCAL_SLOTS_TRAINING
  local claimed=0
  for s in $(local_seeds) $(unstarted_seeds); do
    done_seed "$s" && continue
    pgrep -f "(finetune_simulate|sim-real).R --args 25 $s " > /dev/null && continue
    running=$(pgrep -f 'finetune_simulate.R --args' | wc -l)
    (( running >= slots )) && break
    (( $(free_mem_gb) < LOCAL_MIN_FREE_MEM_GB )) && break
    (
      export TMPDIR="$PWD/output/finetune/tmp/$s"
      mkdir -p "$TMPDIR"
      Rscript scripts/finetune_simulate.R 25 "$s" "$SIM_DIR" \
        > "$SIM_DIR/logs/$s.log" 2>&1 < /dev/null
      rm -rf "$TMPDIR"
    ) &
    echo "$s" >> "$STARTED"
    if ! local_seeds | grep -qx "$s"; then
      echo "$s" >> "$CLAIMED"
      claimed=1
    fi
    log "started local batch $s"
    sleep 5
  done
  if (( claimed )) && [ -n "${REMOTE:-}" ]; then
    # The remote's scheduler restarts without the seeds the laptop took.
    remote "pkill -f '[f]inetune_remote.sh'" > /dev/null 2>&1
    remote_tried=0
  fi
}

sync_remote() {
  [ -z "${REMOTE:-}" ] && return 0
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
[ -n "${REMOTE:-}" ] && remote "pkill -f '[f]inetune_remote.sh'" \
  > /dev/null 2>&1 && log "stopped the previous remote scheduler"
log "starting round from $1: training seeds $TRAIN_FIRST-$TRAIN_LAST, held-out $HELD_FIRST-$HELD_LAST, laptop $LOCAL_FIRST-$LOCAL_LAST"
trainer_args=("$SIM_DIR" "$OUT" --train-seeds "$TRAIN_FIRST" "$TRAIN_LAST"
  --held-out-seeds "$HELD_FIRST" "$HELD_LAST" --outbreaks "$OUTBREAKS"
  --note "${NOTE:-}" --blocks-per-simulation "${BLOCKS_PER_SIMULATION:-12}"
  --prefixes ${PREFIXES:-1000 2000 4000 8000} --min-gain "${MIN_GAIN:-0.01}"
  --stop-from "${STOP_FROM:-8000}" --memory-gb "${MEMORY_GB:-36}")
[ "${PUBLISH:-0}" = 1 ] && trainer_args+=(--publish)
start_trainer() {
  # MPS memory is capped so the caching allocator cannot fill the laptop.
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=$MPS_HIGH_WATERMARK \
  PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 \
  HF_HOME=output/finetune/hf HF_HUB_OFFLINE=1 MPLCONFIGDIR="${TMPDIR:-/tmp}" \
    nohup .venv/bin/python scripts/finetune_stream.py "${trainer_args[@]}" \
    >> "$TRAINER_LOG" 2>&1 < /dev/null &
  trainer=$!
  echo "$(date '+%Y-%m-%d %H:%M:%S') [driver] trainer started, pid $trainer" \
    >> "$TRAINER_LOG"
  adopted=0
  echo "$trainer" > output/finetune/trainer.pid
  log "started the trainer, pid $trainer"
}

trainer_exit() {
  # The exit code of a stopped trainer; a kept one is not this shell's
  # child, so its state tells.
  if (( adopted )); then
    local stage
    stage=$(python3 -c "import json;print(json.load(open('$OUT/stream-status.json'))['stage'])" 2> /dev/null)
    [ "$stage" = finished ] && echo 0 || echo 1
  else
    wait "$trainer"
    echo $?
  fi
}

trainer=$(cat output/finetune/trainer.pid 2> /dev/null)
if [ -n "$trainer" ] && ps -p "$trainer" -o args= | grep -q finetune_stream; then
  # A restarted driver keeps a running trainer and its progress.
  adopted=1
  log "kept the running trainer, pid $trainer"
else
  start_trainer
fi
[ -e "$OUT/train-last.txt" ] && TRAIN_LAST=$(cat "$OUT/train-last.txt")
failures=0
retry_at=0
while true; do
  extend_seeds
  check_remote
  local_slots
  sync_remote
  check_batches
  if (( retry_at )) && (( $(date +%s) >= retry_at )); then
    retry_at=0
    start_trainer
  elif (( ! retry_at )) && ! kill -0 "$trainer" 2> /dev/null; then
    code=$(trainer_exit)
    if [ "$code" -eq 0 ]; then
      log "the trainer finished"
      status "finished"
      exit 0
    fi
    # A failed trainer restarts from its last saved state after a backoff
    # of 5 minutes, doubling to an hour; simulation carries on meanwhile.
    failures=$(( failures + 1 ))
    wait_s=$(( 300 * 2 ** (failures - 1) ))
    (( wait_s > 3600 )) && wait_s=3600
    retry_at=$(( $(date +%s) + wait_s ))
    last_error="trainer failed (exit $code, failure $failures), restart in $(( wait_s / 60 )) min"
    log "ERROR: $last_error; the end of $TRAINER_LOG:"
    tail -n 20 "$TRAINER_LOG" | sed 's/^/    /' >> "$LOG"
  fi
  status
  sleep 180
done
