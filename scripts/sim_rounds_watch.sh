#!/usr/bin/env bash
# Watch scripts/sim_rounds.sh: write a mailbox line if it dies before
# $ROOT/STOP, or if no batch finishes for STALL_MIN minutes.
#
#   scripts/sim_rounds_watch.sh DRIVER_PID
set -uo pipefail
ROOT=${ROOT:-$HOME/bdbv-rounds}
MAILBOX=${MAILBOX:-$HOME/bdbv-archie-mailbox.md}
STALL_MIN=${STALL_MIN:-120}
pid=$1
say() { echo "$(date '+%F %H:%M') archie: sim rounds watcher: $*" >> "$MAILBOX"; }
stalled=0
while kill -0 "$pid" 2> /dev/null; do
  latest=$(find "$ROOT/rounds" -name manifest.csv -mmin "-$STALL_MIN" 2> /dev/null | head -n 1)
  if [ -z "$latest" ] && [ "$stalled" = 0 ]; then
    say "no batch finished in $STALL_MIN min (driver pid $pid still running); check $ROOT/rounds/*/sims/logs"
    stalled=1
  elif [ -n "$latest" ]; then
    stalled=0
  fi
  sleep 600
done
[ -f "$ROOT/STOP" ] || say "driver pid $pid is gone without $ROOT/STOP; check the last round's logs and restart with scripts/sim_rounds.sh (finished batches are kept and skipped)"
