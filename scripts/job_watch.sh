#!/usr/bin/env bash
# Watch a background job and write its progress and end to the mailbox.
#
#   scripts/job_watch.sh LABEL PID LOG [PROGRESS_REGEX] [MINUTES] [RELAUNCH]
#
# Every MINUTES (default 60) while PID runs, the last line of LOG matching
# PROGRESS_REGEX (default "step [0-9]+") is written to the mailbox if it
# changed. When PID ends, a line says whether LOG reached "finished" or
# what its last lines were. With RELAUNCH (a command), a job that ended
# without "finished" is started once more under nohup and watched again,
# for jobs that resume from saved state. Lines go to $MAILBOX, by default
# ~/bdbv-archie-mailbox.md, as "YYYY-MM-DD HH:MM archie: LABEL: ...".
set -u
label=$1
pid=$2
log=$3
progress=${4:-"step [0-9]+"}
minutes=${5:-60}
relaunch=${6:-}
mailbox=${MAILBOX:-$HOME/bdbv-archie-mailbox.md}

say() {
  echo "$(date '+%Y-%m-%d %H:%M') archie: $label: $*" >> "$mailbox"
}

alive() {
  # An ended job its parent has not reaped yet is a zombie: not running.
  kill -0 "$1" 2> /dev/null && [[ "$(ps -o stat= -p "$1")" != Z* ]]
}

watch_pid() {
  local last=""
  say "watching pid $1, log $log"
  while alive "$1"; do
    sleep "${WATCH_SECONDS:-$((minutes * 60))}"
    local now
    now=$(grep -E "$progress" "$log" 2> /dev/null | tail -n 1)
    if [ -n "$now" ] && [ "$now" != "$last" ]; then
      say "progress: $now"
      last=$now
    fi
  done
}

watch_pid "$pid"
if grep -q "finished" "$log" 2> /dev/null; then
  say "finished: $(grep -v -i warn "$log" | tail -n 3 | tr '\n' ' ')"
  exit 0
fi
say "FAILED (pid $pid ended without finishing): $(tail -n 3 "$log" | tr '\n' ' ')"
if [ -n "$relaunch" ]; then
  nohup bash -c "$relaunch" >> "$log" 2>&1 < /dev/null &
  new=$!
  say "relaunched once as pid $new: $relaunch"
  watch_pid "$new"
  if grep -q "finished" "$log" 2> /dev/null; then
    say "finished after relaunch: $(tail -n 3 "$log" | tr '\n' ' ')"
    exit 0
  fi
  say "FAILED again after relaunch: $(tail -n 3 "$log" | tr '\n' ' ')"
fi
exit 1
