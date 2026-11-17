#!/usr/bin/env bash
# Move the rounds to a new grid version between rounds.
#
#   scripts/sim_rounds_handover.sh DRIVER_PID WORKTREE COMMIT NEXT_ROUND LAUNCHER
#
# With $ROOT/STOP set, scripts/sim_rounds.sh ends after its current
# round. This waits for it to exit, checks out COMMIT (detached) in the
# rounds WORKTREE, removes STOP and runs LAUNCHER NEXT_ROUND, writing
# mailbox lines. Run it from anywhere, under nohup.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_config.sh"
pid=$1 worktree=$2 commit=$3 next=$4 launcher=$5
say() { echo "$(date '+%F %H:%M') $HOST_NAME sim rounds handover: $*" >> "$MAILBOX"; }
[ -f "$ROOT/STOP" ] || { say "not started: $ROOT/STOP is not set"; exit 1; }
say "waiting for the driver (pid $pid) to finish its round, then round $next on ${commit:0:9}"
while kill -0 "$pid" 2> /dev/null; do sleep 300; done
# The driver's watcher checks STOP up to 10 minutes after the driver ends.
sleep 660
if ! git -C "$worktree" checkout -q --detach "$commit"; then
  say "could not check out $commit in $worktree; rounds stopped"
  exit 1
fi
rm -f "$ROOT/STOP"
"$launcher" "$next" > /dev/null 2>&1
say "round $next launched on $(git -C "$worktree" log -n1 --format='%h %s' | cut -c1-120)"
