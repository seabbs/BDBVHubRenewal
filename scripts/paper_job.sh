#!/usr/bin/env bash
# Run a command as a named job and report it to the archie mailbox.
# Usage: paper_job.sh NAME LOG CMD...
# Writes a start line, then a done or failed line (with the log's tail).
name=$1 log=$2; shift 2
mb=${MAILBOX:-$HOME/bdbv-archie-mailbox.md}
say() { echo "$(date '+%F %H:%M') paper: $name: $*" >> "$mb"; }
say "started (log $log)"
if "$@" > "$log" 2>&1; then
  say "done"
else
  rc=$?
  say "FAILED (exit $rc): $(tail -n 3 "$log" | tr '\n' ' ' | cut -c1-300)"
  exit "$rc"
fi
