#!/usr/bin/env bash
# Run workflow targets as a detached job that reports to the archie
# mailbox: a start line, then done or FAILED with the log's tail.
# Single-threaded and niced, for shared hosts.
#
# Usage (from anywhere): paper/scripts/run_target.sh NAME TARGET... [-- SNAKEMAKE ARGS]
# Start it with nohup; the log is ~/paper-jobs/NAME.log.
set -uo pipefail
name=$1; shift
paper=$(cd "$(dirname "$0")/.." && pwd)
mb=${MAILBOX:-$HOME/bdbv-archie-mailbox.md}
logs=$HOME/paper-jobs; mkdir -p "$logs"; log=$logs/$name.log
say() { echo "$(date '+%F %H:%M') paper: $name: $*" >> "$mb"; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$paper"
say "started at $(git rev-parse --short=9 HEAD) (log $log)"
if nice -n 19 uv run --group paper snakemake -s workflow/Snakefile \
    --cores 1 --resources heavy=1 --rerun-triggers mtime "$@" > "$log" 2>&1; then
  say "done"
else
  rc=$?
  say "FAILED (exit $rc): $(grep -m1 -iE 'error|exception' "$log" | cut -c1-200) | $(tail -n 2 "$log" | tr '\n' ' ' | cut -c1-200)"
  exit "$rc"
fi
