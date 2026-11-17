#!/usr/bin/env bash
# Optional: run paper workflow targets as a detached job on a shared
# host, single-threaded and niced, with a start line and then done or
# FAILED with the log's tail on its output. With NOTIFY and MAILBOX set in
# config/local.env those lines are also appended to MAILBOX
# (workflow/scripts/notify.sh). Not needed to run the workflow: `task
# paper` does the same in the foreground.
#
# Usage (from anywhere): tools/run_target.sh NAME TARGET... [SNAKEMAKE ARGS]
# Start it with nohup; the log is NAME.log in PAPER_LOGS.
set -uo pipefail
name=$1; shift
paper=$(cd "$(dirname "$0")/../paper" && pwd)
setting() { python3 "$paper/../workflow/flowlib/localconfig.py" "$1"; }
logs=$(setting PAPER_LOGS); mkdir -p "$logs"; log=$logs/$name.log
SAY_PREFIX="paper: $name"
# shellcheck source=workflow/scripts/notify.sh
source "$paper/../workflow/scripts/notify.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$paper"
say "started at $(git rev-parse --short=9 HEAD) (log $log)"
if nice -n 19 "$paper/../.venv/bin/python" -m snakemake -s workflow/Snakefile \
    "$@" --cores 1 --resources heavy=1 --rerun-triggers mtime > "$log" 2>&1; then
  say "done"
else
  rc=$?
  say "FAILED (exit $rc): $(grep -m1 -iE 'error|exception' "$log" | cut -c1-200) | $(tail -n 2 "$log" | tr '\n' ' ' | cut -c1-200)"
  exit "$rc"
fi
