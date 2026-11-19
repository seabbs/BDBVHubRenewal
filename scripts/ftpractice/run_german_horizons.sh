#!/usr/bin/env bash
# German nowcast + forecast horizons for each label, one process each,
# after any checkpoint-scoring unit (scoring first). Env: CUDALIB, OUT,
# MAILBOX. Labels are the arguments.
set -uo pipefail
cd "$(dirname "$0")/../.."
note() { echo "$(date '+%Y-%m-%d %H:%M') [archie bdbv-ftpractice] $*" >> "$MAILBOX"; }
for label in "$@"; do
  while pgrep -f bdbvhubrenewal.evaluation.checkpoint > /dev/null; do sleep 60; done
  if ! LD_LIBRARY_PATH=$CUDALIB DEVICE=cuda .venv/bin/python \
      scripts/ftpractice/german_horizons.py "$label" "$OUT" >> "$OUT/run.log" 2>&1; then
    note "German horizons FAILED for $label, see $OUT/run.log"
    exit 1
  fi
done
note "German horizons done: $* ($OUT)"
