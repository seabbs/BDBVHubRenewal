#!/usr/bin/env bash
# Held-out quantiles for the width/conditioning check, detached, with
# mailbox lines on finish or failure. Machine paths come from the env:
#   CUDALIB (libcuda symlink dir), OUT, SIMS, MAILBOX.
set -uo pipefail
cd "$(dirname "$0")/../.."
cache=$HOME/.cache/bdbvhubrenewal
ckpts=(--checkpoint zero-shot
  --checkpoint r6-n250=$cache/chronos-slots-6d3c64da9bc2
  --checkpoint r6-n500=$cache/chronos-slots-6bb954c7ffe7
  --checkpoint r5-n4000=$cache/chronos-slots-56221d82b0ed)
for extra in "$@"; do ckpts+=(--checkpoint "$extra"); done
note() { echo "$(date '+%Y-%m-%d %H:%M') [archie bdbv-ftpractice] $*" >> "$MAILBOX"; }
if LD_LIBRARY_PATH=$CUDALIB DEVICE=cuda HF_HUB_OFFLINE=1 \
    .venv/bin/python scripts/ftpractice/heldout_quantiles.py "$SIMS" "$OUT" \
    --seeds 32000 32039 --per-simulation 8 "${ckpts[@]}" >> "$OUT/run.log" 2>&1; then
  note "held-out quantiles done ($OUT)"
else
  note "held-out quantiles FAILED, see $OUT/run.log"
fi
