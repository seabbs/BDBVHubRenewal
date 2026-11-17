#!/usr/bin/env bash
# A short pilot of the simulation grid at a fixed length, to size the
# rounds' slots: each batch runs in its own systemd scope and TMPDIR, as
# scripts/sim_rounds.sh runs them, with its peak memory and wall time.
#
#   scripts/sim_pilot.sh OUT_DIR [BATCHES] [BATCH] [WEEKS] [MEM]
#
# Writes OUT_DIR/sims/ (counts, settings, failures, logs) and
# OUT_DIR/pilot.csv: seed, peak_rss_kb, wall_s, peak TMPDIR size in KB and
# the exit status. Draws are ENW_DRAWS (1 by default). Run from the repo
# root with ENW_CACHE and the rlib set as for the rounds.
set -uo pipefail
out=$1 batches=${2:-4} batch=${3:-5} weeks=${4:-52} mem=${5:-14G}
mkdir -p "$out/sims/logs"
echo "seed,peak_rss_kb,wall_s,tmp_kb,status" > "$out/pilot.csv"
run() {
  local seed=$1 tmp
  tmp=$(mktemp -d "$out/tmp-$seed-XXXX")
  ( while [ -d "$tmp" ]; do du -sk "$tmp" 2> /dev/null | cut -f1; sleep 5; done ) \
    > "$out/sims/logs/$seed.tmp" &
  TMPDIR=$tmp ENW_DRAWS=${ENW_DRAWS:-1} ENW_WEEKS=$weeks \
    systemd-run --user --scope -q -p MemoryMax="$mem" -p MemorySwapMax=0 \
    /usr/bin/time -f "peak_rss_kb %M wall_s %e" -o "$out/sims/logs/$seed.time" \
    nice -n 19 Rscript scripts/finetune_simulate.R "$batch" "$seed" "$out/sims" \
    > "$out/sims/logs/$seed.log" 2>&1
  local status=$?
  rm -rf "$tmp"
  local rss wall top
  rss=$(awk '/peak_rss_kb/ {print $2}' "$out/sims/logs/$seed.time")
  wall=$(awk '/peak_rss_kb/ {print $4}' "$out/sims/logs/$seed.time")
  top=$(sort -n "$out/sims/logs/$seed.tmp" | tail -n 1)
  echo "$seed,${rss:-},${wall:-},${top:-0},$status" >> "$out/pilot.csv"
}
for i in $(seq 1 "$batches"); do
  run $((90000 + i)) &
done
wait
