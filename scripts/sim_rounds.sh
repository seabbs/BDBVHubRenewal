#!/usr/bin/env bash
# Simulate fine-tuning rounds continuously on this host.
#
#   scripts/sim_rounds.sh FIRST_ROUND [LAST_ROUND]
#
# Each round r is ROUND_BATCHES training batches and HELD_BATCHES held-out
# batches of BATCH simulations (scripts/finetune_simulate.R, exactly
# DRAWS observation draws each), run SLOTS at a time, each in a systemd
# scope of SLOT_MEM. A batch that fails is retried once at RETRY_MEM.
# Outputs go to $ROOT/rounds/r<r>/:
#
#   sims/          counts-<seed>.csv.gz, settings-<seed>.csv,
#                  failures-<seed>.csv and logs/<seed>.log
#   seeds.txt      training seeds, one a line (scripts/finetune_round.sh)
#   held-out.txt   first and last held-out seed
#   manifest.csv   one row per batch: seed, kind, grid and its commit,
#                  draws, sha256 of each file, memory and time
#   qc/            the sim-QC review (scripts/sim_qc.py)
#   READY          written when the round is simulated, recorded and
#                  reviewed, so the trainer can pull it
#
# Every simulation is kept: QC exclusions are a list, never deletions.
# Each round is also archived as a compressed release asset (sims-r<r>)
# when ARCHIVE=1. Progress and failures are written to $MAILBOX. A file
# $ROOT/STOP ends the run after the current round. Run from the repo
# root of a clean worktree pinned to the grid's commit.
set -uo pipefail

ROOT=${ROOT:-$HOME/bdbv-rounds}
MAILBOX=${MAILBOX:-$HOME/bdbv-archie-mailbox.md}
BATCH=${BATCH:-25}
ROUND_BATCHES=${ROUND_BATCHES:-40}
HELD_BATCHES=${HELD_BATCHES:-4}
DRAWS=10
SLOTS=${SLOTS:-2}
SLOT_MEM=${SLOT_MEM:-16G}
RETRY_MEM=${RETRY_MEM:-20G}
SEED_BASE=${SEED_BASE:-30000}
ARCHIVE=${ARCHIVE:-1}
REPO=${REPO:-seabbs/BDBVHubRenewal}

say() { echo "$(date '+%F %H:%M') archie: sim rounds: $*" >> "$MAILBOX"; }

# Seeds of round $1: training then held-out, unique across rounds.
round_seeds() {
  local first=$((SEED_BASE + $1 * 100))
  seq "$first" $((first + ROUND_BATCHES + HELD_BATCHES - 1))
}

sha() { if [ -f "$1" ]; then sha256sum "$1" | cut -d' ' -f1; fi; }

# Run batch seed $1 of round dir $2 at memory $3; 0 if it wrote counts.
run_batch() {
  local seed=$1 dir=$2 mem=$3 log
  log=$dir/sims/logs/$seed.log
  ENW_DRAWS=$DRAWS systemd-run --user --scope -q -p MemoryMax="$mem" \
    -p MemorySwapMax=0 /usr/bin/time -f "peak_rss_kb %M wall_s %e" \
    nice -n 19 Rscript scripts/finetune_simulate.R "$BATCH" "$seed" \
    "$dir/sims" > "$log" 2>&1
  [ -s "$dir/sims/counts-$seed.csv.gz" ] && \
    gzip -t "$dir/sims/counts-$seed.csv.gz" 2> /dev/null
}

# Append batch $1 (kind $3) of round dir $2 to the manifest.
manifest_row() {
  local seed=$1 dir=$2 kind=$3 log rss wall
  log=$dir/sims/logs/$seed.log
  rss=$(grep -o 'peak_rss_kb [0-9]*' "$log" | tail -n 1 | cut -d' ' -f2)
  wall=$(grep -o 'wall_s [0-9.]*' "$log" | tail -n 1 | cut -d' ' -f2)
  echo "$ROUND,$seed,$kind,$GRID,$COMMIT,$DRAWS,$BATCH,$(sha "$dir/sims/counts-$seed.csv.gz"),$(sha "$dir/sims/settings-$seed.csv"),$(sha "$dir/sims/failures-$seed.csv"),${rss:-},${wall:-},$(hostname),$(date -u +%FT%TZ)" >> "$dir/manifest.csv"
}

# Run the seeds of round dir $1 in $SLOTS parallel queues.
simulate_round() {
  local dir=$1 seeds kind total
  mapfile -t seeds < <(round_seeds "$ROUND")
  total=${#seeds[@]}
  slot() {
    local q=$1 s k
    for k in "${!seeds[@]}"; do
      [ $((k % SLOTS)) -eq "$q" ] || continue
      s=${seeds[$k]}
      kind=train
      [ "$k" -ge "$ROUND_BATCHES" ] && kind=held-out
      if [ -f "$dir/sims/counts-$s.csv.gz" ] && \
          grep -q ",$s,$kind," "$dir/manifest.csv"; then
        continue
      fi
      if ! run_batch "$s" "$dir" "$SLOT_MEM"; then
        say "round $ROUND batch $s failed at $SLOT_MEM; retrying at $RETRY_MEM ($(tail -n 2 "$dir/sims/logs/$s.log" | tr '\n' ' ' | cut -c1-200))"
        mv "$dir/sims/logs/$s.log" "$dir/sims/logs/$s.first.log"
        if ! run_batch "$s" "$dir" "$RETRY_MEM"; then
          say "round $ROUND batch $s failed again; skipped (log $dir/sims/logs/$s.log)"
          continue
        fi
      fi
      manifest_row "$s" "$dir" "$kind"
      n=$(grep -c "^$ROUND," "$dir/manifest.csv")
      if [ $((n % 10)) -eq 0 ]; then
        say "round $ROUND: $n of $total batches simulated"
      fi
    done
  }
  for q in $(seq 0 $((SLOTS - 1))); do slot "$q" & done
  wait
}

first=$1
last=${2:-1000000}
COMMIT=$(git rev-parse HEAD)
GRID=$(grep -o '^grid_name <- "[^"]*"' scripts/finetune_simulate.R | cut -d'"' -f2)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  say "not started: the worktree has uncommitted changes"
  exit 1
fi
for ((ROUND = first; ROUND <= last; ROUND++)); do
  [ -f "$ROOT/STOP" ] && { say "stopped before round $ROUND ($ROOT/STOP)"; break; }
  dir=$ROOT/rounds/r$ROUND
  mkdir -p "$dir/sims/logs" "$dir/qc"
  [ -f "$dir/manifest.csv" ] || echo "round,seed,kind,grid,grid_commit,draws,batch,counts_sha256,settings_sha256,failures_sha256,peak_rss_kb,wall_s,host,finished" > "$dir/manifest.csv"
  say "round $ROUND started: $((ROUND_BATCHES * BATCH)) training and $((HELD_BATCHES * BATCH)) held-out simulations, $DRAWS draws each, grid $GRID at ${COMMIT:0:9}, $SLOTS slots of $SLOT_MEM, into $dir"
  t0=$(date +%s)
  simulate_round "$dir"
  mapfile -t seeds < <(round_seeds "$ROUND")
  printf '%s\n' "${seeds[@]:0:ROUND_BATCHES}" > "$dir/seeds.txt"
  echo "${seeds[ROUND_BATCHES]} ${seeds[-1]}" > "$dir/held-out.txt"
  n_sims=$(cat "$dir"/sims/settings-*.csv 2> /dev/null | grep -vc '^outbreak')
  if ! uv run python scripts/sim_qc.py "$dir" > "$dir/qc/run.log" 2>&1; then
    say "round $ROUND: sim QC failed ($(tail -n 1 "$dir/qc/run.log" | cut -c1-200)); round not marked ready"
    continue
  fi
  if [ "$ARCHIVE" = 1 ]; then
    tar -C "$dir" -czf "$dir.tar.gz" sims manifest.csv seeds.txt held-out.txt qc
    if gh release view "sims-r$ROUND" -R "$REPO" > /dev/null 2>&1 || \
        gh release create "sims-r$ROUND" -R "$REPO" --prerelease \
          --title "Simulations, round $ROUND" \
          --notes "Fine-tuning simulations of round $ROUND (grid $GRID at $COMMIT), with their manifest and QC record." > /dev/null 2>&1; then
      gh release upload "sims-r$ROUND" "$dir.tar.gz" "$dir/manifest.csv" -R "$REPO" --clobber > /dev/null 2>&1 \
        || say "round $ROUND: archive upload failed; the files stay in $dir"
    else
      say "round $ROUND: could not create release sims-r$ROUND; the files stay in $dir"
    fi
  fi
  date -u +%FT%TZ > "$dir/READY"
  say "round $ROUND ready for training: $n_sims simulations in $(( ($(date +%s) - t0) / 60 )) min, $(cat "$dir/qc/summary.txt" 2> /dev/null); seeds $dir/seeds.txt, held-out $(cat "$dir/held-out.txt")"
done
