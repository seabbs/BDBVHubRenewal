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
#   seeds.txt      training seeds, one a line (scripts/finetune_continuous.sh)
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

source "$(dirname "${BASH_SOURCE[0]}")/local_config.sh"
BATCH=${BATCH:-25}
ROUND_BATCHES=${ROUND_BATCHES:-40}
HELD_BATCHES=${HELD_BATCHES:-4}
DRAWS=${DRAWS:-10}
SLOTS=${SLOTS:-3}
MIN_FREE_GB=${MIN_FREE_GB:-20}
MIN_DISK_GB=${MIN_DISK_GB:-16}
SLOT_STAGGER=${SLOT_STAGGER:-120}
SLOT_MEM=${SLOT_MEM:-16G}
RETRY_MEM=${RETRY_MEM:-20G}
SEED_BASE=${SEED_BASE:-30000}
ARCHIVE=${ARCHIVE:-1}
REPO=${REPO:-seabbs/BDBVHubRenewal}

say() { echo "$(date '+%F %H:%M') $HOST_NAME sim rounds: $*" >> "$MAILBOX"; }

# Seeds of round $1: training then held-out, unique across rounds.
round_seeds() {
  local first=$((SEED_BASE + $1 * 100))
  seq "$first" $((first + ROUND_BATCHES + HELD_BATCHES - 1))
}

sha() { if [ -f "$1" ]; then sha256 "$1"; fi; }

# Run batch seed $1 of round dir $2 at memory $3; 0 if the run exited
# cleanly and wrote readable counts and settings. R and Stan write their
# temporary files (standata up to 1 GB) to a TMPDIR of the batch's own,
# removed when the batch ends however it ends, so a batch killed at its
# memory cap leaves nothing behind.
run_batch() {
  local seed=$1 dir=$2 mem=$3 log tmp rc=0
  log=$dir/sims/logs/$seed.log
  tmp=$dir/sims/tmp/$seed
  rm -rf "$tmp"
  mkdir -p "$tmp"
  TMPDIR=$tmp ENW_DRAWS=$DRAWS systemd-run --user --scope -q \
    -p MemoryMax="$mem" -p MemorySwapMax=0 \
    /usr/bin/time -f "peak_rss_kb %M wall_s %e" \
    nice -n 19 Rscript scripts/finetune_simulate.R "$BATCH" "$seed" \
    "$dir/sims" > "$log" 2>&1 || rc=1
  rm -rf "$tmp"
  [ "$rc" = 0 ] && [ -s "$dir/sims/settings-$seed.csv" ] && \
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

# Seeds another host claims in the mailbox (lines with "claim" naming
# seeds, as 30123 or 30120-30125), which this host skips.
claimed_seeds() {
  grep -i 'claim' "$MAILBOX" 2> /dev/null \
    | grep -v -E "^[0-9-]+ [0-9:]+ $HOST_NAME[ :(]" \
    | grep -oE '\b[0-9]{5}(-[0-9]{5})?\b' | while read -r r; do
      if [[ $r == *-* ]]; then seq "${r%-*}" "${r#*-}"; else echo "$r"; fi
    done | sort -u
}

# Whether batch seed $1 of round dir $2 is recorded in the manifest.
recorded() { grep -q "^$ROUND,$1," "$2/manifest.csv"; }

# Available memory in GB.
free_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }

# Free disk in GB where the rounds are written.
disk_gb() { df -Pk "$ROOT" | awk 'NR == 2 {print int($4 / 1048576)}'; }

# Whether there is room for a batch: memory and disk.
room() {
  [ "$(free_gb)" -ge "$MIN_FREE_GB" ] && [ "$(disk_gb)" -ge "$MIN_DISK_GB" ]
}

# Wait for room, saying once when the disk is why.
wait_for_room() {
  local said=0
  until room; do
    if [ "$(disk_gb)" -lt "$MIN_DISK_GB" ] && [ "$said" = 0 ]; then
      say "round $ROUND: launching paused, $(disk_gb) GB disk free (below $MIN_DISK_GB GB)"
      said=1
    fi
    sleep 60
  done
  [ "$said" = 1 ] && say "round $ROUND: launching resumed, $(disk_gb) GB disk free"
  return 0
}

# Whether a simulation of seed $1 into round dir $2 is already running
# (a batch left by an earlier driver).
running() { pgrep -f "finetune_simulate.R $BATCH $1 $2/sims" > /dev/null; }

# Record a finished batch left by an earlier driver.
reconcile() {
  local dir=$1 k s kind
  for k in "${!seeds[@]}"; do
    s=${seeds[$k]}
    kind=train
    [ "$k" -ge "$ROUND_BATCHES" ] && kind=held-out
    recorded "$s" "$dir" && continue
    if [ -s "$dir/sims/settings-$s.csv" ] && \
        grep -q 'peak_rss_kb' "$dir/sims/logs/$s.log" 2> /dev/null && \
        gzip -t "$dir/sims/counts-$s.csv.gz" 2> /dev/null; then
      manifest_row "$s" "$dir" "$kind"
    fi
  done
}

# Run the seeds of round dir $1 with $SLOTS slots. Each slot takes the
# next seed no slot holds (an atomic lock directory), skips seeds the
# laptop claims, and starts a batch only with $MIN_FREE_GB GB of memory
# and $MIN_DISK_GB GB of disk available.
simulate_round() {
  local dir=$1 total
  mapfile -t seeds < <(round_seeds "$ROUND")
  total=${#seeds[@]}
  # Locks of an earlier driver are stale, except for batches still running.
  rm -rf "$dir"/sims/.lock-*
  # So are the temporary folders of batches no longer running.
  for tmp in "$dir"/sims/tmp/*; do
    [ -d "$tmp" ] || continue
    running "$(basename "$tmp")" "$dir" || rm -rf "$tmp"
  done
  slot() {
    local s k kind n
    for k in "${!seeds[@]}"; do
      s=${seeds[$k]}
      kind=train
      [ "$k" -ge "$ROUND_BATCHES" ] && kind=held-out
      recorded "$s" "$dir" && continue
      if claimed_seeds | grep -qx "$s"; then continue; fi
      mkdir "$dir/sims/.lock-$s" 2> /dev/null || continue
      if running "$s" "$dir"; then
        while running "$s" "$dir"; do sleep 60; done
        reconcile "$dir"
        # A batch an earlier driver left that then failed is run again.
        recorded "$s" "$dir" && continue
        say "round $ROUND batch $s, left running by an earlier driver, failed; running it again"
      fi
      wait_for_room
      if ! run_batch "$s" "$dir" "$SLOT_MEM"; then
        say "round $ROUND batch $s failed at $SLOT_MEM; retrying at $RETRY_MEM ($(tail -n 2 "$dir/sims/logs/$s.log" | tr '\n' ' ' | cut -c1-200))"
        mv "$dir/sims/logs/$s.log" "$dir/sims/logs/$s.first.log"
        wait_for_room
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
  for _ in $(seq 1 "$SLOTS"); do
    slot &
    # Stagger the slots so each sees the memory the last one took.
    sleep "$SLOT_STAGGER"
  done
  wait
  reconcile "$dir"
}

[[ "${BASH_SOURCE[0]}" != "$0" ]] && return 0

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
  n_sims=$(cat "$dir"/sims/settings-*.csv 2> /dev/null | grep -vc '^outbreak')
  if ! uv run python scripts/sim_qc.py "$dir" > "$dir/qc/run.log" 2>&1; then
    say "round $ROUND: sim QC failed ($(tail -n 1 "$dir/qc/run.log" | cut -c1-200)); round not marked ready"
    continue
  fi
  if [ "$ARCHIVE" = 1 ]; then
    tar -C "$dir" -czf "$dir.tar.gz" sims manifest.csv qc
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
  # The seed lists come last, with READY: the trainer pulls a round once
  # they exist, so it never trains on an unreviewed round.
  mapfile -t seeds < <(round_seeds "$ROUND")
  printf '%s\n' "${seeds[@]:0:ROUND_BATCHES}" > "$dir/seeds.txt"
  echo "${seeds[ROUND_BATCHES]} ${seeds[-1]}" > "$dir/held-out.txt"
  date -u +%FT%TZ > "$dir/READY"
  claimed=$(claimed_seeds | grep -xF -f <(round_seeds "$ROUND") | tr '\n' ' ')
  [ -n "$claimed" ] && say "round $ROUND: seeds claimed by another host, not simulated here: $claimed"
  say "round $ROUND ready for training: $n_sims simulations in $(( ($(date +%s) - t0) / 60 )) min, $(cat "$dir/qc/summary.txt" 2> /dev/null); seeds $dir/seeds.txt, held-out $(cat "$dir/held-out.txt")"
done
