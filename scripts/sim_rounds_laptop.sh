#!/usr/bin/env bash
# Simulate claimed batches of the rounds host's round on this machine.
#
#   SIM_TREE=<worktree at the grid commit> nohup \
#     scripts/sim_rounds_laptop.sh ROUND FIRST_SEED LAST_SEED \
#     > output/finetune/laptop-rounds.out 2>&1 &
#
# Claims seeds FIRST_SEED to LAST_SEED of round ROUND with a "claim" line
# in the shared mailbox, which the rounds host's scripts/sim_rounds.sh then
# skips. Each batch runs scripts/finetune_simulate.R from SIM_TREE, a clean
# worktree at the grid commit of the rounds host's latest round, SLOTS at a
# time at the lowest priority. Each finished batch is copied to
# $REMOTE_ROOT/rounds/r<ROUND>/sims/ and only then given its row in that
# round's manifest.csv, in scripts/sim_rounds.sh's format, so a reader of
# the manifest sees complete batches only. macOS has no memory cgroups:
# a watchdog kills a batch whose processes pass SLOT_MEM_GB, and a failed
# batch is retried once at RETRY_MEM_GB. Run from the repo root.
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/local_config.sh"
LOCAL=${LOCAL:-$PWD/output/finetune/laptop-rounds}
ENW_LIB=${ENW_LIB:-$PWD/output/finetune/rlib}
ENW_CACHE=${ENW_CACHE:-$PWD/output/finetune/enwcache}
BATCH=${BATCH:-21}
DRAWS=1
SLOTS=${SLOTS:-2}
SLOT_MEM_GB=${SLOT_MEM_GB:-16}
RETRY_MEM_GB=${RETRY_MEM_GB:-20}
HEADER="round,seed,kind,grid,grid_commit,draws,batch,counts_sha256,settings_sha256,failures_sha256,peak_rss_kb,wall_s,host,finished"

# Total resident memory in KB of process $1 and its descendants.
tree_rss() {
  ps -A -o pid=,ppid=,rss= | awk -v root="$1" '
    { parent[$1] = $2; rss[$1] = $3 }
    END {
      s = 0
      for (p in parent) {
        q = p
        while (q > 1) { if (q == root) { s += rss[p]; break }; q = parent[q] }
      }
      print s
    }'
}

# Process $1 and its descendants.
tree_pids() {
  ps -A -o pid=,ppid= | awk -v root="$1" '
    { parent[$1] = $2 }
    END {
      for (p in parent) {
        q = p
        while (q > 1) { if (q == root) { print p; break }; q = parent[q] }
      }
    }'
}

sha() { if [ -f "$1" ]; then sha256 "$1"; fi; }

# The peak memory (KB) and wall time (s) of a `/usr/bin/time -l` log $1,
# as scripts/sim_rounds.sh's GNU time line.
time_line() {
  awk '
    / real / && wall == "" { wall = $1 }
    /maximum resident set size/ { rss = int($1 / 1024) }
    END { if (rss != "") print "peak_rss_kb " rss " wall_s " wall }' "$1"
}

# Manifest row of batch seed $1 (kind $2) from its files under $3.
manifest_row() {
  local seed=$1 kind=$2 sims=$3 log rss wall
  log=$sims/logs/$seed.log
  rss=$(grep -o 'peak_rss_kb [0-9]*' "$log" | tail -n 1 | cut -d' ' -f2)
  wall=$(grep -o 'wall_s [0-9.]*' "$log" | tail -n 1 | cut -d' ' -f2)
  echo "$ROUND,$seed,$kind,$GRID,$COMMIT,$DRAWS,$BATCH,$(sha "$sims/counts-$seed.csv.gz"),$(sha "$sims/settings-$seed.csv"),$(sha "$sims/failures-$seed.csv"),${rss:-},${wall:-},$HOST_NAME,$(date -u +%FT%TZ)"
}

[[ "${BASH_SOURCE[0]}" != "$0" ]] && return 0

ROUND=$1
FIRST=$2
LAST=$3
SIM_TREE=${SIM_TREE:?set SIM_TREE to a worktree at the grid commit}
RDIR=$REMOTE_ROOT/rounds/r$ROUND
DIR=$LOCAL/r$ROUND
mkdir -p "$DIR/sims/logs"

say() {
  echo "$(date '+%F %H:%M') sim round $ROUND: $*"
  post "sim round $ROUND: $*"
}

# The grid and commit of the rounds host's latest recorded batch: this machine
# must simulate the same grid version.
latest=$(on_rounds "tail -q -n 1 \$(ls -d $REMOTE_ROOT/rounds/r*/manifest.csv | sort -V | tail -n 1)")
GRID=$(echo "$latest" | cut -d, -f4)
COMMIT=$(git -C "$SIM_TREE" rev-parse HEAD)
if [ "$COMMIT" != "$(echo "$latest" | cut -d, -f5)" ]; then
  echo "$SIM_TREE is at $COMMIT, not the rounds host's grid commit ($latest)" >&2
  exit 1
fi
if [ -n "$(git -C "$SIM_TREE" status --porcelain --untracked-files=no)" ]; then
  echo "$SIM_TREE has uncommitted changes" >&2
  exit 1
fi
if [ "$GRID" != "$(grep -o '^grid_name <- "[^"]*"' "$SIM_TREE/scripts/finetune_simulate.R" | cut -d'"' -f2)" ]; then
  echo "$SIM_TREE's grid is not $GRID" >&2
  exit 1
fi

claim="claim $FIRST-$LAST for round $ROUND (grid $GRID)"
on_rounds "mkdir -p $RDIR/sims/logs && { [ -f $RDIR/manifest.csv ] || echo $HEADER > $RDIR/manifest.csv; }"
if ! on_rounds "grep -qF '$HOST_NAME $claim' $REMOTE_MAILBOX"; then
  post "$claim"
fi
say "started: seeds $FIRST-$LAST, $BATCH simulations each, $DRAWS draws, grid $GRID at ${COMMIT:0:9}, $SLOTS slots of $SLOT_MEM_GB GB, host $HOST_NAME"

# Run batch seed $1 capped at $2 GB; 0 if it exited cleanly and wrote
# readable counts and settings.
run_batch() {
  local seed=$1 cap=$(($2 * 1024 * 1024)) log pid rss
  log=$DIR/sims/logs/$seed.log
  (
    cd "$SIM_TREE" && ENW_DRAWS=$DRAWS ENW_SEED_BASE=${SEED_BASE:-30000} ENW_LIB=$ENW_LIB ENW_CACHE=$ENW_CACHE \
      exec /usr/bin/time -l nice -n 19 Rscript scripts/finetune_simulate.R \
      "$BATCH" "$seed" "$DIR/sims"
  ) > "$log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2> /dev/null; do
    rss=$(tree_rss "$pid")
    if [ "${rss:-0}" -gt "$cap" ]; then
      echo "killed: $rss KB over the $2 GB cap" >> "$log"
      kill $(tree_pids "$pid") 2> /dev/null
    fi
    sleep 10
  done
  wait "$pid" || return 1
  time_line "$log" >> "$log"
  [ -s "$DIR/sims/settings-$seed.csv" ] && \
    [ -s "$DIR/sims/counts-$seed.csv.gz" ] && \
    gzip -t "$DIR/sims/counts-$seed.csv.gz" 2> /dev/null
}

# Copy batch $1 (kind $2) to the rounds host's round, then record it there.
publish() {
  local seed=$1 kind=$2 f
  local files=("$DIR/sims/counts-$seed.csv.gz" "$DIR/sims/settings-$seed.csv")
  f=$DIR/sims/failures-$seed.csv
  [ -f "$f" ] && files+=("$f")
  rsync -a "${files[@]}" "$(at_rounds "$RDIR/sims/")" && \
    rsync -a "$DIR/sims/logs/$seed.log" "$(at_rounds "$RDIR/sims/logs/")" && \
    on_rounds "echo '$(manifest_row "$seed" "$kind" "$DIR/sims")' >> $RDIR/manifest.csv"
}

recorded() { on_rounds "grep -q '^$ROUND,$1,' $RDIR/manifest.csv"; }

slot() {
  local s
  for s in $(seq "$FIRST" "$LAST"); do
    mkdir "$DIR/sims/.lock-$s" 2> /dev/null || continue
    recorded "$s" && continue
    if ! run_batch "$s" "$SLOT_MEM_GB"; then
      say "batch $s failed at $SLOT_MEM_GB GB; retrying at $RETRY_MEM_GB GB ($(tail -n 2 "$DIR/sims/logs/$s.log" | tr '\n' ' ' | tr -d '"$`' | cut -c1-160))"
      mv "$DIR/sims/logs/$s.log" "$DIR/sims/logs/$s.first.log"
      if ! run_batch "$s" "$RETRY_MEM_GB"; then
        say "batch $s failed again; skipped (log $DIR/sims/logs/$s.log)"
        continue
      fi
    fi
    until publish "$s" block; do
      echo "$(date) copying batch $s failed; trying again"
      sleep 300
    done
    echo "$(date) batch $s recorded"
  done
}

# Locks of an earlier run are stale.
rm -rf "$DIR"/sims/.lock-*
t0=$(date +%s)
for _ in $(seq 1 "$SLOTS"); do
  slot &
  sleep 60
done
wait
n=$(on_rounds "grep -c ',$HOST_NAME,' $RDIR/manifest.csv")
say "finished: $n batches of seeds $FIRST-$LAST recorded in $RDIR/manifest.csv in $(( ($(date +%s) - t0) / 60 )) min"
