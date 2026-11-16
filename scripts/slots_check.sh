#!/bin/bash
# Run the slots check (scripts/slots_check.py) on slots checkpoints as
# they land, then commit and push the evidence.
#
# Usage: scripts/slots_check.sh LABEL [--dry]
#
# LABEL is a slots checkpoint's settings hash, as `<hash>`, `slots-<hash>`
# or `chronos-slots-<hash>`: the runner waits for it, checks it and
# exits. Or `auto`: the runner checks every slots checkpoint once as it
# lands (n250, n500, n1000, ... n8000), oldest first, and keeps watching.
# A checkpoint is found, polling every $POLL seconds:
#   - locally, as an nK folder of $CKPTS whose files match its
#     sha256sums.txt, so a half-copied folder is never used; or
#   - as the GitHub release chronos-slots-<hash>, downloaded and checked
#     by foundation.finetuned_adapter.
# Whichever path finds it, the checkpoint is labelled slots-<hash>.
#
# One run per label: each label holds a lock ($RUN_ROOT/locks), and a
# label whose final summary exists is skipped, so a second caller (the
# auto watcher, or checkpoint-scoring's slots_followups.sh) exits 0.
# In auto mode a failed checkpoint is marked (`failed` in its run folder)
# and skipped, and the watcher goes on with the next.
#
# Each checkpoint is scored against zero-shot Chronos-2 and the no-slots
# checkpoint with the same n_outbreaks (n1000 de129103cacc, n2000
# 7962fd718a47 in ~/bdbv-rounds/checkpoints), or zero-shot alone where
# there is none (n250, n500); slots_check.py picks it from the settings.
# Two phases, each in $SHARDS shards, up to $MAX_PAR at once:
#   headline  given and swap_places for the slots checkpoint and its
#             comparator; then an interim summary and mailbox line;
#   rest      the other conditions and models; then the final summary,
#             commit, push and mailbox line.
# Each shard runs under
#   systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 nice -n 19
# with one thread, and starts only when compute-budget.sh is not red and
# at least $MIN_FREE_GB + $SHARD_GB GB is available.
# Outputs per checkpoint LABEL (slots-<hash>):
#   evidence/slots-check/LABEL.csv           summary rows (final)
#   evidence/slots-check/LABEL-blocks.csv.gz per-block rows (final)
#   ~/slots-check/LABEL/summary.csv          summary, interim then final
#                                            (column `stage`)
# The evidence is committed on branch slots-check as seabbs-bot and
# pushed; the push is checked against origin. Start, interim, failure and
# finish lines go to $MAILBOX.
#
# --dry scores two blocks of each set in one shard per phase, writes only
# under $RUN_ROOT, prints the mailbox lines and never commits or pushes.
#
# Environment: PY (python with chronos and peft), CKPTS, RUN_ROOT,
# PAGE_ROOT, SHARDS, MAX_PAR, DRY_SHARDS, POLL, MIN_FREE_GB, SHARD_GB, MAILBOX,
# SCORE_ARGS (passed on to slots_check.py score).
set -uo pipefail

label=${1:?usage: slots_check.sh LABEL|auto [--dry]}
dry=${2:-}
[ -z "$dry" ] || [ "$dry" = "--dry" ] || { echo "unknown flag $dry"; exit 2; }

here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here" || exit 1
PY=${PY:-/home/seabbs/code/seabbs/BDBVHubRenewal/.worktrees/checkpoint-scoring/.venv/bin/python}
CKPTS=${CKPTS:-$HOME/bdbv-rounds/checkpoints/slots}
RUN_ROOT=${RUN_ROOT:-$HOME/bdbv-slots-check}
PAGE_ROOT=${PAGE_ROOT:-$HOME/slots-check}
SHARDS=${SHARDS:-3}
MAX_PAR=${MAX_PAR:-3}
DRY_SHARDS=${DRY_SHARDS:-1}
POLL=${POLL:-300}
MIN_FREE_GB=${MIN_FREE_GB:-16}
SHARD_GB=${SHARD_GB:-6}
MAILBOX=${MAILBOX:-$HOME/bdbv-archie-mailbox.md}
SCORE_ARGS=${SCORE_ARGS:-}
REPO=seabbs/BDBVHubRenewal
BRANCH=slots-check
BUDGET=$HOME/.claude/hooks/compute-budget.sh
LOCKS=$RUN_ROOT/locks
mkdir -p "$LOCKS"

want=${label#chronos-}
want=${want#slots-}
[ "$want" = auto ] && want=""

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

mail() {
  local line
  line="$(date '+%Y-%m-%d %H:%M') archie: slots-check: $*"
  if [ -n "$dry" ]; then
    echo "mailbox (dry, not written): $line"
  else
    echo "$line" >> "$MAILBOX"
  fi
}

fail() {
  log "FAILED ($1): ${*:2}"
  mail "FAILED ($1): ${*:2}"
  return 1
}

# --- preflight -----------------------------------------------------------
preflight() {
  [ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] \
    || { fail "$label" "$here is not on branch $BRANCH"; return 1; }
  [ -z "$(git status --porcelain --untracked-files=no)" ] \
    || { fail "$label" "uncommitted changes in $here"; return 1; }
  "$PY" -c "import chronos, peft, torch, safetensors" 2> /dev/null \
    || { fail "$label" "$PY cannot import chronos, peft, torch, safetensors"; return 1; }
}

# --- which checkpoints are done or running -------------------------------
run_dir() { echo "$RUN_ROOT/slots-$1${dry:+-dry}"; }

finished() {
  # The label's final summary exists.
  [ -f "$(run_dir "$1")/finished" ] && return 0
  [ -z "$dry" ] && [ -f "$PAGE_ROOT/slots-$1/summary.csv" ] \
    && grep -q ',final$' "$PAGE_ROOT/slots-$1/summary.csv"
}

locked() { ! flock -n "$LOCKS/slots-$1${dry:+-dry}.lock" true; }

# --- find checkpoints ----------------------------------------------------
settings_hash() {
  "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["hash"])' \
    "$1" 2> /dev/null
}

verified() {
  # A checkpoint folder whose files all match its sha256sums.txt.
  local d=$1 sha name f
  [ -f "$d/sha256sums.txt" ] && [ -f "$d/settings.json" ] || return 1
  while read -r sha name; do
    [ -n "$sha" ] || continue
    name=${name#\*}
    f="$d/$name"
    [ -f "$f" ] || f="$d/finetuned-ckpt/$name"
    [ -f "$f" ] || return 1
    [ "$(sha256sum < "$f" | cut -d' ' -f1)" = "$sha" ] || return 1
  done < "$d/sha256sums.txt"
}

candidates() {
  # "<hash> <adapter>" for each landed slots checkpoint, oldest n first.
  local d h tag
  for d in $(find "$CKPTS" -maxdepth 1 -name "n*" -type d 2> /dev/null \
    | sort -V); do
    h=$(settings_hash "$d/settings.json") || continue
    verified "$d" && echo "$h $d/finetuned-ckpt"
  done
  for tag in $(gh release list -R "$REPO" --limit 200 \
    --json tagName,createdAt \
    -q '[.[] | select(.tagName | startswith("chronos-slots-"))]
        | sort_by(.createdAt) | .[].tagName' 2> /dev/null); do
    echo "${tag#chronos-slots-} release"
  done
}

download() {
  # The adapter folder of release chronos-slots-<hash>, checked.
  PYTHONPATH=src "$PY" -c '
import sys
from bdbvhubrenewal.forecasters import foundation as f
print(f.finetuned_adapter(f.resolve_checkpoint("slots-" + sys.argv[1])))
' "$1" 2> /dev/null | tail -1
}

next_checkpoint() {
  # The first landed checkpoint still to check (only LABEL's, if given),
  # leaving out those done or held by another caller.
  local h where seen=" "
  while read -r h where; do
    [ -n "$h" ] || continue
    case "$seen" in *" $h "*) continue ;; esac
    seen="$seen$h "
    [ -z "$want" ] || [ "$h" = "$want" ] || continue
    finished "$h" && continue
    locked "$h" && continue
    # A checkpoint whose check failed waits until its marker is removed.
    [ -f "$(run_dir "$h")/failed" ] && continue
    if [ "$where" = release ]; then
      where=$(download "$h") && [ -d "$where" ] || continue
    fi
    echo "$h $where"
    return 0
  done < <(candidates)
  return 1
}

# --- shards --------------------------------------------------------------
room() {
  local verdict avail
  verdict=$("$BUDGET" 2> /dev/null | tail -1)
  avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
  log "budget ${verdict:-unknown}, ${avail} GB available"
  [ "$verdict" != red ] && [ "$avail" -ge $((MIN_FREE_GB + SHARD_GB)) ]
}

phase() {
  # Run one phase's shards, up to MAX_PAR at once; fail if any fails.
  local name=$1 adapter=$2 run=$3 which=$4 shards=$5 i pid
  local -a extra pids=()
  extra=(--cache "$RUN_ROOT/blocks.pkl" --phase "$which")
  [ -n "$dry" ] && extra+=(--limit 2)
  # shellcheck disable=SC2206
  extra+=($SCORE_ARGS)
  for i in $(seq 0 $((shards - 1))); do
    [ -f "$run/$which-shard-$i.done" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ] || ! room; do
      sleep 60
    done
    log "$name $which shard $i/$shards"
    (
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
        systemd-run --user --scope -q -p "MemoryMax=${SHARD_GB}G" \
        -p MemorySwapMax=0 nice -n 19 \
        "$PY" -u scripts/slots_check.py score "$run/$which-shard-$i.csv" \
        --slots "$adapter" --label "$name" --shard "$i" "$shards" \
        "${extra[@]}" > "$run/$which-shard-$i.log" 2>&1 < /dev/null \
        && touch "$run/$which-shard-$i.done"
    ) &
    pids+=($!)
    # Let the shard load its models before the next memory check.
    sleep 90
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  for i in $(seq 0 $((shards - 1))); do
    [ -f "$run/$which-shard-$i.done" ] \
      || { fail "$name" "$which shard $i failed; log $run/$which-shard-$i.log"; return 1; }
  done
}

summarise() {
  # Summarise the shards so far; prints the mailbox reading.
  local name=$1 run=$2 stage=$3 csv=$4 blocks=$5 page=$6
  PYTHONPATH=src "$PY" scripts/slots_check.py summarise "$run" \
    --label "$name" --commit "$commit" --csv "$csv" --blocks "$blocks" \
    --copy "$page" --stage "$stage"
}

# --- check one checkpoint ------------------------------------------------
check() {
  local adapter=$2 name="slots-$1" run shards t0 minutes csv blocks page
  local headline sha remote fd
  run=$(run_dir "$1")
  mkdir -p "$run"
  exec {fd}> "$LOCKS/$name${dry:+-dry}.lock"
  if ! flock -n "$fd"; then
    log "$name is being checked by another caller"
    exec {fd}>&-
    return 0
  fi
  if finished "$1"; then
    log "$name is already checked"
    exec {fd}>&-
    return 0
  fi
  shards=$SHARDS
  [ -n "$dry" ] && shards=$DRY_SHARDS
  if [ -n "$dry" ]; then
    csv="$run/$name.csv"
    blocks="$run/$name-blocks.csv.gz"
    page="$run/summary.csv"
  else
    mkdir -p evidence/slots-check "$PAGE_ROOT/$name"
    csv="evidence/slots-check/$name.csv"
    blocks="evidence/slots-check/$name-blocks.csv.gz"
    page="$PAGE_ROOT/$name/summary.csv"
  fi
  t0=$(date +%s)
  mail "found ${name} at ${adapter}; headline phase in ${shards} shard(s) into ${run}"
  phase "$name" "$adapter" "$run" headline "$shards" \
    || { exec {fd}>&-; return 1; }
  headline=$(summarise "$name" "$run" interim "$run/interim.csv" \
    "$run/interim-blocks.csv.gz" "$page") \
    || { fail "$name" "interim summary failed"; exec {fd}>&-; return 1; }
  minutes=$((($(date +%s) - t0) / 60))
  mail "-> bdbv-paper: interim ${name} after ${minutes} min (given and swap only; final to follow): ${page}. ${headline//$'\n'/ | }"
  phase "$name" "$adapter" "$run" rest "$shards" \
    || { exec {fd}>&-; return 1; }
  headline=$(summarise "$name" "$run" final "$csv" "$blocks" "$page") \
    || { fail "$name" "final summary failed"; exec {fd}>&-; return 1; }
  minutes=$((($(date +%s) - t0) / 60))
  echo "$headline"
  if [ -n "$dry" ]; then
    mail "dry run of ${name} done in ${minutes} min: ${csv}; ${headline//$'\n'/ | }"
    touch "$run/finished"
    exec {fd}>&-
    return 0
  fi
  {
    # One commit at a time, whichever caller.
    flock 9
    git add "$csv" "$blocks" \
      && git -c user.name=seabbs-bot -c user.email=signin@samabbott.co.uk \
        commit -q -m "evidence: slots check of $name" \
        -m "scripts/slots_check.sh at $commit; shards in $run." \
      && {
        git push -q origin "$BRANCH" || {
          git fetch -q origin "$BRANCH" \
            && git merge -q --no-edit "origin/$BRANCH" \
            && git push -q origin "$BRANCH"
        }
      }
  } 9> "$LOCKS/git.lock" \
    || { fail "$name" "commit or push of the evidence failed"; exec {fd}>&-; return 1; }
  sha=$(git log -n 1 --format=%H -- "$csv")
  remote=$(git ls-remote origin "refs/heads/$BRANCH" | cut -f1)
  git merge-base --is-ancestor "$sha" "$remote" 2> /dev/null \
    || { fail "$name" "$sha is not on origin/$BRANCH ($remote)"; exec {fd}>&-; return 1; }
  touch "$run/finished"
  exec {fd}>&-
  mail "-> bdbv-paper: finished ${name} in ${minutes} min, ${sha:0:9} on origin/${BRANCH}: ${csv}, ${blocks}, ${page}. Swap = whole lag series moved between places, ratio of loss to given (90% CI); slots loss by source against no-slots at the same n and zero-shot: ${headline//$'\n'/ | }"
  log "$name done: $sha"
}

# --- main ----------------------------------------------------------------
if [ -n "$want" ] && finished "$want"; then
  log "slots-$want is already checked"
  exit 0
fi
if [ -n "$want" ] && locked "$want"; then
  log "slots-$want is being checked by another caller"
  exit 0
fi
preflight || exit 1
commit=$(git rev-parse HEAD)
mail "started for ${label} at code ${commit:0:9}${dry:+ (dry run)}; watching $CKPTS and chronos-slots-* releases"
while true; do
  if found=$(next_checkpoint); then
    preflight || exit 1
    if ! check "${found%% *}" "${found#* }"; then
      [ -n "$want" ] && exit 1
      touch "$(run_dir "${found%% *}")/failed"
      mail "slots-${found%% *} is skipped until $(run_dir "${found%% *}")/failed is removed; watching on"
    fi
    [ -n "$want" ] && exit 0
    continue
  fi
  log "no new slots checkpoint for ${label}; next look in ${POLL}s"
  sleep "$POLL"
done
