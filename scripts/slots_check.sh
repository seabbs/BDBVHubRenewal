#!/bin/bash
# Run the slots check (scripts/slots_check.py) on slots checkpoints as
# they land, then commit and push the evidence.
#
# Usage: scripts/slots_check.sh LABEL [--dry]
#
# LABEL is a slots checkpoint's settings hash, as `<hash>`, `slots-<hash>`
# or `chronos-slots-<hash>`: the runner waits for it, checks it and
# exits. Or `auto`: the runner checks every slots checkpoint as it lands
# (n250, n500, n1000, ...), oldest first, and keeps watching.
# A checkpoint is found, polling every $POLL seconds:
#   - locally, as an nK folder of $CKPTS whose files match its
#     sha256sums.txt, so a half-copied folder is never used; or
#   - as the GitHub release chronos-slots-<hash>, downloaded and checked
#     by foundation.finetuned_adapter.
# Each checkpoint is scored against zero-shot Chronos-2 and the no-slots
# checkpoint with the same n_outbreaks (n1000 de129103cacc, n2000
# 7962fd718a47 in ~/bdbv-rounds/checkpoints), or zero-shot alone where
# there is none (n250, n500); slots_check.py picks it from the settings.
# Shards run one at a time, each under
#   systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 nice -n 19
# with one thread, started only when compute-budget.sh is not red and at
# least $MIN_FREE_GB GB would stay free with the shard at its measured
# peak, $SHARD_NEED_GB GB (2.8 GB on a 255-row block; 6 GB is the cap).
# Outputs per checkpoint LABEL (slots-<hash>):
#   evidence/slots-check/LABEL.csv          summary rows
#   evidence/slots-check/LABEL-blocks.csv.gz per-block rows
#   ~/slots-check/LABEL/summary.csv         summary, before the commit
# The evidence is committed on branch slots-check as seabbs-bot and
# pushed; the push is checked against origin. Start, failure and finish
# lines go to $MAILBOX.
#
# --dry scores two blocks of each set in one shard, prints the mailbox
# lines instead of writing them, and never commits or pushes.
#
# Environment: PY (python with chronos and peft), CKPTS, RUN_ROOT,
# PAGE_ROOT, SHARDS, POLL, MIN_FREE_GB, SHARD_GB, SHARD_NEED_GB, MAILBOX,
# SCORE_ARGS
# (passed on to slots_check.py score).
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
SHARDS=${SHARDS:-2}
POLL=${POLL:-300}
MIN_FREE_GB=${MIN_FREE_GB:-16}
SHARD_GB=${SHARD_GB:-6}
SHARD_NEED_GB=${SHARD_NEED_GB:-3}
MAILBOX=${MAILBOX:-$HOME/bdbv-archie-mailbox.md}
SCORE_ARGS=${SCORE_ARGS:-}
REPO=seabbs/BDBVHubRenewal
BRANCH=slots-check
BUDGET=$HOME/.claude/hooks/compute-budget.sh

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
  log "FAILED: $*"
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
preflight || exit 1
commit=$(git rev-parse HEAD)

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

done_already() { [ -f "$RUN_ROOT/slots-$1${dry:+-dry}/finished" ]; }

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
  # The first landed checkpoint still to check (only LABEL's, if given).
  local h where seen=" "
  while read -r h where; do
    [ -n "$h" ] || continue
    case "$seen" in *" $h "*) continue ;; esac
    seen="$seen$h "
    [ -z "$want" ] || [ "$h" = "$want" ] || continue
    done_already "$h" && continue
    if [ "$where" = release ]; then
      where=$(download "$h") && [ -d "$where" ] || continue
    fi
    echo "$h $where"
    return 0
  done < <(candidates)
  return 1
}

room() {
  local verdict avail
  verdict=$("$BUDGET" 2> /dev/null | tail -1)
  avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
  log "budget ${verdict:-unknown}, ${avail} GB available"
  [ "$verdict" != red ] && [ "$avail" -ge $((MIN_FREE_GB + SHARD_NEED_GB)) ]
}

# --- check one checkpoint ------------------------------------------------
check() {
  local adapter=$2 name="slots-$1" run shards extra i t0 minutes
  local csv blocks page headline sha remote
  run="$RUN_ROOT/$name${dry:+-dry}"
  mkdir -p "$run"
  shards=$SHARDS
  extra=(--cache "$RUN_ROOT/blocks.pkl")
  if [ -n "$dry" ]; then
    shards=1
    extra+=(--limit 2)
  fi
  # shellcheck disable=SC2206
  extra+=($SCORE_ARGS)
  t0=$(date +%s)
  mail "found ${name} at ${adapter}; scoring ${shards} shard(s) into ${run}"
  for i in $(seq 0 $((shards - 1))); do
    [ -f "$run/shard-$i.done" ] && continue
    until room; do sleep 300; done
    log "$name shard $i/$shards"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
      systemd-run --user --scope -q -p "MemoryMax=${SHARD_GB}G" \
      -p MemorySwapMax=0 nice -n 19 \
      "$PY" -u scripts/slots_check.py score "$run/shard-$i.csv" \
      --slots "$adapter" --label "$name" --shard "$i" "$shards" \
      "${extra[@]}" > "$run/shard-$i.log" 2>&1 < /dev/null \
      || { fail "$name" "shard $i failed; log $run/shard-$i.log"; return 1; }
    touch "$run/shard-$i.done"
  done
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
  headline=$(PYTHONPATH=src "$PY" scripts/slots_check.py summarise "$run" \
    --label "$name" --commit "$commit" --csv "$csv" --blocks "$blocks" \
    --copy "$page") || { fail "$name" "summary of $run failed"; return 1; }
  minutes=$((($(date +%s) - t0) / 60))
  echo "$headline"
  if [ -n "$dry" ]; then
    mail "dry run of ${name} done in ${minutes} min: ${csv}; ${headline//$'\n'/ | }"
    touch "$run/finished"
    return 0
  fi
  git add "$csv" "$blocks" || { fail "$name" "cannot stage the evidence"; return 1; }
  git -c user.name=seabbs-bot -c user.email=signin@samabbott.co.uk \
    commit -q -m "evidence: slots check of $name" \
    -m "scripts/slots_check.sh at $commit; shards in $run." \
    || { fail "$name" "commit of the evidence failed"; return 1; }
  git push -q origin "$BRANCH" || {
    git fetch -q origin "$BRANCH" \
      && git merge -q --no-edit "origin/$BRANCH" \
      && git push -q origin "$BRANCH"
  } || { fail "$name" "push failed"; return 1; }
  sha=$(git rev-parse HEAD)
  remote=$(git ls-remote origin "refs/heads/$BRANCH" | cut -f1)
  [ "$remote" = "$sha" ] \
    || { fail "$name" "origin/$BRANCH is $remote, not $sha"; return 1; }
  touch "$run/finished"
  mail "-> bdbv-paper: finished ${name} in ${minutes} min, ${sha:0:9} on origin/${BRANCH}: ${csv}, ${blocks}, ${page}. Swap = whole lag series moved between places, ratio of loss to given (90% CI); slots loss by source against no-slots at the same n and zero-shot: ${headline//$'\n'/ | }"
  log "$name done: $sha"
}

mail "started for ${label} at code ${commit:0:9}${dry:+ (dry run)}; watching $CKPTS and chronos-slots-* releases"
while true; do
  if found=$(next_checkpoint); then
    preflight || exit 1
    check "${found%% *}" "${found#* }" || exit 1
    [ -n "$want" ] && exit 0
    continue
  fi
  log "no new slots checkpoint for ${label}; next look in ${POLL}s"
  sleep "$POLL"
done
