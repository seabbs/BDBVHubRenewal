#!/bin/bash
# Run the slots place-swap check (scripts/slots_check.py) on a slots
# checkpoint once it exists, then commit and push its evidence.
#
# Usage: scripts/slots_check.sh LABEL [--dry]
#
# LABEL is the checkpoint's settings hash, as `<hash>`, `slots-<hash>` or
# `chronos-slots-<hash>`, or `auto` for the first slots checkpoint to
# appear. The runner waits, polling every $POLL seconds, until the
# adapter is found:
#   - locally, in the nK folder of $CKPTS whose settings.json has the
#     hash and whose files match its sha256sums.txt (with `auto`, the
#     smallest such nK); or
#   - as the GitHub release chronos-slots-<hash> (with `auto`, the oldest
#     chronos-slots-* release), downloaded and checked by
#     foundation.finetuned_adapter.
# Shards then run one at a time, each under
#   systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 nice -n 19
# with one thread, each started only when compute-budget.sh is not red
# and at least $MIN_FREE_GB GB would stay free with the shard at its
# $SHARD_GB GB cap.
# The summary goes to evidence/slots-check.csv and the per-block rows to
# evidence/slots-check-blocks.csv.gz, committed on branch slots-check as
# seabbs-bot and pushed; the push is checked against origin.
# Start, failure and finish lines go to $MAILBOX.
#
# --dry scores two blocks of each set in one shard, prints the mailbox
# lines instead of writing them, and never commits or pushes.
#
# Environment: PY (python with chronos and peft), CKPTS, RUN_ROOT,
# SHARDS, POLL, MIN_FREE_GB, SHARD_GB, MAILBOX, SCORE_ARGS (passed on to
# slots_check.py score).
set -uo pipefail

label=${1:?usage: slots_check.sh LABEL|auto [--dry]}
dry=${2:-}
[ -z "$dry" ] || [ "$dry" = "--dry" ] || { echo "unknown flag $dry"; exit 2; }

here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here" || exit 1
PY=${PY:-/home/seabbs/code/seabbs/BDBVHubRenewal/.worktrees/checkpoint-scoring/.venv/bin/python}
CKPTS=${CKPTS:-$HOME/bdbv-rounds/checkpoints/slots}
RUN_ROOT=${RUN_ROOT:-$HOME/bdbv-slots-check}
SHARDS=${SHARDS:-6}
POLL=${POLL:-600}
MIN_FREE_GB=${MIN_FREE_GB:-16}
SHARD_GB=${SHARD_GB:-6}
MAILBOX=${MAILBOX:-$HOME/bdbv-archie-mailbox.md}
SCORE_ARGS=${SCORE_ARGS:-}
REPO=seabbs/BDBVHubRenewal
BRANCH=slots-check
BUDGET=$HOME/.claude/hooks/compute-budget.sh

hash=${label#chronos-}
hash=${hash#slots-}
[ "$hash" = auto ] && hash=""

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
  mail "FAILED (${label}): $*"
  exit 1
}

# --- preflight -----------------------------------------------------------
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] \
  || fail "$here is not on branch $BRANCH"
[ -z "$(git status --porcelain --untracked-files=no)" ] \
  || fail "uncommitted changes in $here"
commit=$(git rev-parse HEAD)
"$PY" -c "import chronos, peft, torch, safetensors" 2> /dev/null \
  || fail "$PY cannot import chronos, peft, torch and safetensors"

# --- find the adapter ----------------------------------------------------
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

local_adapter() {
  local d h
  for d in $(find "$CKPTS" -maxdepth 1 -name "n*" -type d 2> /dev/null | sort -V); do
    h=$(settings_hash "$d/settings.json") || continue
    if [ -z "$hash" ] || [ "$h" = "$hash" ]; then
      verified "$d" && { echo "$h $d/finetuned-ckpt"; return 0; }
    fi
  done
  return 1
}

release_adapter() {
  local h=$hash tag dir
  if [ -z "$h" ]; then
    tag=$(gh release list -R "$REPO" --limit 200 \
      --json tagName,createdAt \
      -q '[.[] | select(.tagName | startswith("chronos-slots-"))]
          | sort_by(.createdAt) | .[0].tagName // empty' 2> /dev/null)
    [ -n "$tag" ] || return 1
    h=${tag#chronos-slots-}
  fi
  dir=$(PYTHONPATH=src "$PY" -c '
import sys
from bdbvhubrenewal.forecasters import foundation as f
print(f.finetuned_adapter(f.resolve_checkpoint("slots-" + sys.argv[1])))
' "$h" 2> /dev/null | tail -1) || return 1
  [ -d "$dir" ] || return 1
  echo "$h $dir"
}

mail "started for ${label} at code ${commit:0:9}${dry:+ (dry run)}; waiting for the slots adapter (local $CKPTS or release chronos-slots-${hash:-*})"
found=""
until found=$(local_adapter) || found=$(release_adapter); do
  log "no slots adapter for ${label} yet; next look in ${POLL}s"
  sleep "$POLL"
done
hash=${found%% *}
adapter=${found#* }
label="slots-$hash"
log "adapter $adapter ($label)"

# --- score ---------------------------------------------------------------
run="$RUN_ROOT/$label${dry:+-dry}"
mkdir -p "$run"
shards=$SHARDS
extra=(--cache "$RUN_ROOT/blocks.pkl")
if [ -n "$dry" ]; then
  shards=1
  extra+=(--limit 2)
fi
# shellcheck disable=SC2206
extra+=($SCORE_ARGS)

room() {
  local verdict avail
  verdict=$("$BUDGET" 2> /dev/null | tail -1)
  avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
  log "budget ${verdict:-unknown}, ${avail} GB available"
  [ "$verdict" != red ] && [ "$avail" -ge $((MIN_FREE_GB + SHARD_GB)) ]
}

t0=$(date +%s)
mail "found ${label} at ${adapter}; scoring ${shards} shard(s) into ${run}"
for i in $(seq 0 $((shards - 1))); do
  [ -f "$run/shard-$i.done" ] && continue
  until room; do sleep 300; done
  log "shard $i/$shards"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
    systemd-run --user --scope -q -p "MemoryMax=${SHARD_GB}G" \
    -p MemorySwapMax=0 nice -n 19 \
    "$PY" -u scripts/slots_check.py score "$run/shard-$i.csv" \
    --slots "$adapter" --label "$label" --shard "$i" "$shards" \
    "${extra[@]}" > "$run/shard-$i.log" 2>&1 < /dev/null \
    || fail "shard $i of $label failed; log $run/shard-$i.log"
  touch "$run/shard-$i.done"
done

mkdir -p evidence
csv=evidence/slots-check.csv
blocks=evidence/slots-check-blocks.csv.gz
if [ -n "$dry" ]; then
  csv="$run/slots-check.csv"
  blocks="$run/slots-check-blocks.csv.gz"
fi
headline=$(PYTHONPATH=src "$PY" scripts/slots_check.py summarise "$run" \
  --label "$label" --commit "$commit" --csv "$csv" --blocks "$blocks") \
  || fail "summary of $run failed"
minutes=$((($(date +%s) - t0) / 60))
echo "$headline"
if [ -n "$dry" ]; then
  mail "dry run of ${label} done in ${minutes} min: ${csv}; ${headline//$'\n'/ | }"
  exit 0
fi

# --- commit and push -----------------------------------------------------
git add "$csv" "$blocks" || fail "cannot stage the evidence"
git -c user.name=seabbs-bot -c user.email=signin@samabbott.co.uk commit -q \
  -m "evidence: slots place-swap check of $label" \
  -m "scripts/slots_check.sh $label at $commit; shards in $run." \
  || fail "commit of the evidence failed"
sha=$(git rev-parse HEAD)
git push -q origin "$BRANCH" || {
  git fetch -q origin "$BRANCH" \
    && git merge -q --no-edit "origin/$BRANCH" \
    && git push -q origin "$BRANCH"
} || fail "push of $sha failed"
sha=$(git rev-parse HEAD)
remote=$(git ls-remote origin "refs/heads/$BRANCH" | cut -f1)
[ "$remote" = "$sha" ] || fail "origin/$BRANCH is $remote, not $sha"
mail "-> bdbv-paper: finished ${label} in ${minutes} min, ${sha:0:9} on origin/${BRANCH}: ${csv} and ${blocks}. Swap of whole lag series between places (ratio of loss to given, 90% CI; identical = blocks with bit-identical output): ${headline//$'\n'/ | }"
log "done: $sha"
