#!/usr/bin/env bash
# The aggregation arms of one fine-tuned Chronos-2 checkpoint.
#
#   scripts/agg/run_finetuned.sh LABEL [--dry]
#
# LABEL is slots-<hash> (release chronos-slots-<hash>) or a plain hash
# (release chronos-ft-<hash>), pinned with BDBV_CHRONOS_FT=LABEL.
# scripts/slots_followups.sh starts this from this worktree once a slots
# checkpoint's scores land.
#
# Waits for the release and its sha256sums.txt, then, one process at a
# time, each in a memory-capped scope at nice 19 with one thread:
#   1. the sanity check, once per version of src/: zero-shot
#      fm_chronos_daily_ragged, joint, at BDBV
#      origins 2026-10-17 and 2026-11-01, against its existing rows
#      (10-17 is the origin with scored zones);
#   2. joint, level and bottom for fm_chronos_{daily,weekly}_ragged_ft on
#      the BDBV release-backtest origins 2026-10-09 to 2026-11-01 and the
#      80 German development dates;
#   3. the paired arm from bottom (the rolling copula, as at c83d85461);
#   4. scores, then the paper's rows: results/agg/aggregation-scores.csv.gz
#      keeps every zero-shot and reference row and adds this checkpoint's,
#      with checkpoint=LABEL; committed and pushed to agg-slots.
# The data are the hub at ad1bb4af13bf (2026-11-14, the zero-shot rows'
# data: the 2026-11-08 release, D = 35) and the German hub at the commit in
# config/germany.toml, fetched once into AGG_ROOT.
# Runs queue on AGG_ROOT/lock, one checkpoint after another; each adds or
# replaces its own rows and keeps the others'. Resumes from
# AGG_ROOT/cache/LABEL. --dry fits one BDBV origin, one German
# date and the daily model only, into AGG_ROOT/cache/dry-LABEL, and never
# commits or pushes. The dry origin, 2026-10-17, has every level scored.
set -Eeuo pipefail

label=${1:?usage: run_finetuned.sh LABEL [--dry]}
dry=${2:-}
here=$(cd "$(dirname "$0")/../.." && pwd)
cd "$here"
py=${AGG_PYTHON:-$(dirname "$here")/checkpoint-scoring/.venv/bin/python}
export PYTHONPATH=$here/src
root=${AGG_ROOT:-$HOME/agg-slots}
mailbox=${MAILBOX:-$HOME/bdbv-archie-mailbox.md}
hub_ref=ad1bb4af13bf
cap=${AGG_CAP:-6G}
cap_bdbv_daily=${AGG_CAP_BDBV_DAILY:-$cap}
models=(fm_chronos_daily_ragged_ft fm_chronos_weekly_ragged_ft)
origins=(2026-10-09 2026-10-17 2026-10-24 2026-11-01)
dates=()
run=$label
if [ "$dry" = "--dry" ]; then
  models=(fm_chronos_daily_ragged_ft)
  origins=(2026-10-17)
  dates=(2022-02-09)
  run=dry-$label
elif [ -n "$dry" ]; then
  echo "unknown option $dry" >&2
  exit 2
fi
cache=$root/cache/$run
mkdir -p "$cache/log"
logfile=$cache/log/run.log
exec > >(tee -a "$logfile") 2>&1

say() {
  echo "$(date '+%Y-%m-%d %H:%M') archie: agg-slots: $run: $*" >> "$mailbox"
  echo "$(date '+%F %T') $*"
}
trap 'say "FAILED at line $LINENO (exit $?); log $logfile"' ERR

# One run at a time over every checkpoint: later ones queue here, so
# each checkpoint the slots run publishes runs in turn.
mkdir -p "$root"
exec 8> "$root/lock"
if ! flock -n 8; then
  say "queued behind another run (lock $root/lock)"
  flock 8
fi

read -r env tag < <("$py" -c "
import sys
from bdbvhubrenewal.evaluation.groupings import checkpoint_label
print(*checkpoint_label(sys.argv[1]))" "$label")
if [ -z "${tag:-}" ]; then
  say "FAILED: $label is no checkpoint label"
  exit 2
fi
export BDBV_CHRONOS_FT=$env
say "started (release $tag, cache $cache, head $(git rev-parse --short HEAD))"

# The release, with its adapter and checksums.
waited=0
until gh release view "$tag" -R seabbs/BDBVHubRenewal --json assets \
  --jq '[.assets[].name] | index("sha256sums.txt") and
        index("adapter_model.safetensors") and index("adapter_config.json")' \
  2> /dev/null | grep -q true; do
  if [ $((waited % 3600)) -eq 0 ]; then
    say "waiting for release $tag"
  fi
  sleep 600
  waited=$((waited + 600))
done

# The data, pinned.
if [ ! -f "$root/data/target.csv" ]; then
  GITHUB_TOKEN=${GITHUB_TOKEN:-$(gh auth token)} "$py" -m bdbvhubrenewal.cli \
    data --raw "$root/data/raw" --out "$root/data" --ref "$hub_ref"
fi
if [ ! -f "$root/germany/COVID-19_hospitalizations.csv" ]; then
  GITHUB_TOKEN=${GITHUB_TOKEN:-$(gh auth token)} "$py" -m bdbvhubrenewal.cli \
    germany-fetch --out "$root/germany"
fi

# One heavy process: capped, nice 19, one thread, CPU only.
heavy() {
  local mem=$1
  shift
  local gb=${mem%G}
  systemd-run --user --scope -q -p "MemoryMax=$mem" -p MemorySwapMax=0 \
    nice -n 19 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= AGG_CAP_GB="$gb" \
    "$py" scripts/agg/finetuned.py "$@"
}

# Progress, at most hourly, while the fits run.
progress() {
  # Best effort: a failed count never stops the watcher or the run.
  trap - ERR
  set +eE +o pipefail
  while sleep 3600; do
    local b g
    b=$(find "$cache/bdbv-fits" -name release-samples.csv.gz 2> /dev/null |
      wc -l)
    g=$(find "$cache/germany-samples" -name '*.csv.gz' 2> /dev/null | wc -l)
    say "progress: $b BDBV and $g German fits done"
  done
}
progress 8>&- &
watcher=$!
trap 'trap - ERR; pkill -P $watcher 2> /dev/null || true; kill $watcher 2> /dev/null || true' EXIT

existing=$here/results/agg/aggregation-scores.csv.gz
git show "origin/evidence/fm-agg:results/agg/aggregation-scores.csv.gz" \
  > "$root/zero-shot-scores.csv.gz"

# 1. Sanity: a zero-shot arm reproduces its existing rows.
# Rerun whenever the package code changes; its fits are kept per version.
code=$(git rev-parse --short HEAD:src)
sane=$root/cache/zero-shot-sanity/$code
if [ ! -f "$sane/ok" ]; then
  heavy "$cap_bdbv_daily" bdbv-fit --data "$root/data" \
    --model fm_chronos_daily_ragged --groupings joint \
    --origins 2026-10-17 2026-11-01 --out "$sane/fits"
  heavy "$cap" bdbv-score --data "$root/data" --fits "$sane/fits" \
    --out "$sane/scores.csv"
  heavy "$cap" sanity --existing "$root/zero-shot-scores.csv.gz" \
    --scores "$sane/scores.csv" --out "$sane/compare.csv" \
    --tolerance "${AGG_SANITY_TOLERANCE:-0}" | tee "$sane/result.txt"
  touch "$sane/ok"
fi
say "sanity: $(tail -n 1 "$sane/result.txt")"

# 2. The fitted arms.
for model in "${models[@]}"; do
  mem=$cap
  [ "$model" = fm_chronos_daily_ragged_ft ] && mem=$cap_bdbv_daily
  heavy "$mem" bdbv-fit --data "$root/data" --model "$model" \
    --checkpoint "$label" --origins "${origins[@]}" --out "$cache/bdbv-fits"
done
for model in "${models[@]}"; do
  heavy "$cap" germany-fit --data "$root/germany" --model "$model" \
    --checkpoint "$label" --dates "${dates[@]}" \
    --out "$cache/germany-samples"
done
say "progress: fits done; pairing and scoring"

# 3 and 4. The paired arm, the scores and the paper's rows.
if [ ! -f "$cache/bdbv-paired/.done" ]; then
  rm -rf "$cache/bdbv-paired"
  systemd-run --user --scope -q -p "MemoryMax=$cap" -p MemorySwapMax=0 \
    nice -n 19 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$py" scripts/agg/bdbv_paired.py --data "$root/data" \
    --fits "$cache/bdbv-fits" --out "$cache/bdbv-paired" \
    > "$cache/log/bdbv-paired.log"
  touch "$cache/bdbv-paired/.done"
fi
mkdir -p "$cache/scores"
heavy "$cap" bdbv-score --data "$root/data" \
  --fits "$cache/bdbv-fits" "$cache/bdbv-paired" --out "$cache/scores/bdbv.csv"
heavy "$cap" germany-score --data "$root/germany" \
  --samples "$cache/germany-samples" --out "$cache/scores/germany.csv"
subset=()
out=$existing
if [ -n "$dry" ]; then
  subset=(--subset)
  out=$cache/aggregation-scores.csv.gz
fi
heavy "$cap" export --existing "$existing" --checkpoint "$label" \
  --bdbv "$cache/scores/bdbv.csv" --germany "$cache/scores/germany.csv" \
  "${subset[@]}" --out "$out"

if [ -n "$dry" ]; then
  say "dry run finished: rows in $out; timings in $cache/*/timings.csv"
  exit 0
fi

# Commit and push the evidence to agg-slots.
git fetch -q origin agg-slots || true
if git rev-parse -q --verify origin/agg-slots > /dev/null &&
  ! git merge-base --is-ancestor origin/agg-slots HEAD; then
  git merge --no-edit origin/agg-slots
fi
cat >> results/agg/README.md << EOF
- \`$label\` (release \`$tag\`): rows added $(date +%F) by \`scripts/agg/run_finetuned.sh $label\` from code at $(git rev-parse --short HEAD).
EOF
git add -f results/agg/aggregation-scores.csv.gz results/agg/README.md
git commit -q -m "Evidence: aggregation arms of fine-tuned checkpoint $label (BDBV and the German development half)" \
  -- results/agg/aggregation-scores.csv.gz results/agg/README.md
git push -q origin HEAD:agg-slots
sha=$(git rev-parse HEAD)
remote=$(git ls-remote origin refs/heads/agg-slots | cut -f1)
if [ "$remote" != "$sha" ]; then
  say "FAILED: pushed $sha but origin/agg-slots is $remote"
  exit 1
fi
rule="every arm with slots_rule none (no slots)"
case $label in
  slots-*)
    rule="slots_rule cased on BDBV joint, bottom (zones_summed, zones_copula) and level's zones: those blocks hold over 128 places, so only places with cases have slots, outside the slots' training range (training blocks held about 95 targets); slots_rule all elsewhere"
    ;;
esac
say "finished: commit $sha on agg-slots; results/agg/aggregation-scores.csv.gz (evidence key aggregation_scores; keep rows with checkpoint in final_checkpoints, here checkpoint == '$label', plus checkpoint == 'zero-shot'; other checkpoints' rows are kept); $rule; aggregation_cost unchanged at results/agg/aggregation-cost.csv (no checkpoint column; keep dropping its _ft rows)"
