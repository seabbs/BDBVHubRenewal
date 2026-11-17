#!/usr/bin/env bash
# Gather a dispatched Pages run's (`paper_target: paper`) inputs for a
# local paper run, as its build and paper steps do: the release backtest
# with the stored fits (results/forecasts.csv.gz), and the German
# benchmark's development half (results/germany) and held-out half
# (results/germany/held-out). Waits for the run's forecast and held-out
# jobs, printing each stage (also to MAILBOX with NOTIFY set in
# config/local.env; workflow/scripts/notify.sh).
#
# Usage (from anywhere): paper/scripts/gather_heldout.sh RUN_ID
set -uo pipefail
run=$1
root=$(cd "$(dirname "$0")/../.." && pwd)
setting() { python3 "$root/workflow/flowlib/localconfig.py" "$1"; }
SAY_PREFIX="paper: CI gather (run $run)"
# shellcheck source=workflow/scripts/notify.sh
source "$root/workflow/scripts/notify.sh"
repo=seabbs/BDBVHubRenewal
bdbv="$root/.venv/bin/bdbvhubrenewal"
cd "$root"
jobs() {
  gh run view "$run" -R "$repo" --json jobs --jq "[.jobs[] | select(.name | test(\"^(Held-out German|Forecast)\")) | $1] | $2"
}
say "waiting for the forecast and held-out jobs"
for _ in $(seq 1 96); do
  [ "$(jobs 'select(.status != "completed")' length)" = 0 ] && break
  sleep 300
done
bad=$(jobs 'select(.conclusion != "success" and .conclusion != "skipped") | .name' 'join(", ")')
[ -n "$bad" ] && { say "FAILED: jobs not successful: $bad"; exit 1; }
rm -rf output/ci && mkdir -p output/ci
for p in inputs 'parts-*' 'heldout-*'; do
  gh run download "$run" -R "$repo" -p "$p" -D output/ci/dl || {
    say "FAILED: download of $p"; exit 1; }
done
# `inputs`: data and results with the stored fits loaded.
cp -R output/ci/dl/inputs/data/. data/
mkdir -p results && cp -R output/ci/dl/inputs/results/. results/
mkdir -p output/parts && cp -R output/ci/dl/parts-* output/parts/ 2>/dev/null
p=output/parts/parts
if "$bdbv" score \
     "$p"-development-{randomwalk,randomwalknowcast} \
     "$p"-recovery-randomwalk "$p"-recovery-randomwalknowcast-{0,1,2} \
     "$p"-zones-{randomwalk,randomwalknowcast} \
     "$p"-zones-pprandomwalknowcast-{0,1} \
     "$p"-scenario-{randomwalk,randomwalknowcast}-{falling,flat,rising} &&
   "$bdbv" concat-results --out results/germany \
     output/parts/parts-germany-*/germany &&
   "$bdbv" germany-fetch &&
   "$bdbv" germany-teams --out results/germany &&
   "$bdbv" germany-paper --results results/germany &&
   "$bdbv" concat-results --out results/germany/held-out \
     output/ci/dl/heldout-*/germany/held-out &&
   "$bdbv" germany-teams --held-out --out results/germany &&
   "$bdbv" germany-paper --results results/germany/held-out; then
  say "done: results/forecasts.csv.gz, results/germany and results/germany/held-out in $root"
else
  say "FAILED: gathering (log $(setting PAPER_LOGS)/gather-heldout.log)"; exit 1
fi
