#!/usr/bin/env bash
# Gather the German benchmark's held-out half from a dispatched Pages run
# (`paper_target: paper`), whose jobs upload it as heldout-* artifacts,
# into ../results/germany/held-out, as the Pages paper step does.
# Waits for the held-out jobs; reports to the archie mailbox.
#
# Usage (from anywhere): paper/scripts/gather_heldout.sh RUN_ID
set -uo pipefail
run=$1
root=$(cd "$(dirname "$0")/../.." && pwd)
mb=${MAILBOX:-$HOME/bdbv-archie-mailbox.md}
say() { echo "$(date '+%F %H:%M') paper: held-out German gather (run $run): $*" >> "$mb"; }
repo=seabbs/BDBVHubRenewal
bdbv="$root/.venv/bin/bdbvhubrenewal"
cd "$root"
say "waiting for the held-out jobs"
for _ in $(seq 1 96); do
  left=$(gh run view "$run" -R "$repo" --json jobs \
    --jq '[.jobs[] | select(.name | startswith("Held-out German")) | select(.status != "completed")] | length')
  [ "$left" = 0 ] && break
  sleep 300
done
bad=$(gh run view "$run" -R "$repo" --json jobs \
  --jq '[.jobs[] | select(.name | startswith("Held-out German")) | select(.conclusion != "success") | .name] | join(", ")')
[ -n "$bad" ] && { say "FAILED: held-out jobs not successful: $bad"; exit 1; }
rm -rf output/heldout && mkdir -p output/heldout
if ! gh run download "$run" -R "$repo" -p 'heldout-*' -D output/heldout; then
  say "FAILED: artifact download"; exit 1
fi
if "$bdbv" concat-results --out results/germany/held-out \
     output/heldout/heldout-*/germany/held-out &&
   "$bdbv" germany-teams --held-out --out results/germany &&
   "$bdbv" germany-paper --results results/germany/held-out; then
  say "done: results/germany/held-out/paper-forecasts.csv.gz in $root"
else
  say "FAILED: gathering (see the job's log)"; exit 1
fi
