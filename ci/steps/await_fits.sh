#!/usr/bin/env bash
# Wait for the fit store to hold this round's candidate fits.
#
# The submission does not fit a round the store can hold. Fit store
# (fit-store.yml) fits every fitted model at each real origin, and each
# candidate reads the stored fit of the same key (`fit-store unpack`).
# This lists the candidates' fits the store lacks at the round's origin.
# If there are any, it starts Fit store and checks the store branch until
# it has them, Fit store has finished without them, or the wait runs
# out. Fit store runs one at a time (its concurrency group), so a
# release is fitted once, whichever workflow saw it first.
#
#   MODELS            the fitted candidates' models (space-separated)
#   AS_OF             the round's origin (the data vintage)
#   WAIT_MINUTES      the longest wait: the fit timeout from
#                     ci/policy.toml [timing], so the gate still starts
#                     reserve_minutes before the window closes
#   DISPATCH=true     start Fit store when fits are missing
#   FIT_REF           the branch Fit store runs on (default main)
#   FIT_STORE_BRANCH  the store branch (default fit-store; a test from a
#                     branch names a fit-store-scratch* branch)
#   STORE             where to fetch the store (default output/fit-store)
#   POLL_SECONDS      between checks (default 120)
#   SKIP_FETCH=true   use STORE as it is (the dry run)
#
# Writes $WORK/await/missing.json, the fits still missing. Exits 0 unless
# a step breaks: the gate rules out a candidate whose fit is missing.
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

: "${MODELS:?}" "${AS_OF:?}"
store=${STORE:-output/fit-store}
branch=${FIT_STORE_BRANCH:-fit-store}
ref=${FIT_REF:-main}
out="$WORK/await"
mkdir -p "$out"
budget=$(( ${WAIT_MINUTES:-0} * 60 ))
started=$(date -u +%s)

# The number of candidate fits the store lacks at AS_OF. A fit stored as
# failed is not missing: unpack says so and the gate rules it out.
missing() {
  if [ "${SKIP_FETCH:-false}" != true ]; then
    rm -rf "$store"
    FIT_STORE_BRANCH=$branch "$CI_DIR/../scripts/fit-store.sh" fetch "$store"
  fi
  # shellcheck disable=SC2086  # MODELS is a list
  cli fit-store plan --store "$store" --models $MODELS --origins "$AS_OF" \
    --out "$out/missing.json" > "$out/plan.log"
  python3 -c 'import json, sys
print(len(json.load(open(sys.argv[1]))["include"]))' "$out/missing.json"
}

# True once a Fit store run created at or after $1 has finished and none
# is queued or running on the branch.
fit_store_done() {
  local runs
  runs=$(gh run list -R "${GITHUB_REPOSITORY:?}" -w fit-store.yml -b "$ref" \
    -L 10 --json status,createdAt) || return 1
  python3 -c 'import json, sys
runs, since = json.loads(sys.argv[1]), sys.argv[2]
busy = any(r["status"] != "completed" for r in runs)
done = any(r["status"] == "completed" and r["createdAt"] >= since
           for r in runs)
sys.exit(0 if done and not busy else 1)' "$runs" "$1"
}

n=$(missing)
if [ "$n" = 0 ]; then
  note "Every candidate's fit at $AS_OF is in the fit store."
  exit 0
fi
note "$n candidate fit(s) at $AS_OF are not in the fit store yet."
if [ "${DISPATCH:-false}" != true ]; then
  note "Fit store is not started from this run, so they are ruled out."
  exit 0
fi

since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
args=(fit-store.yml --ref "$ref" -R "${GITHUB_REPOSITORY:?}")
if [ "$branch" != fit-store ]; then
  # A scratch store fits only this round's candidates.
  args+=(-f "store_branch=$branch" -f "models=$MODELS" -f "origins=$AS_OF")
fi
gh workflow run "${args[@]}"
note "Started Fit store on \`$ref\`; waiting up to ${WAIT_MINUTES:-0} minutes."

while :; do
  sleep "${POLL_SECONDS:-120}"
  n=$(missing)
  if [ "$n" = 0 ]; then
    note "Fit store stored every candidate's fit after $(( ($(date -u +%s) - started) / 60 )) minutes."
    exit 0
  fi
  if [ $(( $(date -u +%s) - started )) -ge "$budget" ]; then
    note "Stopped waiting with $n fit(s) missing: the gate needs the rest of the window."
    exit 0
  fi
  if fit_store_done "$since"; then
    n=$(missing)
    [ "$n" = 0 ] && exit 0
    note "Fit store finished without $n fit(s), so they are ruled out."
    exit 0
  fi
done
