#!/usr/bin/env bash
# Make a GitHub release of each round in the ledger whose window has
# closed and whose hub pull requests are all merged or closed, tagged
# round-<reference date> at the commit it was submitted from. A round
# whose tag exists is skipped. Runs on every trigger.
#
#   DRY_RUN        true to print what would be released (default true)
#   LEDGER_BRANCH  default submission-ledger
#   GH_TOKEN       a token that can write releases here and read the hub
# Sets the step output released=<number of releases made>.
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

LEDGER_BRANCH=${LEDGER_BRANCH:-submission-ledger}
REPO=${GITHUB_REPOSITORY:-seabbs/BDBVHubRenewal}
ledger="$WORK/release-ledger"
mkdir -p "$ledger"
if ! git fetch -q --depth 1 origin "$LEDGER_BRANCH" 2> /dev/null; then
  note "No ledger yet: nothing to release."
  echo "released=0" >> "${GITHUB_OUTPUT:-/dev/null}"
  exit 0
fi
git show FETCH_HEAD:state.json > "$ledger/state.json"

flags=()
[ "${DRY_RUN:-true}" = false ] || flags+=(--dry-run)
sci python -m submission_ci.release_rounds \
  --state "$ledger/state.json" --repo "$REPO" "${flags[@]}"
