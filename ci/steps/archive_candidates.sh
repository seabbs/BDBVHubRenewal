#!/usr/bin/env bash
# Archive every candidate model's forecasts for the round on the ledger
# branch, beside the files record.sh archived (`fit-store archive`).
# The stored models' forecasts are those the fit store holds now; the
# other candidates, such as randomwalk, are forecast now. Each row says
# when its forecast was made. This runs after record.sh, apart from it,
# so a failure here never stops a round being recorded.
#
#   REFERENCE_DATE, AS_OF  from the earlier jobs
#   LEDGER_BRANCH    default submission-ledger
#   STORE            where to fetch the fit store (default output/fit-store)
#   SKIP_FETCH=true  use STORE as it is (the dry run)
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

: "${REFERENCE_DATE:?}" "${AS_OF:?}"
store=${STORE:-output/fit-store}
if [ "${SKIP_FETCH:-false}" != true ]; then
  rm -rf "$store"
  "$CI_DIR/../scripts/fit-store.sh" fetch "$store"
fi
made="$WORK/candidates.csv.gz"
cli fit-store archive --store "$store" --origins "$AS_OF" --out "$made"

ledger="$WORK/ledger"
ledger_worktree "$ledger"
archive="$ledger/archive/$REFERENCE_DATE/${GITHUB_RUN_ID:-local}"
mkdir -p "$archive"
cp "$made" "$archive/candidates.csv.gz"
ledger_push "$ledger" "Archive the candidates' forecasts for $REFERENCE_DATE"
note "Ledger: every candidate's forecasts at $AS_OF archived for $REFERENCE_DATE."
