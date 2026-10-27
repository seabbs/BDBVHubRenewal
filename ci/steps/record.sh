#!/usr/bin/env bash
# Record a live run's outcome on the ledger branch and archive the
# round's small files there, so the next run knows which data each round
# used and the submissions outlive the 90-day artefact limit.
#
#   LEDGER_BRANCH  default submission-ledger
#   REFERENCE_DATE, DATA_COMMIT, AS_OF, ACTION, PR_URL, PRODUCED_BY,
#   SKIPPED, RUN   from the earlier jobs
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

: "${REFERENCE_DATE:?}" "${DATA_COMMIT:?}"
LEDGER_BRANCH=${LEDGER_BRANCH:-submission-ledger}
ledger="$WORK/ledger"
rm -rf "$ledger"
git worktree prune
if git fetch -q --depth 1 origin "$LEDGER_BRANCH" 2> /dev/null; then
  git worktree add -q -B "$LEDGER_BRANCH" "$ledger" FETCH_HEAD
else
  git worktree add -q --orphan -b "$LEDGER_BRANCH" "$ledger"
fi

if [ "${SKIPPED:-}" = "true" ]; then
  status=skipped
else
  case "${ACTION:-}" in
    created | updated | unchanged-on-hub) status=submitted ;;
    *) status=failed ;;
  esac
fi
sci detect-new-vintage record --state "$ledger/state.json" \
  --reference-date "$REFERENCE_DATE" --data-commit "$DATA_COMMIT" \
  --as-of "${AS_OF:-}" --status "$status" --produced-by "${PRODUCED_BY:-}" \
  --pr-url "${PR_URL:-}" --run "${RUN:-}"

archive="$ledger/archive/$REFERENCE_DATE/${GITHUB_RUN_ID:-local}"
mkdir -p "$archive"
gate="$WORK/gate"
for f in provenance.json provenance.md pr-body.md; do
  [ -f "$gate/$f" ] && cp "$gate/$f" "$archive/"
done
if [ -d "$gate/selected" ]; then
  for f in run.json checks.json fit-summary.json nowcast.png; do
    [ -f "$gate/selected/$f" ] && cp "$gate/selected/$f" "$archive/"
  done
  find "$gate/selected/submission" -name '*.csv' -exec cp {} "$archive/" \;
fi

git -C "$ledger" add -A
git -C "$ledger" -c user.name="github-actions[bot]" \
  -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
  commit -q -m "Record $REFERENCE_DATE: $status"
git -C "$ledger" push -q origin "$LEDGER_BRANCH"
note "Ledger: $REFERENCE_DATE recorded as $status."
