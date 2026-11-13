#!/usr/bin/env bash
# Record a live run's outcome on the ledger branch and archive the
# round's small files there, so the next run knows which data each round
# used and the submissions outlive the 90-day artefact limit.
#
#   LEDGER_BRANCH  default submission-ledger
#   REFERENCE_DATE, DATA_COMMIT, AS_OF, ACTION, PR_URL, PRODUCED_BY,
#   SKIPPED, RUN   from the earlier jobs
#   GITHUB_SHA     the commit the round was submitted from
#   RENEWAL_ID, RENEWAL_ACTION, RENEWAL_PR_URL, RENEWAL_PRODUCED_BY,
#   RENEWAL_SKIPPED and the same for BASELINE_: each hub model that ran
#   BASELINE_GATE  baseline's gate folder, when it ran
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

: "${REFERENCE_DATE:?}" "${DATA_COMMIT:?}" "${GITHUB_SHA:?}"
ledger="$WORK/ledger"
ledger_worktree "$ledger"

# A track's status from its SKIPPED and ACTION.
status_of() {
  if [ "$1" = "true" ]; then
    echo skipped
    return
  fi
  case "$2" in
    created | updated | unchanged-on-hub) echo submitted ;;
    *) echo failed ;;
  esac
}
status=$(status_of "${SKIPPED:-}" "${ACTION:-}")
sci detect-new-vintage record --state "$ledger/state.json" \
  --reference-date "$REFERENCE_DATE" --data-commit "$DATA_COMMIT" \
  --as-of "${AS_OF:-}" --status "$status" --produced-by "${PRODUCED_BY:-}" \
  --pr-url "${PR_URL:-}" --run "${RUN:-}" --code-commit "$GITHUB_SHA"

archive="$ledger/archive/$REFERENCE_DATE/${GITHUB_RUN_ID:-local}"
# The gate's small files: provenance, PR body, the selected file and plot.
archive_gate() {
  local gate=$1 dest=$2 f
  [ -d "$gate" ] || return 0
  mkdir -p "$dest"
  for f in provenance.json provenance.md pr-body.md; do
    [ -f "$gate/$f" ] && cp "$gate/$f" "$dest/"
  done
  if [ -d "$gate/selected" ]; then
    for f in run.json checks.json fit-summary.json nowcast.png; do
      [ -f "$gate/selected/$f" ] && cp "$gate/selected/$f" "$dest/"
    done
    find "$gate/selected/submission" -name '*.csv' -exec cp {} "$dest/" \;
  fi
}
mkdir -p "$archive"
archive_gate "$WORK/gate" "$archive"
# baseline's files sit beside the round's (BASELINE_GATE); when it
# is the only hub model that ran, they are the round's own.
baseline_archive=$archive
[ -d "$WORK/gate" ] && baseline_archive="$archive/baseline"
[ -n "${BASELINE_GATE:-}" ] && archive_gate "$BASELINE_GATE" "$baseline_archive"

# Each hub model that ran, under its id: status, PR and files.
#   record_model ID ACTION PR_URL PRODUCED_BY SKIPPED ARCHIVE
record_model() {
  [ -n "$1" ] || return 0
  sci detect-new-vintage record-model --state "$ledger/state.json" \
    --reference-date "$REFERENCE_DATE" --model-id "$1" \
    --status "$(status_of "$5" "$2")" --pr-url "$3" --produced-by "$4" \
    --run "${RUN:-}" --code-commit "$GITHUB_SHA" --archive "${6#"$ledger"/}"
}
record_model "${RENEWAL_ID:-}" "${RENEWAL_ACTION:-}" "${RENEWAL_PR_URL:-}" \
  "${RENEWAL_PRODUCED_BY:-}" "${RENEWAL_SKIPPED:-}" "$archive"
record_model "${BASELINE_ID:-}" "${BASELINE_ACTION:-}" \
  "${BASELINE_PR_URL:-}" "${BASELINE_PRODUCED_BY:-}" \
  "${BASELINE_SKIPPED:-}" "$baseline_archive"

ledger_push "$ledger" "Record $REFERENCE_DATE: $status"
note "Ledger: $REFERENCE_DATE recorded as $status."
