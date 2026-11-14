#!/usr/bin/env bash
# Walk the fallback chain over the candidates in $WORK/candidates and
# write $WORK/gate (provenance and the selected candidate).
#
#   VALIDATE=true|false  run the hub validation hook (default true)
#   NO_SHADOWS=true      leave the shadow candidates out (track.sh)
#   ONLY_MODEL=true      submit $MODEL or nothing, never another candidate
#                        in its place (track.sh)
# Exits 3 when every candidate is ruled out and the round is skipped.
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

args=(--policy "$POLICY" --candidates "$WORK/candidates" --model "$MODEL"
  --out "$WORK/gate")
if [ "${NO_SHADOWS:-false}" = "true" ]; then
  args+=(--no-shadows)
fi
if [ "${ONLY_MODEL:-false}" = "true" ]; then
  args+=(--only-model)
fi
if [ "${VALIDATE:-true}" = "true" ]; then
  args+=(--validate-cmd "$CI_DIR/steps/validate_hub.sh {submission}")
fi
status=0
sci python -m submission_ci.select_candidate "${args[@]}" || status=$?
if [ "$status" = 0 ]; then
  # The selected file becomes the file the PR holds. A resubmission keeps
  # the hub's rows unchanged and adds this forecast's.
  : "${REFERENCE_DATE:?}"
  for f in $(find "$WORK/gate/selected/submission" -name '*.csv' | sort); do
    append_previous "$f"
  done
  # The reported counts, for the PR body's summary against recent weeks.
  cp "$WORK/inputs/time-series.csv" "$WORK/gate/time-series.csv"
fi
exit "$status"
