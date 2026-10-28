#!/usr/bin/env bash
# Walk the fallback chain over the candidates in $WORK/candidates and
# write $WORK/gate (provenance and the selected candidate).
#
#   RETRIED=true         the retry stage ran
#   VALIDATE=true|false  run the hub validation hook (default true)
#   NO_SHADOWS=true      leave the shadow candidates out (track.sh)
# Exits 3 when every candidate is ruled out and the round is skipped.
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

args=(select --policy "$POLICY" --candidates "$WORK/candidates" --model "$MODEL"
  --out "$WORK/gate")
if [ "${RETRIED:-false}" = "true" ]; then
  args+=(--retried)
fi
if [ "${NO_SHADOWS:-false}" = "true" ]; then
  args+=(--no-shadows)
fi
if [ "${VALIDATE:-true}" = "true" ]; then
  args+=(--validate-cmd "$CI_DIR/steps/validate_hub.sh {submission}")
fi
sci python -m submission_ci.select_candidate "${args[@]}"
