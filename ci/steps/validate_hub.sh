#!/usr/bin/env bash
# Hub validation hook: validate_hub.sh <submission.csv>
#
# Stages the file under the submitting model id, as it will sit in the PR
# (`<model_id>/<reference_date>-<model_id>.csv`, so a fallback's file is
# renamed), with the metadata beside it, and runs both validators of the
# hub validation bundle against the hub checkout:
#
#   1. `bdbvhubrenewal validate` (hub_validate.py, Python, fast);
#   2. scripts/validate-hub.R (hubValidations, the hub's own checks),
#      unless R_VALIDATION=false.
#
# Either failing fails the candidate. Set HUB_VALIDATOR to replace both
# with one command, called as `$HUB_VALIDATOR <hub_path> <csv> <metadata>`
# (the dry-run harness uses `true`).
#
#   HUB_PATH      checkout of the hub at HUB_REF (hub-config, model-metadata)
#   METADATA      model metadata YAML for MODEL_ID
#   CHECK_WINDOW  true to include the submission window check
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

submission=${1:?usage: validate_hub.sh <submission.csv>}
: "${HUB_PATH:?}" "${METADATA:?}" "${REFERENCE_DATE:?}"
stage="$WORK/hub-validation"
rm -rf "$stage" && mkdir -p "$stage/$MODEL_ID"
csv="$stage/$MODEL_ID/$REFERENCE_DATE-$MODEL_ID.csv"
yml="$stage/$MODEL_ID.yml"
cp "$submission" "$csv"
cp "$METADATA" "$yml"

if [ -n "${HUB_VALIDATOR:-}" ]; then
  $HUB_VALIDATOR "$HUB_PATH" "$csv" "$yml"
  exit
fi

window=(--skip-window)
if [ "${CHECK_WINDOW:-false}" = "true" ]; then
  window=()
fi
echo "== hub_validate (Python)"
cli validate --hub "$HUB_PATH" --metadata-dir "$stage" ${window[@]+"${window[@]}"} \
  --json "$stage/py-validation.json" "$csv" "$yml"
if [ "${R_VALIDATION:-true}" != "false" ]; then
  echo "== hubValidations (R)"
  VALIDATION_JSON="$stage/r-validation.json" \
    Rscript scripts/validate-hub.R "$HUB_PATH" "$csv" "$yml"
fi
