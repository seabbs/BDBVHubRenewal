#!/usr/bin/env bash
# Hub validation hook: validate_hub.sh <submission.csv>
#
# Stages the file under the submitting model id (a fallback's file is
# renamed, as it will be in the PR) and runs the hub's own checks with
# hubValidations, through the main repo's scripts/validate-hub.R. Set
# HUB_VALIDATOR to replace the command, for example with a Python
# validator; it is called as `$HUB_VALIDATOR <hub_path> <csv> <metadata>`.
#
#   HUB_PATH      checkout of the hub at HUB_REF (hub-config, model-metadata)
#   METADATA      model metadata YAML for MODEL_ID
#   CHECK_WINDOW  true to include the submission window check
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

submission=${1:?usage: validate_hub.sh <submission.csv>}
: "${HUB_PATH:?}" "${METADATA:?}" "${REFERENCE_DATE:?}"
stage="$WORK/hub-validation"
rm -rf "$stage" && mkdir -p "$stage"
cp "$submission" "$stage/$REFERENCE_DATE-$MODEL_ID.csv"
cp "$METADATA" "$stage/$MODEL_ID.yml"
${HUB_VALIDATOR:-Rscript scripts/validate-hub.R} "$HUB_PATH" \
  "$stage/$REFERENCE_DATE-$MODEL_ID.csv" "$stage/$MODEL_ID.yml"
