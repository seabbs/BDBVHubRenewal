#!/usr/bin/env bash
# Check and submit the baseline hub model (baseline) on its own:
# random walk on a nowcast under its own id, with health zones required.
# Run after `prepare`, apart from the renewal chain.
#
# There is no fallback. When the model cannot run, or a check or the hub
# validation rules it out, the round is skipped for this id and says so.
# Nothing else is ever submitted under it: not plain random walk, not
# any other model.
#
#   TRACK_MODEL   the model (default randomwalknowcast)
#   TRACK_ID      its hub id (default: `bdbvhubrenewal hub-id --model`)
#   As for select.sh and submit.sh: HUB_PATH, VALIDATE, CHECK_WINDOW,
#   DRY_RUN, FORK, HUB_REF, DATA_COMMIT, AS_OF, REFERENCE_DATE.
# Exits 3 when the model is ruled out and the round is skipped.
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

TRACK_MODEL=${TRACK_MODEL:-randomwalknowcast}
TRACK_ID=${TRACK_ID:-$(cli hub-id --model "$TRACK_MODEL")}
main_work=$WORK
export WORK="$main_work/track-$TRACK_MODEL"
rm -rf "$WORK" && mkdir -p "$WORK/candidates"
cp -R "$main_work/inputs" "$WORK/inputs"
export MODEL=$TRACK_MODEL MODEL_ID=$TRACK_ID

# Plain random walk is built only as the reference the model's forecast
# is checked against. It is a shadow here and is never selected. Neither
# needs a fit.
for id in randomwalk "$TRACK_MODEL"; do
  label="random walk baseline"
  [ "$id" = randomwalk ] || label="random walk on a nowcast"
  CAND_ID=$id CAND_KIND=randomwalk CAND_CONFIG=$id CAND_MODEL=$id \
    CAND_LABEL=$label CAND_SAMPLER='{}' REQUIRE_ZONES=true \
    "$(dirname "$0")/candidate.sh"
done

cli metadata --model "$TRACK_MODEL" --abbr "${TRACK_ID#*-}" \
  --out "$WORK/metadata"
export METADATA="$WORK/metadata/$TRACK_ID.yml"
status=0
NO_SHADOWS=true ONLY_MODEL=true "$(dirname "$0")/select.sh" || status=$?
if [ "$status" != 0 ]; then
  reason=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['reason'])" \
    "$WORK/gate/provenance.json" 2> /dev/null || echo "see the candidate logs")
  echo "::warning::$TRACK_ID: round skipped. $reason"
  note "**$TRACK_ID: round skipped, nothing submitted.** $TRACK_MODEL did not clear the checks and there is no fallback. $reason"
  exit "$status"
fi
"$(dirname "$0")/submit.sh"
