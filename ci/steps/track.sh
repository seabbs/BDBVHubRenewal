#!/usr/bin/env bash
# Check and submit the baseline hub model (baseline) on its own:
# persistence on a nowcast, falling back to plain persistence, under its
# own id, with health zones required. Run after `prepare`, apart from the
# renewal chain.
#
#   TRACK_MODEL   the model (default persistencenow)
#   TRACK_ID      its hub id (default: `bdbvhubrenewal hub-id --model`)
#   As for select.sh and submit.sh: HUB_PATH, VALIDATE, CHECK_WINDOW,
#   DRY_RUN, FORK, HUB_REF, DATA_COMMIT, AS_OF, REFERENCE_DATE.
# Exits 3 when both candidates are ruled out.
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

TRACK_MODEL=${TRACK_MODEL:-persistencenow}
TRACK_ID=${TRACK_ID:-$(cli hub-id --model "$TRACK_MODEL")}
main_work=$WORK
export WORK="$main_work/track-$TRACK_MODEL"
rm -rf "$WORK" && mkdir -p "$WORK/candidates"
cp -R "$main_work/inputs" "$WORK/inputs"
export MODEL=$TRACK_MODEL MODEL_ID=$TRACK_ID

# Both candidates again, here with health zones required: persistence
# first, as the other is checked against it. Neither needs a fit.
for id in persistence "$TRACK_MODEL"; do
  label="persistence baseline"
  [ "$id" = persistence ] || label="persistence on a nowcast"
  CAND_ID=$id CAND_KIND=persistence CAND_CONFIG=$id CAND_MODEL=$id \
    CAND_LABEL=$label CAND_SAMPLER='{}' REQUIRE_ZONES=true \
    "$(dirname "$0")/candidate.sh"
done

cli metadata --model "$TRACK_MODEL" --abbr "${TRACK_ID#*-}" \
  --out "$WORK/metadata"
export METADATA="$WORK/metadata/$TRACK_ID.yml"
status=0
"$(dirname "$0")/select.sh" || status=$?
if [ "$status" != 0 ]; then
  note "$TRACK_ID: no candidate cleared the checks; nothing submitted."
  exit "$status"
fi
"$(dirname "$0")/submit.sh"
