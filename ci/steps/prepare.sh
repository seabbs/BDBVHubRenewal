#!/usr/bin/env bash
# Pin the inputs every candidate shares, then build the persistence
# candidate, which needs no fit and is also the baseline the other
# candidates are checked against.
#
#   HUB_REF         hub commit to read the data at (from the data check)
#   AS_OF           data vintage
#   REFERENCE_DATE  round
#   OBSERVED        optional local time-series.csv (skips the download)
#   TASKS           optional local tasks.json (skips the download)
#   PREVIOUS        optional last week's submission file
#   SKIP_DATA=true  do not run `bdbv-forecasts data` (fixtures)
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

: "${HUB_REF:?}" "${AS_OF:?}" "${REFERENCE_DATE:?}"
inputs="$WORK/inputs"
mkdir -p "$inputs"
raw="https://raw.githubusercontent.com/$HUB/$HUB_REF"

if [ "${SKIP_DATA:-false}" != "true" ]; then
  cli data --ref "$HUB_REF"
fi
if [ -n "${OBSERVED:-}" ]; then
  cp "$OBSERVED" "$inputs/time-series.csv"
else
  curl -fsSL --retry 3 "$raw/target-data/time-series.csv" \
    -o "$inputs/time-series.csv"
fi
if [ -n "${TASKS:-}" ]; then
  cp "$TASKS" "$inputs/tasks.json"
else
  curl -fsSL --retry 3 "$raw/hub-config/tasks.json" -o "$inputs/tasks.json"
fi

# Last week's submission, for the consistency check: the newest file of
# this model on the hub before this round. A round whose PR has not been
# merged yet is not on the hub; the check then compares with the one
# before it, or skips.
if [ -n "${PREVIOUS:-}" ]; then
  cp "$PREVIOUS" "$inputs/previous.csv"
elif command -v gh > /dev/null; then
  # gh prints the error body on stdout when the model has no folder yet
  # (a 404), so keep the answer only if the call worked and it is a file.
  prev=""
  if out=$(gh api "repos/$HUB/contents/model-output/$MODEL_ID?ref=$HUB_REF" \
    --jq "[.[] | select(.name | endswith(\".csv\")) |
      select(.name < \"$REFERENCE_DATE\") | .name] | sort | last // empty" \
    2> /dev/null); then
    prev=$out
  fi
  if [[ "$prev" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}-.*\.csv$ ]]; then
    curl -fsSL --retry 3 "$raw/model-output/$MODEL_ID/$prev" \
      -o "$inputs/previous.csv"
  fi
fi

cat > "$inputs/context.json" << EOF
{"hub_ref": "$HUB_REF", "as_of": "$AS_OF",
 "reference_date": "$REFERENCE_DATE", "model": "$MODEL",
 "model_id": "$MODEL_ID"}
EOF

# Whether the national data add up to the provinces. A data check that
# only warns: it is reported once, in the job summary and the PR body.
if ! cli coherence --vintage "$AS_OF" --out "$inputs/coherence.json"; then
  echo "::warning::bdbv-forecasts coherence failed; data coherence is n/a"
  rm -f "$inputs/coherence.json"
fi
sci check-coherence --coherence "$inputs/coherence.json" --out "$inputs"

CAND_ID=persistence CAND_KIND=persistence CAND_CONFIG=persistence \
  CAND_LABEL="persistence baseline" CAND_SAMPLER='{}' \
  "$(dirname "$0")/candidate.sh"
