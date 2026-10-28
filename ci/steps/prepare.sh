#!/usr/bin/env bash
# Pin the inputs every candidate shares, then build the baseline
# candidates, which need no fit: persistence first, since every other
# candidate is checked against it, then persistence on a nowcast.
#
#   HUB_REF         hub commit to read the data at (from the data check)
#   AS_OF           data vintage
#   REFERENCE_DATE  round
#   OBSERVED        optional local time-series.csv (skips the download)
#   TASKS           optional local tasks.json (skips the download)
#   PREVIOUS        optional last week's submission file
#   SKIP_DATA=true  do not run `bdbvhubrenewal data` (fixtures)
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"
resolve_model

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

# The BVDOutbreakSize release this vintage may use, cached in the data
# directory so it travels with the inputs. Only the shadow `bvd` candidate
# reads it; a failure leaves that variant as the main model.
cli bvd --vintages "$AS_OF" \
  || echo "::warning::could not cache the BVDOutbreakSize release"

cat > "$inputs/context.json" << EOF
{"hub_ref": "$HUB_REF", "as_of": "$AS_OF",
 "reference_date": "$REFERENCE_DATE", "model": "$MODEL",
 "model_id": "$MODEL_ID"}
EOF

# Whether the national data add up to the provinces. A data check that
# only warns: it is reported once, in the job summary and the PR body.
if ! cli coherence --vintage "$AS_OF" --out "$inputs/coherence.json"; then
  echo "::warning::bdbvhubrenewal coherence failed; data coherence is n/a"
  rm -f "$inputs/coherence.json"
fi
sci check-coherence --coherence "$inputs/coherence.json" --out "$inputs"

sci python -m submission_ci.policy --policy "$POLICY" --stage baseline \
  --hub-ref "$HUB_REF" --model "$MODEL" --root "${SRC_ROOT:-.}" \
  > "$WORK/matrix-baseline.json"
python3 - "$WORK/matrix-baseline.json" << 'EOF' > "$WORK/legs-baseline.txt"
import json, sys
legs = json.load(open(sys.argv[1]))["include"]
for e in sorted(legs, key=lambda e: e["id"] != "persistence"):
    print("\t".join([e["id"], e["kind"], e["config"], e["label"],
                     e["model"] or "-"]))
EOF
while IFS=$'\t' read -r id kind config label cmodel; do
  [ "$cmodel" = - ] && cmodel=""
  CAND_ID=$id CAND_KIND=$kind CAND_CONFIG=$config CAND_LABEL=$label \
    CAND_MODEL=$cmodel CAND_SAMPLER='{}' "$(dirname "$0")/candidate.sh"
done < "$WORK/legs-baseline.txt"
