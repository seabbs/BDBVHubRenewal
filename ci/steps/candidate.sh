#!/usr/bin/env bash
# Produce and check one candidate of the fallback chain. One matrix leg.
#
#   CAND_ID, CAND_KIND, CAND_CONFIG, CAND_LABEL, CAND_SAMPLER (JSON)
#   FIT_CACHE_HIT=true  the fit directory was restored from the cache
#
# A bad fit or a failed check does not fail this step: the verdicts are
# written to the candidate's directory and the select step walks the
# chain. Only a broken step (missing inputs) exits non-zero.
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

: "${CAND_ID:?}" "${CAND_KIND:?}" "${CAND_CONFIG:?}"
CAND_SAMPLER=${CAND_SAMPLER:-'{}'}
CAND_LABEL=${CAND_LABEL:-$CAND_ID}
inputs="$WORK/inputs"
[ -f "$inputs/context.json" ] || { echo "::error::run prepare.sh first"; exit 1; }
ctx() { python3 -c "import json,sys; print(json.load(open('$inputs/context.json'))['$1'])"; }
AS_OF=$(ctx as_of)
REFERENCE_DATE=$(ctx reference_date)

out="$WORK/candidates/$CAND_ID"
fit="$WORK/fits/$CAND_ID"
mkdir -p "$out" "$fit"
export CAND_ID  # read by the fixture CLI to pick a scenario

note "### Candidate \`$CAND_ID\`: $CAND_LABEL"

fit_ok=true
if [ "$CAND_KIND" = "persistence" ]; then
  : # no fit
elif [ "${FIT_CACHE_HIT:-false}" = "true" ] && [ -f "$fit/diagnostics.json" ]; then
  note "Fit restored from the cache, not refitted."
else
  start=$SECONDS
  # Heartbeat, as in BVDOutbreakSize's docs.yml: lift the newest
  # progress line into the job log once a minute, only when it moved,
  # so a stalled fit is distinguishable from a slow one.
  (
    last=""
    while sleep "${HEARTBEAT_SECONDS:-60}"; do
      line=$(tail -n 1 "$fit/progress.log" 2> /dev/null) || true
      [ -n "$line" ] && [ "$line" != "$last" ] || continue
      printf '[%3dm] %s\n' "$(( (SECONDS - start) / 60 ))" "$line"
      last="$line"
    done
  ) &
  heartbeat=$!
  trap 'kill "$heartbeat" 2> /dev/null || true' EXIT
  if ! cli fit --model "$MODEL" --config "$CAND_CONFIG" --vintage "$AS_OF" \
      --sampler-json "$CAND_SAMPLER" --out "$fit"; then
    fit_ok=false
    echo "::warning::fit for $CAND_ID exited non-zero"
  fi
  kill "$heartbeat" 2> /dev/null || true
  note "Fit took $(( SECONDS - start )) s."
fi

if [ "$CAND_KIND" = "persistence" ]; then
  cli nowcast --model persistence --vintage "$AS_OF" \
    --out "$out/submission" || echo "::warning::persistence nowcast failed"
elif $fit_ok; then
  cli nowcast --model "$MODEL" --config "$CAND_CONFIG" --fit "$fit" \
    --vintage "$AS_OF" --out "$out/submission" \
    || echo "::warning::nowcast for $CAND_ID failed"
fi
sub=$(find "$out/submission" -name '*.csv' 2> /dev/null | sort | head -n 1)

if [ "$CAND_KIND" != "persistence" ]; then
  cp "$fit/diagnostics.json" "$out/" 2> /dev/null || true
  sci summarise-fit --id "$CAND_ID" --diagnostics "$out/diagnostics.json" \
    --out "$out" --policy "$POLICY" --no-fail
fi

baseline=$(find "$WORK/candidates/persistence/submission" -name '*.csv' \
  2> /dev/null | sort | head -n 1)
checks=(--id "$CAND_ID" --kind "$CAND_KIND"
  --submission "${sub:-$out/submission/missing.csv}"
  --reference-date "$REFERENCE_DATE"
  --observed "$inputs/time-series.csv" --as-of "$AS_OF"
  --tasks "$inputs/tasks.json" --locations "${EXPECTED_LOCATIONS:-}"
  --policy "$POLICY" --out "$out" --no-fail)
if [ -n "$baseline" ]; then
  checks+=(--baseline "$baseline")
fi
if [ -f "$inputs/previous.csv" ]; then
  checks+=(--previous "$inputs/previous.csv")
fi
sci check-forecast "${checks[@]}"

if [ -n "$sub" ]; then
  cli plot-submission --submission "$sub" --vintage "$AS_OF" \
    --out "$out/nowcast.png" || echo "::warning::plot for $CAND_ID failed"
fi

cat > "$out/run.json" << EOF
{"candidate": "$CAND_ID", "kind": "$CAND_KIND", "label": "$CAND_LABEL",
 "model": "$( [ "$CAND_KIND" = persistence ] && echo persistence || echo "$MODEL")",
 "config": "$CAND_CONFIG", "sampler": $CAND_SAMPLER,
 "package_version": "$(cli --version 2> /dev/null || echo unknown)",
 "as_of": "$AS_OF", "reference_date": "$REFERENCE_DATE"}
EOF
