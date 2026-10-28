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
# The data check from prepare, carried with each candidate so the PR
# body can report it.
cp "$inputs/coherence-summary.json" "$out/" 2> /dev/null || true

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
  # Read when JAX starts, so set per candidate, in case a candidate asks
  # for more chains than the runner has vCPUs.
  XLA_FLAGS=$(xla_flags_for "$CAND_SAMPLER")
  export XLA_FLAGS
  note "XLA host devices: ${XLA_FLAGS##*=}, for $(sampler_chains "$CAND_SAMPLER") chains."
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

# Posterior predictive checks, from the ppc.json the fit wrote beside its
# diagnostics. n/a for a model with no posterior; an error for a sampled
# fit that wrote none.
ppc_args=(--id "$CAND_ID" --kind "$CAND_KIND" --ppc "$out/ppc.json"
  --out "$out" --policy "$POLICY" --no-fail)
if [ "$CAND_KIND" != "persistence" ]; then
  cp "$fit/ppc.json" "$out/" 2> /dev/null || true
  ppc_args+=(--diagnostics "$out/diagnostics.json")
fi
sci check-ppc "${ppc_args[@]}"
if python3 -c "import json, sys; sys.exit(not json.load(open(sys.argv[1])).get('available'))" \
    "$out/ppc.json" 2> /dev/null; then
  if cli plot-ppc --ppc "$out/ppc.json" --out "$out/ppc.png"; then
    note "Posterior predictive plot: \`ppc.png\` in the \`candidate-$CAND_ID\` artefact."
  else
    echo "::warning::posterior predictive plot for $CAND_ID failed"
  fi
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
# Written by the nowcast beside the hub file. The gate rebuilds the
# national rows as the sum of the provinces plus its unassigned part, or,
# without it, plus the data's national gap from coherence.json.
checks+=(--national-sidecar "$out/submission/national.json")
if [ -f "$inputs/coherence.json" ]; then
  checks+=(--coherence "$inputs/coherence.json")
fi
sci check-forecast "${checks[@]}"

if [ -n "$sub" ]; then
  cli plot-submission --submission "$sub" --vintage "$AS_OF" \
    --out "$out/nowcast.png" || echo "::warning::plot for $CAND_ID failed"
fi

# A config that runs a stand-in model (for example persistence in place
# of the model) says so in its diagnostics or fit.json.
stand_in=$(python3 - "$out/diagnostics.json" "$fit/fit.json" << 'PY'
import json, sys
for path in sys.argv[1:]:
    try:
        value = json.load(open(path)).get("stand_in")
    except Exception:
        continue
    if value:
        print(json.dumps(value))
        break
else:
    print("null")
PY
)
if [ "$stand_in" != null ]; then
  note "**Stand-in:** $(python3 -c 'import json, sys; print(json.loads(sys.argv[1]))' "$stand_in")"
fi

cat > "$out/run.json" << EOF
{"candidate": "$CAND_ID", "kind": "$CAND_KIND", "label": "$CAND_LABEL",
 "model": "$( [ "$CAND_KIND" = persistence ] && echo persistence || echo "$MODEL")",
 "config": "$CAND_CONFIG", "sampler": $CAND_SAMPLER,
 "package_version": "$(cli --version 2> /dev/null || echo unknown)",
 "as_of": "$AS_OF", "reference_date": "$REFERENCE_DATE",
 "stand_in": $stand_in}
EOF
