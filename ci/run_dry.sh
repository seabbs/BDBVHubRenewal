#!/usr/bin/env bash
# Local dry run of the submission workflow, end to end, on fixture data.
#
#   ci/run_dry.sh              every scenario, with assertions
#   ci/run_dry.sh retry        one scenario, printing its PR body
#
# It runs the same step scripts as .github/workflows/submit.yml, in the
# same order: detect -> prepare (persistence) -> primary candidates ->
# triage -> retries -> select -> submit (DRY_RUN=true). The model is the
# fixture CLI (fixtures/fake_bdbv_hub_renewal.py) and `gh`/`git` are never
# called for writing. Outputs go to a new $TMPDIR/submission-ci-dry.XXXXXX/
# <scenario>/ for each run, so concurrent dry runs never share files.
set -euo pipefail

CI_DIR=$(cd "$(dirname "$0")" && pwd)
export CI_DIR
FIX="$CI_DIR/fixtures"
tmp=${TMPDIR:-/tmp}
ROOT=$(mktemp -d "${tmp%/}/submission-ci-dry.XXXXXX")
echo "Dry-run outputs: $ROOT"
export UV_CACHE_DIR=${UV_CACHE_DIR:-${TMPDIR:-/tmp}/uvcache}
export FORECAST_CLI="python3 $FIX/fake_bdbv_hub_renewal.py"
# MODEL and MODEL_ID are left unset, so `prepare` resolves them from the
# fixture CLI (`chosen-model`, `hub-id`) as the workflow does.
unset MODEL MODEL_ID
export HEARTBEAT_SECONDS=1
# A GitHub-hosted runner's 4 vCPUs. The fixture fit fails, as a
# sequential run would be too slow, when there are fewer XLA devices
# than chains.
export CPUS=4
# Monday 2 November 2026, 10:00 UTC: inside the window for 2026-11-07.
NOW=${NOW:-2026-11-02T10:00:00+00:00}

# name | FAKE_SCENARIOS | expected selection ("skip" for a skipped round)
#   | extra environment (optional) | text the PR body must hold (optional)
SCENARIOS=(
  "main|main=pass|main||All checks passed."
  "chosenpersistence|main=pass|persistence|FAKE_CHOSEN_MODEL=persistence|Shadow candidates"
  "coherence|main=pass|main|FAKE_COHERENCE_GAP=2026-10-24:2|Handled: national (CD) differs from the sum of the provinces"
  "warn|main=warn|main"
  "standin|main=standin|main||Stand-in: "
  "ppcwarn|main=ppcwarn|main||All checks passed, with warnings on ppc."
  "ppcfail|main=ppcfail|main||All checks passed, with warnings on ppc."
  "noppc|main=noppc|main||All checks passed, with warnings on ppc."
  "implausible|main=implausible|main||with warnings on"
  "national|main=badnational|main||Handled: national (CD) rebuilt as the sum of the provinces"
  "latestream|main=latestream|main||Handled: delays: no delay data for this vintage"
  "loose|main=loose,retry-1=pass|retry-1||Handled: main model refitted once (max R-hat is 1.2"
  "loosekept|main=loose,retry-1=fail|main||Handled: main model kept after one refit did not help"
  "retry|main=fail,retry-1=pass|retry-1||Main model refitted once: main model was ruled out"
  "fallback|main=fail,retry-1=fail|fallback||Fallback model used ("
  # Tuesday 19:00 UTC, four hours before the close: no time to refit, so
  # a failed main model goes straight to the fallback.
  "late|main=fail,retry-1=pass|fallback|NOW=2026-11-03T19:00:00+00:00|Fallback model used ("
  "persistence|main=fail,retry-1=crash,fallback=fail|persistence"
  "skip|main=fail,retry-1=fail,fallback=fail,persistence=negative|skip"
)

sci() { uv run --quiet --project "$CI_DIR" "$@"; }

# A stand-in `gh` that records calls and answers nothing, so the dry run
# never touches GitHub. The submit step must only have looked up PRs.
mkdir -p "$ROOT/bin"
# shellcheck disable=SC2016  # expanded when the stub runs
printf '#!/usr/bin/env bash\necho "$*" >> "${GH_LOG:?}"\nexit 1\n' \
  > "$ROOT/bin/gh"
chmod +x "$ROOT/bin/gh"
export PATH="$ROOT/bin:$PATH"

# A stand-in main repo tree for the fit cache key (policy [cache] inputs).
SRC_ROOT="$ROOT/main-repo"
mkdir -p "$SRC_ROOT/src" "$SRC_ROOT/config"
echo "model = 1" > "$SRC_ROOT/src/model.py"
echo "x = 1" > "$SRC_ROOT/config/model.toml"
touch "$SRC_ROOT/pyproject.toml" "$SRC_ROOT/uv.lock"

run_scenario() {
  local name=$1 scenarios=$2 expect=$3 extra=${4:-} want=${5:-}
  # A scenario may move the clock; the next one gets it back.
  local NOW=$NOW
  export WORK="$ROOT/$name"
  export FAKE_SCENARIOS=$scenarios
  export FAKE_COHERENCE_GAP="" FAKE_CHOSEN_MODEL=""
  local kv
  for kv in $extra; do export "${kv?}"; done
  export GH_LOG="$WORK/gh.log"
  export GITHUB_OUTPUT="$WORK/github-output" GITHUB_STEP_SUMMARY="$WORK/summary.md"
  rm -rf "$WORK" && mkdir -p "$WORK"
  echo "=== $name ($scenarios) ==="

  # 1. Data check, as in the `detect` job.
  sci detect-new-vintage detect --state "$FIX/hub/state.json" \
    --fixture "$FIX/hub" --now "$NOW" --out "$WORK/decision.json" > /dev/null
  get() { python3 -c "import json;print(json.load(open('$WORK/decision.json'))['$1'])"; }
  [ "$(get should_run)" = "True" ] || { echo "detect said no: $(get reason)"; return 1; }
  HUB_REF=$(get hub_ref) DATA_COMMIT=$(get data_commit) AS_OF=$(get as_of)
  REFERENCE_DATE=$(get reference_date)
  export HUB_REF DATA_COMMIT AS_OF REFERENCE_DATE

  # 2. Shared inputs and the persistence candidate (`prepare` job).
  SKIP_DATA=true OBSERVED="$FIX/hub/time-series.csv" TASKS="$FIX/hub/tasks.json" \
    PREVIOUS="$FIX/hub/previous.csv" EXPECTED_LOCATIONS="CD,CD-IT,CD-NK,CD-HU" \
    "$CI_DIR/steps/prepare.sh" > "$WORK/prepare.log"

  # The model `prepare` resolved (the latest model choice).
  local model
  model=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['model'])" \
    "$WORK/inputs/context.json")

  # 3. Primary matrix legs (`fit` job), one after another here.
  run_stage() {
    sci python -m submission_ci.policy --policy "$CI_DIR/policy.toml" \
      --stage "$1" --hub-ref "$HUB_REF" --root "$SRC_ROOT" --model "$model" \
      --reference-date "$REFERENCE_DATE" --now "$NOW" > "$WORK/matrix-$1.json"
    python3 - "$WORK/matrix-$1.json" << 'EOF' > "$WORK/legs-$1.txt"
import json, sys
for e in json.load(open(sys.argv[1]))["include"]:
    print("\t".join([e["id"], e["kind"], e["config"], e["label"], e["sampler"]]))
EOF
    while IFS=$'\t' read -r id kind config label sampler; do
      CAND_ID=$id CAND_KIND=$kind CAND_CONFIG=$config CAND_LABEL=$label \
        CAND_SAMPLER=$sampler EXPECTED_LOCATIONS="CD,CD-IT,CD-NK,CD-HU" \
        "$CI_DIR/steps/candidate.sh" > "$WORK/candidate-$id.log" 2>&1
    done < "$WORK/legs-$1.txt"
  }
  run_stage primary

  # 4. Triage, then the retry matrix only if `main` was ruled out.
  local retried=false
  if sci python -m submission_ci.select_candidate triage --model "$model" \
      --policy "$CI_DIR/policy.toml" --candidates "$WORK/candidates" \
      --reference-date "$REFERENCE_DATE" --now "$NOW" \
      | grep -q need_retry=true; then
    run_stage retry
    retried=true
  fi

  # As downloaded by actions/download-artifact: one `candidate-<id>`
  # directory per artefact, laid out again by unpack_candidates.sh.
  for d in "$WORK"/candidates/*; do
    [ "${d##*/}" = persistence ] || mv "$d" "$WORK/candidates/candidate-${d##*/}"
  done
  "$CI_DIR/steps/unpack_candidates.sh" > /dev/null

  # The hub id `prepare` resolved, as the gate's metadata step reads it.
  model_id=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['model_id'])" \
    "$WORK/inputs/context.json")

  # 5. Select, with a stand-in for hubValidations (`gate` job).
  local status=0
  RETRIED=$retried HUB_PATH="$FIX/hub" METADATA="$FIX/hub/$model_id.yml" \
    HUB_VALIDATOR="true" "$CI_DIR/steps/select.sh" > "$WORK/select.log" \
    || status=$?
  local got
  got=$(python3 -c "import json;print(json.load(open('$WORK/gate/provenance.json'))['selected'] or 'skip')")
  if [ "$got" != "$expect" ]; then
    echo "FAIL: $name selected $got, expected $expect"; cat "$WORK/select.log"
    return 1
  fi
  if [ "$expect" = "skip" ]; then
    [ "$status" = 3 ] || { echo "FAIL: skip should exit 3, got $status"; return 1; }
    echo "ok: round skipped (exit 3); reason in $WORK/gate/provenance.md"
    return 0
  fi

  # 6. Submit, dry run: build the body and print the planned PR.
  DRY_RUN=true FORK=seabbs-bot/BDBV-Modeling-Hub METADATA="$FIX/hub/$model_id.yml" MODEL_ID="$model_id" \
    "$CI_DIR/steps/submit.sh" > "$WORK/submit.log" 2>&1 \
    || { echo "FAIL: submit"; cat "$WORK/submit.log"; return 1; }
  grep -q '"action": "dry-run"' "$WORK/submit.log" \
    || { echo "FAIL: no dry-run outcome"; cat "$WORK/submit.log"; return 1; }
  if grep -v '^pr list' "$WORK/gh.log" 2> /dev/null; then
    echo "FAIL: dry run made gh calls other than a PR lookup"; return 1
  fi
  # Short enough to scan: at most six visible lines outside <details>,
  # the docs link for the round, and the footer last.
  local shown
  shown=$(python3 - "$WORK/gate/pr-body.md" << 'EOF'
import sys
depth = n = 0
for line in open(sys.argv[1]):
    depth += line.count("<details>") - line.count("</details>")
    if "<details>" in line or "</details>" in line:
        continue
    if depth == 0 and line.strip() and not line.startswith("<!--"):
        n += 1
print(n)
EOF
)
  [ "$shown" -le 6 ] || { echo "FAIL: $name PR body shows $shown lines"; return 1; }
  grep -q "^This round in our docs: .*/submission/rounds/$REFERENCE_DATE/" \
    "$WORK/gate/pr-body.md" || { echo "FAIL: $name has no docs link"; return 1; }
  tail -n 1 "$WORK/gate/pr-body.md" | grep -q "@seabbs"
  # A posterior predictive plot whenever the checks were available.
  if python3 -c "import json, sys; sys.exit(not json.load(open(sys.argv[1])).get('available'))" \
      "$WORK/gate/selected/ppc.json" 2> /dev/null \
      && [ ! -f "$WORK/gate/selected/ppc.png" ]; then
    echo "FAIL: $name has no ppc.png"; return 1
  fi
  if [ -n "$want" ] && ! grep -qF -- "$want" "$WORK/gate/pr-body.md"; then
    echo "FAIL: $name PR body lacks: $want"; return 1
  fi
  # Every fit job's timeout leaves the gate 45 minutes before the close:
  # 350 on Monday morning, 240 - 45 = 195 at 19:00 UTC on Tuesday.
  local timeout
  timeout=$(python3 -c "import json, sys; print(sorted({e['timeout'] for e in json.load(open(sys.argv[1]))['include']}))" \
    "$WORK/matrix-primary.json")
  case "$name" in
    late) [ "$timeout" = "['195']" ] && [ "$retried" = false ] \
      || { echo "FAIL: late timeouts $timeout, retried $retried"; return 1; } ;;
    *) [ "$timeout" = "['350']" ] \
      || { echo "FAIL: $name timeouts $timeout"; return 1; } ;;
  esac
  echo "ok: selected $got; body $WORK/gate/pr-body.md"
}

ledger() {
  # record.sh against a throwaway repo with a local `origin`: the round
  # from the `retry` scenario is written to the ledger branch, twice.
  local repo="$ROOT/ledger-repo" origin="$ROOT/ledger-origin.git"
  rm -rf "$repo" "$origin"
  git init -q --bare "$origin"
  git init -q -b main "$repo"
  git -C "$repo" remote add origin "$origin"
  for run in 101 102; do
    (cd "$repo" && WORK="$ROOT/retry" GITHUB_RUN_ID=$run \
      REFERENCE_DATE=2026-11-07 DATA_COMMIT=dddd AS_OF=2026-10-31 \
      ACTION=created PR_URL=https://example/pr/1 PRODUCED_BY="retry 1" \
      "$CI_DIR/steps/record.sh" > /dev/null)
  done
  local files
  files=$(git -C "$origin" ls-tree -r --name-only submission-ledger)
  echo "$files" | grep -q "archive/2026-11-07/102/provenance.json" \
    || { echo "FAIL: ledger archive"; echo "$files"; return 1; }
  git -C "$origin" show submission-ledger:state.json \
    | python3 -c "import json,sys; r=json.load(sys.stdin)['rounds']['2026-11-07']; assert r['status']=='submitted' and len(r['history'])==2"
  echo "=== ledger ===" && echo "ok: state.json and archive/2026-11-07/{101,102} on submission-ledger"
}

no_new_data() {
  # A round that already ran on today's data must not run again.
  export WORK="$ROOT/no-new-data"
  rm -rf "$WORK" && mkdir -p "$WORK"
  sci detect-new-vintage record --state "$WORK/state.json" \
    --reference-date 2026-11-07 --data-commit "$(printf 'd%.0s' {1..40})" \
    --as-of 2026-10-31 --status submitted
  GITHUB_OUTPUT="$WORK/out" sci detect-new-vintage detect \
    --state "$WORK/state.json" --fixture "$FIX/hub" --now "$NOW" > /dev/null
  grep -q "should_run=false" "$WORK/out" \
    || { echo "FAIL: no-new-data ran"; return 1; }
  echo "=== no-new-data ===" && echo "ok: $(grep '^reason=' "$WORK/out")"
}

failed=0
if [ $# -gt 0 ]; then
  for s in "${SCENARIOS[@]}"; do
    IFS='|' read -r name sc expect extra want <<< "$s"
    if [ "$name" = "$1" ]; then
      run_scenario "$name" "$sc" "$expect" "$extra" "$want" || failed=1
      [ -f "$ROOT/$name/gate/pr-body.md" ] && cat "$ROOT/$name/gate/pr-body.md"
    fi
  done
  exit $failed
fi
for s in "${SCENARIOS[@]}"; do
  IFS='|' read -r name sc expect extra want <<< "$s"
  run_scenario "$name" "$sc" "$expect" "$extra" "$want" || failed=1
done
no_new_data || failed=1
ledger || failed=1
[ $failed = 0 ] && echo "All dry-run scenarios passed." || echo "Some scenarios FAILED."
exit $failed
