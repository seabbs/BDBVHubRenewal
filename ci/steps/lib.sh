# shellcheck shell=bash
# Shared settings for the step scripts. Sourced, not run.
#
# The workflow and run_dry.sh call the same steps, so the dry run
# exercises the workflow's logic rather than a copy of it.
#
#   CI_DIR         this ci/ directory (the submission-ci uv project)
#   WORK           where every output of the run goes (default output/ci)
#   FORECAST_CLI   the main repo's CLI (default `uv run bdbvhubrenewal`)
#   MODEL          main repo model name (default: `[submission] model` in
#                  config/hub.toml)
#   MODEL_ID       hub model id (default: config/hub.toml,
#                  `bdbvhubrenewal hub-id`)

set -euo pipefail

CI_DIR=${CI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WORK=${WORK:-output/ci}
FORECAST_CLI=${FORECAST_CLI:-uv run bdbvhubrenewal}
MODEL=${MODEL:-}
# The main repo's CLI reads and writes the tidy data here, so the data
# the prepare job pins travel with the `inputs` artefact.
export BDBV_DATA_DIR=${BDBV_DATA_DIR:-$WORK/inputs/data}
export BDBV_RAW_DIR=${BDBV_RAW_DIR:-$WORK/raw}
MODEL_ID=${MODEL_ID:-}
POLICY=${POLICY:-$CI_DIR/policy.toml}
HUB=${HUB:-InsightNet-US/BDBV-Modeling-Hub}

# The submission-ci tools, run from their own small environment.
sci() {
  uv run --quiet --project "$CI_DIR" "$@"
}

# Run the main repo's CLI (word-split on purpose), outside the
# submission-ci environment that `sci` may have activated.
cli() {
  # shellcheck disable=SC2086
  env -u VIRTUAL_ENV $FORECAST_CLI "$@"
}

# After `prepare`, MODEL and MODEL_ID come from the run's context.json,
# so every job of a run uses the values `prepare` resolved.
ctx_value() {
  python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" \
    "$WORK/inputs/context.json" "$1" 2> /dev/null || true
}
[ -n "$MODEL" ] || MODEL=$(ctx_value model)
[ -n "$MODEL_ID" ] || MODEL_ID=$(ctx_value model_id)

# Resubmission in the same forecast week: if the hub's main branch already
# has this model's file for the round, rewrite $1 as that file unchanged
# plus its rows, with one new forecast_date (the hub's rule). The hub's
# copy is read only, from raw GitHub, or from PREVIOUS_CHECKOUT (a local
# copy of the hub, for the dry run). Sets PREVIOUS_FILE when there is one
# and fails when the rows cannot be appended.
#   MODEL_ID, REFERENCE_DATE
append_previous() {
  local csv=$1 prev="$WORK/hub-previous.csv" args found
  PREVIOUS_FILE=""
  rm -f "$prev"
  args=(--model-id "$MODEL_ID" --reference-date "$REFERENCE_DATE" --out "$prev")
  if [ -n "${PREVIOUS_CHECKOUT:-}" ]; then
    args+=(--checkout "$PREVIOUS_CHECKOUT")
  fi
  found=$(sci python -m submission_ci.forecast_date fetch "${args[@]}")
  [ "$found" = found ] || return 0
  sci python -m submission_ci.forecast_date append --previous "$prev" \
    --new "$csv" --out "$csv.appended"
  mv "$csv.appended" "$csv"
  PREVIOUS_FILE=$prev
  export PREVIOUS_FILE
}

# Before `prepare`: the chosen model and the configured hub id, from the
# main repo. Called by `prepare` and by the gate's metadata step.
resolve_model() {
  # The renewal id's model: `[submission] model` in config/hub.toml.
  [ -n "$MODEL" ] || MODEL=$(python3 -c 'import sys, tomllib
print(tomllib.load(open(sys.argv[1], "rb"))["submission"]["model"])' \
    "${HUB_TOML:-$CI_DIR/../config/hub.toml}")
  [ -n "$MODEL_ID" ] || MODEL_ID=$(cli hub-id)
}

# The number of chains in the sampler settings $1 (JSON; default 4).
sampler_chains() {
  python3 -c 'import json, sys
print(int(json.loads(sys.argv[1] or "{}").get("num_chains", 4)))' "${1:-}"
}

# XLA_FLAGS for a fit with the sampler settings $1: one XLA host device
# per chain, and at least one per CPU (CPUS overrides the count). NumPyro runs chains in parallel
# only when there are as many devices as chains; otherwise it runs them
# one after another. Other flags in XLA_FLAGS are kept.
xla_flags_for() {
  local chains cpus n rest
  chains=$(sampler_chains "${1:-}")
  cpus=${CPUS:-$(nproc 2> /dev/null || sysctl -n hw.ncpu 2> /dev/null || echo 1)}
  n=$(( chains > cpus ? chains : cpus ))
  rest=$(printf '%s' "${XLA_FLAGS:-}" \
    | sed -E 's/--xla_force_host_platform_device_count=[0-9]+//g; s/^ +//; s/ +$//')
  echo "${rest:+$rest }--xla_force_host_platform_device_count=$n"
}

# The round's window, so fit timeouts leave the gate time
# before it closes (ci/policy.toml [timing]). Word-split on purpose.
deadline_args() {
  local args="--reference-date ${REFERENCE_DATE:-}"
  [ "${IGNORE_WINDOW:-false}" = true ] && args="$args --ignore-window"
  [ -n "${NOW:-}" ] && args="$args --now $NOW"
  echo "$args"
}

# Check out the ledger branch (LEDGER_BRANCH, default submission-ledger)
# as a worktree at $1, as a new orphan branch when there is none yet.
ledger_worktree() {
  local dir=$1 branch=${LEDGER_BRANCH:-submission-ledger}
  rm -rf "$dir"
  git worktree prune
  if git fetch -q --depth 1 origin "$branch" 2> /dev/null; then
    git worktree add -q -B "$branch" "$dir" FETCH_HEAD
  else
    git worktree add -q --orphan -b "$branch" "$dir"
  fi
}

# Commit everything in the ledger worktree $1 with message $2 and push it.
ledger_push() {
  git -C "$1" add -A
  git -C "$1" -c user.name="github-actions[bot]" \
    -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
    commit -q -m "$2"
  git -C "$1" push -q origin "${LEDGER_BRANCH:-submission-ledger}"
}

note() {
  echo "$*"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    echo "$*" >> "$GITHUB_STEP_SUMMARY"
  fi
}
