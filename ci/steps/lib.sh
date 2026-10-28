# shellcheck shell=bash
# Shared settings for the step scripts. Sourced, not run.
#
# The workflow and run_dry.sh call the same steps, so the dry run
# exercises the workflow's logic rather than a copy of it.
#
#   CI_DIR         this ci/ directory (the submission-ci uv project)
#   WORK           where every output of the run goes (default output/ci)
#   FORECAST_CLI   the main repo's CLI (default `uv run bdbv-hub-renewal`)
#   MODEL          main repo model name (default: the latest entry of
#                  results/model-choice.yml, `bdbv-hub-renewal chosen-model`)
#   MODEL_ID       hub model id (default: config/hub.toml,
#                  `bdbv-hub-renewal hub-id`)

set -euo pipefail

CI_DIR=${CI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WORK=${WORK:-output/ci}
FORECAST_CLI=${FORECAST_CLI:-uv run bdbv-hub-renewal}
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

# Before `prepare`: the chosen model and the configured hub id, from the
# main repo. Called by `prepare` and by the gate's metadata step.
resolve_model() {
  [ -n "$MODEL" ] || MODEL=$(cli chosen-model)
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

# The round's window, so fit timeouts and the retry leave the gate time
# before it closes (ci/policy.toml [timing]). Word-split on purpose.
deadline_args() {
  local args="--reference-date ${REFERENCE_DATE:-}"
  [ "${IGNORE_WINDOW:-false}" = true ] && args="$args --ignore-window"
  [ -n "${NOW:-}" ] && args="$args --now $NOW"
  echo "$args"
}

note() {
  echo "$*"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    echo "$*" >> "$GITHUB_STEP_SUMMARY"
  fi
}
