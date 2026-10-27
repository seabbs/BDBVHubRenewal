# Shared settings for the step scripts. Sourced, not run.
#
# The workflow and run_dry.sh call the same steps, so the dry run
# exercises the workflow's logic rather than a copy of it.
#
#   CI_DIR         this ci/ directory (the submission-ci uv project)
#   WORK           where every output of the run goes (default output/ci)
#   FORECAST_CLI   the main repo's CLI (default `uv run bdbv-forecasts`)
#   MODEL          main repo model name (default renewal)
#   MODEL_ID       hub model id (default epiforecasts-$MODEL)

set -euo pipefail

CI_DIR=${CI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WORK=${WORK:-output/ci}
FORECAST_CLI=${FORECAST_CLI:-uv run bdbv-forecasts}
MODEL=${MODEL:-renewal}
MODEL_ID=${MODEL_ID:-epiforecasts-$MODEL}
POLICY=${POLICY:-$CI_DIR/policy.toml}
HUB=${HUB:-InsightNet-US/BDBV-Modeling-Hub}

# The submission-ci tools, run from their own small environment.
sci() {
  uv run --quiet --project "$CI_DIR" "$@"
}

# Run the main repo's CLI (word-split on purpose).
cli() {
  # shellcheck disable=SC2086
  $FORECAST_CLI "$@"
}

note() {
  echo "$*"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    echo "$*" >> "$GITHUB_STEP_SUMMARY"
  fi
}
