#!/usr/bin/env bash
# Run the paper's evaluation workflow and render the paper into the docs.
#
#   scripts/paper.sh [development|paper]
#
# The one command for the paper, run by `task paper` and by the paper
# step of .github/workflows/pages.yml, so a local run and CI do the same.
# `development` (the default) never reads the held-out data; `paper`
# needs it (paper/README.md). CORES sets Snakemake's cores (default 4).
#
# The Ebola study reads our stored fits from the fit store's history, so
# the fit-store branch is fetched first. Snakemake then runs the target's
# missing steps (`--rerun-triggers mtime` reuses the backtest forecasts
# CI has already made) and renders main.qmd and si.qmd, which are copied
# into docs/paper/ for the site. The chronos group is installed too, as
# the German rule runs the Chronos-2 models. Needs uv, git and Quarto,
# and a GitHub token (GITHUB_TOKEN, GH_TOKEN or a gh login).
set -euo pipefail

target=${1:-development}
case $target in
  development | paper) ;;
  *)
    echo "Unknown target '$target': give development or paper." >&2
    exit 2
    ;;
esac
cd "$(dirname "$0")/.."

# The data steps call the GitHub API, which rate limits anonymous calls.
# CI sets GITHUB_TOKEN; locally take GH_TOKEN or gh's login.
if [ -z "${GITHUB_TOKEN:-}" ]; then
  GITHUB_TOKEN=${GH_TOKEN:-$(gh auth token 2>/dev/null || true)}
  export GITHUB_TOKEN
fi

git fetch -q --no-tags origin \
  +refs/heads/fit-store:refs/remotes/origin/fit-store

cd paper
uv run --group paper --group chronos snakemake -s workflow/Snakefile render \
  --config render="$target" --cores "${CORES:-4}" --resources heavy=1 \
  --rerun-triggers mtime

mkdir -p ../docs/paper/figures ../docs/paper/build
cp -R main.html main.pdf main_files si.html si.pdf si_files ../docs/paper/
cp figures/*.svg figures/*.pdf ../docs/paper/figures/
# Image inputs, such as a screenshot, when there are any.
find figures -maxdepth 1 -name "*.png" -exec cp {} ../docs/paper/figures/ \;
cp -R build/figures ../docs/paper/build/
