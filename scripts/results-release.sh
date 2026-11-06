#!/usr/bin/env bash
# Results releases: every model's results with each hub release and each
# code release, from what the Pages build made.
#
#   scripts/results-release.sh tag RESULTS
#       print `tag=<name>` for the release to make, or `tag=` for none.
#       A published code release vX.Y.Z gives results-vX.Y.Z. A build of
#       main gives results-<hub release> once every stored fit at that
#       release is of this code (RESULTS/fit-store.json). A tag whose
#       release is already published gives none.
#   scripts/results-release.sh collect BUILD DIR
#       copy BUILD/results and a zip of BUILD/data into DIR
#   scripts/results-release.sh publish TAG DIR SITE
#       zip the built site SITE into DIR and publish DIR as release TAG,
#       replacing a draft left by a failed attempt
#
# Needs GH_TOKEN (and GITHUB_REPOSITORY, GITHUB_SHA) for `tag` and
# `publish`. Uses git, gh, zip and Python's standard library only.
set -euo pipefail

repo=${GITHUB_REPOSITORY:-seabbs/BDBVHubRenewal}
site_url="https://samabbott.co.uk/BDBVHubRenewal/"

# Whether release $1 is published: true, false (a draft) or none.
release_state() {
  gh release view "$1" -R "$repo" --json isDraft \
    -q 'if .isDraft then "false" else "true" end' 2> /dev/null || echo none
}

case "$1" in
  tag)
    results=$2
    if [ "${GITHUB_EVENT_NAME:-}" = release ]; then
      case "${GITHUB_REF_NAME:-}" in
        v*) tag="results-$GITHUB_REF_NAME" ;;
        *) tag="" ;;
      esac
    else
      origin=$(python3 -I - "$results/fit-store.json" << 'EOF'
import json
import sys

status = json.load(open(sys.argv[1]))
latest = max(s["origin"] for s in status)
ready = all(s["current"] for s in status if s["origin"] == latest)
print(latest if ready else "")
EOF
      )
      tag=${origin:+results-$origin}
    fi
    if [ -n "$tag" ] && [ "$(release_state "$tag")" = true ]; then
      tag=""
    fi
    echo "tag=$tag"
    ;;
  collect)
    build=$2 dir=$3
    mkdir -p "$dir"
    cp "$build"/results/*.csv "$build"/results/*.csv.gz \
      "$build"/results/*.json "$dir"/
    (cd "$build/results" && zip -qr - hub hub-comparison) > "$dir/hub.zip"
    (cd "$build/data" && zip -qr - .) > "$dir/data.zip"
    ;;
  publish)
    tag=$2 dir=$3 site=$4
    stage=$(mktemp -d "${TMPDIR:-/tmp}/site.XXXXXX")
    cp -R "$site" "$stage/BDBVHubRenewal"
    printf '%s\n' \
      "Run \`python3 -m http.server\` in this folder and open" \
      "http://localhost:8000/BDBVHubRenewal/" > "$stage/README.txt"
    (cd "$stage" && zip -qr - README.txt BDBVHubRenewal) > "$dir/site.zip"
    hub_release=$(python3 -I -c "import json, sys; print(max(s['origin'] for s in json.load(open(sys.argv[1]))))" "$dir/fit-store.json")
    notes="$stage/notes.md"
    cat > "$notes" << EOF
Results of every model at hub release $hub_release, built from commit $GITHUB_SHA.

- \`hub.zip\`: each model's forecast at the latest release in the hub's format, samples and quantiles for the country, provinces and health zones (\`hub/<model>-samples.csv.gz\`, \`hub/<model>-quantiles.csv.gz\`), and the scored hub comparison (\`hub-comparison/\`).
- \`forecasts.csv.gz\`, \`heldout-forecasts.csv.gz\`: every model's forecasts at each real origin.
- \`*-scores.csv\`, \`*-energy.csv\`, \`nowcast-*.csv\`, \`zone-*.csv\`: the scores behind the Backtest, Nowcast and Comparison pages.
- \`backtest-fits.csv\`, \`delay-summary.csv\`, \`fit-store.json\`: the stored fits' diagnostics and learnt delays, and the fit behind each model and origin.
- \`data.zip\`: the tidy hub data the build used.
- \`site.zip\`: the rendered site; see \`README.txt\` inside to view it offline.

Site: $site_url
EOF
    # A draft until every file is uploaded, so nobody sees half a release.
    if [ "$(release_state "$tag")" = false ]; then
      gh release delete "$tag" -R "$repo" --yes
    fi
    gh release create "$tag" "$dir"/* -R "$repo" --target "$GITHUB_SHA" \
      --title "$tag ($(echo "$GITHUB_SHA" | cut -c1-7))" \
      --notes-file "$notes" --latest=false --draft
    gh release edit "$tag" -R "$repo" --draft=false --latest=false
    ;;
  *)
    echo "usage: $0 tag RESULTS | collect BUILD DIR | publish TAG DIR SITE" >&2
    exit 2
    ;;
esac
