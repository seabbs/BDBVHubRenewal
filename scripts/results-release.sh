#!/usr/bin/env bash
# Results releases: every model's results with each hub release and each
# code release, as the Pages build makes them.
#
#   scripts/results-release.sh tag
#       print `tag=<name>` for the release this build should make, or
#       `tag=` for none. A published code release vX.Y.Z gives
#       results-vX.Y.Z. A build of main gives results-<hub release> once
#       every stored fit at that release is of this code
#       (results/fit-store.json), unless that release exists.
#   scripts/results-release.sh collect DIR
#       copy the build's results and a zip of the data into DIR
#   scripts/results-release.sh publish TAG DIR SITE
#       zip the built site SITE into DIR and publish DIR as release TAG
#
# Needs GH_TOKEN (and GITHUB_REPOSITORY, GITHUB_SHA) for `tag` and
# `publish`. Uses git, gh, zip and Python's standard library only.
set -euo pipefail

repo=${GITHUB_REPOSITORY:-seabbs/BDBVHubRenewal}
site_url="https://samabbott.co.uk/BDBVHubRenewal/"

case "$1" in
  tag)
    if [ "${GITHUB_EVENT_NAME:-}" = release ]; then
      case "${GITHUB_REF_NAME:-}" in
        v*) echo "tag=results-$GITHUB_REF_NAME" ;;
        *) echo "tag=" ;;
      esac
      exit 0
    fi
    origin=$(python3 -I - results/fit-store.json << 'EOF'
import json
import sys

status = json.load(open(sys.argv[1]))
latest = max(s["origin"] for s in status)
ready = all(s["current"] for s in status if s["origin"] == latest)
print(latest if ready else "")
EOF
    )
    if [ -z "$origin" ] || gh release view "results-$origin" -R "$repo" > /dev/null 2>&1; then
      echo "tag="
    else
      echo "tag=results-$origin"
    fi
    ;;
  collect)
    dir=$2
    mkdir -p "$dir"
    cp results/*.csv results/*.csv.gz results/*.json "$dir"/
    cp -R results/hub "$dir"/hub
    cp -R results/hub-comparison "$dir"/hub-comparison
    (cd data && zip -qr - . -x 'raw/*') > "$dir/data.zip"
    (cd "$dir" && zip -qr hub.zip hub hub-comparison && rm -rf hub hub-comparison)
    ;;
  publish)
    tag=$2 dir=$3 site=$4
    stage=$(mktemp -d "${TMPDIR:-/tmp}/site.XXXXXX")
    cp -R "$site" "$stage/BDBVHubRenewal"
    printf '%s\n' \
      "Run \`python3 -m http.server\` in this folder and open" \
      "http://localhost:8000/BDBVHubRenewal/" > "$stage/README.txt"
    (cd "$stage" && zip -qr - README.txt BDBVHubRenewal) > "$dir/site.zip"
    hub_release=$(python3 -I -c "import json; print(max(s['origin'] for s in json.load(open('$dir/fit-store.json'))))")
    notes="$dir/../notes.md"
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
    gh release create "$tag" "$dir"/* -R "$repo" --target "$GITHUB_SHA" \
      --title "$tag ($(echo "$GITHUB_SHA" | cut -c1-7))" \
      --notes-file "$notes" --latest=false --draft
    gh release edit "$tag" -R "$repo" --draft=false --latest=false
    ;;
  *)
    echo "usage: $0 tag | collect DIR | publish TAG DIR SITE" >&2
    exit 2
    ;;
esac
