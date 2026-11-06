#!/usr/bin/env bash
# Attach every model's results to the round releases.
#
#   scripts/results-release.sh attach BUILD SITE
#
# submit.yml's `release` job publishes `round-<reference date>` from the
# commit the round was submitted from. This adds the Pages build's
# results to each such release whose round used the build's latest hub
# release, once every stored fit at that release is of this code
# (BUILD/results/fit-store.json): the hub tables, the results tables, a
# zip of BUILD/data and the built site SITE as site.zip. site.zip goes
# last, so a release with it is complete; one without it is filled again
# by the next build.
#
# Needs GH_TOKEN and GITHUB_SHA. Uses git, gh, zip and Python's
# standard library only.
set -euo pipefail

repo=${GITHUB_REPOSITORY:-seabbs/BDBVHubRenewal}
site_url="https://samabbott.co.uk/BDBVHubRenewal/"

case "$1" in
  attach)
    build=$2 site=$3
    if ! git fetch -q --depth 1 origin submission-ledger 2> /dev/null; then
      echo "No ledger: no round to attach results to."
      exit 0
    fi
    state=$(mktemp "${TMPDIR:-/tmp}/state.XXXXXX")
    git show FETCH_HEAD:state.json > "$state"
    rounds=$(python3 -I - "$build/results/fit-store.json" "$state" << 'EOF'
import json
import sys

status = json.load(open(sys.argv[1]))
latest = max(s["origin"] for s in status)
if all(s["current"] for s in status if s["origin"] == latest):
    for ref, entry in json.load(open(sys.argv[2]))["rounds"].items():
        if entry.get("as_of") == latest:
            print(ref)
EOF
    )
    for ref in $rounds; do
      tag="round-$ref"
      assets=$(gh release view "$tag" -R "$repo" --json isDraft,assets \
        -q 'select(.isDraft | not) | [.assets[].name] | join(" ")' \
        2> /dev/null) || continue
      case " $assets " in
        *" site.zip "*) continue ;;
      esac
      dir=$(mktemp -d "${TMPDIR:-/tmp}/round.XXXXXX")
      cp "$build"/results/*.csv "$build"/results/*.csv.gz \
        "$build"/results/*.json "$dir"/
      (cd "$build/results" && zip -qr - hub hub-comparison) > "$dir/hub.zip"
      (cd "$build/data" && zip -qr - .) > "$dir/data.zip"
      stage=$(mktemp -d "${TMPDIR:-/tmp}/site.XXXXXX")
      cp -R "$site" "$stage/BDBVHubRenewal"
      printf '%s\n' \
        "Run \`python3 -m http.server\` in this folder and open" \
        "http://localhost:8000/BDBVHubRenewal/" > "$stage/README.txt"
      (cd "$stage" && zip -qr - README.txt BDBVHubRenewal) > "$stage/site.zip"
      gh release upload "$tag" "$dir"/* -R "$repo" --clobber
      notes="$stage/notes.md"
      gh release view "$tag" -R "$repo" --json body -q .body |
        sed '/^## Results$/,$d' > "$notes"
      cat >> "$notes" << EOF

## Results

Every model's results at this round's hub release, built from commit $GITHUB_SHA.

- \`hub.zip\`: each model's forecast in the hub's format, samples and quantiles for the country, provinces and health zones (\`hub/<model>-samples.csv.gz\`, \`hub/<model>-quantiles.csv.gz\`), and the scored hub comparison (\`hub-comparison/\`).
- \`forecasts.csv.gz\`, \`heldout-forecasts.csv.gz\`: every model's forecasts at each real origin.
- \`*-scores.csv\`, \`*-energy.csv\`, \`nowcast-*.csv\`, \`zone-*.csv\`: the scores behind the Backtest, Nowcast and Comparison pages.
- \`backtest-fits.csv\`, \`delay-summary.csv\`, \`fit-store.json\`: the stored fits' diagnostics and learnt delays, and the fit behind each model and origin.
- \`data.zip\`: the tidy hub data the build used.
- \`site.zip\`: the rendered site; see \`README.txt\` inside to view it offline.

Site: $site_url
EOF
      gh release edit "$tag" -R "$repo" --notes-file "$notes"
      gh release upload "$tag" "$stage/site.zip" -R "$repo" --clobber
      echo "Attached results to $tag."
    done
    ;;
  *)
    echo "usage: $0 attach BUILD SITE" >&2
    exit 2
    ;;
esac
