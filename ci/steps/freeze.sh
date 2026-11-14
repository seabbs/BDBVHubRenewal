#!/usr/bin/env bash
# Freeze the docs page of every submitted round once each hub model's pull
# request is merged or closed, as for its release (release.sh). The page is written once to `pages/` on the ledger branch, and
# later docs builds copy it as it is. Runs on every trigger, whether or
# not there is new data.
#
#   LEDGER_BRANCH  default submission-ledger
#   GH_TOKEN       any token that can read the hub's pull requests
# Sets the step output frozen=<number of pages written>.
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

LEDGER_BRANCH=${LEDGER_BRANCH:-submission-ledger}
ledger="$WORK/freeze-ledger"
rm -rf "$ledger"
git worktree prune
if ! git fetch -q --depth 1 origin "$LEDGER_BRANCH" 2> /dev/null; then
  note "No ledger yet: nothing to freeze."
  echo "frozen=0" >> "${GITHUB_OUTPUT:-/dev/null}"
  exit 0
fi
git worktree add -q --detach "$ledger" FETCH_HEAD

# Rounds with no frozen page yet, and the pull request of each submitted
# hub model.
pending=$(python3 - "$ledger" << 'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rounds = json.loads((root / "state.json").read_text()).get("rounds", {})
for ref, r in sorted(rounds.items()):
    urls = [m["pr_url"] for m in r.get("models", {}).values()
            if m.get("status") == "submitted" and m.get("pr_url")]
    if urls and not (root / "pages" / f"{ref}.md").exists():
        print(ref, *urls)
PY
)

frozen=0
while read -r ref urls; do
  [ -n "$ref" ] || continue
  done_all=true
  for url in $urls; do
    state=$(gh pr view "$url" --json state -q .state 2> /dev/null || echo unknown)
    case "$state" in MERGED | CLOSED) ;; *) done_all=false ;; esac
  done
  if [ "$done_all" = true ]; then
    cli freeze-round --ledger "$ledger" --round "$ref"
    frozen=$((frozen + 1))
  fi
done <<< "$pending"

if [ "$frozen" -gt 0 ]; then
  git -C "$ledger" add -A
  git -C "$ledger" -c user.name="github-actions[bot]" \
    -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
    commit -q -m "Freeze $frozen round page(s)"
  git -C "$ledger" push -q origin "HEAD:$LEDGER_BRANCH"
fi
note "Froze $frozen round page(s)."
echo "frozen=$frozen" >> "${GITHUB_OUTPUT:-/dev/null}"
