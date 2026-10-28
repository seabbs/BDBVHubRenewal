#!/usr/bin/env bash
# Freeze the docs page of every submitted round whose hub pull request has
# merged. The page is written once to `pages/` on the ledger branch, and
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

# Submitted rounds with a pull request and no frozen page yet.
pending=$(python3 - "$ledger" << 'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rounds = json.loads((root / "state.json").read_text()).get("rounds", {})
for ref, r in sorted(rounds.items()):
    if (r.get("status") == "submitted" and r.get("pr_url")
            and not (root / "pages" / f"{ref}.md").exists()):
        print(ref, r["pr_url"])
PY
)

frozen=0
while read -r ref url; do
  [ -n "$ref" ] || continue
  state=$(gh pr view "$url" --json state -q .state 2> /dev/null || echo unknown)
  if [ "$state" = MERGED ]; then
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
