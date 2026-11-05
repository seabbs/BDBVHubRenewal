#!/usr/bin/env bash
# The fit store branch (`bdbvhubrenewal.evaluation.store`).
#
#   scripts/fit-store.sh fetch DIR        copy the branch's files into DIR
#                                         (empty if there is no branch)
#   scripts/fit-store.sh push DIR KEYS    add DIR's fits to the branch,
#                                         prune it and push
#
# KEYS is the JSON list of current keys, `<model>/<origin>/<key>`, that
# `fit-store plan --keys` writes. For each of those pairs the push keeps
# the current key's file and the newest good fit of older code, and
# deletes the rest. FIT_STORE_BRANCH names the branch (default
# fit-store). Uses git and Python's standard library only.
set -euo pipefail

action=$1
dir=$2
branch=${FIT_STORE_BRANCH:-fit-store}

fetch_tip() {
  git fetch -q --depth 1 origin "$branch" 2> /dev/null
}

case "$action" in
  fetch)
    mkdir -p "$dir"
    if fetch_tip; then
      git archive FETCH_HEAD | tar -x -C "$dir"
    fi
    ;;
  push)
    keys=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
    files=$(cd "$dir" && find . -name '*.json.gz' | sort)
    if [ -z "$files" ]; then
      echo "No stored fits to push."
      exit 0
    fi
    msg="Store $(echo "$files" | sed 's|^\./||; s|\.json\.gz$||' | paste -sd ' ' -)"
    work=$(mktemp -d "${TMPDIR:-/tmp}/fit-store.XXXXXX")
    for attempt in 1 2 3 4 5; do
      if fetch_tip; then
        git worktree add -q --detach "$work" FETCH_HEAD
      else
        git worktree add -q --detach "$work"
        git -C "$work" switch -q --orphan "$branch-new"
      fi
      cp -R "$dir"/. "$work"/
      python3 - "$work" "$keys" << 'EOF'
import gzip
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for entry in json.loads(Path(sys.argv[2]).read_text()):
    model, origin, key = entry.split("/")
    older = []
    for path in (root / model / origin).glob("*.json.gz"):
        if path.name == f"{key}.json.gz":
            continue
        meta = json.loads(gzip.decompress(path.read_bytes()))["meta"]
        older.append((meta["status"] == "ok", meta["fitted_at"], path))
    good = sorted(f for f in older if f[0])
    keep = good[-1][2] if good else None
    for _, _, path in older:
        if path != keep:
            path.unlink()
EOF
      git -C "$work" add -A
      if git -C "$work" diff --cached --quiet; then
        git worktree remove --force "$work"
        echo "Nothing to store."
        exit 0
      fi
      git -C "$work" \
        -c user.name='github-actions[bot]' \
        -c user.email='41898282+github-actions[bot]@users.noreply.github.com' \
        commit -q -m "$msg"
      pushed=0
      git -C "$work" push -q origin "HEAD:refs/heads/$branch" && pushed=1
      git worktree remove --force "$work"
      if [ "$pushed" = 1 ]; then
        echo "$msg"
        exit 0
      fi
      sleep $((attempt * 5))
    done
    echo "::error::could not push to $branch"
    exit 1
    ;;
  *)
    echo "usage: $0 fetch DIR | push DIR KEYS" >&2
    exit 2
    ;;
esac
