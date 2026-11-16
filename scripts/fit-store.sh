#!/usr/bin/env bash
# The fit store branch (`bdbvhubrenewal.evaluation.store`).
#
#   scripts/fit-store.sh guard EVENT REF SCRATCH
#       print `true` if the run may write a store, `false` if it only
#       plans, and fail if SCRATCH is not a scratch branch name
#   scripts/fit-store.sh fetch DIR
#       copy the branch's files into DIR (empty if there is no branch),
#       with the commit fetched in DIR/.commit, and fail if the remote
#       cannot be read
#   scripts/fit-store.sh push DIR KEYS PLAN [REGISTERED]
#       add DIR's planned fits to the branch, prune it and push
#
# KEYS is the JSON list of current keys, `<model>/<origin>/<key>`, PLAN
# the matrix of planned fits and REGISTERED the list of registered
# stored models, as `fit-store plan --keys --out --registered` write
# them. Only planned files are copied. For each pair in KEYS the push
# keeps the current key's file and the newest good fit of older code,
# and deletes the rest. With REGISTERED, it also deletes each top-level
# model folder not in it: a model removed from the registry. FIT_STORE_BRANCH names the branch (default
# fit-store). Uses git and Python's standard library only.
set -euo pipefail

action=$1
branch=${FIT_STORE_BRANCH:-fit-store}
scratch_prefix=fit-store-scratch

fetch_tip() {
  git fetch -q --depth 1 origin "$branch"
}

case "$action" in
  guard)
    event=$2 ref=$3 scratch=${4:-}
    if [ -n "$scratch" ] && [[ "$scratch" != "$scratch_prefix"* ]]; then
      echo "::error::store_branch must start with $scratch_prefix" >&2
      exit 1
    fi
    if [ "$event" = pull_request ]; then
      echo false
    elif [ -n "$scratch" ] || [ "$ref" = refs/heads/main ]; then
      echo true
    else
      echo false
    fi
    ;;
  fetch)
    dir=$2
    mkdir -p "$dir"
    # Exit 2 means the branch does not exist: an empty store, not an error.
    rc=0
    git ls-remote -q --exit-code --heads origin "$branch" > /dev/null || rc=$?
    if [ "$rc" = 2 ]; then
      exit 0
    elif [ "$rc" != 0 ]; then
      echo "::error::could not read $branch from origin" >&2
      exit 1
    fi
    if ! fetch_tip; then
      echo "::error::could not fetch $branch" >&2
      exit 1
    fi
    git archive FETCH_HEAD | tar -x -C "$dir"
    git rev-parse FETCH_HEAD > "$dir/.commit"
    ;;
  push)
    dir=$(cd "$2" && pwd)
    keys=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
    plan=$(cd "$(dirname "$4")" && pwd)/$(basename "$4")
    registered=""
    if [ -n "${5:-}" ]; then
      registered=$(cd "$(dirname "$5")" && pwd)/$(basename "$5")
    fi
    work=$(mktemp -d "${TMPDIR:-/tmp}/fit-store.XXXXXX")
    for attempt in 1 2 3 4 5; do
      if fetch_tip; then
        git worktree add -q --detach "$work" FETCH_HEAD
      else
        git worktree add -q --detach "$work"
        git -C "$work" switch -q --orphan "$branch-new"
      fi
      msg=$(python3 -I - "$work" "$dir" "$keys" "$plan" "$registered" << 'EOF'
import gzip
import json
import shutil
import sys
from pathlib import Path

work, new = Path(sys.argv[1]), Path(sys.argv[2])
keys = json.loads(Path(sys.argv[3]).read_text())
plan = json.loads(Path(sys.argv[4]).read_text())["include"]
stored = []
for fit in plan:
    name = f"{fit['model']}/{fit['origin']}/{fit['key']}.json.gz"
    if (new / name).is_file():
        (work / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(new / name, work / name)
        stored.append(name.removesuffix(".json.gz"))
for entry in keys:
    model, origin, key = entry.split("/")
    older = []
    for path in (work / model / origin).glob("*.json.gz"):
        if path.name == f"{key}.json.gz":
            continue
        meta = json.loads(gzip.decompress(path.read_bytes()))["meta"]
        older.append((meta["status"] == "ok", meta["fitted_at"], path))
    good = sorted(f for f in older if f[0])
    keep = good[-1][2] if good else None
    for _, _, path in older:
        if path != keep:
            path.unlink()
removed = []
if sys.argv[5]:
    registered = set(json.loads(Path(sys.argv[5]).read_text()))
    for path in sorted(work.iterdir()):
        if path.is_dir() and not path.name.startswith("."):
            if path.name not in registered:
                shutil.rmtree(path)
                removed.append(path.name)
msg = "Store " + " ".join(stored)
if removed:
    msg += "; prune removed models " + " ".join(removed)
print(msg)
EOF
      )
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
    echo "usage: $0 guard EVENT REF SCRATCH | fetch DIR | push DIR KEYS PLAN [REGISTERED]" >&2
    exit 2
    ;;
esac
