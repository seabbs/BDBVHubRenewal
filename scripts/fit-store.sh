#!/usr/bin/env bash
# The fit store branch (`bdbvhubrenewal.evaluation.store`).
#
#   scripts/fit-store.sh fetch DIR   copy the branch's files into DIR
#                                    (empty if the branch does not exist)
#   scripts/fit-store.sh push DIR    commit DIR's files on top of the
#                                    branch and push, retrying when
#                                    another fit job pushed first
#
# Each fit is its own file, so concurrent pushes never touch the same
# path and a retry on the new tip always applies.
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
      git -C "$work" add -A
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
    echo "usage: $0 fetch|push DIR" >&2
    exit 2
    ;;
esac
