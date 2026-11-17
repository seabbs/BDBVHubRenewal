#!/usr/bin/env bash
# Run a command capped at a memory size, where there are no memory
# cgroups (macOS): the cap is a job prefix (JOB_PREFIX in
# config/local.env, `workflow/scripts/mem_cap.sh {mem}`).
#
#   workflow/scripts/mem_cap.sh MEM COMMAND [ARGS...]
#
# MEM is a size such as 16384M or 16G. Every CAP_POLL seconds (10) the
# resident memory of the command and its descendants is summed; past
# MEM they are killed and this exits 137. Otherwise it exits as the
# command does.
set -uo pipefail
mem=$1
shift
poll=${CAP_POLL:-10}
n=${mem%[GgMmKk]}
case $mem in
  *[Gg]) cap=$((n * 1048576)) ;;
  *[Mm]) cap=$((n * 1024)) ;;
  *) cap=$n ;;
esac

# Each process of the tree rooted at $1, with its resident memory (KB).
tree() {
  ps -A -o pid=,ppid=,rss= | awk -v root="$1" '
    { parent[$1] = $2; rss[$1] = $3 }
    END {
      for (p in parent) {
        q = p
        while (q > 1) { if (q == root) { print p, rss[p]; break }; q = parent[q] }
      }
    }'
}

"$@" &
pid=$!
trap 'kill $(tree "$pid" | cut -d" " -f1) 2> /dev/null' TERM INT
while kill -0 "$pid" 2> /dev/null; do
  rss=$(tree "$pid" | awk '{s += $2} END {print s + 0}')
  if [ "$rss" -gt "$cap" ]; then
    echo "mem_cap: killed, $rss KB over the $mem cap" >&2
    # shellcheck disable=SC2046
    kill $(tree "$pid" | cut -d' ' -f1) 2> /dev/null
    wait "$pid" 2> /dev/null
    exit 137
  fi
  sleep "$poll"
done
wait "$pid"
