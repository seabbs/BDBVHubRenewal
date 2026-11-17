#!/usr/bin/env bash
# Run a command and end its stderr with its peak memory and wall time,
# as `peak_rss_kb <KB> wall_s <s>` (flowlib.manifest reads it), with GNU
# or BSD (macOS) time.
#
#   workflow/scripts/timed.sh COMMAND [ARGS...]
set -uo pipefail
if /usr/bin/time -f "" true 2> /dev/null; then
  exec /usr/bin/time -f "peak_rss_kb %M wall_s %e" "$@"
fi
# Its stderr passes through as it is written and is kept for the end.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
out=$tmp/stderr
mkfifo "$tmp/pipe"
tee "$out" >&2 < "$tmp/pipe" &
tee_pid=$!
/usr/bin/time -l "$@" 2> "$tmp/pipe"
rc=$?
wait "$tee_pid"
# BSD time gives the maximum resident set size in bytes.
awk '
  / real / { wall = $1 }
  /maximum resident set size/ { rss = int($1 / 1024) }
  END { if (rss != "") print "peak_rss_kb " rss " wall_s " wall }' "$out" >&2
exit "$rc"
