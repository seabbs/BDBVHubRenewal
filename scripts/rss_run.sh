#!/usr/bin/env bash
# Run a command on pinned cores, at nice 19, under a systemd memory cap,
# logging the resident memory (GB) of its process tree every 30 s.
#
# Usage: scripts/rss_run.sh OUTDIR CORES MEMMAX COMMAND...
#   OUTDIR  gets log.txt (the command's output) and rss.log
#           (seconds, GB).
# Set MAILBOX to append a line when it ends.
set -euo pipefail
out=$1
cores=$2
memmax=$3
shift 3
mkdir -p "$out"
n=$(awk -F, '{print NF}' <<< "$cores")
start=$(date +%s)
unit=rss-run-$(basename "$out")-$$
systemd-run --user --scope --quiet --unit="$unit" -p MemoryMax="$memmax" \
  -p MemorySwapMax=0 \
  env XLA_FLAGS=--xla_force_host_platform_device_count=$n \
  nice -n 19 taskset -c "$cores" "$@" > "$out/log.txt" 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  # The scope's memory: every process the command started.
  b=$(systemctl --user show -p MemoryCurrent --value "$unit.scope" \
    2>/dev/null || echo 0)
  case $b in ''|*[!0-9]*) b=0 ;; esac
  echo "$(( $(date +%s) - start )) $(awk -v b="$b" \
    'BEGIN {printf "%.2f", b / 1e9}')" >> "$out/rss.log"
  sleep 30
done
if wait "$pid"; then status=done; else status="FAILED ($?)"; fi
peak=$(sort -k2 -n "$out/rss.log" | tail -n 1 | cut -d' ' -f2)
if [ -n "${MAILBOX:-}" ]; then
  echo "$(date '+%Y-%m-%d %H:%M') archie renewal-conv rss_run $out" \
    "($(( ($(date +%s) - start) / 60 )) min) $status, peak RSS $peak GB" \
    >> "$MAILBOX"
fi
