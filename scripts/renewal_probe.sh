#!/usr/bin/env bash
# Run scripts/renewal_probe.py detached on pinned cores, at nice 19, in a
# systemd scope with a hard memory cap, and append a mailbox line when it
# ends.
#
# Usage: scripts/renewal_probe.sh LABEL CORES MEMMAX ARGS...
#   LABEL   output/probe/LABEL holds the probe's files and log.
#   CORES   taskset list, e.g. 12,13 (one per chain).
#   MEMMAX  systemd MemoryMax, e.g. 12G; the probe is killed past it.
# Set MAILBOX to the mailbox file (default ~/bdbv-archie-mailbox.md).
set -euo pipefail
label=$1
cores=$2
memmax=$3
shift 3
out=output/probe/$label
mkdir -p "$out"
n=$(awk -F, '{print NF}' <<< "$cores")
mailbox=${MAILBOX:-$HOME/bdbv-archie-mailbox.md}
start=$(date +%s)
if systemd-run --user --scope --quiet -p MemoryMax="$memmax" \
  -p MemorySwapMax=0 \
  env XLA_FLAGS=--xla_force_host_platform_device_count=$n \
  nice -n 19 taskset -c "$cores" \
  uv run python scripts/renewal_probe.py --out "$out" "$@" \
  > "$out/log.txt" 2>&1; then
  status="done: $(tail -n 1 "$out/log.txt")"
else
  peak=$(sort -k2 -n "$out/rss.log" 2>/dev/null | tail -n 1 | cut -d' ' -f2)
  status="FAILED, peak RSS ${peak:-?} GB of $memmax (log $PWD/$out/log.txt)"
fi
mins=$(( ($(date +%s) - start) / 60 ))
echo "$(date '+%Y-%m-%d %H:%M') archie renewal-conv probe $label" \
  "(${mins} min) $status" >> "$mailbox"
