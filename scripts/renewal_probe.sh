#!/usr/bin/env bash
# Run scripts/renewal_probe.py detached on two pinned cores, at nice 19,
# and append a mailbox line when it ends.
#
# Usage: scripts/renewal_probe.sh LABEL CORES ARGS...
#   LABEL  output/probe/LABEL holds the probe's files and log.
#   CORES  taskset list, e.g. 12,13 (one per chain).
# Set MAILBOX to the mailbox file (default ~/bdbv-archie-mailbox.md).
set -euo pipefail
label=$1
cores=$2
shift 2
out=output/probe/$label
mkdir -p "$out"
n=$(awk -F, '{print NF}' <<< "$cores")
mailbox=${MAILBOX:-$HOME/bdbv-archie-mailbox.md}
start=$(date +%s)
if XLA_FLAGS=--xla_force_host_platform_device_count=$n \
  nice -n 19 taskset -c "$cores" \
  uv run python scripts/renewal_probe.py --out "$out" "$@" \
  > "$out/log.txt" 2>&1; then
  status="done: $(tail -n 1 "$out/log.txt")"
else
  status="FAILED (log $PWD/$out/log.txt)"
fi
mins=$(( ($(date +%s) - start) / 60 ))
echo "$(date '+%Y-%m-%d %H:%M') archie renewal-conv probe $label" \
  "(${mins} min) $status" >> "$mailbox"
