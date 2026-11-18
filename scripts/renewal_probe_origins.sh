#!/usr/bin/env bash
# Run scripts/renewal_probe.sh for several origins, one after another.
#
# Usage: scripts/renewal_probe_origins.sh PREFIX CORES MEMMAX ORIGINS ARGS...
#   PREFIX   each origin's output goes to output/probe/PREFIX-ORIGIN.
#   ORIGINS  space-separated, e.g. "2026-11-15 2026-11-08".
#   ARGS     passed to scripts/renewal_probe.py (not --origin or --out).
set -uo pipefail
prefix=$1
cores=$2
memmax=$3
origins=$4
shift 4
for o in $origins; do
  scripts/renewal_probe.sh "$prefix-$o" "$cores" "$memmax" --origin "$o" "$@"
done
