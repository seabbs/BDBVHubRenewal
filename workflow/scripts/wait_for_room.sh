#!/usr/bin/env bash
# Wait until MIN_FREE_GB of memory is available, MIN_DISK_GB of disk is
# free at PATH and the 1-minute load average is below MAX_LOAD, so a job
# on a shared machine starts only with room to spare.
#
#   workflow/scripts/wait_for_room.sh MIN_FREE_GB MIN_DISK_GB PATH [MAX_LOAD]
#
# MAX_LOAD left out or empty is the machine's core count. Polls every
# ROOM_POLL seconds (60); gives up after ROOM_TRIES polls (unset: never).
# Exits 2 if it cannot read memory, disk or load, as an empty reading
# would otherwise count as no room and wait forever.
set -euo pipefail
min_free=$1
min_disk=$2
where=$3
max_load=${4:-$(getconf _NPROCESSORS_ONLN)}
poll=${ROOM_POLL:-60}
tries=${ROOM_TRIES:-}
# Available memory: Linux's MemAvailable, or on macOS the free, inactive,
# speculative and purgeable pages.
free_gb() {
  if [ -r /proc/meminfo ]; then
    awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo
  else
    vm_stat | awk '/page size of/ {ps = $8}
      /Pages (free|inactive|speculative|purgeable)/ {gsub(/\./, "", $NF); n += $NF}
      END {print int(n * ps / 1073741824)}'
  fi
}
disk_gb() { df -Pk "$where" | awk 'NR == 2 {print int($4 / 1048576)}'; }
# The 1-minute load average, from uptime on Linux and macOS alike.
load() { uptime | sed -E 's/.*load averages?: *//; s/,? .*//'; }
room() {
  local free disk now
  free=$(free_gb || true)
  disk=$(disk_gb || true)
  now=$(load || true)
  if ! [[ $free =~ ^[0-9]+$ && $disk =~ ^[0-9]+$ && $now =~ ^[0-9.]+$ ]]; then
    echo "cannot read memory ('$free'), disk ('$disk') or load ('$now')" \
      "at $where" >&2
    exit 2
  fi
  [ "$free" -ge "$min_free" ] && [ "$disk" -ge "$min_disk" ] &&
    awk -v now="$now" -v max="$max_load" 'BEGIN {exit !(now < max)}'
}
n=0
until room; do
  n=$((n + 1))
  if [ -n "$tries" ] && [ "$n" -ge "$tries" ]; then
    echo "no room: $(free_gb) GB memory (need $min_free)," \
      "$(disk_gb) GB disk (need $min_disk), load $(load) (below $max_load)" >&2
    exit 1
  fi
  sleep "$poll"
done
