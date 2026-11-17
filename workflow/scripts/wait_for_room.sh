#!/usr/bin/env bash
# Wait until MIN_FREE_GB of memory is available and MIN_DISK_GB of disk
# is free at PATH, so a job on a shared machine starts only with room
# to spare.
#
#   workflow/scripts/wait_for_room.sh MIN_FREE_GB MIN_DISK_GB PATH
#
# Polls every ROOM_POLL seconds (60); gives up after ROOM_TRIES polls
# (unset: never). Exits 2 if it cannot read memory or disk, as an empty
# reading would otherwise count as no room and wait forever.
set -euo pipefail
min_free=$1
min_disk=$2
where=$3
poll=${ROOM_POLL:-60}
tries=${ROOM_TRIES:-}
free_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
disk_gb() { df -Pk "$where" | awk 'NR == 2 {print int($4 / 1048576)}'; }
room() {
  local free disk
  free=$(free_gb || true)
  disk=$(disk_gb || true)
  if ! [[ $free =~ ^[0-9]+$ && $disk =~ ^[0-9]+$ ]]; then
    echo "cannot read memory ('$free') or disk ('$disk') at $where" >&2
    exit 2
  fi
  [ "$free" -ge "$min_free" ] && [ "$disk" -ge "$min_disk" ]
}
n=0
until room; do
  n=$((n + 1))
  if [ -n "$tries" ] && [ "$n" -ge "$tries" ]; then
    echo "no room: $(free_gb) GB memory (need $min_free), $(disk_gb) GB disk (need $min_disk)" >&2
    exit 1
  fi
  sleep "$poll"
done
