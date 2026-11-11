#!/usr/bin/env bash
# Watch an archie run: exit when its log says DONE/SCORED, or when the
# process is gone or archie rebooted.
log=$1; pat=$2; boot0=$(ssh archie 'cat /proc/sys/kernel/random/boot_id')
while true; do
  sleep 180
  out=$(ssh -o ConnectTimeout=20 archie "cat /proc/sys/kernel/random/boot_id; pgrep -f '$pat' >/dev/null && echo ALIVE || echo GONE; tail -2 ~/fm-chronos-evidence/$log" 2>&1) || { echo "SSH FAILED: $out"; continue; }
  boot=$(echo "$out" | sed -n 1p)
  [ "$boot" != "$boot0" ] && { echo "ARCHIE REBOOTED"; echo "$out"; exit 2; }
  echo "$out" | grep -q "DONE\|SCORED" && { echo "FINISHED"; echo "$out"; exit 0; }
  echo "$out" | grep -q GONE && { echo "PROCESS DIED"; echo "$out"; exit 1; }
done
