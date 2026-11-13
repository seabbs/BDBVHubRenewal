#!/usr/bin/env bash
# Time and peak memory of one fit of every Chronos-2 model, with and
# without zones, at the latest vintage, one process each.
set -u
export GH_TOKEN=$(gh auth token)
out=${1:-results/timings-cpu.txt}
: > "$out"
for m in fm_chronos_weekly_complete fm_chronos_weekly_ragged fm_chronos_weekly_ragged_ft \
         fm_chronos_daily_complete fm_chronos_daily_ragged fm_chronos_daily_ragged_ft; do
  for flag in "" --plain; do
    /usr/bin/time -v uv run python ev-scripts/fit_one.py $m $flag > /tmp/fmz-one.log 2> /tmp/fmz-time.log
    line=$(grep -E "^fm_chronos" /tmp/fmz-one.log)
    rss=$(grep "Maximum resident" /tmp/fmz-time.log | awk '{print $NF}')
    wall=$(grep "Elapsed (wall" /tmp/fmz-time.log | awk '{print $NF}')
    echo "$line max_rss_gb=$(awk "BEGIN{printf \"%.1f\", $rss/1048576}") wall=$wall" | tee -a "$out"
    [ -z "$line" ] && tail -5 /tmp/fmz-time.log
  done
done
