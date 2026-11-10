#!/usr/bin/env bash
# Tree-depth experiments (depth-exp). Usage: run.sh CORE "ANCHOR:STEP" ...
# STEP is geom or METRIC,DEPTH,SEED for exp.py. Logs go to out/.
set -u
here=$(cd "$(dirname "$0")" && pwd)
core=$1; shift
export WARM=${WARM:-250} DRAWS=${DRAWS:-150}
for job in "$@"; do
  anchor=${job%%:*}; step=${job#*:}
  log="$here/out/log_${anchor}_${step//,/_}.txt"
  echo "$(date -Is) start $job" >> "$here/out/progress.txt"
  if [ "$step" = geom ]; then
    BDBV_WALK_ANCHOR=$anchor taskset -c "$core" uv run python "$here/geom.py" renewal > "$log" 2>&1
  else
    IFS=, read -r metric depth seed <<< "$step"
    BDBV_WALK_ANCHOR=$anchor METRIC=$metric DEPTH=$depth SEED=$seed \
      taskset -c "$core" uv run python "$here/exp.py" > "$log" 2>&1
  fi
  echo "$(date -Is) done $job exit $?" >> "$here/out/progress.txt"
done
