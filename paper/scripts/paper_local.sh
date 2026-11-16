#!/usr/bin/env bash
# The full paper run on a shared host, as a detached job: wait for the
# held-out inputs, then run scripts/paper.sh paper single-threaded and
# niced, reporting each stage to the archie mailbox.
#
# Usage (from anywhere): nohup paper/scripts/paper_local.sh [TARGET] &
#
# Waits for (each checked every 10 minutes, for up to 24 h):
# - the held-out synthetic simulations of the paper run's seeds, in the
#   folder `synthetic.simulations.paper` of the workflow config names;
# - the German held-out half, results/germany/held-out/paper-forecasts.csv.gz
#   (paper/scripts/gather_heldout.sh brings it from a dispatched CI run).
# QUARTO_BIN picks a Quarto (default: quarto on PATH).
set -uo pipefail
target=${1:-paper}
root=$(cd "$(dirname "$0")/../.." && pwd)
mb=${MAILBOX:-$HOME/bdbv-archie-mailbox.md}
log=$HOME/paper-jobs/paper-local-$target.log
mkdir -p "$HOME/paper-jobs"
say() { echo "$(date '+%F %H:%M') paper: local $target run: $*" >> "$mb"; }
cd "$root"
export PATH="$root/.venv/bin:${QUARTO_BIN:+$(dirname "$QUARTO_BIN"):}$PATH"
unset VIRTUAL_ENV
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CORES=1

ready() {
  [ "$target" != paper ] && return 0
  python - <<'PY'
import sys
from pathlib import Path
import yaml
sys.path.insert(0, "paper")
from paperpipe.config import run_settings
config = yaml.safe_load(Path("paper/workflow/config.yaml").read_text())
sims = Path("paper") / config["synthetic"]["simulations"]["paper"]
seeds = run_settings(config, "paper").synthetic_seeds
need = [sims / f"{k}-{s}.{e}" for s in seeds
        for k, e in (("counts", "csv.gz"), ("settings", "csv"))]
need.append(Path("results/germany/held-out/paper-forecasts.csv.gz"))
missing = [str(p) for p in need if not p.exists()]
print(", ".join(missing))
sys.exit(1 if missing else 0)
PY
}
for _ in $(seq 1 144); do
  if waiting=$(ready); then break; fi
  [ -z "${said:-}" ] && { say "waiting for: $waiting"; said=1; }
  sleep 600
done
ready > /dev/null || { say "FAILED: still waiting after 24 h"; exit 1; }
say "started at $(git rev-parse --short=9 HEAD) (log $log)"
if nice -n 19 scripts/paper.sh "$target" > "$log" 2>&1; then
  say "done: docs/paper/main.html and si.html in $root"
else
  say "FAILED: $(grep -m1 -iE 'error|exception' "$log" | cut -c1-250)"
  exit 1
fi
