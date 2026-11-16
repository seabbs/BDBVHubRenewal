#!/usr/bin/env bash
# Follow-up jobs for one slots checkpoint, after its scores land.
#
#   scripts/slots_followups.sh slots-<hash>
#
# scripts/checkpoint_poll.py starts this, detached, once a slots
# checkpoint's rows land. It runs, one after another and one checkpoint at
# a time (flock), the slots check (lag swap and held-out loss by source,
# branch slots-check) and then the aggregation arms for the fine-tuned
# models (branch agg-slots), each from its own worktree beside this one.
# A runner that is missing is reported in the mailbox and skipped. Each
# runner commits its own evidence and writes its own mailbox lines.
set -u
label=$1
here=$(cd "$(dirname "$0")/.." && pwd)
trees=$(dirname "$here")
mailbox=${MAILBOX:-$HOME/bdbv-archie-mailbox.md}
say() {
  echo "$(date '+%Y-%m-%d %H:%M') archie: slots-followups: $label: $*" >> "$mailbox"
}
# The site's Germany page reads the checkpoint's development forecasts
# from the germany-ft-forecasts branch (scripts/germany_ft_store.py,
# #255); quick, so before the lock.
store=$trees/germany-site-ft/scripts/germany_ft_store.py
[ -f "$here/scripts/germany_ft_store.py" ] && store=$here/scripts/germany_ft_store.py
if [ -f "$store" ]; then
  if uv run --project "$(dirname "$(dirname "$store")")" python "$store" \
    "$HOME/ckpt-scores/cache" "$label" \
    --repo "$(dirname "$(dirname "$store")")"; then
    say "German development forecasts stored for the site (germany-ft-forecasts)"
  else
    say "FAILED to store the German forecasts for the site ($store)"
  fi
else
  say "skipped the site's German store: no germany_ft_store.py yet"
fi
exec 9> "$HOME/.slots-followups.lock"
flock 9
for job in "slots-check:scripts/slots_check.sh" "agg-slots:scripts/agg/run_finetuned.sh"; do
  tree=$trees/${job%%:*}
  runner=$tree/${job#*:}
  if [ ! -x "$runner" ]; then
    say "skipped ${job%%:*}: no runner at $runner"
    continue
  fi
  say "started ${job%%:*} ($runner)"
  if (cd "$tree" && "$runner" "$label"); then
    say "${job%%:*} finished"
  else
    say "FAILED ${job%%:*} (exit $?); see its log and mailbox lines"
  fi
done
