# shellcheck shell=bash
# say MESSAGE...: print a timestamped line, the script's log, and append
# it to MAILBOX too only when both NOTIFY and MAILBOX are set in
# config/local.env (or the environment). It never fails its caller.
# Source it after setting SAY_PREFIX (for example "paper: gather").
_notify_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mapfile -t _notify_cfg < <(
  python3 "$_notify_root/workflow/flowlib/localconfig.py" NOTIFY MAILBOX \
    HOST_NAME 2> /dev/null
)
say() {
  local line
  line="$(date '+%F %H:%M') ${_notify_cfg[2]:-} ${SAY_PREFIX:-}: $*"
  echo "$line"
  if [ -n "${_notify_cfg[0]:-}" ] && [ -n "${_notify_cfg[1]:-}" ]; then
    { echo "$line" >> "${_notify_cfg[1]}"; } 2> /dev/null ||
      echo "notify: could not write ${_notify_cfg[1]}" >&2
  fi
  return 0
}
