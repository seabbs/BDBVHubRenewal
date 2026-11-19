# shellcheck shell=bash
# Machine settings of the fine-tuning and simulation scripts, sourced by
# each of them. Which machines run what is local, so it is read from an
# untracked file, `config/local.env` (or $LOCAL_CONFIG), of KEY=VALUE
# lines; see `config/local.env.example`. A variable already in the
# environment wins over the file. With no file everything runs on this
# machine:
#
#   REMOTE          host holding the rounds and the shared mailbox, as ssh
#                   names it; empty: this machine
#   ROOT, MAILBOX   the rounds folder and mailbox of the machine that
#                   simulates (workflow/rules/simulate.smk)
#   REMOTE_ROOT, REMOTE_MAILBOX
#                   the same as seen from here: paths on REMOTE, or ROOT
#                   and MAILBOX when there is none
#   HOST_NAME       this machine's name in mailbox lines and manifests
#   INBOX           an optional local copy of this machine's lines

_local_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOCAL_CONFIG=${LOCAL_CONFIG:-$_local_root/config/local.env}
if [ -f "$LOCAL_CONFIG" ]; then
  while IFS= read -r _line || [ -n "$_line" ]; do
    case $_line in '' | '#'*) continue ;; esac
    _key=${_line%%=*}
    [[ $_key =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
    [ -n "${!_key+x}" ] && continue
    eval "export $_line"
  done < "$LOCAL_CONFIG"
fi
REMOTE=${REMOTE:-}
HOST_NAME=${HOST_NAME:-$(hostname -s)}
ROOT=${ROOT:-$HOME/bdbv-rounds}
MAILBOX=${MAILBOX:-$HOME/bdbv-mailbox.md}
if [ -n "$REMOTE" ]; then
  REMOTE_ROOT=${REMOTE_ROOT:-bdbv-rounds}
  REMOTE_MAILBOX=${REMOTE_MAILBOX:-bdbv-mailbox.md}
else
  REMOTE_ROOT=${REMOTE_ROOT:-$ROOT}
  REMOTE_MAILBOX=${REMOTE_MAILBOX:-$MAILBOX}
fi
INBOX=${INBOX:-}

# Run shell command $1 on the rounds host.
on_rounds() {
  if [ -n "$REMOTE" ]; then
    ssh -o ConnectTimeout=20 "$REMOTE" "$1"
  else
    bash -c "$1"
  fi
}

# The rsync address of path $1 on the rounds host.
at_rounds() {
  if [ -n "$REMOTE" ]; then echo "$REMOTE:$1"; else echo "$1"; fi
}

# Append a line, from this machine, to the shared mailbox.
post() {
  local line
  line="$(date '+%F %H:%M') $HOST_NAME $*"
  if [ -n "$INBOX" ]; then echo "$line" >> "$INBOX"; fi
  on_rounds "echo $(printf '%q' "$line") >> $REMOTE_MAILBOX"
}

# The sha256 of file $1, with GNU or BSD tools.
sha256() {
  if command -v sha256sum > /dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}
