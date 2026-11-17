"""Optional notification lines, and the guard on where evidence is pushed.

The workflow needs neither. A line is appended to `MAILBOX` only when
both `NOTIFY` and `MAILBOX` are set in the local config, and a line that
cannot be written is a warning on stderr, never a failure of the rule
that posted it.
"""

from __future__ import annotations

import datetime as dt
import sys
from collections.abc import Mapping
from pathlib import Path


def post(message: str, cfg: Mapping[str, str]) -> None:
    """Append `<time> <host> workflow: <message>` to the mailbox."""
    if not (cfg.get("NOTIFY") and cfg.get("MAILBOX")):
        return
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"{stamp} {cfg.get('HOST_NAME', '')} workflow: {message}\n"
    try:
        with Path(cfg["MAILBOX"]).expanduser().open("a") as f:
            f.write(line)
    except OSError as err:
        print(
            f"notify: could not write {cfg['MAILBOX']}: {err}", file=sys.stderr
        )


def check_branch(current: str, allowed: str) -> None:
    """Raise unless evidence is landed from the branch it belongs to."""
    if current in ("main", "master"):
        raise SystemExit(f"never land evidence on {current}")
    if current != allowed:
        raise SystemExit(
            f"land runs on the evidence branch {allowed}, not {current}"
        )
