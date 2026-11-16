"""Mailbox lines, and the guard on where evidence is pushed.

Lines go to `MAILBOX` only when `NOTIFY` is set in the local config, so
CI and test runs never write one.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from pathlib import Path


def post(message: str, cfg: Mapping[str, str]) -> None:
    """Append `<time> <host> workflow: <message>` to the mailbox."""
    if not cfg.get("NOTIFY"):
        return
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"{stamp} {cfg['HOST_NAME']} workflow: {message}\n"
    with Path(cfg["MAILBOX"]).expanduser().open("a") as f:
        f.write(line)


def check_branch(current: str, allowed: str) -> None:
    """Raise unless evidence is landed from the branch it belongs to."""
    if current in ("main", "master"):
        raise SystemExit(f"never land evidence on {current}")
    if current != allowed:
        raise SystemExit(
            f"land runs on the evidence branch {allowed}, not {current}"
        )
