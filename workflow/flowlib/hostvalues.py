"""Machine values in tracked text: home folders, hosts, IPs, host:port.

Machine values live in config/local.env (`flowlib.localconfig`), never
in tracked files. `problems` lists what a text holds of them. Documented
defaults such as `$HOME/...`, `~/...` or a port number in a setting are
allowed; a literal address with a port is not. Standard library only.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_HOME = re.compile(r"/(?:home|Users)/[a-z][\w.-]*")
_IP = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_PORT = re.compile(
    r"(?<![\w.])(?:localhost|(?:\d{1,3}\.){3}\d{1,3}"
    r"|[a-z0-9-]+(?:\.[a-z0-9-]+)+):\d{2,5}\b"
)
_LOOPBACK = ("127.0.0.1", "0.0.0.0")


def problems(text: str, hosts: Iterable[str] = ()) -> list[str]:
    """Return each machine value in `text`, as `line N: value`."""
    names = [re.compile(rf"\b{re.escape(h)}\b", re.I) for h in hosts]
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        hits = [m.group() for m in _HOME.finditer(line)]
        hits += [m.group() for m in _PORT.finditer(line)]
        hits += [
            m.group()
            for m in _IP.finditer(line)
            if m.group() not in _LOOPBACK and f"{m.group()}:" not in line
        ]
        hits += [m.group() for r in names for m in r.finditer(line)]
        found += [f"line {n}: {h}" for h in hits]
    return found
