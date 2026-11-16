"""A round's manifest, one row per batch, as `scripts/sim_rounds.sh` wrote.

Each batch gets its row in a file of its own, and the round's
`manifest.csv` joins them in seed order. A row the old driver already
wrote is kept as it is, so switching to the workflow keeps its record.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

HEADER = (
    "round,seed,kind,grid,grid_commit,draws,batch,counts_sha256,"
    "settings_sha256,failures_sha256,peak_rss_kb,wall_s,host,finished"
)


def _sha(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last(log: Path, name: str) -> str:
    if not log.is_file():
        return ""
    found = re.findall(rf"{name} ([0-9.]+)", log.read_text())
    return found[-1] if found else ""


def row(
    sims: Path | str,
    *,
    round_: int,
    seed: int,
    kind: str,
    grid: str,
    commit: str,
    draws: int,
    batch: int,
    host: str,
    finished: str,
) -> str:
    """Return a batch's manifest row (no newline).

    The memory and time are the last `/usr/bin/time` record in the
    batch's log, so a retried batch gives its last attempt's.
    """
    sims = Path(sims)
    log = sims / "logs" / f"{seed}.log"
    fields = [
        round_,
        seed,
        kind,
        grid,
        commit,
        draws,
        batch,
        _sha(sims / f"counts-{seed}.csv.gz"),
        _sha(sims / f"settings-{seed}.csv"),
        _sha(sims / f"failures-{seed}.csv"),
        _last(log, "peak_rss_kb"),
        _last(log, "wall_s"),
        host,
        finished,
    ]
    return ",".join(str(f) for f in fields)


def existing(manifest: Path | str, round_: int, seed: int) -> str | None:
    """Return the row a manifest already holds for a batch, if any."""
    manifest = Path(manifest)
    if not manifest.is_file():
        return None
    start = f"{round_},{seed},"
    for line in manifest.read_text().splitlines():
        if line.startswith(start):
            return line
    return None


def join(rows: Iterable[Path | str], out: Path | str) -> None:
    """Write the batches' row files as one manifest, in seed order."""
    lines = [Path(r).read_text().strip() for r in rows]
    lines.sort(key=lambda x: int(x.split(",")[1]))
    Path(out).write_text("\n".join([HEADER, *lines]) + "\n")
