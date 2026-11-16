"""Where the dashboard reads checkpoint scores from.

On the machine that scores, `SCORE_CACHE` is read as it is. On another
machine (`REMOTE` set in config/local.env, the host as ssh names it),
the score table and the German forecast caches are copied, read-only,
from `REMOTE:REMOTE_ROOT/scores` (or `REMOTE_SCORE_CACHE`) into a local
mirror with rsync. Nothing is ever written back.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

TABLE = "tables/checkpoint-scores.csv"
"""The score table, under the scores root."""


class Source:
    """The scores root the dashboard reads, local or mirrored."""

    def __init__(self, cfg: Mapping[str, str], mirror: Path | str) -> None:
        self.remote = cfg.get("REMOTE", "")
        if self.remote:
            self.path = cfg.get("REMOTE_SCORE_CACHE") or (
                f"{cfg.get('REMOTE_ROOT') or 'bdbv-rounds'}/scores"
            )
            self.root = Path(mirror)
        else:
            self.path = ""
            self.root = Path(cfg["SCORE_CACHE"]).expanduser()

    @property
    def scores(self) -> Path:
        """Return the score table's path."""
        return self.root / TABLE

    @property
    def germany(self) -> Path:
        """Return the German forecast caches' folder."""
        return self.root / "germany"

    def command(self) -> list[str] | None:
        """Return the rsync command that refreshes the mirror, if any."""
        if not self.remote:
            return None
        return [
            "rsync",
            "-a",
            "--timeout=60",
            "--include=tables/",
            f"--include={TABLE}",
            "--include=germany/***",
            "--exclude=*",
            f"{self.remote}:{self.path.rstrip('/')}/",
            f"{self.root}/",
        ]

    def refresh(self) -> str:
        """Refresh the mirror; return what happened, for the page."""
        cmd = self.command()
        if cmd is None:
            return "local"
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
        except (OSError, subprocess.SubprocessError) as err:
            return f"mirror failed: {err}"
        if out.returncode != 0:
            return f"mirror failed: {out.stderr.strip()[:200]}"
        return f"mirrored from {self.remote}"
