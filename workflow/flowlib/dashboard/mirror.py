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
        self.remote_mailbox = cfg.get("REMOTE_MAILBOX", "")
        local_box = cfg.get("MAILBOX") or ""
        self._local_mailbox = (
            Path(local_box).expanduser() if local_box else None
        )
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

    @property
    def mailbox(self) -> Path | None:
        """Return the mailbox the page reads: mirrored, or this machine's."""
        if self.remote and self.remote_mailbox:
            return self.root / "mailbox.md"
        return None if self.remote else self._local_mailbox

    def mailbox_command(self) -> list[str] | None:
        """Return the rsync command that copies the remote mailbox, if any."""
        if not (self.remote and self.remote_mailbox):
            return None
        return [
            "rsync",
            "-a",
            "--timeout=60",
            f"{self.remote}:{self.remote_mailbox}",
            str(self.root / "mailbox.md"),
        ]

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
        status = f"mirrored from {self.remote}"
        box = self.mailbox_command()
        if box:
            try:
                got = subprocess.run(
                    box, capture_output=True, text=True, timeout=60
                )
            except (OSError, subprocess.SubprocessError) as err:
                return f"{status}; mailbox failed: {err}"
            if got.returncode != 0:
                return f"{status}; mailbox failed: {got.stderr.strip()[:200]}"
        return status
