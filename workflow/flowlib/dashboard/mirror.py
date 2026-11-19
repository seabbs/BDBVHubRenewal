"""Where the dashboard reads scores, forecasts, truth and notes from.

Each source is a local path, read where it is, or `host:path`, copied
read-only with rsync over ssh into the mirror folder; nothing is ever
written back. The sources are the score table (`scores`), each dataset's
per-checkpoint forecast folders (`datasets[].source`) and truth table
(`datasets[].truth_source`), and an optional notes file (`mailbox`).
Unset, the score table and forecasts are this machine's scoring cache
(`SCORE_CACHE`, or `REMOTE:REMOTE_ROOT/scores` when `REMOTE` is set) and
the notes file is `MAILBOX` (or `REMOTE:REMOTE_MAILBOX`).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TABLE = "tables/checkpoint-scores.csv"
"""The score table, under a scoring cache."""

REMOTE = re.compile(r"^[\w.-]+:")
"""A `host:path` source."""

RSYNC = [
    "rsync",
    "-a",
    "--timeout=60",
    "-e",
    "ssh -o BatchMode=yes -o ConnectTimeout=10",
]
"""rsync over ssh that never waits on a password or host-key prompt."""


def is_remote(source: str) -> bool:
    """Return whether a source is `host:path`."""
    return bool(REMOTE.match(source))


class Source:
    """The paths the dashboard reads, local or mirrored."""

    def __init__(
        self,
        cfg: Mapping[str, str],
        mirror: Path | str,
        datasets: Sequence[Mapping[str, Any]] = (),
        scores: str | None = None,
        mailbox: Mapping[str, Any] | None = None,
        root: Path | str = ".",
        extras: Sequence[str] = (),
    ) -> None:
        self.root = Path(mirror)
        self.base = Path(root)
        self.failed: list[str] = []
        remote = cfg.get("REMOTE", "")
        if remote:
            path = cfg.get("REMOTE_SCORE_CACHE") or (
                f"{cfg.get('REMOTE_ROOT') or 'bdbv-rounds'}/scores"
            )
            cache = f"{remote}:{path}"
        else:
            cache = str(Path(cfg.get("SCORE_CACHE") or "scores").expanduser())
        cache = cache.rstrip("/")
        self._cache = cache
        # name -> (source, local path, folder?, required?)
        self._items: dict[str, tuple[str, Path, bool, bool]] = {}
        self._add(
            "scores",
            scores or f"{cache}/{TABLE}",
            "checkpoint-scores.csv",
            folder=False,
        )
        for ds in datasets:
            key = ds["key"]
            # A samples dataset's copy is kept apart from its summaries.
            local = f"{key}-samples" if ds.get("format") == "samples" else key
            # A dataset that says why it may have no nowcasts (`missing`)
            # does not fail the mirror while it has none.
            self._add(
                f"dataset:{key}",
                ds.get("source") or f"{cache}/{key}",
                local,
                folder=True,
                required=not ds.get("missing"),
            )
            if ds.get("truth_source"):
                self._add(
                    f"truth:{key}",
                    ds["truth_source"],
                    f"{key}-truth.csv",
                    folder=False,
                )
        box = (mailbox or {}).get("source") or (
            f"{remote}:{cfg['REMOTE_MAILBOX']}"
            if remote and cfg.get("REMOTE_MAILBOX")
            else cfg.get("MAILBOX") or ""
        )
        if box:
            self._add(
                "mailbox", box, "mailbox.md", folder=False, required=False
            )
        # Optional files shown as they are (tables, notes); a failed copy
        # is reported but does not fail the page.
        for i, path in enumerate(extras):
            self._add(
                f"extra:{i}",
                path,
                f"extra-{i}-{Path(path).name}",
                folder=False,
                required=False,
            )
        self.mailbox_lines = int((mailbox or {}).get("lines") or 400)

    def _add(
        self,
        name: str,
        source: str,
        local: str,
        folder: bool,
        required: bool = True,
    ) -> None:
        if is_remote(source):
            path = self.root / local
        else:
            path = Path(source).expanduser()
            if not path.is_absolute():
                path = self.base / path
        self._items[name] = (source, path, folder, required)

    def _path(self, name: str) -> Path | None:
        item = self._items.get(name)
        return item[1] if item else None

    @property
    def scores(self) -> Path:
        """Return the score table's path."""
        return self._items["scores"][1]

    def dataset(self, key: str) -> Path:
        """Return a dataset's per-checkpoint forecast folders.

        A dataset no source names is its folder in the scoring cache.
        """
        if f"dataset:{key}" not in self._items:
            self._add(
                f"dataset:{key}", f"{self._cache}/{key}", key, folder=True
            )
        return self._items[f"dataset:{key}"][1]

    def truth(self, key: str) -> Path | None:
        """Return a dataset's truth table, if it has one."""
        return self._path(f"truth:{key}")

    def extra(self, i: int) -> Path | None:
        """Return the `i`th extra file (a table or note)."""
        return self._path(f"extra:{i}")

    @property
    def mailbox(self) -> Path | None:
        """Return the notes file, if one is configured."""
        return self._path("mailbox")

    def commands(self) -> list[list[str]]:
        """Return the rsync command of each remote source."""
        out = []
        for source, path, folder, _ in self._items.values():
            if not is_remote(source):
                continue
            src = source.rstrip("/") + "/" if folder else source
            dst = f"{path}/" if folder else str(path)
            out.append([*RSYNC, src, dst])
        return out

    def refresh(self) -> str:
        """Copy each remote source; return what happened, for the page.

        The status starts `mirror failed` if a required source (scores,
        forecasts, truth) could not be copied; a failed notes file is
        reported but optional.
        """
        self.failed = []
        hard = []
        copied = 0
        for name, (source, path, folder, required) in self._items.items():
            if not is_remote(source):
                continue
            (path if folder else path.parent).mkdir(parents=True, exist_ok=True)
            cmd = [
                *RSYNC,
                source.rstrip("/") + "/" if folder else source,
                f"{path}/" if folder else str(path),
            ]
            try:
                got = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600
                )
                err = got.stderr.strip()[:200] if got.returncode else ""
            except (OSError, subprocess.SubprocessError) as e:
                err = str(e)[:200]
            if err:
                self.failed.append(f"{path.name} failed: {err}")
                if required:
                    hard.append(name)
            else:
                copied += 1
        if not copied and not self.failed:
            return "local"
        if hard:
            return "mirror failed: " + "; ".join(self.failed)
        return "; ".join([f"mirrored {copied} source(s)", *self.failed])
