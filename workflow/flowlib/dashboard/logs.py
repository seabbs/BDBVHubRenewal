"""A training run's state: its log, checkpoints and trainer process.

A run's folder is `scripts/finetune_stream.py`'s output folder: its logs,
`train/n<N>.log` per segment or an older run's `train.log` (a line per
logged step), `train/n<N>/settings.json` per checkpoint and `trainer.pid`.
Everything here only reads.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

STEP = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \[train\] step (\d+): "
    r"loss ([\d.eE+-]+), held-out \([^)]*\) ([\d.eE+-]+)"
    r"(?: \(nowcast ([\d.eE+-]+), forecast ([\d.eE+-]+)\))?"
    r"(?:, by source ((?:[\w+]+ [\d.eE+-]+(?:, |$))+))?"
)
"""A logged step of the trainer: time, step, loss, held-out loss, its
nowcast and forecast parts if logged, and the held-out loss by source
(any source names) every few hundred steps. The bias relations' means
and ranges after it are read with `BIAS`."""

STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \[\w+\] (.*)$")
"""Any line of the trainer: its time and message."""

SOURCE = re.compile(r"([\w+]+) ([\d.eE+-]+)")
"""One source and its loss in a step's `by source` list."""

NUM = r"[+-]?[\d.]+(?:e[+-]?\d+)?"
BIAS = re.compile(rf"(\w+?)_? ({NUM}) \[({NUM}), ({NUM})\]")
"""One bias relation's mean and range in a step's `bias` list."""

CHUNK = re.compile(
    r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \[train\] outbreaks (\d+)-(\d+): "
    r"\d+ blocks, \d+ series, (\d+) steps"
)
"""A chunk the trainer plans: its first and last simulation and steps."""

OLD_LOGS = ("train.log", "train-launcher.log")
"""An older run's single log, as written and as renamed beside its
segment logs."""


def _float(x: str | None) -> float | None:
    return float(x) if x is not None else None


class Log:
    """A log file, read incrementally: whole lines, new ones appended."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.offset = 0
        self._rows: dict[int, dict[str, Any]] = {}
        self._chunks: dict[int, tuple[int, int]] = {}

    def _update(self) -> None:
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self.offset:
            self.offset, self._rows, self._chunks = 0, {}, {}
        if size == self.offset:
            return
        with self.path.open("rb") as f:
            f.seek(self.offset)
            data = f.read()
        end = data.rfind(b"\n") + 1
        self.offset += end
        for line in data[:end].decode("utf-8", "replace").splitlines():
            c = CHUNK.match(line)
            if c:
                a, b, n = map(int, c.groups())
                self._chunks[a - 1] = (b, n)
                continue
            m = STEP.match(line)
            if not m:
                continue
            g = m.groups()
            cut = line.find(", bias ")
            biased = line[cut + 7 :] if cut >= 0 else ""
            t = datetime.strptime(g[0], "%Y-%m-%d %H:%M:%S").timestamp()
            self._rows[int(g[1])] = {
                "t": t,
                "step": int(g[1]),
                "loss": _float(g[2]),
                "held": _float(g[3]),
                "parts": (
                    {"nowcast": float(g[4]), "forecast": float(g[5])}
                    if g[4] is not None
                    else {}
                ),
                "src": {k: float(v) for k, v in SOURCE.findall(g[6] or "")},
                "bias": {
                    k: {"mean": float(v), "min": float(lo), "max": float(hi)}
                    for k, v, lo, hi in BIAS.findall(biased)
                },
            }

    def rows(self) -> list[dict[str, Any]]:
        """Return the logged steps so far, in step order."""
        self._update()
        return [self._rows[k] for k in sorted(self._rows)]

    def chunks(self) -> dict[int, tuple[int, int]]:
        """Return the planned chunks: simulations before -> (last, steps).

        A chunk planned twice (a resumed segment) keeps its newest line.
        """
        self._update()
        return dict(self._chunks)


class RunLog:
    """A run's logged steps from every log it writes, each read as a `Log`.

    A run writes `train/n<N>.log`, one per segment, and an older run
    `train.log` (`train-launcher.log` once renamed). New segment logs are
    found on each read, a file reached by two names (a link) is read
    once, and a step logged twice keeps its newest row.
    """

    def __init__(self, folder: Path | str) -> None:
        self.folder = Path(folder)
        self._logs: dict[Path, Log] = {}

    def _paths(self) -> list[Path]:
        found = sorted(
            self.folder.glob("train/n*.log"), key=lambda p: _size(p.name)
        )
        old = [self.folder / n for n in OLD_LOGS]
        return [p for p in old if p.exists()] + found

    def _each(self) -> list[Log]:
        out = []
        for path in self._paths():
            real = path.resolve()
            if real not in self._logs:
                self._logs[real] = Log(real)
            if self._logs[real] not in out:
                out.append(self._logs[real])
        return out

    def chunks(self) -> dict[int, tuple[int, int]]:
        """Return every log's planned chunks (`Log.chunks`)."""
        merged: dict[int, tuple[int, int]] = {}
        for log in self._each():
            merged |= log.chunks()
        return merged

    def last_line(self) -> tuple[float, str] | None:
        """Return the time and text of the newest log's last line.

        The newest log is the one written last, the running segment's
        `train/n<N>.log`: its last line may be any of the trainer's lines,
        not only a logged step, so a checkpoint's evaluation or a wait
        for simulations does not look stale.
        """
        paths = [p for p in self._paths() if p.exists()]
        if not paths:
            return None
        newest = max(paths, key=lambda p: (p.stat().st_mtime, _size(p.name)))
        with newest.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            lines = f.read().decode("utf-8", "replace").splitlines()
        for line in reversed(lines):
            m = STAMP.match(line)
            if m:
                t = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S").timestamp()
                return t, m[2]
        return None

    def rows(self) -> list[dict[str, Any]]:
        """Return the logged steps of every log, in step order."""
        merged: dict[int, dict[str, Any]] = {}
        for log in self._each():
            for row in log.rows():
                if (
                    row["step"] not in merged
                    or row["t"] >= merged[row["step"]]["t"]
                ):
                    merged[row["step"]] = row
        return [merged[k] for k in sorted(merged)]


class JsonCache:
    """JSON files, read again only when their size or time changes."""

    def __init__(self) -> None:
        self._seen: dict[str, tuple[tuple[int, int], Any]] = {}

    def get(self, path: Path | str) -> Any:
        """Return a JSON file's content, or None if it is missing or bad."""
        try:
            st = Path(path).stat()
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size)
        p = str(path)
        if p not in self._seen or self._seen[p][0] != key:
            try:
                value = json.loads(Path(path).read_text())
            except (OSError, ValueError):
                value = None
            self._seen[p] = (key, value)
        return self._seen[p][1]


CACHE = JsonCache()


def _size(name: str) -> int:
    digits = re.sub(r"\D", "", name)
    return int(digits) if digits else 0


def checkpoints(folder: Path | str) -> list[tuple[str, dict, float | None]]:
    """Return each saved checkpoint's name, settings and time, by size."""
    out = []
    for d in sorted(Path(folder).glob("train/n*"), key=lambda p: _size(p.name)):
        s = CACHE.get(d / "settings.json")
        if isinstance(s, dict):
            try:
                mtime = (d / "settings.json").stat().st_mtime
            except OSError:
                mtime = None
            out.append((d.name, s, mtime))
    return out


def trainer_args(folder: Path | str) -> str:
    """Return the running trainer's command line, or an empty string."""
    try:
        pid = int((Path(folder) / "trainer.pid").read_text().split()[0])
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return ""
    return out.stdout


def prefixes(
    folder: Path | str, args: str, default: Sequence[int]
) -> list[int]:
    """Return the run's checkpoint sizes.

    The trainer's `--prefixes` if its command line is given, else the
    saved checkpoints with `default`.
    """
    m = re.search(r"--prefixes((?:\s+\d+)+)", args)
    if m:
        return sorted(int(x) for x in m.group(1).split())
    saved = {_size(d.name) for d in Path(folder).glob("train/n*")}
    return sorted((saved - {0}) | set(default))


FRESH_S = 600
"""Seconds within which a stream status shows the trainer running: it is
written every 10 steps, and the log tile turns red after 10 minutes."""


def _pid_state(folder: Path | str) -> tuple[int | None, bool | None]:
    """Return `trainer.pid`'s pid and whether it runs (None if unknown)."""
    try:
        pid = int((Path(folder) / "trainer.pid").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None, None
    try:
        os.kill(pid, 0)
    except PermissionError:
        return pid, True
    except OSError:
        return pid, False
    return pid, True


def trainer_state(folder: Path | str) -> tuple[str, bool | None]:
    """Return a description of the trainer process and whether it runs.

    It runs if `trainer.pid`'s process does, or if the stream status
    (`train/stream-status.json`) was written in the last `FRESH_S`
    seconds: a trainer started by Snakemake writes no pid file, so one
    left by an old launcher says nothing. A finished or stopped segment
    is neither running nor dead.
    """
    pid, running = _pid_state(folder)
    if running:
        return f"alive (pid {pid})", True
    path = Path(folder) / "train" / "stream-status.json"
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        age = None
    if age is not None and age <= FRESH_S:
        return f"alive (status {age / 60:.0f} min ago)", True
    stage = str((CACHE.get(path) or {}).get("stage", ""))
    if stage == "finished" or stage.startswith("stopped at"):
        return stage, None
    if pid is None and age is None:
        return "unknown", None
    return f"DEAD (pid {pid})" if pid else "DEAD (no recent status)", False
