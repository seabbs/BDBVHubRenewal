"""What this machine's workflow run is doing, for the monitor.

Each tick (`workflow/scripts/tick.py`) writes `<ROOT>/status/<host>.json`:
the step it runs, whether a run is going, when it last wrote, and the
running and recent jobs of its Snakemake run (rule, wildcards, the
resources that place a job, start and end), read from the run's own log
(`.snakemake/log`). With a rounds host (`REMOTE`) the file is copied to
its `status/` folder too, so the monitor (`workflow/scripts/monitor.py`)
sees every machine in one place. Standard library only.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

RESOURCES = ("mem_mb", "gpu", "sim_slots", "threads")
"""The resources shown: those that decide where and when a job runs."""

RECENT = 20
"""Finished or failed jobs shown."""

_STAMP = re.compile(r"^\[(\w{3} \w{3} +\d+ \d\d:\d\d:\d\d \d{4})\]$")
_START = re.compile(r"^(?:local)?rule (\w+):$")
_DONE = re.compile(r"^Finished jobid: (\d+)")
_ERROR = re.compile(r"^Error in rule (\w+):$")


def _time(stamp: str) -> str:
    return datetime.strptime(stamp, "%a %b %d %H:%M:%S %Y").isoformat()


def parse_log(text: str) -> list[dict]:
    """Return each job of a Snakemake log, its last attempt's state.

    A job is `running` from its start until `Finished` (`done`) or an
    error (`failed`); a retry starts it again under the same jobid.
    """
    jobs: dict[int, dict] = {}
    stamp = ""
    opened: dict | None = None
    current: dict | None = None
    erred = False
    for raw in text.splitlines():
        line = raw.strip()
        if m := _STAMP.match(line):
            stamp, opened, current, erred = _time(m[1]), None, None, False
        elif m := _START.match(line):
            opened = {"rule": m[1], "started": stamp, "state": "running"}
        elif m := _DONE.match(line):
            if (job := jobs.get(int(m[1]))) is not None:
                job.update(state="done", ended=stamp)
        elif _ERROR.match(line):
            erred = True
        elif line.startswith("jobid:"):
            jobid = int(line.split(":", 1)[1])
            if opened is not None:
                current = jobs[jobid] = {
                    "jobid": jobid,
                    "wildcards": "",
                    "resources": {},
                    "ended": "",
                    **opened,
                }
            elif erred and jobid in jobs:
                jobs[jobid].update(state="failed", ended=stamp)
        elif current is not None and line.startswith("wildcards:"):
            current["wildcards"] = line.split(":", 1)[1].strip()
        elif current is not None and line.startswith("resources:"):
            pairs = (p.split("=", 1) for p in line.split(":", 1)[1].split(","))
            current["resources"] = {
                k.strip(): v.strip()
                for k, v in (p for p in pairs if len(p) == 2)
                if k.strip() in RESOURCES
            }
    return list(jobs.values())


def newest_log(repo: Path | str) -> Path | None:
    """Return the newest Snakemake log of the worktree `repo`, if any."""
    logs = sorted(Path(repo, ".snakemake", "log").glob("*.snakemake.log"))
    return logs[-1] if logs else None


def host(cfg: Mapping[str, str]) -> str:
    """Return the machine's name: HOST_NAME, else its host name."""
    return cfg.get("HOST_NAME") or socket.gethostname().split(".")[0]


def status(
    cfg: Mapping[str, str],
    step: str,
    log: Path | None,
    alive: bool,
    now: str | None = None,
) -> dict:
    """Return this machine's status: its step, run and jobs."""
    jobs = parse_log(log.read_text(errors="replace")) if log else []
    running = sorted(
        (j for j in jobs if j["state"] == "running"),
        key=lambda j: j["started"],
    )
    ended = sorted(
        (j for j in jobs if j["state"] != "running"),
        key=lambda j: j["ended"],
        reverse=True,
    )
    return {
        "host": host(cfg),
        "step": step,
        "alive": alive,
        "updated": now or datetime.now().isoformat(timespec="seconds"),
        "every": int(cfg.get("STATUS_EVERY") or 60),
        "log": log.name if log else "",
        "running": running if alive else [],
        "recent": ended[:RECENT],
    }


def write(got: dict, cfg: Mapping[str, str]) -> Path:
    """Write a status whole to `<ROOT>/status/<host>.json`; return it.

    With a rounds host (`REMOTE`), copy it to its `status/` folder too; a
    copy that fails is left to the next write.
    """
    path = Path(cfg["ROOT"]).expanduser() / "status" / f"{got['host']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(got, indent=1))
    tmp.replace(path)
    if cfg.get("REMOTE"):
        there = f"{cfg.get('REMOTE_ROOT') or cfg['ROOT']}/status/"
        subprocess.run(
            ["rsync", "-a", str(path), f"{cfg['REMOTE']}:{there}"],
            capture_output=True,
            check=False,
        )
    return path
