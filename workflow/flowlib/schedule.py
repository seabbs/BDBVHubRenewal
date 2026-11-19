"""Scheduled workflow runs: what one tick runs, and the timer that runs it.

A tick (`workflow/scripts/tick.py`) runs this machine's targets
(`SCHEDULE_TARGETS` in config/local.env, for example `score land` where
checkpoints are scored or `finetune publish` where they are trained).
The pool targets (`POOL`) are one Snakemake run at nice 19 with the
default profile and the machine's cores, memory and GPU less its
headroom (flowlib.machine), so whichever queued job fits starts as
another ends, by the rules' priorities (the trainer, then simulation,
then scoring). The outward targets, which write outside this machine,
follow in their configured order; `land` and `release` only after a
pool run that succeeded (`NEEDS_POOL`). Work that appears while a run is
going (a new checkpoint, a landed round) waits for the next tick's run,
so its delay is at most the longest running job. Snakemake locks the
worktree it runs in, so a machine has one driver worktree.
It runs only when there is room (`MIN_FREE_GB`, `MIN_DISK_GB`) and never
beside another tick (a lock). Snakemake reruns only what changed, so
`land` commits only new tables.

A timer runs the tick every `SCHEDULE_EVERY` minutes (default 60): a
cron line or a systemd user timer on Linux, a launchd agent on macOS
(`SCHEDULE_KIND`, or chosen from the platform). The units are made at
install (`workflow/scripts/schedule.py install`), from where the
repository is on this machine, so nothing machine-specific is committed.
"""

from __future__ import annotations

import fcntl
import os
import plistlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from flowlib import compute

TARGETS = (
    "simulate",
    "push",
    "release",
    "score",
    "land",
    "finetune",
    "publish",
    "dashboard",
)
"""The targets a tick may run, each with workflow/profiles/default."""

POOL = ("simulate", "push", "finetune", "score", "dashboard")
"""Targets that share one Snakemake run and the machine's budget."""

NEEDS_POOL = {"land", "release"}
"""Outward targets that run only after a pool run that succeeded: each
would otherwise rerun the failed scoring or simulation it depends on.
`publish` reads only saved checkpoints, so it always runs."""

LISTINGS = {"score": "release_list"}
"""The list each target plans from, refreshed each tick: `score` the
checkpoint releases. It is refreshed in a run of its own, the tick's
first step: Snakemake 9 ends a run after a forced checkpoint without the
jobs it unlocks. A failed refresh leaves the last list, and the pool
still runs. `finetune` lists its batches itself each time a run plans."""

MARK = "# bdbvhubrenewal workflow tick"
"""Ends this repository's cron line, so it can be replaced or removed."""

LABEL = "org.bdbvhubrenewal.workflow-tick"
"""The systemd unit and launchd agent name."""

TICK = "workflow/scripts/tick.py"


def _snakemake(cfg: Mapping[str, str], targets: list[str]) -> list[str]:
    """Return the Snakemake run of `targets` on this machine's budget."""
    extra = []
    if "finetune" in targets:
        # One segment a run, so the next run plans afresh.
        extra += ["--config", "next_segment_only=True"]
    return [
        "nice",
        "-n",
        "19",
        "uv",
        "run",
        "--group",
        "paper",
        "snakemake",
        *targets,
        *compute.args(cfg),
        *extra,
    ]


def plan(cfg: Mapping[str, str]) -> list[tuple[str, list[str], bool]]:
    """Return each step of a tick, in order.

    A step is its name, its command and whether a failure lets the tick go
    on (the refresh and the pool run). The listings the pool plans from
    are refreshed first (`LISTINGS`), then the pool targets are one run
    named by its targets, then each outward target is a step of its own.

    Raises
    ------
    ValueError
        If a target is not one a tick may run.
    """
    targets = cfg.get("SCHEDULE_TARGETS", "").split()
    bad = [t for t in targets if t not in TARGETS]
    if bad:
        raise ValueError(f"targets {bad} are not among {sorted(TARGETS)}")
    pool = [t for t in targets if t in POOL]
    lists = [LISTINGS[t] for t in pool if t in LISTINGS]
    out = []
    if lists:
        cmd = [*_snakemake(cfg, lists), "--forcerun", *lists]
        out.append(("refresh", cmd, True))
    if pool:
        out.append((" ".join(pool), _snakemake(cfg, pool), True))
    out += [(t, _snakemake(cfg, [t]), False) for t in targets if t not in POOL]
    return out


def _free_gb(meminfo: Path) -> int | None:
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1048576
    except OSError:
        return None
    return None


def room(
    cfg: Mapping[str, str],
    where: Path | str,
    meminfo: Path | str = "/proc/meminfo",
) -> tuple[bool, str]:
    """Return whether there is memory and disk for a tick, and why not.

    Memory is checked where `/proc/meminfo` exists (Linux); disk at
    `where`.
    """
    need_mem = int(cfg.get("MIN_FREE_GB") or 0)
    need_disk = int(cfg.get("MIN_DISK_GB") or 0)
    free = _free_gb(Path(meminfo))
    if free is not None and free < need_mem:
        return False, f"{free} GB memory available, {need_mem} GB needed"
    st = os.statvfs(where)
    disk = st.f_bavail * st.f_frsize // 2**30
    if disk < need_disk:
        return False, f"{disk} GB disk free, {need_disk} GB needed"
    return True, ""


@contextmanager
def locked(path: Path | str) -> Iterator[bool]:
    """Hold a lock on `path` if no other tick does; yield whether it is held."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def cron_spec(every: int) -> str:
    """Return a cron schedule for every `every` minutes, off the hour."""
    if every < 60 and 60 % every == 0:
        return f"*/{every} * * * *"
    if every % 60 == 0:
        hours = every // 60
        return "7 * * * *" if hours == 1 else f"7 */{hours} * * *"
    return f"*/{every} * * * *"


def _command(root: Path, path: str) -> str:
    log = root / "output" / "workflow-tick.log"
    env = f"PATH={path} " if path else ""
    return f"cd {root} && {env}python3 {TICK} >> {log} 2>&1"


def _log(root: Path) -> Path:
    """Return the tick's log, making its folder.

    A timer cannot open a log in a folder that does not exist, and so
    never runs the tick.
    """
    (root / "output").mkdir(parents=True, exist_ok=True)
    return root / "output" / "workflow-tick.log"


def cron_line(root: Path | str, every: int, path: str = "") -> str:
    """Return this repository's cron line; `path` is the PATH to run with."""
    _log(Path(root))
    return f"{cron_spec(every)} {_command(Path(root), path)} {MARK}"


def merge_crontab(old: str, line: str | None) -> str:
    """Return a crontab with this repository's line replaced (or removed)."""
    kept = [x for x in old.splitlines() if not x.rstrip().endswith(MARK)]
    if line:
        kept.append(line)
    return "".join(f"{x}\n" for x in kept)


def systemd_units(
    root: Path | str, every: int, path: str = ""
) -> tuple[str, str]:
    """Return the systemd user service and timer of the tick."""
    root = Path(root)
    log = _log(root)
    env = f"Environment=PATH={path}\n" if path else ""
    service = (
        "[Unit]\nDescription=bdbv-hub-modelling workflow tick\n\n"
        "[Service]\nType=oneshot\n"
        f"WorkingDirectory={root}\n{env}"
        f"ExecStart=/usr/bin/env python3 {TICK}\n"
        f"StandardOutput=append:{log}\n"
        f"StandardError=append:{log}\n"
    )
    timer = (
        "[Unit]\nDescription=Run the bdbv-hub-modelling workflow tick\n\n"
        f"[Timer]\nOnBootSec=5min\nOnUnitActiveSec={every}min\n"
        "Persistent=true\n\n[Install]\nWantedBy=timers.target\n"
    )
    return service, timer


def launchd_plist(root: Path | str, every: int, path: str = "") -> bytes:
    """Return the launchd agent of the tick; `path` is the PATH to run with."""
    root = Path(root)
    log = str(_log(root))
    extra = {"EnvironmentVariables": {"PATH": path}} if path else {}
    return plistlib.dumps(
        {
            **extra,
            "Label": LABEL,
            "ProgramArguments": ["/usr/bin/env", "python3", TICK],
            "WorkingDirectory": str(root),
            "StartInterval": every * 60,
            "StandardOutPath": log,
            "StandardErrorPath": log,
            "RunAtLoad": False,
        }
    )


def timer_kind(
    cfg: Mapping[str, str], platform: str, has_systemd: bool = False
) -> str:
    """Return the timer to use: `SCHEDULE_KIND`, else by platform."""
    if cfg.get("SCHEDULE_KIND"):
        return cfg["SCHEDULE_KIND"]
    if platform == "darwin":
        return "launchd"
    return "systemd" if has_systemd else "cron"
