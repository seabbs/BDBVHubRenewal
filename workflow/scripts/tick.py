"""Run this machine's scheduled workflow targets once (one tick).

    python3 workflow/scripts/tick.py

What a tick runs, and in what order, is `flowlib.schedule.plan`'s
(`SCHEDULE_TARGETS` in config/local.env). It is skipped when another
tick is running (a lock in output/) or there is too little memory or
disk (`MIN_FREE_GB`, `MIN_DISK_GB`). Standard library only, so a timer
can run it with any python3.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workflow"))

from flowlib import localconfig, runstatus, schedule  # noqa: E402


def say(text: str) -> None:
    """Print a timestamped line (the tick's log)."""
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}", flush=True)


def report(cfg: dict[str, str], step: str, alive: bool) -> None:
    """Write this machine's status for the monitor (`flowlib.runstatus`).

    The newest run's jobs are shown, as running only while it is alive.
    """
    log = runstatus.newest_log(ROOT)
    runstatus.write(runstatus.status(cfg, step, log, alive), cfg)


def watch(cmd: list[str], step: str, cfg: dict[str, str]) -> int:
    """Run a step, writing the status every STATUS_EVERY seconds."""
    every = int(cfg.get("STATUS_EVERY") or 60)
    proc = subprocess.Popen(cmd, cwd=ROOT)
    while True:
        try:
            return proc.wait(timeout=every)
        except subprocess.TimeoutExpired:
            report(cfg, step, alive=True)


def main() -> int:
    """Run one tick; return its exit code."""
    cfg = localconfig.read()
    steps = schedule.plan(cfg)
    if not steps:
        say("no SCHEDULE_TARGETS set; nothing to run")
        return 0
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    with schedule.locked(out / "workflow-tick.lock") as held:
        if not held:
            # The running tick reports for itself.
            say("another tick is running; skipped")
            return 0
        ok, why = schedule.room(cfg, out)
        if not ok:
            say(f"skipped: {why}")
            report(cfg, f"skipped: {why}", alive=False)
            return 0
        failed = 0
        for target, cmd, pool in steps:
            if failed and target in schedule.NEEDS_POOL:
                say(f"{target}: skipped after a failed run")
                continue
            say(f"{target}: start")
            report(cfg, target, alive=True)
            rc = watch(cmd, target, cfg)
            say(f"{target}: {'done' if rc == 0 else f'FAILED (exit {rc})'}")
            if rc and not pool:
                # An outward target's failure stops the tick.
                report(cfg, f"failed: {target}", alive=False)
                return rc
            failed = failed or rc
        report(cfg, "idle", alive=False)
    return failed


if __name__ == "__main__":
    sys.exit(main())
