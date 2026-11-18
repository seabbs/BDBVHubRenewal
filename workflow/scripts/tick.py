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

from flowlib import localconfig, schedule  # noqa: E402


def say(text: str) -> None:
    """Print a timestamped line (the tick's log)."""
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}", flush=True)


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
            say("another tick is running; skipped")
            return 0
        ok, why = schedule.room(cfg, out)
        if not ok:
            say(f"skipped: {why}")
            return 0
        failed = 0
        for target, cmd, pool in steps:
            if failed and target in schedule.NEEDS_POOL:
                say(f"{target}: skipped after a failed run")
                continue
            say(f"{target}: start")
            rc = subprocess.run(cmd, cwd=ROOT).returncode
            say(f"{target}: {'done' if rc == 0 else f'FAILED (exit {rc})'}")
            if rc and not pool:
                # An outward target's failure stops the tick.
                return rc
            failed = failed or rc
    return failed


if __name__ == "__main__":
    sys.exit(main())
