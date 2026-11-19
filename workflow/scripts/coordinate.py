"""The persistent coordinator run: one tick after another until stopped.

    python3 workflow/scripts/coordinate.py      # or `task run`

Each pass is a tick (workflow/scripts/tick.py: this machine's
SCHEDULE_TARGETS, with its lock and room checks). As soon as a pass
ends the next starts, after `COORDINATE_PAUSE` seconds (60), or
`COORDINATE_BACKOFF` (600) after a failed pass, so new releases and
batches are planned as soon as the run before ends. No timer runs it:
start it once (under nohup, tmux or a service of the machine's own)
and stop it with `touch output/coordinator.stop`, which ends it after
the running pass, or by its pid. Standard library only.
"""

import sys
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workflow"))
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))

from flowlib import localconfig  # noqa: E402

STOP = ROOT / "output" / "coordinator.stop"


def run(
    tick: Callable[[], int],
    stop: Path,
    pause: float,
    backoff: float,
    sleep: Callable[[float], object] = time.sleep,
) -> int:
    """Run `tick` until `stop` exists; remove it and return 0."""
    while not stop.exists():
        rc = tick()
        if stop.exists():
            break
        sleep(backoff if rc else pause)
    stop.unlink(missing_ok=True)
    return 0


def main() -> int:
    """Run the coordinator with this machine's settings."""
    import tick

    cfg = localconfig.read()
    return run(
        tick.main,
        STOP,
        pause=float(cfg.get("COORDINATE_PAUSE") or 60),
        backoff=float(cfg.get("COORDINATE_BACKOFF") or 600),
    )


if __name__ == "__main__":
    sys.exit(main())
