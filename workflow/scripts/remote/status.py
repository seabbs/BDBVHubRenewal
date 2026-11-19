"""Report a placed job's state: running, success or failed.

    python3 workflow/scripts/remote/status.py JOBID

A job is done when its runner has written its exit code (after pulling
its files back); running while its runner lives; failed otherwise.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowlib import compute


def state(jid: str) -> str:
    """Return the state of job `jid`."""
    where = compute.state_dir(Path.cwd())
    done = where / f"{jid}.exit"
    if done.is_file():
        return "success" if done.read_text().strip() == "0" else "failed"
    try:
        os.kill(int((where / f"{jid}.pid").read_text()), 0)
    except (OSError, ValueError):
        # The runner may have just written its exit code.
        if done.is_file():
            return state(jid)
        return "failed"
    return "running"


if __name__ == "__main__":
    print(state(sys.argv[-1]))
