"""Cancel placed jobs: stop each runner and what it runs.

    python3 workflow/scripts/remote/cancel.py JOBID...

A runner leads its own process group, its ssh in it, so the job on
the worker ends with its ssh session.
"""

import contextlib
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowlib import compute

if __name__ == "__main__":
    where = compute.state_dir(Path.cwd())
    for jid in sys.argv[1:]:
        with contextlib.suppress(OSError, ValueError):
            os.killpg(int((where / f"{jid}.pid").read_text()), signal.SIGTERM)
