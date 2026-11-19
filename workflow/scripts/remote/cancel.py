"""Cancel placed jobs: stop each runner and what it runs.

    python3 workflow/scripts/remote/cancel.py JOBID...

A runner leads its own process group, its ssh in it, so killing the
group stops the runner. The job's Snakemake on a worker is stopped by
the pid it recorded there (flowlib.compute.remote_pid_file), as a
session that ends does not always end the command it ran.
"""

import contextlib
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowlib import compute, localconfig


def stop_remote(where: Path, jid: str) -> None:
    """Stop job `jid`'s Snakemake on the worker it was placed on."""
    placed = where / f"{jid}.placed"
    if not placed.is_file():
        return
    rec = json.loads((where / f"{jid}.json").read_text())
    worker = compute.workers(localconfig.read(), rec["checkout"]).get(
        json.loads(placed.read_text())["worker"]
    )
    if worker is None:
        return
    pid = shlex.quote(compute.remote_pid_file(worker, rec["job"]))
    subprocess.run(
        compute.on(worker, f'[ -f {pid} ] && kill "$(cat {pid})"'),
        capture_output=True,
        timeout=30,
    )


if __name__ == "__main__":
    where = compute.state_dir(Path.cwd())
    for jid in sys.argv[1:]:
        with contextlib.suppress(OSError, ValueError):
            os.killpg(int((where / f"{jid}.pid").read_text()), signal.SIGTERM)
        with contextlib.suppress(
            OSError, ValueError, KeyError, subprocess.TimeoutExpired
        ):
            stop_remote(where, jid)
