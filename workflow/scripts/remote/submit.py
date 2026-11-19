"""Place a Snakemake job on a worker (COMPUTE=ssh; flowlib.compute).

    python3 workflow/scripts/remote/submit.py JOBSCRIPT

Run by the coordinator's Snakemake in its checkout. Fails at once if
the job's worker (its group's, else every worker) does not answer over
ssh. Otherwise records the job, starts its runner (run.py) detached and
prints the job's id; the runner waits for room on a worker, then runs it.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowlib import compute, localconfig


def main() -> int:
    """Submit the job in `sys.argv[-1]`; return the exit code."""
    script = Path(sys.argv[-1])
    cfg = localconfig.read()
    checkout = Path.cwd()
    found = compute.workers(cfg, checkout)
    job, argv = compute.read_jobscript(script.read_text())
    group = compute.group_of(job["rule"])
    placed = compute.groups(cfg, list(found))
    target = placed.get(group, "any")
    names = list(found) if target == "any" else [target]
    up = [n for n in names if compute.reachable(found[n])]
    if not up:
        hosts = ", ".join(found[n].host for n in names)
        print(f"{job['rule']}: no worker answers ({hosts})", file=sys.stderr)
        return 1
    state = compute.state_dir(checkout)
    state.mkdir(parents=True, exist_ok=True)
    jid = f"{job['jobid']}-{time.time_ns()}"
    record = {
        "job": job,
        "argv": argv,
        "target": target if target != "any" else "any",
        "workers": up,
        "checkout": str(checkout),
        "jobscript": script.read_text(),
    }
    (state / f"{jid}.json").write_text(json.dumps(record))
    with (state / f"{jid}.log").open("ab") as log:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("run.py")), jid],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=os.environ,
        )
    (state / f"{jid}.pid").write_text(str(proc.pid))
    print(jid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
