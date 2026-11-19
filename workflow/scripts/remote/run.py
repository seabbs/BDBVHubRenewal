"""Run one placed job on its worker (started by submit.py).

    python3 workflow/scripts/remote/run.py JOBID

Waits until the job fits on a worker beside the jobs placed there
(flowlib.compute.place; budgets from the launch), then runs it. On this
machine's own root and checkout the jobscript runs as it is. Elsewhere:
the worker's checkout must be at this commit; the inputs are pushed by
rsync; the worker's Snakemake runs the job with the coordinator's run
settings; then the log, and on success the outputs and
`params.copy_back`, are pulled back. The exit code is written last, so
the status script reports success only once the files are here.
"""

import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowlib import compute, localconfig


def say(text: str) -> None:
    """Print a timestamped line to the job's log."""
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}", flush=True)


def busy(state: Path, me: str) -> dict[str, list[dict]]:
    """Return the needs of the jobs running on each worker now."""
    out: dict[str, list[dict]] = {}
    for placed in state.glob("*.placed"):
        jid = placed.stem
        if jid == me or (state / f"{jid}.exit").exists():
            continue
        rec = json.loads(placed.read_text())
        try:
            os.kill(rec["pid"], 0)
        except OSError:
            continue
        out.setdefault(rec["worker"], []).append(rec["need"])
    return out


def wait_for_room(state: Path, jid: str, rec: dict) -> str:
    """Return the worker the job is placed on, once it fits."""
    need = compute.need(rec["job"])
    path = state / "budgets.json"
    budgets = json.loads(path.read_text()) if path.is_file() else {}
    budgets = {n: budgets.get(n, {}) for n in rec["workers"]}
    poll = float(os.environ.get("REMOTE_POLL") or 30)
    waited = False
    while True:
        with (state / "lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            name = compute.place(rec["target"], need, budgets, busy(state, jid))
            if name:
                mine = {"worker": name, "need": need, "pid": os.getpid()}
                (state / f"{jid}.placed").write_text(json.dumps(mine))
                return name
        if not waited:
            say(f"waiting for room on {rec['target']}")
            waited = True
        time.sleep(poll)


def rsync(src: str, dst: str, missing_ok: bool = False) -> int:
    """Copy one file or folder; return rsync's exit code."""
    cmd = ["rsync", "-a", "-e", " ".join(compute.SSH)]
    if missing_ok:
        cmd.append("--ignore-missing-args")
    return subprocess.run([*cmd, src, dst]).returncode


def files(value) -> list[str]:
    """Return a job field (a path, a list or a dict of them) as paths."""
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, str):
        value = [value]
    return [str(v) for v in value or []]


def elsewhere(cfg: dict, worker: compute.Worker, rec: dict) -> int:
    """Run the job on another root or checkout; return its exit code."""
    job, checkout = rec["job"], rec["checkout"]
    q = shlex.quote
    here = subprocess.run(
        ["git", "-C", checkout, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    there = subprocess.run(
        compute.on(worker, f"git -C {q(worker.checkout)} rev-parse HEAD"),
        capture_output=True,
        text=True,
    )
    if there.returncode or there.stdout.strip() != here:
        say(
            f"{worker.name}'s checkout is at commit {there.stdout.strip()!r},"
            f" not the coordinator's {here}"
        )
        return 1
    at = compute.mapper(cfg["ROOT"], checkout, worker)
    inputs = files(job.get("input"))
    back = files(job.get("output")) + files(job.get("log"))
    extra = files(job.get("params", {}).get("copy_back"))
    dirs = {str(Path(at(p)).parent) for p in inputs + back + extra}
    mk = "mkdir -p " + " ".join(q(d) for d in sorted(dirs))
    if subprocess.run(compute.on(worker, mk)).returncode:
        say(f"cannot make the job's folders on {worker.name}")
        return 1

    def addr(path: str) -> str:
        return path if worker.local else f"{worker.host}:{path}"

    for path in inputs:
        src, dst = path, at(path)
        if Path(path).is_dir():
            src, dst = path.rstrip("/") + "/", dst.rstrip("/") + "/"
        if rsync(src, addr(dst)):
            say(f"cannot push {path} to {worker.name}")
            return 1
    env = compute.job_env(cfg, worker)
    snakemake = cfg.get("WORKER_SNAKEMAKE") or "uv run --group paper snakemake"
    command = (
        f"cd {q(worker.checkout)} && env -u LOCAL_CONFIG "
        + " ".join(f"{k}={q(v)}" for k, v in env.items())
        + f" {snakemake} "
        + " ".join(q(a) for a in rec["argv"])
    )
    say(f"running on {worker.name}: {job['rule']}")
    rc = subprocess.run(compute.on(worker, command, compute.SSH_JOB)).returncode
    pull = back + extra if rc == 0 else files(job.get("log"))
    for path in pull:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        got = rsync(addr(at(path)), path, missing_ok=True)
        if got and rc == 0:
            say(f"cannot pull {path} from {worker.name}")
            rc = 1
    return rc


def main() -> int:
    """Run job `sys.argv[1]`, writing its exit code; return it."""
    jid = sys.argv[1]
    state = compute.state_dir(Path.cwd())
    rec = json.loads((state / f"{jid}.json").read_text())
    rc = 1
    try:
        cfg = localconfig.read()
        found = compute.workers(cfg, rec["checkout"])
        name = wait_for_room(state, jid, rec)
        worker = found[name]
        same = Path(worker.root) == Path(cfg["ROOT"]) and Path(
            worker.checkout
        ) == Path(rec["checkout"])
        if worker.local and same:
            script = state / f"{jid}.sh"
            script.write_text(rec["jobscript"])
            say(f"running here: {rec['job']['rule']}")
            rc = subprocess.run(["sh", str(script)]).returncode
        else:
            rc = elsewhere(cfg, worker, rec)
    finally:
        tmp = state / f"{jid}.exit.tmp"
        tmp.write_text(f"{rc}\n")
        tmp.replace(state / f"{jid}.exit")
        say(f"exit {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
