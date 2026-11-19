"""Where the workflow's jobs run: this machine, or workers over ssh.

`COMPUTE` in config/local.env picks how (`single` by default):

- `single`: one machine, `workflow/profiles/default` with its budget
  (flowlib.machine), as before.
- `ssh`: this machine is the coordinator, the one Snakemake run, and
  places each job on a worker (`workflow/profiles/ssh`, the
  cluster-generic executor with `workflow/scripts/remote/`). `WORKERS`
  lists them as `name=host:ROOT:CHECKOUT` (`host` as ssh names it, or
  `local`, which may leave out ROOT and CHECKOUT for this machine's).
  `WORKER_GROUPS` maps rule groups to a worker
  (`train=laptop score=archie`); a group left out, or mapped to `any`,
  runs on the first worker with room. The pool is the sum of the
  workers' budgets, each read from the worker itself at launch.
- any other name: its profile, `workflow/profiles/<name>`, alone (a
  cluster or cloud executor with its own resources; none ships).

A rule names no machine: it declares its threads, memory and GPU, and
its group is `RULE_GROUPS`'. Local rules always run on the coordinator.
A job on another root or checkout has its inputs pushed by rsync before
it runs and its outputs, log and `params.copy_back` files pulled back
after, all at the same paths under the worker's ROOT or checkout. Its
checkout must be at the coordinator's commit. It runs with the
coordinator's run settings (`job_env`) and its own machine settings.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parents[1]))

from flowlib import localconfig, machine

MODES = ("single", "ssh")
"""The modes this module sets the pool of; any other `COMPUTE` names a
profile, `workflow/profiles/<name>`, whose executor brings its own."""

GROUPS = ("simulate", "train", "publish", "score", "any")
"""The rule groups `WORKER_GROUPS` in the local config can place."""

RULE_GROUPS = {
    "simulate_batch": "simulate",
    "epinowcast_lib": "simulate",
    "train_segment": "train",
    "real_validation": "train",
    "verify_checkpoint": "publish",
    "publish_checkpoint": "publish",
    "score_germany": "score",
    "slots_check_shard": "score",
    "agg_sanity": "score",
    "agg_fit": "score",
    "agg_rows": "score",
}
"""Each placed rule's group; any other rule is in `any`."""

GPU_RULES = frozenset({"train_segment", "score_germany"})
"""The rules taking the `gpu` resource. Placed on a worker by their
group, each also takes that worker's own `gpu_<worker>`, so the pooled
GPUs are not shared across workers."""

MACHINE_KEYS = frozenset(
    {
        "HOST_NAME",
        "MAILBOX",
        "INBOX",
        "NOTIFY",
        "JOB_PREFIX",
        "SNAKEMAKE_PROFILE",
        "HEADROOM_CORES",
        "HEADROOM_GB",
        "CORES",
        "MEM_MB",
        "SIM_SLOTS",
        "DEVICE",
        "TARGETS",
        "SCHEDULE_TARGETS",
        "SCHEDULE_EVERY",
        "SCHEDULE_KIND",
        "COORDINATE_PAUSE",
        "COORDINATE_BACKOFF",
        "STATUS_EVERY",
        "MONITOR_HOST",
        "MONITOR_PORT",
        "MONITOR_REFRESH",
        "DASHBOARD_RUNS",
        "MIN_FREE_GB",
        "MIN_DISK_GB",
        "PAPER_LOGS",
        "COMPUTE",
        "WORKERS",
        "WORKER_GROUPS",
        "WORKER_SNAKEMAKE",
    }
)
"""Local settings a worker keeps its own of; it takes the rest of
`localconfig.DEFAULTS` from the coordinator."""

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
# A job's ssh gives up after 10 minutes without an answer (a laptop
# asleep), so the job fails and its retry resumes it.
SSH_JOB = [*SSH, "-o", "ServerAliveInterval=60", "-o", "ServerAliveCountMax=10"]


@dataclass(frozen=True)
class Worker:
    """A machine jobs run on: its ssh host (or `local`), ROOT and checkout."""

    name: str
    host: str
    root: str
    checkout: str

    @property
    def local(self) -> bool:
        """Whether the worker is this machine."""
        return self.host == "local"


def mode(
    cfg: Mapping[str, str], checkout: str | Path = localconfig.ROOT
) -> str:
    """Return `COMPUTE` (`single` when unset).

    Raises `ValueError` for a name not in `MODES` without a profile in
    `workflow/profiles/<name>`.
    """
    got = cfg.get("COMPUTE") or "single"
    profile = f"workflow/profiles/{got}"
    if got not in MODES and not (Path(checkout) / profile).is_dir():
        raise ValueError(f"COMPUTE={got}: no profile {profile}")
    return got


def workers(
    cfg: Mapping[str, str], checkout: str | Path = localconfig.ROOT
) -> dict[str, Worker]:
    """Return the workers of `WORKERS`, in order (only for `ssh`)."""
    if mode(cfg, checkout) != "ssh":
        return {}
    out = {}
    for spec in cfg.get("WORKERS", "").split():
        name, sep, rest = spec.partition("=")
        host, _, paths = rest.partition(":")
        root, _, where = paths.partition(":")
        if not sep or not host:
            raise ValueError(f"not a worker (name=host:ROOT:CHECKOUT): {spec}")
        if host == "local":
            root = root or cfg["ROOT"]
            where = where or str(checkout)
        if not root or not where:
            raise ValueError(f"worker {name} needs host:ROOT:CHECKOUT: {spec}")
        out[name] = Worker(name, host, root, where)
    return out


def groups(cfg: Mapping[str, str], names: Sequence[str]) -> dict[str, str]:
    """Return `WORKER_GROUPS`: each group's worker, or `any`."""
    out = {}
    for spec in cfg.get("WORKER_GROUPS", "").split():
        group, _, worker = spec.partition("=")
        if group not in GROUPS:
            raise ValueError(f"no group {group} (groups: {GROUPS})")
        if worker != "any" and worker not in names:
            raise ValueError(f"group {group} on {worker}: not in WORKERS")
        out[group] = worker
    return out


def group_of(rule: str) -> str:
    """Return a rule's group (`any` if it has none)."""
    return RULE_GROUPS.get(rule, "any")


def mapper(
    root: str | Path, checkout: str | Path, worker: Worker
) -> Callable[[str], str]:
    """Return the function giving a coordinator path's place on `worker`.

    A path under the coordinator's ROOT goes under the worker's; one
    relative or under the checkout goes under the worker's checkout.
    Another path raises `ValueError`, as the worker has no place for it.
    """
    root, checkout = Path(root), Path(checkout)

    def at(path: str) -> str:
        p = Path(path)
        if not p.is_absolute():
            return str(Path(worker.checkout) / p)
        for base, there in ((root, worker.root), (checkout, worker.checkout)):
            if p.is_relative_to(base):
                return str(Path(there) / p.relative_to(base))
        raise ValueError(f"{path} is under neither ROOT nor the checkout")

    return at


def job_env(cfg: Mapping[str, str], worker: Worker) -> dict[str, str]:
    """Return the settings a job on `worker` runs with.

    The coordinator's run settings, paths under its ROOT moved to the
    worker's, and `COMPUTE=single`; the worker keeps its own machine
    settings (`MACHINE_KEYS`, and any not in `localconfig.DEFAULTS`).
    """
    root = cfg["ROOT"]
    out = {}
    for key, _ in localconfig.DEFAULTS:
        if key in MACHINE_KEYS or key not in cfg:
            continue
        value = cfg[key]
        if value == root or value.startswith(root.rstrip("/") + "/"):
            value = worker.root + value[len(root.rstrip("/")) :]
        out[key] = value
    out["COMPUTE"] = "single"
    return out


def fits(
    need: Mapping[str, int],
    budget: Mapping[str, int],
    running: Sequence[Mapping[str, int]],
) -> bool:
    """Whether a job needing `need` fits beside `running` in `budget`."""
    return all(
        need.get(k, 0) + sum(r.get(k, 0) for r in running) <= budget[k]
        for k in ("cores", "mem_mb", "gpu")
        if k in budget
    )


def place(
    target: str,
    need: Mapping[str, int],
    budgets: Mapping[str, Mapping[str, int]],
    busy: Mapping[str, Sequence[Mapping[str, int]]],
) -> str | None:
    """Return the worker to run on now, or None to wait.

    `target` is the group's worker, or `any` for the first worker in
    order where the job fits.
    """
    names = list(budgets) if target == "any" else [target]
    for name in names:
        if fits(need, budgets.get(name, {}), busy.get(name, [])):
            return name
    return None


def on(worker: Worker, command: str, ssh: Sequence[str] = SSH) -> list[str]:
    """Return the argv running shell `command` on `worker`."""
    if worker.local:
        return ["bash", "-c", command]
    return [*ssh, worker.host, command]


def reachable(worker: Worker) -> bool:
    """Whether `worker` answers over ssh within 10 s (`local` always)."""
    if worker.local:
        return True
    got = subprocess.run(on(worker, "true"), capture_output=True)
    return got.returncode == 0


def budget_of(
    worker: Worker, checkout: str | Path, cfg: Mapping[str, str]
) -> dict[str, int]:
    """Return a worker's budget, as its own local config gives it.

    This machine's, from `cfg`, for a local worker in this checkout.
    """
    if worker.local and Path(worker.checkout) == Path(checkout):
        return {**machine.budget(cfg), "gpu": max(1, machine.gpus())}
    cmd = f"cd {shlex.quote(worker.checkout)} && "
    cmd += "python3 workflow/flowlib/machine.py budget"
    got = subprocess.run(
        on(worker, cmd), capture_output=True, text=True, check=True
    )
    return json.loads(got.stdout)


def state_dir(checkout: str | Path | None = None) -> Path:
    """Return the folder of the placed jobs' records (`REMOTE_STATE`)."""
    if os.environ.get("REMOTE_STATE"):
        return Path(os.environ["REMOTE_STATE"])
    return Path(checkout or localconfig.ROOT) / "output" / "compute"


def args(
    cfg: Mapping[str, str],
    path: str | None = None,
    budgets: Mapping[str, Mapping[str, int]] | None = None,
    checkout: str | Path = localconfig.ROOT,
) -> list[str]:
    """Return the Snakemake options of the profile and its pool.

    For `ssh` without `budgets`, each worker's is read from it and kept
    in `state_dir()/budgets.json` for the submit script. Another
    `COMPUTE` is its profile alone.
    """
    got = mode(cfg, checkout)
    if got not in MODES:
        return ["--workflow-profile", f"workflow/profiles/{got}"]
    if got == "single":
        return [
            "--workflow-profile",
            "workflow/profiles/default",
            *machine.args(cfg, path=path),
        ]
    found = workers(cfg, checkout)
    if budgets is None:
        budgets = {}
        for name, worker in found.items():
            try:
                budgets[name] = budget_of(worker, checkout, cfg)
            except (subprocess.CalledProcessError, OSError, ValueError):
                # Its jobs fail at submit until it answers again.
                print(
                    f"worker {name} ({worker.host}) did not answer:"
                    " left out of the pool",
                    file=sys.stderr,
                )
        state = state_dir(checkout)
        state.mkdir(parents=True, exist_ok=True)
        (state / "budgets.json").write_text(json.dumps(budgets) + "\n")
    keys = ("cores", "mem_mb", "gpu")
    total = {k: sum(b[k] for b in budgets.values()) for k in keys}
    sets = [
        f"{rule}:gpu_{worker}=1"
        for group, worker in groups(cfg, list(found)).items()
        if worker in budgets
        for rule, of in RULE_GROUPS.items()
        if of == group and rule in GPU_RULES
    ]
    return [
        "--workflow-profile",
        "workflow/profiles/ssh",
        "--jobs",
        str(max(1, total["cores"])),
        *(["--set-resources", *sets] if sets else []),
        "--resources",
        f"mem_mb={total['mem_mb']}",
        f"gpu={total['gpu']}",
        *(f"gpu_{n}={b['gpu']}" for n, b in budgets.items()),
        f"sim_slots={cfg.get('SIM_SLOTS') or 2}",
        "awaiting=1",
    ]


# Options of a jobscript's Snakemake that name the coordinator's paths:
# the worker's own Snakemake finds its own.
_DROP = {
    "--snakefile",
    "--wait-for-files",
    "--scheduler-solver-path",
    "--runtime-source-cache-path",
    "--local-storage-prefix",
}


def read_jobscript(text: str) -> tuple[dict, list[str]]:
    """Return a jobscript's job and the Snakemake arguments to run it.

    The arguments are the jobscript's, less those naming the
    coordinator's paths, with `--cores` the job's threads.
    """
    job = None
    command = None
    for line in text.splitlines():
        if line.startswith("# properties = "):
            job = json.loads(line.removeprefix("# properties = "))
        elif "-m snakemake " in line:
            command = line.split("-m snakemake ", 1)[1].split(" && ")[0]
    if job is None or command is None:
        raise ValueError("not a Snakemake jobscript")
    argv: list[str] = []
    drop = False
    for token in shlex.split(command):
        if token.startswith("--"):
            drop = token in _DROP
            if not drop:
                argv.append(token)
        elif not drop:
            argv.append(token)
    if "--cores" in argv:
        argv[argv.index("--cores") + 1] = str(job.get("threads", 1))
    return job, argv


def need(job: Mapping) -> dict[str, int]:
    """Return a job's cores, memory (MB) and GPUs."""
    res = job.get("resources", {})
    return {
        "cores": int(job.get("threads", 1)),
        "mem_mb": int(res.get("mem_mb", 0)),
        "gpu": int(res.get("gpu", 0)),
    }


if __name__ == "__main__":
    print(" ".join(shlex.quote(a) for a in args(localconfig.read())))
