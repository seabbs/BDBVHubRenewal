"""Score each published checkpoint (slots run first), for good.

Every `POLL_SECONDS` the poller lists the repository's releases and takes
the first checkpoint `checkpoint_scores.checkpoint_queue` gives: the new
continuous run's releases, oldest first (published from
`checkpoint_scores.PUBLISHED_FROM`). Before any, it scores and lands the
zero-shot reference (randomwalknowcast and base Chronos-2). It runs
the checkpoint's units (`units_for`, each a `checkpoint_scores.py`
process in its own systemd scope of `SCOPE_GB`, at nice 19 with one
thread), starting one only while `can_launch` allows, rebuilds the
rule's rows, commits evidence/checkpoint-scores.csv and
evidence/checkpoints.csv, pushes them to the `checkpoint-scoring`
branch, checks the commit is on origin, and writes a mailbox line for
bdbv-paper. When no checkpoint is waiting it extends the scored
checkpoints' BDBV rows to origins new releases add, and then runs the
daily target (`daily_units`): the reference's first (about 4.5 h), then
the one-model checkpoints', each landed again with its daily rows. Units
run with the 7-day target only otherwise, so 7-day rows never wait.
Units resume from the cache, so a restart loses only the forecasts in
flight.

Usage: python scripts/checkpoint_poll.py CACHE

Run from the repository root of the `checkpoint-scoring` worktree. Lines
starting `progress:` in its log are what scripts/job_watch.sh reports.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import checkpoint_scores as cs

LABEL = "ckpt-scores"
BRANCH = "checkpoint-scoring"
REPO = "seabbs/BDBVHubRenewal"
EVIDENCE = Path("evidence/checkpoint-scores.csv")
CHECKPOINTS = Path("evidence/checkpoints.csv")
MAILBOX = Path(
    os.environ.get("MAILBOX", Path.home() / "bdbv-archie-mailbox.md")
)
BUDGET = Path.home() / ".claude/hooks/compute-budget.sh"
POLL_SECONDS = 30 * 60
TICK_SECONDS = 60
PAUSE_SECONDS = 5 * 60
"""How often a slots checkpoint's German run checks for a newer one."""
MAX_UNITS = 3
SCOPE_GB = 6
UNIT_PEAK_GB = 5
"""A unit's measured peak resident memory, which the gate keeps beside
`KEEP_FREE_GB`; `SCOPE_GB` stays its hard cap."""
KEEP_FREE_GB = 16
MIN_DISK_GB = 3
DATA_HOURS = 6
MAX_FAILURES = 2
REFERENCE_STAMP = "reference.landed"
"""In the cache: the commit the zero-shot reference rows landed at."""


def can_launch(
    budget: Mapping[str, Any], mem_avail_gb: float, running: int = 0
) -> bool:
    """Whether a unit may start: memory to keep free, and a free core.

    With no free core but the box not red, one unit still runs at a time
    (nice 19, one thread), so scoring never stalls for good behind the
    simulation rounds.
    """
    free_core = int(budget.get("max_parallel", 0)) >= 1 or running == 0
    return (
        budget.get("verdict") != "red"
        and free_core
        and mem_avail_gb >= KEEP_FREE_GB + UNIT_PEAK_GB
    )


def units_for(checkpoint: str, scored: set[str]) -> list[tuple[str, str]]:
    """Return a checkpoint's units: fast ones first, then the slow.

    Held-out settings and BDBV are fast; the held-out proper scores and
    Germany are slow.

    BDBV units of scored checkpoints are included so their rows cover
    the same origins; finished units return at once from the cache. The
    reference's held-out proper scores are here too, as a cache whose
    reference landed before them never runs `reference_pending`.
    """
    return [
        ("heldout", checkpoint),
        ("bdbv", cs.REFERENCE),
        ("bdbv", checkpoint),
        ("heldout_scores", cs.REFERENCE),
        ("heldout_scores", checkpoint),
        ("germany", cs.REFERENCE),
        ("germany", checkpoint),
        *(("bdbv", s) for s in sorted(scored)),
    ]


DAILY_JOB = "daily target"
"""The idle-time job: the reference's daily units, then the checkpoints'
(`daily_units`), pausable by a fresh slots checkpoint."""

COMPLETE_JOB = "complete-data reference"
"""The job that adds the complete-data models to the zero-shot reference."""


def next_job(
    queue: list[str],
    complete_pending: bool,
    fresh: set[str] | None = None,
    zero_init: set[str] | None = None,
) -> str | None:
    """Return what to run next.

    The slots run's checkpoints whose held-out and BDBV units have not run
    (`fresh`) first, then its other checkpoints, then the complete-data
    reference, then the no-slots checkpoints, and last the zero-init slots
    run (`zero_init`), which runs only with spare capacity.
    """
    fresh, zero = fresh or set(), zero_init or set()
    slots = [lb for lb in queue if lb.startswith("slots-") and lb not in zero]
    for label in slots:
        if label in fresh:
            return label
    if slots:
        return slots[0]
    if complete_pending:
        return COMPLETE_JOB
    rest = [lb for lb in queue if lb not in zero]
    if rest:
        return rest[0]
    return queue[0] if queue else None


def should_pause(
    current: str,
    slots: list[str],
    fresh: set[str],
    zero_init: set[str] | None = None,
) -> bool:
    """Whether to pause `current`'s German run for another slots
    checkpoint whose held-out and BDBV units have not run, never for one of
    the zero-init run."""
    zero = zero_init or set()
    return any(lb != current and lb in fresh and lb not in zero for lb in slots)


HOLD = Path.home() / "ckpt-scores" / "hold-slots-run"
"""While this file exists, the slots run's checkpoints wait (its adapters
carry parent_scale, which slots.py drops until bdbv-covariates' fix)."""


HELD: dict[str, list[str]] = {"labels": []}
"""The checkpoints last reported as held, so each is said once."""


def unheld(
    queue: list[str], zero_init: set[str], hold: Path = HOLD
) -> list[str]:
    """Return the queue less the slots run's checkpoints while `hold` exists."""
    if not hold.exists():
        return list(queue)
    return [
        lb for lb in queue if not lb.startswith("slots-") or lb in zero_init
    ]


def scoreable(queue: list[str], runs: Mapping[str, str]) -> list[str]:
    """Return the queue less the slots checkpoints this code cannot load.

    A slots checkpoint is scored only if its run is current
    (`cs.CURRENT_RUNS`); one not yet classified waits (`unclassified`).
    """
    return [
        lb
        for lb in queue
        if not lb.startswith("slots-") or runs.get(lb) in cs.CURRENT_RUNS
    ]


def unclassified(queue: list[str], runs: Mapping[str, str]) -> list[str]:
    """Return the slots checkpoints whose settings are not fetched yet."""
    return [lb for lb in queue if lb.startswith("slots-") and lb not in runs]


def runs_of(cache: Path) -> dict[str, str]:
    """Return each fetched checkpoint's run, from its settings.json."""
    out = {}
    for f in (cache / "heldout").glob("*.json"):
        if f.name.endswith("-release.json"):
            continue
        out[f.stem] = cs.run_of(json.loads(f.read_text()))
    return out


GERMAN_PARALLEL = 2
"""Slots checkpoints whose German units run at once (each under 6 GB)."""


def german_order(
    labels: list[str], cached: Mapping[str, int], n: Mapping[str, int]
) -> list[str]:
    """Return the order of German runs: the one furthest on, then newest.

    The run with the most dates cached goes first (it finishes soonest),
    then the rest by outbreaks, most first.
    """
    rest = sorted(labels, key=lambda lb: -n.get(lb, 0))
    going = [lb for lb in labels if cached.get(lb, 0) > 0]
    if not going:
        return rest
    first = max(going, key=lambda lb: (cached[lb], n.get(lb, 0)))
    return [first, *[lb for lb in rest if lb != first]]


def choose(
    queue: list[str],
    runs: Mapping[str, str],
    zero_init: set[str],
    hold: Path,
    complete_pending: bool,
    fresh: set[str],
) -> str | None:
    """Return the next job, or None when nothing may run (idle).

    The queue loses checkpoints this code cannot load (`scoreable`) and
    held ones (`unheld`) before `next_job` picks.
    """
    left = unheld(scoreable(queue, runs), zero_init, hold)
    return next_job(left, complete_pending, fresh & set(left), zero_init)


def extendable(labels: list[str], runs: Mapping[str, str]) -> list[str]:
    """Return the labels whose BDBV rows may grow to new origins here:
    the reference, no-slots checkpoints and the current run's."""
    ok = cs.CURRENT_RUNS | {"no-slots"}
    return [lb for lb in labels if lb == cs.REFERENCE or runs.get(lb) in ok]


def daily_units(
    cache: Path, labels: list[str], runs: Mapping[str, str]
) -> list[Unit]:
    """Return the daily-target units still to run, an idle-time job.

    The reference's first: every checkpoint's daily rows divide by its
    randomwalknowcast. Then those of `labels` that take the daily target
    by default (`cs.targets_for`, the one-model run) and whose BDBV rows
    may grow here (`extendable`), BDBV before Germany.
    """
    ref = [(study, cs.REFERENCE, "daily") for study in ("bdbv", "germany")]
    todo = [u for u in ref if not cs.unit_done(cache, *u)]
    if todo:
        return todo
    out: list[Unit] = []
    for lb in extendable(labels, runs):
        if lb == cs.REFERENCE or "daily" not in cs.targets_for(cache, lb):
            continue
        units = [(study, lb, "daily") for study in ("bdbv", "germany")]
        out += [u for u in units if not cs.unit_done(cache, *u)]
    return out


def zero_init_labels(cache: Path) -> set[str]:
    """Return the scored slots checkpoints of the zero-init run."""
    out = set()
    for f in (cache / "heldout").glob("slots-*.json"):
        if f.name.endswith("-release.json"):
            continue
        settings = json.loads(f.read_text())
        if cs.run_of(settings) == "slots-zeroinit":
            out.add(f.stem)
    return out


def fresh_slots(cache: Path, labels: list[str]) -> set[str]:
    """Return the slots labels whose held-out or BDBV unit is not done."""
    return {
        lb
        for lb in labels
        if lb.startswith("slots-")
        and not all(
            cs.unit_done(cache, study, lb) for study in ("heldout", "bdbv")
        )
    }


FAST = ("heldout", "bdbv")
"""Studies whose units take minutes."""


def split_fast(
    units: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split units into the fast ones (held-out, BDBV) and the slow.

    The held-out proper scores take about 45 minutes a checkpoint, so they
    run with the German units, after the interim BDBV rows land.
    """
    fast = [u for u in units if u[0] in FAST]
    return fast, [u for u in units if u[0] not in FAST]


def reference_pending(cache: Path) -> list[tuple[str, str]]:
    """Return the zero-shot reference's units, until its rows have landed."""
    if (cache / REFERENCE_STAMP).exists():
        return []
    return [
        ("germany", cs.REFERENCE),
        ("bdbv", cs.REFERENCE),
        ("heldout_scores", cs.REFERENCE),
    ]


FOLLOW_UPS = Path(__file__).parent / "slots_followups.sh"
"""Started, detached, once a slots checkpoint's rows land."""


def follow_ups(checkpoint: str) -> list[str] | None:
    """Return the follow-up jobs' command for a slots checkpoint, or None."""
    if not checkpoint.startswith("slots-"):
        return None
    return [str(FOLLOW_UPS), checkpoint]


def mailbox_line(text: str, when: str | None = None) -> str:
    """Return a mailbox line, as scripts/job_watch.sh writes them."""
    when = when or dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"{when} archie: {LABEL}: {text}\n"


# Side effects ----------------------------------------------------------


def log(text: str) -> None:
    """Print a stamped log line."""
    print(f"{dt.datetime.now():%Y-%m-%d %H:%M} {text}", flush=True)


def say(text: str) -> None:
    """Append a line to the mailbox and the log."""
    with MAILBOX.open("a") as f:
        f.write(mailbox_line(text))
    log(f"mailbox: {text}")


def budget() -> dict[str, Any]:
    """Read the compute budget; a failed read holds new units back."""
    try:
        out = subprocess.run(
            [str(BUDGET), "--json"], capture_output=True, text=True, check=False
        )
        return dict(json.loads(out.stdout))
    except (OSError, ValueError):
        return {"verdict": "red", "max_parallel": 0}


def mem_avail_gb() -> float:
    """Return MemAvailable in GB."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return 0.0


def gh_env() -> dict[str, str]:
    """Return the environment with a GitHub token for API reads."""
    env = dict(os.environ)
    token = subprocess.run(
        ["gh", "auth", "token"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if token:
        env["GITHUB_TOKEN"] = token
        env["GH_TOKEN"] = token
    return env


def releases() -> list[dict[str, str]]:
    """List the repository's releases."""
    out = subprocess.run(
        [
            *("gh", "release", "list", "-R", REPO, "--limit", "200"),
            *("--json", "tagName,publishedAt"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return list(json.loads(out.stdout))


def unit_env(label: str) -> dict[str, str]:
    """Return a unit's environment: one thread, offline model, pinned."""
    env = gh_env()
    threads = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    env.update(dict.fromkeys(threads, "1"))
    env["HF_HUB_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    env["BDBV_CHRONOS_FT"] = (
        label if label != cs.REFERENCE else cs.REFERENCE_PIN
    )
    return env


Unit = tuple[str, ...]
"""(study, label), the 7-day target, or (study, label, target)."""


def target_of(unit: Unit) -> str:
    """Return a unit's target, `7day` where it names none."""
    return unit[2] if len(unit) > 2 else "7day"


def unit_command(cache: Path, unit: Unit) -> list[str]:
    """Return a unit's command, its target always given.

    Without one, `checkpoint_scores.py` would take the label's own
    targets (`cs.targets_for`), so a 7-day unit could run the daily one.
    """
    study, label = unit[0], unit[1]
    return [
        "systemd-run",
        "--user",
        "--scope",
        "-q",
        "-p",
        f"MemoryMax={SCOPE_GB}G",
        "-p",
        "MemorySwapMax=0",
        "nice",
        "-n",
        "19",
        sys.executable,
        str(Path(__file__).parent / "checkpoint_scores.py"),
        study,
        str(cache),
        label,
        target_of(unit),
    ]


def launch(cache: Path, unit: Unit) -> subprocess.Popen:
    """Start one unit in its own memory-capped scope."""
    study, label = unit[0], unit[1]
    logs = cache.parent / "log"
    logs.mkdir(parents=True, exist_ok=True)
    name = f"{study}-{label}" + ("-daily" if target_of(unit) == "daily" else "")
    out = (logs / f"{name}.log").open("a")
    cmd = unit_command(cache, unit)
    return subprocess.Popen(
        cmd, stdout=out, stderr=subprocess.STDOUT, env=unit_env(label)
    )


def next_unit(
    pending: list[Unit],
    running: Iterable[Unit],
    limits: Mapping[str, int] | None,
) -> Unit | None:
    """Return the first pending unit whose study has a free lane.

    `limits` caps the running units per study; with none, the first.
    """
    if not limits:
        return pending[0] if pending else None
    busy: dict[str, int] = {}
    for unit in running:
        busy[unit[0]] = busy.get(unit[0], 0) + 1
    for unit in pending:
        if busy.get(unit[0], 0) < limits.get(unit[0], len(pending)):
            return unit
    return None


def run_units(
    cache: Path,
    units: list[Unit],
    name: str,
    pause: Any = None,
    paused: list[str] | None = None,
    on_done: Any = None,
    max_units: int = 0,
    limits: Mapping[str, int] | None = None,
) -> bool:
    """Run the units not yet done, gated; return whether all finished.

    With `pause`, a callable checked every `PAUSE_SECONDS`, a True answer
    stops the running units (their finished dates stay in the cache),
    appends `name` to `paused` and returns False.
    """
    pending = [u for u in units if not cs.unit_done(cache, *u)]
    total, ok = len(pending), True
    running: dict[Unit, subprocess.Popen] = {}
    waited = False
    checked = time.monotonic()
    while pending or running:
        if pause is not None and time.monotonic() - checked > PAUSE_SECONDS:
            checked = time.monotonic()
            if pause():
                for proc in running.values():
                    proc.terminate()
                for proc in running.values():
                    proc.wait()
                log(f"progress: {name}: paused for a newer slots checkpoint")
                if paused is not None:
                    paused.append(name)
                return False
        for unit, proc in list(running.items()):
            if proc.poll() is None:
                continue
            del running[unit]
            done = proc.returncode == 0 and cs.unit_done(cache, *unit)
            ok &= done
            left = len(pending) + len(running)
            state = "done" if done else f"FAILED ({proc.returncode})"
            log(f"progress: {name}: {' '.join(unit)} {state}, "
                f"{total - left}/{total} units")  # fmt: skip
            if done and on_done is not None:
                on_done(unit)
        nxt = next_unit(pending, running, limits)
        if nxt is not None and len(running) < (max_units or MAX_UNITS):
            if nxt[0] == "heldout" or can_launch(
                budget(), mem_avail_gb(), len(running)
            ):
                unit = nxt
                pending.remove(unit)
                running[unit] = launch(cache, unit)
                log(f"progress: {name}: started {' '.join(unit)}")
                waited = False
                continue
            if not waited:
                log(f"progress: {name}: waiting for a free core or memory")
                waited = True
        time.sleep(TICK_SECONDS)
    return ok


def refresh_data(stamp: Path) -> None:
    """Refetch the BDBV data at most every `DATA_HOURS`, no unit running."""
    if (
        stamp.exists()
        and time.time() - stamp.stat().st_mtime < DATA_HOURS * 3600
    ):
        return
    run = subprocess.run(
        [str(Path(sys.executable).parent / "bdbvhubrenewal"), "data"],
        env=gh_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode == 0:
        stamp.touch()
        log("data refreshed")
    else:
        log(f"data refresh failed: {run.stderr.strip()[-200:]}")


def git(*args: str) -> subprocess.CompletedProcess:
    """Run git in the worktree."""
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )


LANDED: dict[str, bool] = {"changed": False}
"""Whether the last `land` made a new evidence commit."""


def interim_note(checkpoint: str, sha: str, changed: bool) -> str | None:
    """Return bdbv-paper's line for interim rows, or None if unchanged."""
    if not changed:
        return None
    return (
        f"bdbv-paper: {checkpoint} held-out and BDBV rows landed at {sha} on "
        f"origin/{BRANCH}; German first half follows, written per date to "
        "the cache"
    )


def land(cache: Path, hashes: list[str], what: str) -> str | None:
    """Rebuild the rows, commit, push and verify; return the commit."""
    table = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "checkpoint_scores.py"),
            "table",
            str(cache),
            str(EVIDENCE),
            str(CHECKPOINTS),
            *hashes,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=unit_env(cs.REFERENCE),
    )
    if table.returncode != 0:
        say(f"FAILED building rows for {what}: {table.stderr.strip()[-300:]}")
        return None
    log(table.stdout)
    git("add", str(EVIDENCE), str(CHECKPOINTS))
    LANDED["changed"] = False
    if git("diff", "--cached", "--quiet").returncode == 0:
        log(f"no change in the evidence for {what}")
        return git("rev-parse", "HEAD").stdout.strip()
    commit = git(
        "commit",
        "-q",
        "-m",
        f"evidence: {what}, scored under Sam's 2026-11-16 rule "
        "(scripts/checkpoint_poll.py)",
    )
    if commit.returncode != 0:
        git("reset", "-q", str(EVIDENCE), str(CHECKPOINTS))
        say(f"FAILED committing {what} (not pushed): "
            f"{(commit.stdout + commit.stderr).strip()[-300:]}")  # fmt: skip
        return None
    sha = git("rev-parse", "HEAD").stdout.strip()
    LANDED["changed"] = True
    push = git("push", "-q", "origin", f"HEAD:{BRANCH}")
    if push.returncode != 0:
        say(f"FAILED pushing {sha[:9]} ({what}): {push.stderr.strip()[-300:]}")
        return None
    git("fetch", "-q", "origin", BRANCH)
    on = git("merge-base", "--is-ancestor", sha, "FETCH_HEAD").returncode == 0
    if not on:
        say(f"FAILED: {sha[:9]} ({what}) is not on origin/{BRANCH}")
        return None
    return sha


def note_paper(sha: str, checkpoint: str, cache: Path) -> None:
    """Tell bdbv-paper the commit to pin."""
    settings = json.loads(
        (cache / "heldout" / f"{checkpoint}.json").read_text()
    )
    release = json.loads(
        (cache / "heldout" / f"{checkpoint}-release.json").read_text()
    )
    outbreaks, grid = cs.outbreaks_and_grid(settings, release)
    say(paper_note(sha, checkpoint, outbreaks, grid, cs.run_of(settings)))


def paper_note(
    sha: str, checkpoint: str, outbreaks: int, grid: str, run: str
) -> str:
    """Return bdbv-paper's line for a landed checkpoint."""
    return (
        f"bdbv-paper: rows for {checkpoint} ({outbreaks} outbreaks, grid "
        f"{grid}, run {run}) landed at {sha} on origin/{BRANCH} (verified), "
        f"evidence/checkpoint-scores.csv; bump `checkpoint_scores` in "
        f"paper/workflow/config.yaml to it. Filter rule == '{cs.RULE}' "
        "(older rows are rule 'samples-100') and target == '7day' for the "
        "hub target (target 'daily' rows compare our models only); "
        f"final_checkpoints from run == 'one-model'. germany uses rel_wis "
        "and rel_log_wis, ebola rel_crps and rel_log_crps (rel_wis empty), "
        "held_out loss and loss_zero_shot; curves by grid."
    )


def main(cache: Path) -> None:
    """Poll for good."""
    cache.mkdir(parents=True, exist_ok=True)
    failures: dict[str, int] = {}
    log(f"progress: poller started in {Path.cwd()}, cache {cache}")
    while True:
        free = shutil.disk_usage(cache).free / 1024**3
        if free < MIN_DISK_GB:
            say(f"STOPPED: {free:.1f} GB disk free (< {MIN_DISK_GB} GB)")
            raise SystemExit(1)
        refresh_data(cache / "data.stamp")
        try:
            listed = releases()
        except (subprocess.CalledProcessError, ValueError) as err:
            log(f"release list failed: {err}")
            time.sleep(POLL_SECONDS)
            continue
        pending = reference_pending(cache)
        if pending:
            log("progress: zero-shot reference started")
            if not run_units(cache, pending, "zero-shot reference"):
                say("FAILED the zero-shot reference run; retrying later")
                time.sleep(POLL_SECONDS)
                continue
            sha = land(cache, [], "zero-shot reference")
            if sha is None:
                raise SystemExit(1)
            (cache / REFERENCE_STAMP).write_text(sha + "\n")
            say(f"zero-shot reference rows landed at {sha[:9]} on "
                f"{BRANCH}; scoring works end to end")  # fmt: skip
            continue
        evidence = cs.pd.read_csv(EVIDENCE, dtype={"checkpoint": str})
        scored = cs.scored_checkpoints(evidence)
        # Every rebuild carries each checkpoint with rows, scored or not.
        keep = scored | cs.checkpoints_with_rows(evidence)
        queue = [
            h
            for h in cs.checkpoint_queue(listed, scored)
            if failures.get(h, 0) < MAX_FAILURES
        ]
        runs = runs_of(cache)
        new = unclassified(queue, runs)
        if new:
            # Their settings say which run they are; held-out units only.
            run_units(cache, [("heldout", lb) for lb in new], "settings")
            runs = runs_of(cache)
        skipped = [lb for lb in queue if lb not in scoreable(queue, runs)]
        if skipped and skipped != HELD.get("skipped"):
            say("not scored on this code (superseded runs): "
                + ", ".join(f"{lb} ({runs.get(lb, '?')})" for lb in skipped))  # fmt: skip
        HELD["skipped"] = skipped
        queue = scoreable(queue, runs)
        zero = zero_init_labels(cache)
        held = [lb for lb in queue if lb not in unheld(queue, zero)]
        if held and held != HELD["labels"]:
            say(f"held until the parent_scale fix is merged: {', '.join(held)}")
        HELD["labels"] = held
        fresh = fresh_slots(cache, queue)
        job = choose(
            queue,
            runs,
            zero,
            HOLD,
            not (cache / cs.COMPLETE_STAMP).exists(),
            fresh,
        )
        queue = unheld(queue, zero)
        paused: list[str] = []

        def newer_slots(current: str | None = job) -> bool:
            # Any long job yields to a slots-run checkpoint whose held-out
            # and BDBV units have not run.
            try:
                labels = cs.checkpoint_queue(releases(), scored)
            except (subprocess.CalledProcessError, ValueError):
                return False
            labels = scoreable(
                unheld(labels, zero_init_labels(cache)), runs_of(cache)
            )
            return should_pause(
                current or "",
                labels,
                fresh_slots(cache, labels),
                zero_init_labels(cache),
            )

        if job is None:
            stale = [
                s
                for s in extendable([cs.REFERENCE, *sorted(scored)], runs)
                if not cs.unit_done(cache, "bdbv", s)
            ]
            if stale and scored:
                units = [("bdbv", s) for s in stale]
                if run_units(cache, units, "new BDBV origins"):
                    land(cache, sorted(keep), "BDBV rows to new origins")
                else:
                    time.sleep(POLL_SECONDS)
                continue
            daily = daily_units(cache, sorted(scored), runs)
            if daily and failures.get(DAILY_JOB, 0) < MAX_FAILURES:
                labels = sorted({u[1] for u in daily})
                log(f"progress: {DAILY_JOB} started: {', '.join(labels)}")
                if run_units(cache, daily, DAILY_JOB, newer_slots, paused):
                    land(cache, sorted(keep),
                         f"daily target of {', '.join(labels)}")  # fmt: skip
                elif not paused:
                    failures[DAILY_JOB] = failures.get(DAILY_JOB, 0) + 1
                    say(f"FAILED the {DAILY_JOB} ({', '.join(labels)}); "
                        f"logs in {cache.parent / 'log'}")  # fmt: skip
                    time.sleep(POLL_SECONDS)
                continue
            log("progress: idle, every published checkpoint scored")
            time.sleep(POLL_SECONDS)
            continue
        if job == COMPLETE_JOB:
            log(f"progress: {COMPLETE_JOB} started")
            units = [
                ("bdbv", cs.COMPLETE_LABEL),
                ("germany", cs.COMPLETE_LABEL),
            ]
            if not run_units(cache, units, COMPLETE_JOB, newer_slots, paused):
                if paused:
                    continue
                say(f"FAILED the {COMPLETE_JOB}; retrying later")
                time.sleep(POLL_SECONDS)
                continue
            cs.merge_complete(cache)
            sha = land(cache, sorted(keep), COMPLETE_JOB)
            if sha is not None:
                say(f"bdbv-paper: {COMPLETE_JOB} landed at {sha} on origin/"
                    f"{BRANCH} (verified): fm_chronos_daily_complete and "
                    "fm_chronos_weekly_complete, checkpoint 'zero-shot', "
                    "German first half (rel_wis) and BDBV (rel_crps), on the "
                    "cells every model and checkpoint made; bump "
                    "`checkpoint_scores` to it")  # fmt: skip
            continue
        checkpoint = job
        log(f"progress: {checkpoint} started, {len(queue)} waiting")
        units = units_for(checkpoint, scored)
        fast, _ = split_fast(units)
        fast_due = any(not cs.unit_done(cache, *u) for u in fast)
        if checkpoint.startswith("slots-") and fast_due:
            if run_units(cache, fast, checkpoint):
                sha = land(cache, sorted(keep | {checkpoint}),
                           f"{checkpoint} held-out and BDBV (German running)")  # fmt: skip
                note = interim_note(checkpoint, sha or "", LANDED["changed"])
                if sha is not None and note is not None:
                    say(note)
            # The loop picks what goes on next (an older German run, say).
            continue
        if checkpoint.startswith("slots-"):
            # Every due German run of the live run's checkpoints, in
            # german_order, two at a time (a freed slot takes the next);
            # each lands as soon as its own unit ends.
            meta = {
                lb: json.loads((cache / "heldout" / f"{lb}.json").read_text())
                for lb in queue
                if lb.startswith("slots-")
                and (cache / "heldout" / f"{lb}.json").exists()
            }
            due = [
                lb
                for lb in meta
                if runs.get(lb) in cs.CURRENT_RUNS
                and not (
                    cs.unit_done(cache, "germany", lb)
                    and cs.unit_done(cache, "heldout_scores", lb)
                )
                and all(cs.unit_done(cache, *u) for u in split_fast(
                    units_for(lb, scored))[0])
            ]  # fmt: skip
            cached = {
                lb: len(list((cache / "germany" / lb).glob("*.csv.gz")))
                for lb in due
            }
            outbreaks = {lb: int(m.get("n_outbreaks", 0)) for lb, m in meta.items()}
            pick = german_order(due, cached, outbreaks)
            if pick:
                # Two German lanes and one held-out-scores lane: the first
                # two German runs, then the reference's and each
                # checkpoint's held-out scores, then the other German runs.
                hs = [("heldout_scores", cs.REFERENCE)]
                hs += [("heldout_scores", lb) for lb in pick]
                germ = [
                    ("germany", cs.REFERENCE),
                    *(("germany", lb) for lb in pick[:GERMAN_PARALLEL]),
                    *hs,
                    *(("germany", lb) for lb in pick[GERMAN_PARALLEL:]),
                ]

                def landed(unit: Unit) -> None:
                    lb = unit[1]
                    if lb == cs.REFERENCE and unit[0] == "germany":
                        return
                    hashes = keep if lb == cs.REFERENCE else keep | {lb}
                    what = f"{lb} {unit[0]}"
                    sha = land(cache, sorted(hashes), what)
                    if sha is not None and lb != cs.REFERENCE:
                        keep.add(lb)
                        note_paper(sha, lb, cache)

                say(f"German runs: {', '.join(pick)} "
                    f"({', '.join(f'n{outbreaks[lb]} {cached[lb]}/80' for lb in pick)})")  # fmt: skip
                run_units(
                    cache, germ, "German", newer_slots, paused, landed,
                    max_units=GERMAN_PARALLEL + 1,
                    limits={"germany": GERMAN_PARALLEL, "heldout_scores": 1},
                )  # fmt: skip
                continue
        if not run_units(cache, units, checkpoint, newer_slots, paused):
            if paused:
                continue
            failures[checkpoint] = failures.get(checkpoint, 0) + 1
            if failures[checkpoint] >= MAX_FAILURES:
                say(f"FAILED scoring {checkpoint} twice; skipped until "
                    f"restart. Logs in {cache.parent / 'log'}")  # fmt: skip
            continue
        sha = land(cache, sorted(keep | {checkpoint}), checkpoint)
        if sha is None:
            failures[checkpoint] = MAX_FAILURES
            continue
        log(f"progress: {checkpoint} landed at {sha}")
        note_paper(sha, checkpoint, cache)
        cmd = follow_ups(checkpoint)
        if cmd is not None:
            out = cache.parent / "log" / f"followups-{checkpoint}.log"
            with out.open("a") as f:
                subprocess.Popen(
                    cmd, stdout=f, stderr=f, stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )  # fmt: skip
            say(f"{checkpoint}: started the aggregation arms and the "
                f"slots check ({cmd[0]}); log {out}")  # fmt: skip


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser().resolve())
