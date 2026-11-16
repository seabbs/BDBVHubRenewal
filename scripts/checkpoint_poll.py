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
checkpoints' BDBV rows to origins new releases add. Units resume from
the cache, so a restart loses only the forecasts in flight.

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
from collections.abc import Mapping
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
MAX_UNITS = 3
SCOPE_GB = 6
KEEP_FREE_GB = 16
MIN_DISK_GB = 3
DATA_HOURS = 6
MAX_FAILURES = 2
REFERENCE_STAMP = "reference.landed"
"""In the cache: the commit the zero-shot reference rows landed at."""


def can_launch(budget: Mapping[str, Any], mem_avail_gb: float) -> bool:
    """Whether a unit may start: a free core, and memory to keep free."""
    return (
        budget.get("verdict") != "red"
        and int(budget.get("max_parallel", 0)) >= 1
        and mem_avail_gb >= KEEP_FREE_GB + SCOPE_GB
    )


def units_for(checkpoint: str, scored: set[str]) -> list[tuple[str, str]]:
    """Return a checkpoint's units: held-out first, then the long ones.

    BDBV units of scored checkpoints are included so their rows cover
    the same origins; finished units return at once from the cache.
    """
    return [
        ("heldout", checkpoint),
        ("germany", cs.REFERENCE),
        ("germany", checkpoint),
        ("bdbv", cs.REFERENCE),
        ("bdbv", checkpoint),
        *(("bdbv", s) for s in sorted(scored)),
    ]


def reference_pending(cache: Path) -> list[tuple[str, str]]:
    """Return the zero-shot reference's units, until its rows have landed."""
    if (cache / REFERENCE_STAMP).exists():
        return []
    return [("germany", cs.REFERENCE), ("bdbv", cs.REFERENCE)]


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


def launch(cache: Path, unit: tuple[str, str]) -> subprocess.Popen:
    """Start one unit in its own memory-capped scope."""
    study, label = unit
    logs = cache.parent / "log"
    logs.mkdir(parents=True, exist_ok=True)
    out = (logs / f"{study}-{label}.log").open("a")
    cmd = [
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
    ]
    return subprocess.Popen(
        cmd, stdout=out, stderr=subprocess.STDOUT, env=unit_env(label)
    )


def run_units(cache: Path, units: list[tuple[str, str]], name: str) -> bool:
    """Run the units not yet done, gated; return whether all finished."""
    pending = [u for u in units if not cs.unit_done(cache, *u)]
    total, ok = len(pending), True
    running: dict[tuple[str, str], subprocess.Popen] = {}
    waited = False
    while pending or running:
        for unit, proc in list(running.items()):
            if proc.poll() is None:
                continue
            del running[unit]
            done = proc.returncode == 0 and cs.unit_done(cache, *unit)
            ok &= done
            left = len(pending) + len(running)
            state = "done" if done else f"FAILED ({proc.returncode})"
            log(f"progress: {name}: {unit[0]} {unit[1]} {state}, "
                f"{total - left}/{total} units")  # fmt: skip
        if pending and len(running) < MAX_UNITS:
            if pending[0][0] == "heldout" or can_launch(
                budget(), mem_avail_gb()
            ):
                unit = pending.pop(0)
                running[unit] = launch(cache, unit)
                log(f"progress: {name}: started {unit[0]} {unit[1]}")
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
    say(
        f"bdbv-paper: rows for {checkpoint} ({outbreaks} outbreaks, grid "
        f"{grid}) landed at {sha} on origin/{BRANCH} (verified), "
        f"evidence/checkpoint-scores.csv; bump `checkpoint_scores` in "
        f"paper/workflow/config.yaml to it. Filter rule == '{cs.RULE}' "
        "(older rows are rule 'samples-100'); germany uses rel_wis and "
        "rel_log_wis, ebola rel_crps and rel_log_crps (rel_wis empty), "
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
        queue = [
            h
            for h in cs.checkpoint_queue(listed, scored)
            if failures.get(h, 0) < MAX_FAILURES
        ]
        if not queue:
            stale = [
                s
                for s in [cs.REFERENCE, *sorted(scored)]
                if not cs.unit_done(cache, "bdbv", s)
            ]
            if stale and scored:
                units = [("bdbv", s) for s in stale]
                if run_units(cache, units, "new BDBV origins"):
                    land(cache, sorted(scored), "BDBV rows to new origins")
                else:
                    time.sleep(POLL_SECONDS)
                continue
            log("progress: idle, every published checkpoint scored")
            time.sleep(POLL_SECONDS)
            continue
        checkpoint = queue[0]
        log(f"progress: {checkpoint} started, {len(queue)} waiting")
        if not run_units(cache, units_for(checkpoint, scored), checkpoint):
            failures[checkpoint] = failures.get(checkpoint, 0) + 1
            if failures[checkpoint] >= MAX_FAILURES:
                say(f"FAILED scoring {checkpoint} twice; skipped until "
                    f"restart. Logs in {cache.parent / 'log'}")  # fmt: skip
            continue
        sha = land(cache, sorted(scored | {checkpoint}), checkpoint)
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
