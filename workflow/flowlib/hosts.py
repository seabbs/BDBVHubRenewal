"""Rounds simulated on two hosts: each host's seeds and the push of a batch.

Each host lists the seeds it takes in its local config (`SIM_SEEDS` in
config/local.env): seeds, `first-last` ranges or whole rounds (`r<N>`).
A host with none takes every seed. The rounds host simulates its seeds
of each round and waits for the rest; a helping host pushes each of its
batches to the rounds host's round (`REMOTE`, or this machine when
empty): the files first, then the batch's manifest row, so a reader of
the manifest sees complete batches only.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

from flowlib import manifest

SPEC = re.compile(r"^(r(\d+)|(\d+)(-(\d+))?)$")


def seeds(specs: str, round_seeds: Callable[[int], Iterable[int]]) -> set[int]:
    """Return the seeds of space-separated specs.

    A spec is a seed (`31005`), a range (`31900-31939`) or a round
    (`r19`, its seeds from `round_seeds`). Raises `ValueError` on any
    other spec.
    """
    out: set[int] = set()
    for spec in specs.split():
        found = SPEC.match(spec)
        if not found:
            raise ValueError(f"not a seed, range or round: {spec}")
        if found.group(2):
            out.update(round_seeds(int(found.group(2))))
        else:
            first = int(found.group(3))
            last = int(found.group(5) or first)
            out.update(range(first, last + 1))
    return out


def simulates(seed: int, mine: set[int]) -> bool:
    """Whether a host taking seeds `mine` (empty: all) simulates `seed`."""
    return not mine or seed in mine


def _on(remote: str, command: str) -> None:
    """Run shell `command` on `remote`, or here when it is empty."""
    if remote:
        args = ["ssh", "-o", "ConnectTimeout=20", remote, command]
    else:
        args = ["bash", "-c", command]
    subprocess.run(args, check=True)


def _at(remote: str, path: Path | str) -> str:
    """Return the rsync address of `path` on `remote`."""
    return f"{remote}:{path}" if remote else str(path)


def push(
    sims: Path | str, seed: int, there: Path | str, row: str, remote: str
) -> None:
    """Copy batch `seed` of `sims` to round folder `there`, then its row.

    The counts, settings, failures (if any) and log go to
    `there/sims/`, then `row` is appended to `there/manifest.csv` unless
    the manifest already holds the batch. Raises
    `subprocess.CalledProcessError` if a step fails.
    """
    sims, there = Path(sims), Path(there)
    files = [
        sims / f"counts-{seed}.csv.gz",
        sims / f"settings-{seed}.csv",
    ]
    failures = sims / f"failures-{seed}.csv"
    if failures.is_file():
        files.append(failures)
    q = shlex.quote
    rman = there / "manifest.csv"
    _on(
        remote,
        f"mkdir -p {q(str(there / 'sims' / 'logs'))} && "
        f"{{ [ -f {q(str(rman))} ] || "
        f"echo {q(manifest.HEADER)} > {q(str(rman))}; }}",
    )
    subprocess.run(
        ["rsync", "-a", *map(str, files), _at(remote, f"{there}/sims/")],
        check=True,
    )
    subprocess.run(
        [
            "rsync",
            "-a",
            str(sims / "logs" / f"{seed}.log"),
            _at(remote, f"{there}/sims/logs/"),
        ],
        check=True,
    )
    round_, seed_ = row.split(",")[:2]
    _on(
        remote,
        f"grep -q {q(f'^{round_},{seed_},')} {q(str(rman))} || "
        f"echo {q(row)} >> {q(str(rman))}",
    )
