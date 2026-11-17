"""Rounds simulated on two hosts: the seed split and the push of a batch.

`simulate.hosts` in workflow/config.yaml names the seeds each helping
host simulates, as seeds or `first-last` ranges. A host it names
simulates only those seeds; any other host, the rounds host, simulates
every seed no host is named for and waits for the rest. A helping host
pushes each finished batch to the rounds host's round (`REMOTE`, or
this machine when empty): the files first, then the batch's manifest
row, so a reader of the manifest sees complete batches only.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

from flowlib import manifest


def seeds(specs: Iterable[str | int]) -> set[int]:
    """Return the seeds of specs such as `31005` or `31900-31939`."""
    out: set[int] = set()
    for spec in specs:
        first, _, last = str(spec).partition("-")
        out.update(range(int(first), int(last or first) + 1))
    return out


def check(split: Mapping[str, Iterable[str | int]]) -> None:
    """Raise if a seed is named for more than one host."""
    seen: dict[int, str] = {}
    for host, specs in split.items():
        for seed in sorted(seeds(specs)):
            if seed in seen:
                raise ValueError(
                    f"seed {seed} is named for both {seen[seed]} and {host}"
                )
            seen[seed] = host


def simulates(
    seed: int, host: str, split: Mapping[str, Iterable[str | int]]
) -> bool:
    """Whether `host` simulates `seed` under the split."""
    if host in split:
        return seed in seeds(split[host])
    return not any(seed in seeds(s) for s in split.values())


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
