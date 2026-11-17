"""The sim-QC exclusions of rounds released as `sims-r<N>`.

A round's release holds `excluded-r<N>.csv` once it has been reviewed.
A release without it is a round not yet reviewed, as a local round
without `qc/excluded.csv`; any other failure (no release, `gh` errors)
stops the caller, so training never runs without exclusions it should
have.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path


def fetch_released(
    rounds: Iterable[int],
    repo: str,
    dest: Path | str,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[int, str]:
    """Download each round's released exclusions into `dest`.

    Returns
    -------
    dict
        Each round's outcome: "downloaded" or "not reviewed yet".

    Raises
    ------
    RuntimeError
        If a release is missing or `gh` fails.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for r in sorted(set(rounds)):
        tag, asset = f"sims-r{r}", f"excluded-r{r}.csv"
        view = run(
            ["gh", "release", "view", tag, "-R", repo, "--json", "assets"],
            capture_output=True,
            text=True,
        )
        if view.returncode != 0:
            raise RuntimeError(f"{tag}: {view.stderr.strip()}")
        names = {a["name"] for a in json.loads(view.stdout)["assets"]}
        if asset not in names:
            out[r] = "not reviewed yet"
            continue
        got = run(
            [
                "gh",
                "release",
                "download",
                tag,
                "-R",
                repo,
                "--clobber",
                "-D",
                str(dest),
                "-p",
                asset,
            ],
            capture_output=True,
            text=True,
        )
        if got.returncode != 0:
            raise RuntimeError(f"{tag} {asset}: {got.stderr.strip()}")
        out[r] = "downloaded"
    return out
