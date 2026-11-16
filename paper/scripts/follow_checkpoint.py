"""Pin the workflow to the final fine-tuning run's newest checkpoint.

Lists the published `chronos-ft-*` releases, keeps those whose
settings.json `grid` is GRID or a revision of it (`r4`, `r4.1`), and
prints the newest one's hash and outbreaks. With `--write`, sets
`finetuned` in the workflow config to it. Prints nothing and leaves the
config alone while the run has no published checkpoint.

Usage: python scripts/follow_checkpoint.py GRID [--write CONFIG]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = "seabbs/BDBVHubRenewal"
PREFIX = "chronos-ft-"


def _gh(*args: str) -> str:
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout


def releases() -> list[tuple[str, str, str | None, int | None]]:
    """Return each `chronos-ft-*` release's tag, date, grid and outbreaks."""
    listing = json.loads(
        _gh(
            "release",
            "list",
            "-R",
            REPO,
            "--limit",
            "200",
            "--json",
            "tagName,publishedAt,isDraft",
        )
    )
    out = []
    for r in listing:
        if r["isDraft"] or not r["tagName"].startswith(PREFIX):
            continue
        settings = json.loads(
            _gh(
                "release",
                "download",
                r["tagName"],
                "-R",
                REPO,
                "-p",
                "settings.json",
                "-O",
                "-",
            )
        )
        out.append(
            (
                r["tagName"],
                r["publishedAt"],
                settings.get("grid"),
                settings.get("n_outbreaks"),
            )
        )
    return out


def newest(
    rows: list[tuple[str, str, str | None, int | None]], grid: str
) -> tuple[str, int | None] | None:
    """Return the newest checkpoint of `grid` or its revisions, or None."""
    keep = [
        r
        for r in rows
        if r[0].startswith(PREFIX)
        and r[2] is not None
        and (r[2] == grid or str(r[2]).startswith(f"{grid}."))
    ]
    if not keep:
        return None
    tag, _, _, outbreaks = max(keep, key=lambda r: r[1])
    return tag.removeprefix(PREFIX), outbreaks


def write_pin(config: Path, checkpoint: str) -> None:
    """Set `finetuned` in the workflow config, leaving the rest as it is."""
    text = config.read_text()
    new, n = re.subn(
        r"^finetuned:.*$", f"finetuned: {checkpoint}", text, flags=re.M
    )
    if n != 1:
        raise ValueError(f"{config} has no single `finetuned` entry.")
    config.write_text(new)


def main() -> None:
    """Print, and optionally write, the newest checkpoint of the run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid")
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args()
    found = newest(releases(), args.grid)
    if found is None:
        return
    checkpoint, outbreaks = found
    print(checkpoint, outbreaks)
    if args.write is not None:
        write_pin(args.write, checkpoint)


if __name__ == "__main__":
    main()
