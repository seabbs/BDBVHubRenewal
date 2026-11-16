"""Pin the workflow to the final fine-tuning run's newest checkpoint.

The final run writes each checkpoint to `<run>/n<outbreaks>/` with its
settings.json, and publishes it as a release whose settings.json records
the same `consumed_sha256`: `chronos-slots-<hash>` for the series-slots
run, pinned as `slots-<hash>`, or `chronos-ft-<hash>`, pinned as its hash.
This matches each local checkpoint to its release and, with `--write`,
sets `finetuned` to the newest published one and `final_checkpoints` to
all of them, in the workflow config. Before the run has published a
checkpoint it prints nothing and leaves the config alone.

Usage: python scripts/follow_checkpoint.py RUN_DIR [--write CONFIG]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = "seabbs/BDBVHubRenewal"
# Release tag prefix -> the spec prefix the loader takes for it.
PREFIXES = {"chronos-slots-": "slots-", "chronos-ft-": ""}


def spec_of(tag: str) -> str | None:
    """Return the loader's spec of a fine-tuned release tag, or None."""
    for prefix, spec in PREFIXES.items():
        if tag.startswith(prefix):
            return spec + tag.removeprefix(prefix)
    return None


def _gh(*args: str) -> str:
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout


def releases() -> dict[str, str]:
    """Return each published release's spec by its `consumed_sha256`."""
    listing = json.loads(
        _gh(
            "release",
            "list",
            "-R",
            REPO,
            "--limit",
            "200",
            "--json",
            "tagName,isDraft",
        )
    )
    out: dict[str, str] = {}
    for r in listing:
        tag = r["tagName"]
        if r["isDraft"] or spec_of(tag) is None:
            continue
        settings = json.loads(
            _gh(
                "release",
                "download",
                tag,
                "-R",
                REPO,
                "-p",
                "settings.json",
                "-O",
                "-",
            )
        )
        if settings.get("consumed_sha256"):
            out[settings["consumed_sha256"]] = spec_of(tag)
    return out


def published(run: Path, by_consumed: dict[str, str]) -> list[tuple[str, int]]:
    """Return the run's published checkpoints, by outbreaks, as (hash, n)."""
    out = []
    for d in Path(run).glob("n*/settings.json"):
        settings = json.loads(d.read_text())
        found = by_consumed.get(settings.get("consumed_sha256", ""))
        if found is not None:
            out.append((found, int(settings["n_outbreaks"])))
    return sorted(out, key=lambda r: r[1])


def write_pins(config: Path, checkpoints: list[tuple[str, int]]) -> None:
    """Set `finetuned` and `final_checkpoints`, leaving the rest as it is."""
    text = config.read_text()
    hashes = ", ".join(h for h, _ in checkpoints)
    for key, value in (
        ("finetuned", checkpoints[-1][0]),
        ("final_checkpoints", f"[{hashes}]"),
    ):
        text, n = re.subn(
            rf"^{key}:.*$", f"{key}: {value}", text, flags=re.MULTILINE
        )
        if n != 1:
            raise ValueError(f"{config} has no single `{key}` entry.")
    config.write_text(text)


def main() -> None:
    """Print, and optionally write, the run's published checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args()
    found = published(args.run, releases()) if args.run.exists() else []
    if not found:
        return
    print(" ".join(f"{h}:{n}" for h, n in found))
    if args.write is not None:
        write_pins(args.write, found)


if __name__ == "__main__":
    main()
