"""Which hub models a run submits, from `config/hub.toml`.

Each hub model is opted in on its own with `submit = true`. `DRY_RUN`
stays the global switch: an opted-in model in a dry run builds and
checks everything but pushes nothing.

- `renewal` (`[submission]`): the renewal chain, from `model` there,
  which may fall back down the chain in `ci/policy.toml`.
- `baseline` (`[models.randomwalknowcast]`): random walk on a nowcast.
  It has no fallback. If it cannot run, its
  round is skipped and nothing is submitted under its id.
- `chronos` (`[models.<fm_chronos_*_ragged_ft>]`): fine-tuned Chronos-2,
  run in its own job, with no fallback either.
"""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .common import append_summary, set_outputs

TEAM = "seabbs"
KEYS = {"renewal": "renewal", "rwnowcast": "baseline", "chronos2_ft": "chronos"}
"""Step output name of each hub model's switch."""


@dataclass(frozen=True)
class Track:
    """One hub model and whether this run submits it."""

    name: str
    model_id: str
    model: str
    submit: bool
    runs: bool
    reason: str


def load_tracks(path: Path | str) -> list[Track]:
    """Read the hub models and decide which run."""
    with Path(path).open("rb") as fh:
        cfg = tomllib.load(fh)
    sub = cfg["submission"]
    tracks = [_track(sub["model_abbr"], sub["model"], bool(sub.get("submit")))]
    for model, entry in cfg.get("models", {}).items():
        tracks.append(
            _track(entry["model_abbr"], model, bool(entry.get("submit")))
        )
    return tracks


def _track(abbr: str, model: str, submit: bool) -> Track:
    reason = "opted in" if submit else "not opted in (`submit = false`)"
    return Track(abbr, f"{TEAM}-{abbr}", model, submit, submit, reason)


def main(argv: Sequence[str] | None = None) -> int:
    """Set `<key>=true|false` and `<key>_id` for each hub model."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hub-config", type=Path, default=Path("config/hub.toml"))
    args = p.parse_args(argv)
    tracks = load_tracks(args.hub_config)
    outputs: dict[str, object] = {}
    lines = ["| hub model | submitted this run | why |", "| --- | --- | --- |"]
    for t in tracks:
        key = KEYS.get(t.name, t.name)
        outputs[key] = t.runs
        outputs[f"{key}_id"] = t.model_id
        outputs[f"{key}_model"] = t.model
        lines.append(
            f"| `{t.model_id}` | {'yes' if t.runs else 'no'} | {t.reason} |"
        )
        print(f"{t.model_id}: {'runs' if t.runs else 'skipped'} ({t.reason})")
    set_outputs(outputs)
    append_summary("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
