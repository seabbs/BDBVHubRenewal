"""Which hub models a run submits, from `config/hub.toml`.

Each hub model is opted in on its own with `submit = true`. `DRY_RUN`
stays the global switch: an opted-in model in a dry run builds and
checks everything but pushes nothing.

- `renewal` (`[submission]`): the renewal chain. It runs only when
  opted in and a model has been chosen in `results/model-choice.yml`.
- `baseline` (`[models.persistencenow]`): persistence on a nowcast,
  with plain persistence as its fallback. It needs no model choice.
"""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .common import append_summary, set_outputs

TEAM = "seabbs"
KEYS = {"renewal": "renewal", "baseline": "baseline"}
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


def load_tracks(path: Path | str, chosen: str) -> list[Track]:
    """Read the hub models and decide which run.

    `chosen` is the latest model choice, or empty when none is recorded.
    """
    with Path(path).open("rb") as fh:
        cfg = tomllib.load(fh)
    sub = cfg["submission"]
    tracks = [_track(sub["model_abbr"], chosen, bool(sub.get("submit")), True)]
    for model, entry in cfg.get("models", {}).items():
        tracks.append(
            _track(entry["model_abbr"], model, bool(entry.get("submit")), False)
        )
    return tracks


def _track(abbr: str, model: str, submit: bool, needs_choice: bool) -> Track:
    if not submit:
        runs, reason = False, "not opted in (`submit = false`)"
    elif needs_choice and not model:
        runs, reason = False, "no model choice recorded"
    else:
        runs, reason = True, "opted in"
    return Track(abbr, f"{TEAM}-{abbr}", model, submit, runs, reason)


def main(argv: Sequence[str] | None = None) -> int:
    """Set `<key>=true|false` and `<key>_id` for each hub model."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hub-config", type=Path, default=Path("config/hub.toml"))
    p.add_argument("--chosen", default="", help="The latest model choice")
    args = p.parse_args(argv)
    tracks = load_tracks(args.hub_config, args.chosen)
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
