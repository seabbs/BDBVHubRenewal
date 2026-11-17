"""The Pages workflow's German held-out half runs every benchmark model."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / ".github" / "workflows" / "pages.yml"


def _jobs() -> dict:
    return yaml.safe_load(PAGES.read_text())["jobs"]


def test_the_held_out_matrix_covers_every_german_model() -> None:
    config = tomllib.loads((ROOT / "config" / "germany.toml").read_text())
    models = set(config["nowcast"]["models"])
    matrix = _jobs()["germany-heldout"]["strategy"]["matrix"]["include"]
    held = {row["model"] for row in matrix}
    assert models <= held, sorted(models - held)
    # The Chronos-2 models need the chronos group.
    for row in matrix:
        if row["model"].startswith("fm_chronos"):
            assert row.get("group") == "chronos", row
