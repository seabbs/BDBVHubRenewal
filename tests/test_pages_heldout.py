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
    # The stored fine-tuned model is shown on the development half only:
    # the paper's held-out run waits for the final (slots) checkpoint.
    models = set(config["nowcast"]["models"])
    matrix = _jobs()["germany-heldout"]["strategy"]["matrix"]["include"]
    held = {row["model"] for row in matrix}
    assert models <= held, sorted(models - held)
    # The Chronos-2 models need the chronos group.
    for row in matrix:
        if row["model"].startswith("fm_chronos"):
            assert row.get("group") == "chronos", row


def _build_script() -> str:
    steps = _jobs()["build"]["steps"]
    return "\n".join(str(s.get("run", "")) for s in steps)


def test_the_stored_ft_model_is_not_forecast_by_pages() -> None:
    # Its forecasts come from the checkpoint scorer's store.
    cells = _jobs()["backtest"]["strategy"]["matrix"]["include"]
    live = {c["model"] for c in cells if c.get("germany")}
    assert not live & {
        "fm_chronos_daily_ragged_ft",
        "fm_chronos_weekly_ragged_ft",
    }
    run = _build_script()
    load = run.index("germany-ft-load")
    assert "germany-ft-forecasts" in run
    assert load < run.index("concat-results --out results/germany")
    assert "parts-germany-ft-store/germany" in run


def test_the_built_german_page_is_checked_for_held_out_dates() -> None:
    run = _build_script()
    check = run.index("scripts/check_germany_site.py")
    assert run.index("mkdocs build") < check


def test_a_paper_dispatch_is_never_deployed() -> None:
    # Its pages hold the held-out half, which waits for the paper.
    jobs = _jobs()
    for name in ("deploy", "release"):
        assert "inputs.paper_target != 'paper'" in jobs[name]["if"], name
