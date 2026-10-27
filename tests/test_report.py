from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bdbv_forecasts import report
from bdbv_forecasts.backtest import BacktestConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_relative_section_hidden_with_one_model() -> None:
    scores = pd.read_csv(ROOT / "results" / "backtest-scores.csv")
    assert "Only persistence exists" in report._relative_section(scores)
    two = pd.concat([scores, scores.assign(model="other")])
    text = report._relative_section(two)
    assert "relative_wis" in text
    assert "relative-wis-horizon.svg" in text


def test_persistence_bias_sentence() -> None:
    scores = pd.read_csv(ROOT / "results" / "backtest-scores.csv")
    text = report._persistence_bias(scores)
    assert text.startswith("On real vintages persistence has a mean bias")
    assert report._persistence_bias(scores.iloc[:0]) == ""


def test_heldout_page(tmp_path: Path, target: pd.DataFrame) -> None:
    config = load_config(ROOT / "config" / "backtest.toml")
    config = BacktestConfig(
        model_choice=ROOT / config.model_choice,
        development_until=config.development_until,
    )
    gen, fig = tmp_path / "gen", tmp_path / "fig"
    report._heldout_page(
        fig, gen, ROOT / "results", target, ["persistence"], config
    )
    assert "2026-10-17" in (gen / "heldout-choices.md").read_text()
    status = (gen / "heldout-status.md").read_text()
    assert "pending" in status
    assert (fig / "heldout-cd.svg").exists()
    # Without a run or a record.
    empty = BacktestConfig(model_choice=tmp_path / "none.yml")
    report._heldout_page(fig, gen, tmp_path, target, ["persistence"], empty)
    assert "No model" in (gen / "heldout-choices.md").read_text()
    assert "No held-out" in (gen / "heldout-status.md").read_text()


@pytest.mark.parametrize("display", ["named", "anonymised", "epiforecasts"])
def test_comparison_page(
    tmp_path: Path, target: pd.DataFrame, display: str
) -> None:
    from test_hubcompare import _write_hub

    from bdbv_forecasts.cli import main

    gen, fig = tmp_path / "gen", tmp_path / "fig"
    report._comparison_page(fig, gen, tmp_path / "missing", display, target)
    assert "Not run yet" in (gen / "comparison-scores.md").read_text()
    _write_hub(tmp_path / "hub")
    cfg = tmp_path / "cmp.toml"
    cfg.write_text(
        '[comparison]\nmodels = ["teamA-weekly", "epiforecasts-renewal"]\n'
    )
    assert (
        main(
            [
                "compare-hub",
                "--config",
                str(cfg),
                "--hub-dir",
                str(tmp_path / "hub"),
                "--out",
                str(tmp_path / "out"),
                "--data",
                str(ROOT / "data"),
                "--backtest-config",
                str(ROOT / "config" / "backtest.toml"),
            ]
        )
        == 0
    )
    report._comparison_page(fig, gen, tmp_path / "out", display, target)
    teams = (gen / "comparison-teams.md").read_text()
    scores = (gen / "comparison-scores.md").read_text()
    if display == "named":
        assert "teamA-weekly" in teams
    elif display == "anonymised":
        assert "Team A" in scores
        assert "teamA-weekly" not in scores
    else:
        assert "teamA" not in scores
        assert "Team A" not in scores
    assert (
        "BDBV Modeling Hub" in (gen / "comparison-attribution.md").read_text()
    )
