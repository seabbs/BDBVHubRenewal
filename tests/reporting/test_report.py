from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bdbv_hub_renewal.evaluation.backtest import BacktestConfig, load_config
from bdbv_hub_renewal.reporting import report

ROOT = Path(__file__).resolve().parents[2]


def test_relative_section_hidden_with_one_model() -> None:
    scores = pd.read_csv(ROOT / "results" / "backtest-scores.csv")
    assert "no model other than persistence" in report._relative_section(scores)
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
    from evaluation.test_hubcompare import _write_hub

    from bdbv_hub_renewal.cli import main

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
                "--models",
                "persistence",
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


def test_make_report_writes_every_page(
    tmp_path: Path, target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    import os

    from bdbv_hub_renewal.evaluation.backtest import run_development
    from bdbv_hub_renewal.forecasters import get_model

    config = load_config(ROOT / "config" / "backtest.toml")
    config = BacktestConfig(
        n_samples=20,
        seed=1,
        development_until=config.development_until,
        model_choice=ROOT / config.model_choice,
        reconstructed_dates=("2026-09-26", "2026-10-03"),
        simulated_dates=("2026-08-01",),
    )
    models = [get_model("persistence")]
    out = tmp_path / "output"
    out.mkdir()
    forecasts, _ = run_development(models, target, delays, config)
    forecasts.to_csv(out / "forecasts.csv.gz", index=False)
    old = Path.cwd()
    os.chdir(ROOT)
    try:
        report.make_report(
            figures=tmp_path / "fig",
            generated=tmp_path / "gen",
            output=out,
            hub_comparison=tmp_path / "none",
        )
    finally:
        os.chdir(old)
    gen = tmp_path / "gen"
    for name in (
        "vintages.md",
        "scores-overall.md",
        "relative-section.md",
        "persistence-bias.md",
        "truth-changes.md",
        "heldout-status.md",
        "comparison-scores.md",
        "nowcast-tabs.md",
    ):
        assert (gen / name).exists(), name
    fig = tmp_path / "fig"
    for name in (
        "target.svg",
        "scores-wis-horizon.svg",
        "scores-by-truth-dark.svg",
        "forecasts-real-cd.svg",
        "recovery-wis-horizon.svg",
        "nowcast-cd.svg",
    ):
        assert (fig / name).exists(), name


def test_lopo_section_hidden_without_results(tmp_path: Path) -> None:
    assert report._lopo_section(tmp_path / "missing.csv") == ""
    scores = pd.read_csv(ROOT / "results" / "backtest-scores.csv")
    lopo = scores.loc[scores["location"] != "CD"].assign(
        model="renewal", masked=lambda d: d["location"]
    )
    path = tmp_path / "lopo-scores.csv"
    lopo.to_csv(path, index=False)
    text = report._lopo_section(path)
    assert text.startswith("## Leave one province out")
    assert "renewal" in text
    assert "CD-IT" in text
