from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.backtest import BacktestConfig, load_config
from bdbvhubrenewal.reporting import report

ROOT = Path(__file__).resolve().parents[2]


def test_relative_section_hidden_with_one_model() -> None:
    scores = pd.read_csv(ROOT / "results" / "backtest-scores.csv")
    scores = scores.loc[scores["model"] == "persistence"]
    # Persistence against itself means nothing, so nothing is shown.
    assert report._relative_section(scores) == ""
    two = pd.concat([scores, scores.assign(model="other")])
    text = report._relative_section(two)
    assert text.startswith("## Relative skill")
    assert "relative_wis" in text
    assert "relative-wis-horizon.svg" in text


def test_heldout_page(tmp_path: Path, target: pd.DataFrame) -> None:
    record = tmp_path / "model-choice.yml"
    record.write_text(
        "choices:\n"
        "  - date: 2026-10-10\n"
        "    model: persistence\n"
        "    summary: A test choice.\n"
    )
    config = BacktestConfig(model_choice=record, development_until="2026-10-10")
    gen, fig = tmp_path / "gen", tmp_path / "fig"
    report._heldout_page(
        fig, gen, ROOT / "results", target, ["persistence"], config
    )
    choices = (gen / "heldout-choices.md").read_text()
    assert choices.startswith("## Model choices")
    assert "2026-10-17" in choices
    status = (gen / "heldout-status.md").read_text()
    assert "next vintage" in status
    assert (fig / "heldout-cd.svg").exists()
    # Without a run or a record.
    empty = BacktestConfig(model_choice=tmp_path / "none.yml")
    report._heldout_page(fig, gen, tmp_path, target, ["persistence"], empty)
    assert (gen / "heldout-choices.md").read_text() == ""
    assert (gen / "heldout-status.md").read_text() == ""


@pytest.mark.parametrize("display", ["named", "anonymised", "epiforecasts"])
def test_comparison_page(
    tmp_path: Path, target: pd.DataFrame, display: str
) -> None:
    from evaluation.test_hubcompare import _write_hub

    from bdbvhubrenewal.cli import main

    gen, fig = tmp_path / "gen", tmp_path / "fig"
    report._comparison_page(fig, gen, tmp_path / "missing", display, target)
    assert (gen / "comparison-scores.md").read_text() == ""
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

    from bdbvhubrenewal.evaluation.backtest import run_development
    from bdbvhubrenewal.forecasters import get_model

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
        "backtest-findings.md",
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
        "forecasts-persistence-cd.svg",
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


def test_ladder_page(tmp_path: Path) -> None:
    # Nothing is shown without results, and no progress wording.
    assert report._ladder_page(tmp_path / "missing.csv") == ""
    ladder = pd.DataFrame(
        {
            "step": [0, 1],
            "rung": ["persistence", "S0 national renewal"],
            "description": ["The baseline.", "One national renewal."],
            "decision": ["Reference.", "Kept as the starting point."],
            "real_max_rhat": [None, 1.01],
            "real_min_ess": [None, 400.0],
            "real_divergent": [None, 0],
            "sim_rt_mae": [None, 0.12],
            "real_crps_national": [30.0, 25.0],
            "real_cov90_national_ahead": [0.5, 0.8],
        }
    )
    path = tmp_path / "ladder.csv"
    ladder.to_csv(path, index=False)
    text = report._ladder_page(path)
    for heading in ("## Rungs", "## Sampler", "## Recovery", "## Scores"):
        assert heading in text
    assert "S0 national renewal" in text
    assert "Kept as the starting point." in text
    # Rows run in build order.
    assert text.index("persistence") < text.index("S0 national renewal")
    # Headers are labels, missing cells are blank, and persistence has no
    # sampler row.
    assert "nan" not in text
    assert "real_max_rhat" not in text
    assert "Max R-hat" in text
    sampler = text.split("## Sampler")[1].split("## Recovery")[0]
    assert "persistence" not in sampler


def test_nowcast_recovery_section(tmp_path: Path) -> None:
    assert report._nowcast_recovery_section(tmp_path / "missing.csv") == ""
    rows = pd.DataFrame(
        {
            "sim_day": [188, 188],
            "vintage": ["2026-10-10", "2026-10-10"],
            "version": ["fresh", "recentred"],
            "commit": ["abc", "def"],
            "truth": ["final_vintage", "final_vintage"],
            "n_cells": [20, 20],
            "coverage_50": [0.3, 0.5],
            "coverage_90": [0.6, 0.9],
            "mae_median": [12.0, 6.0],
            "share_below_reported": [0.2, 0.0],
            "divergent_share": [0.0, 0.0],
            "max_rhat": [None, 1.01],
            "step_size": [None, 0.05],
        }
    )
    path = tmp_path / "nowcast-recovery.csv"
    rows.to_csv(path, index=False)
    text = report._nowcast_recovery_section(path)
    assert text.startswith("### Nowcasts of partly reported weeks")
    assert "recentred" in text
    assert "final_vintage" not in text  # truths get plain labels
    assert "final vintage" in text


def test_nowcast_recovery_lead_follows_the_results() -> None:
    path = ROOT / "results" / "nowcast-recovery.csv"
    text = report._nowcast_recovery_section(path)
    rows = pd.read_csv(path)
    final = rows.loc[rows["truth"] == "final_vintage"]
    fresh = final.loc[final["version"] == "fresh", "mae_median"]
    recentred = final.loc[final["version"] == "recentred", "mae_median"]
    # The lead states the error of each version as the results give it.
    for value in [*fresh, *recentred]:
        assert f"{value:.1f}" in text


def _scores() -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "model": "persistence",
            "kind": "real",
            "vintage": "2026-10-09",
            "reference_date": "2026-10-17",
            "location": "CD",
            "target_end_date": ["2026-09-26", "2026-10-17"],
            "horizon": [-3, 0],
            "truth_vintage": "2026-10-24",
            "wis": [10.0, 50.0],
        }
    )
    for col in (
        "dispersion",
        "overprediction",
        "underprediction",
        "log_wis",
        "crps",
        "ae_median",
        "coverage_50",
        "coverage_90",
        "bias",
    ):
        base[col] = 0.0
    return base


def test_score_views() -> None:
    old = _scores()
    # Results without the new columns are final counts as reported, and
    # complete weeks follow K.
    views = report._score_views(old, 3)
    assert list(views) == [
        "complete weeks",
        "all weeks as reported",
        "vintage-matched",
    ]
    assert list(views["complete weeks"]["target_end_date"]) == ["2026-09-26"]
    assert len(views["all weeks as reported"]) == 2
    assert views["vintage-matched"].empty
    new = pd.concat(
        [
            old.assign(scoring="as reported", complete=[True, False]),
            old.assign(scoring="vintage-matched", complete=[True, False]),
        ]
    )
    views = report._score_views(new, 3)
    assert len(views["vintage-matched"]) == 2
    assert len(views["all weeks as reported"]) == 2


def test_overall_scores_table() -> None:
    views = report._score_views(_scores(), 3)
    text = report._overall_scores(views)
    assert "complete weeks" in text
    assert "all weeks as reported" in text
    # A view with no rows is left out.
    assert "vintage-matched" not in text


def test_completeness_note(delays: pd.DataFrame) -> None:
    text = report._completeness_note(3, delays)
    assert "at least 3 weeks before the truth vintage" in text
    assert "2026-10-24 delay data" in text
    assert "96%" in text


def test_score_views_without_lag_prior_origins() -> None:
    scores = _scores().assign(lag_prior=[False, False])
    other = _scores().assign(vintage="2026-08-18", lag_prior=[True, True])
    views = report._score_views(pd.concat([scores, other]), 3)
    own = views["complete weeks, own lag estimate"]
    assert set(own["vintage"]) == {"2026-10-09"}
    assert len(views["complete weeks"]) == 2
    note = report._lag_prior_note(pd.concat([scores, other]))
    assert "2026-08-18" in note
    assert "2026-10-09" not in note
    assert report._lag_prior_note(scores) == ""


def test_sampler_section() -> None:
    fits = pd.DataFrame(
        {
            "model": ["renewal", "renewal", "renewalsp"],
            "kind": "real",
            "vintage": ["2026-08-18", "2026-10-09", "2026-10-09"],
            "fit": ["all", "all", "CD-IT"],
            "num_chains": 2,
            "num_warmup": 300,
            "num_samples": 300,
            "seconds": [100.0, 200.0, 50.0],
            "max_rhat": [1.01, 1.2, 1.02],
            "depth_share": [0.9, 1.0, 0.2],
            "divergent_share": [0.0, 0.01, 0.0],
        }
    )
    submission = {"num_chains": 4, "num_warmup": 1000, "num_samples": 1000}
    text = report._sampler_section(fits, submission)
    assert "2 chains of 300 warm-up and 300 draws" in text
    assert "4 chains of 1000 warm-up and 1000 draws" in text
    assert "shorter than the submission fit" in text
    assert "renewalsp" in text
    assert report._sampler_section(fits.iloc[:0], submission) == ""
    # With the submission's settings the text says they match.
    same = {"num_chains": 2, "num_warmup": 300, "num_samples": 300}
    text = report._sampler_section(fits, same)
    assert "the same as the submission fit" in text
    assert "shorter" not in text


def test_development_log(tmp_path: Path) -> None:
    assert report._development_log(tmp_path / "missing.csv") == ""
    log = pd.DataFrame(
        {
            "date": ["2026-10-10", "2026-10-28"],
            "area": ["evaluation", "model"],
            "decision": ["Old decision.", "New decision."],
            "reason": ["Why then.", "Why now."],
            "evidence": [
                "results/model-choice.yml",
                "https://github.com/seabbs/BDBVHubRenewal/commit/abc1234",
            ],
        }
    )
    path = tmp_path / "log.csv"
    log.to_csv(path, index=False)
    text = report._development_log(path)
    # Newest first, with evidence as links.
    assert text.index("New decision.") < text.index("Old decision.")
    assert (
        "[abc1234](https://github.com/seabbs/BDBVHubRenewal/commit/abc1234)"
        in text
    )
    # The committed log reads.
    assert "fixed 0 to 3 days" in report._development_log(
        ROOT / "results" / "development-log.csv"
    )


def test_backtest_findings() -> None:
    rows = []
    for model, wis, cov in (
        ("persistence", 20.0, 0.6),
        ("renewal", 10.0, 0.9),
        ("renewalsp", 15.0, 0.8),
    ):
        rows.append(
            _scores().assign(
                model=model, wis=wis, coverage_90=cov, complete=True
            )
        )
    primary = pd.concat(rows)
    text = report._backtest_findings(primary)
    assert "renewal 0.50" in text
    assert "renewalsp 0.75" in text
    assert "persistence 0.60" in text
    assert report._backtest_findings(rows[0]) == ""


def test_forecast_tabs() -> None:
    text = report._forecast_tabs(["persistence", "renewal"], {"CD": "DRC"})
    assert text.startswith("## Forecasts against later data")
    assert '=== "renewal"' in text
    assert "figures/forecasts-renewal-cd.svg#only-light" in text
    assert report._forecast_tabs([], {"CD": "DRC"}) == ""


def test_recovery_section(tmp_path: Path) -> None:
    assert report._recovery_section(tmp_path / "missing.csv") == ""
    path = tmp_path / "recovery-scores.csv"
    _scores().assign(kind="simulated").to_csv(path, index=False)
    text = report._recovery_section(path)
    assert text.startswith("## Recovery on simulated vintages")
    assert "recovery-wis-horizon.svg" in text


def _scenario_scores() -> pd.DataFrame:
    rows = []
    for model in ("persistence", "persistencenow"):
        for scenario in ("calibrated_falling", "calibrated_rising"):
            for location in ("CD", "CD-IT"):
                for horizon in (0, 1):
                    for view in ("final", "later"):
                        rows.append(
                            {
                                "model": model,
                                "scenario": scenario,
                                "vintage": "2026-10-24",
                                "location": location,
                                "horizon": horizon,
                                "wis": 10.0 + horizon,
                                "coverage_50": 0.5,
                                "coverage_90": 0.9,
                                "bias": 0.123,
                                "complete": view == "final",
                                "truth_view": view,
                            }
                        )
    return pd.DataFrame(rows)


def test_scenario_scores_hidden_without_results(tmp_path: Path) -> None:
    assert report._scenario_scores(tmp_path / "missing.csv") == ""


def test_scenario_scores_table(tmp_path: Path) -> None:
    path = tmp_path / "scenario-scores.csv"
    _scenario_scores().to_csv(path, index=False)
    text = report._scenario_scores(path)
    assert "final data, complete weeks" in text
    assert "as of 2 weeks later" in text
    assert "calibrated_falling" in text
    assert "national" in text
    assert "province" in text
    assert "persistencenow" in text
    assert "10.5" in text  # mean WIS over horizons 0 and 1
    assert "0.12" in text  # bias to 2 dp


def test_plot_scenarios() -> None:
    from bdbvhubrenewal.reporting.plots import plot_scenarios

    fig = plot_scenarios(
        _scenario_scores(), models=["persistence", "persistencenow"]
    )
    assert len(fig.axes) >= 3
