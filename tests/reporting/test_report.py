from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.evaluation.backtest import BacktestConfig
from bdbvhubrenewal.reporting import report

ROOT = Path(__file__).resolve().parents[2]


def test_heldout_page(
    tmp_path: Path,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    from bdbvhubrenewal.evaluation.heldout import run_heldout

    record = tmp_path / "model-choice.yml"
    record.write_text(
        "choices:\n"
        "  - date: 2026-10-10\n"
        "    model: randomwalk\n"
        "    summary: A test choice.\n"
    )
    config = BacktestConfig(model_choice=record, development_until="2026-10-10")
    res = tmp_path / "res"
    run_heldout(
        target, delays, config, evaluate_holdout=True, out_dir=res, daily=daily
    )
    gen, fig = tmp_path / "gen", tmp_path / "fig"
    report._heldout_page(fig, gen, res, target, ["randomwalk"], config)
    choices = (gen / "heldout-choices.md").read_text()
    assert choices.startswith("## Model choices")
    assert "2026-10-17" in choices
    status = (gen / "heldout-status.md").read_text()
    assert "next vintage" in status
    assert (fig / "heldout-cd.svg").exists()
    # Without a run or a record.
    empty = BacktestConfig(model_choice=tmp_path / "none.yml")
    report._heldout_page(fig, gen, tmp_path, target, ["randomwalk"], empty)
    assert (gen / "heldout-choices.md").read_text() == ""
    assert (gen / "heldout-status.md").read_text() == ""


def test_comparison_page(tmp_path: Path, target: pd.DataFrame) -> None:
    from evaluation.test_hubcompare import _write_hub

    from bdbvhubrenewal.cli import main

    gen, fig = tmp_path / "gen", tmp_path / "fig"
    report._comparison_page(fig, gen, tmp_path / "missing", target)
    assert "compare-hub" in (gen / "comparison-scores.md").read_text()
    _write_hub(tmp_path / "hub")
    out = tmp_path / "out"
    assert (
        main(
            [
                "compare-hub",
                "--hub-dir",
                str(tmp_path / "hub"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--out",
                str(out),
                "--data",
                str(ROOT / "data"),
                "--backtest-config",
                str(ROOT / "config" / "backtest.toml"),
                "--stored",
                str(tmp_path / "none.csv.gz"),
                "--models",
                "randomwalk",
            ]
        )
        == 0
    )
    for name in ("scores.csv", "quantiles.csv.gz", "teams.csv", "summary.json"):
        assert (out / name).exists(), name
    report._comparison_page(fig, gen, out, target)
    teams = (gen / "comparison-teams.md").read_text()
    scores = (gen / "comparison-scores.md").read_text()
    # Other teams are named as the hub names them.
    assert "teamA-weekly" in teams
    assert "CC0-1.0" in teams
    assert "teamA-weekly" in scores
    figures = (gen / "comparison-figures.md").read_text()
    assert "2026-10-24" in figures
    assert (fig / "comparison-2026-10-24.svg").exists()
    # The Backtest summaries lead, with the static figure as the fallback,
    # and randomwalk starts off as the relative reference.
    assert "scoresummary-comparison.json" in scores
    assert "comparison-overall.svg" in scores
    assert 'data-view="place"' in (gen / "comparison-place.md").read_text()
    data = json.loads((fig / "scoresummary-comparison.json").read_text())
    on = {m["name"]: m["on"] for m in data["models"]}
    assert on["randomwalk (ours)"] is False
    assert on["teamA-weekly"] is True
    assert (
        "BDBV Modeling Hub" in (gen / "comparison-attribution.md").read_text()
    )


def test_lopo_section_hidden_without_results(tmp_path: Path) -> None:
    assert report._lopo_section(tmp_path / "missing.csv") == ""
    scores = _two_model_scores()
    lopo = scores.loc[scores["location"] != "CD"].assign(
        model="renewal", masked=lambda d: d["location"]
    )
    path = tmp_path / "lopo-scores.csv"
    lopo.to_csv(path, index=False)
    text = report._lopo_section(path)
    assert text.startswith("## Leave one province out")
    assert "renewal" in text
    assert "CD-IT" in text


def test_zone_section() -> None:
    text = report._zone_section(
        ROOT / "results" / "zone-scores.csv",
        ROOT / "results" / "zone-energy.csv",
    )
    assert text.startswith("## Health zones")
    assert "rel_energy" in text
    assert "randomwalknowcast (summed to provinces)" in text
    assert "week 0" in text


def test_ladder_page(tmp_path: Path) -> None:
    # No progress wording.
    ladder = pd.DataFrame(
        {
            "step": [0, 1],
            "rung": ["randomwalk", "S0 national renewal"],
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
    assert text.index("randomwalk") < text.index("S0 national renewal")
    # Headers are labels, missing cells are blank, and random walk has no
    # sampler row.
    assert "nan" not in text
    assert "real_max_rhat" not in text
    assert "Max R-hat" in text
    sampler = text.split("## Sampler")[1].split("## Recovery")[0]
    assert "randomwalk" not in sampler


def _scores() -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "model": "randomwalk",
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


def _two_model_scores() -> pd.DataFrame:
    rows = []
    for model, wis in (("randomwalk", 10.0), ("renewal", 5.0)):
        for loc in ("CD", "CD-IT"):
            rows.append(_scores().assign(model=model, location=loc, wis=wis))
    return pd.concat(rows, ignore_index=True)


def test_teams_table_links_to_hub_metadata() -> None:
    teams = pd.DataFrame(
        {
            "model": ["teamA-weekly", "epiforecasts-x"],
            "team_name": ["Team A", "Us"],
            "model_name": ["Weekly", "X"],
            "license": ["CC0-1.0", "MIT"],
            "website_url": ["", ""],
        }
    )
    text = report._teams_table(teams, {"x": "epiforecasts-x"})
    assert "| Team " in text
    assert "`teamA-weekly`" in text
    assert "model-metadata/teamA-weekly.yml" in text
    assert "epiforecasts-x" not in text


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
    backtest = {
        "num_chains": 2,
        "num_warmup": 300,
        "num_samples": 300,
        "target_accept": 0.8,
    }
    submission = {
        "renewal": {**backtest, "target_accept": 0.9},
        "renewalbvd": {**backtest, "num_samples": 1000, "target_accept": 0.95},
    }
    text = report._sampler_section(fits, backtest, submission)
    assert (
        "Backtest fits use 2 chains of 300 warm-up and 300 draws at a "
        "target acceptance of 0.8." in text
    )
    assert (
        "- `renewalbvd`: 2 chains of 300 warm-up and 1000 draws at a "
        "target acceptance of 0.95" in text
    )
    assert "same as" not in text
    assert "renewalsp" in text
    assert report._sampler_section(fits.iloc[:0], backtest, submission) == ""


def test_development_log(tmp_path: Path) -> None:
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
        ("randomwalk", 20.0, 0.6),
        ("renewal", 10.0, 0.9),
        ("renewalsp", 15.0, 0.8),
    ):
        rows.append(
            _scores().assign(
                model=model,
                wis=wis,
                coverage_90=cov,
                origin="2026-10-09",
                target="reported",
            )
        )
    primary = pd.concat(rows)
    text = report._backtest_findings(primary)
    assert "renewal 0.50" in text
    assert "renewalsp 0.75" in text
    assert "randomwalk 0.60" in text
    assert report._backtest_findings(rows[0]) == ""


def test_forecast_tabs() -> None:
    text = report._forecast_tabs(["randomwalk", "renewal"], {"CD": "DRC"})
    assert text.startswith("## Forecasts against later data")
    assert '=== "renewal"' in text
    assert "figures/forecasts-renewal-cd.svg#only-light" in text
    assert report._forecast_tabs([], {"CD": "DRC"}) == ""


def test_recovery_section() -> None:
    energy = pd.DataFrame(
        {
            "scale": ["national", "national"],
            "model": ["randomwalk", "randomwalk"],
            "energy": [4.0, 6.0],
            "log_energy": [0.25, 0.75],
        }
    )
    text = report._recovery_section(energy)
    assert text.startswith("## Recovery on simulated vintages")
    assert "recovery-wis-horizon.svg" in text
    assert "mean energy score" in text
    assert "| national | randomwalk |   2 |" in text


def _scenario_scores() -> pd.DataFrame:
    rows = []
    for model in ("randomwalk", "randomwalknowcast"):
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
                                "level": "national"
                                if location == "CD"
                                else "province",
                                "horizon": horizon,
                                "wis": 10.0 + horizon,
                                "log_wis": 0.5,
                                "coverage_50": 0.5,
                                "coverage_90": 0.9,
                                "bias": 0.123,
                                "complete": view == "final",
                                "truth_view": view,
                            }
                        )
    return pd.DataFrame(rows)


def test_scenario_scores_table(tmp_path: Path) -> None:
    path = tmp_path / "scenario-scores.csv"
    _scenario_scores().to_csv(path, index=False)
    energy = tmp_path / "scenario-energy.csv"
    pd.DataFrame(
        {
            "scenario": "calibrated_falling",
            "scale": "national",
            "model": "randomwalk",
            "truth_view": ["final", "as of 2 weeks later"],
            "energy": [3.0, 5.0],
            "log_energy": [0.3, 0.5],
        }
    ).to_csv(energy, index=False)
    text = report._scenario_scores(path, energy)
    assert "Energy score against final data, complete weeks" in text
    assert "final data, complete weeks" in text
    assert "as of 2 weeks later" in text
    assert "calibrated_falling" in text
    assert "national" in text
    assert "province" in text
    assert "randomwalknowcast" in text
    assert "10.5" in text  # mean WIS over horizons 0 and 1
    assert "0.12" in text  # bias to 2 dp


def test_plot_scenarios() -> None:
    from bdbvhubrenewal.reporting.plots import plot_scenarios

    fig = plot_scenarios(
        _scenario_scores(), models=["randomwalk", "randomwalknowcast"]
    )
    assert len(fig.axes) == 2  # one panel per scenario


def test_heldout_page_before_any_held_out_vintage(
    tmp_path: Path, target: pd.DataFrame
) -> None:
    record = tmp_path / "model-choice.yml"
    record.write_text(
        "choices:\n  - {date: 2026-12-31, model: randomwalknowcast}\n"
    )
    res = tmp_path / "res"
    res.mkdir()
    (res / "heldout-run.yml").write_text(
        "run_at: '2026-12-31T00:00:00+00:00'\nscored: []\npending: []\n"
        "latest_vintage: '2026-10-24'\n"
    )
    (res / "heldout-scores.csv").write_text("\n")
    (res / "heldout-forecasts.csv.gz").write_text("\n")
    config = BacktestConfig(model_choice=record, development_until="2026-12-31")
    gen, fig = tmp_path / "gen", tmp_path / "fig"
    report._heldout_page(fig, gen, res, target, ["randomwalk"], config)
    assert "randomwalknowcast" in (gen / "heldout-choices.md").read_text()
    for name in (
        "heldout-status.md",
        "heldout-scores.md",
        "heldout-forecasts.md",
    ):
        assert (gen / name).read_text() == ""


def test_dropdown_wraps_and_indents() -> None:
    text = report._dropdown("a\n\nb\n")
    assert text == '??? note "Tables"\n\n    a\n\n    b\n'
    assert report._dropdown("  \n") == ""
    assert report._dropdown("x", "More").startswith('??? note "More"')


def test_metric_tabs_have_wis_and_log_wis() -> None:
    text = report._metric_tabs("x", "WIS by horizon.")
    assert '=== "WIS"' in text
    assert '=== "WIS on the log scale"' in text
    assert "figures/x-log-wis.svg#only-light" in text


def test_nowcast_groups() -> None:
    target = pd.DataFrame(
        {
            "kind": "real",
            "vintage": "2026-10-24",
            "location": ["CD", "CD-A", "CD-B", "CD-C", "CD-D"],
            "total": [10, 1, 5, 3, 2],
        }
    )
    groups = report._nowcast_groups(
        target, ["CD", "CD-A", "CD-B", "CD-C", "CD-D", "CD-D-01"]
    )
    assert groups["main"] == ["CD", "CD-B", "CD-C", "CD-D"]
    assert groups["other"] == ["CD-A"]
    assert groups["zones"] == ["CD-D-01"]


def test_release_scores_horizon_is_weeks_from_the_reference_date(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {
            "origin": "2026-10-17",
            "target_end_date": ["2026-10-10", "2026-10-17", "2026-10-24"],
        }
    ).to_csv(tmp_path / "nowcast-scores.csv", index=False)
    scores = report._release_scores(tmp_path)
    assert scores["horizon"].tolist() == [-2, -1, 0]
    assert set(scores["vintage"]) == {"2026-10-17"}


def test_summary_covers_only_forecasts_randomwalk_made(tmp_path: Path) -> None:
    scores = report._release_scores(ROOT / "results")
    reported = scores.loc[scores["target"] == "reported"]
    gen = tmp_path / "gen"
    gen.mkdir()
    report._horizon_table(gen, reported, ["randomwalk", "randomwalknowcast"])
    text = (gen / "backtest-summary.md").read_text()
    walk = reported.loc[reported["model"] == "randomwalk", "origin"]
    origins = ", ".join(sorted(set(walk.astype(str))))
    assert f"the origins {origins}" in text
    assert "   n |" in text
    assert "nan" not in text.lower()


def test_backtest_findings_name_the_reported_target() -> None:
    scores = report._release_scores(ROOT / "results")
    text = report._backtest_findings(scores)
    assert text.startswith("On the counts each later release reported")
    assert "randomwalknowcast" in text
