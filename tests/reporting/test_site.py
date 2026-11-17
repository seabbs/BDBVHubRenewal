"""The docs site's settings (`config/site.toml`): the score baseline."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.evaluation import checks, zones
from bdbvhubrenewal.evaluation.hubcompare import load_comparison_config
from bdbvhubrenewal.reporting import nowcast_plots, plots, report, scoresummary
from bdbvhubrenewal.reporting.site import load_site_config, site_baseline

ROOT = Path(__file__).parents[2]


def test_the_site_scores_against_randomwalknowcast() -> None:
    assert load_site_config().baseline == "randomwalknowcast"
    assert site_baseline() == "randomwalknowcast"


def test_a_site_config_sets_the_baseline(tmp_path: Path) -> None:
    path = tmp_path / "site.toml"
    path.write_text('[scores]\nbaseline = "renewal"\n')
    assert load_site_config(path).baseline == "renewal"


def test_reporting_defaults_follow_the_site_baseline() -> None:
    for fn in (
        report._summaries,
        report._scored_on,
        report._without_baseline,
        report._checks_chart,
        scoresummary.summary_rows,
        nowcast_plots.plot_relative_overall,
        checks.check_summary,
    ):
        default = inspect.signature(fn).parameters["baseline"].default
        assert default is None, fn.__qualname__
    rel = pd.DataFrame({"model": ["randomwalk", "randomwalknowcast"]})
    assert list(report._without_baseline(rel)["model"]) == ["randomwalk"]


def test_scored_on_names_the_site_baseline() -> None:
    rows = pd.DataFrame(
        {
            "model": ["randomwalknowcast", "randomwalk"],
            "origin": ["a", "a"],
            "truth_vintage": ["b", "b"],
            "pool": ["national", "national"],
            "truth": [5.0, 5.0],
        }
    )
    assert "The randomwalknowcast baseline has 1 national" in (
        report._scored_on(rows)
    )


def test_axis_labels_name_the_site_baseline() -> None:
    assert plots._metric_label("relative_crps") == (
        "CRPS relative to randomwalknowcast"
    )
    assert plots._metric_label("crps") == "Mean CRPS"


def test_comparisons_zones_and_history_use_randomwalknowcast() -> None:
    assert load_comparison_config().baseline == "randomwalknowcast"
    assert zones.ZONE_BASELINE == "randomwalknowcast"
    spec = importlib.util.spec_from_file_location(
        "score_history", ROOT / "scripts" / "score_history.py"
    )
    assert spec is not None
    assert spec.loader is not None
    history = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(history)
    assert history.DEFAULT_BASELINE == "randomwalknowcast"


def test_no_table_filter_defaults_to_randomwalk() -> None:
    source = (ROOT / "src/bdbvhubrenewal/reporting/report.py").read_text()
    assert '{"baseline": "randomwalk"}' not in source


def test_the_site_shows_one_chronos_model() -> None:
    from bdbvhubrenewal.reporting.site import site_models

    names = [
        "randomwalk",
        "renewal",
        "fm_chronos_daily_ragged_ft",
        "fm_chronos_weekly_ragged_ft",
        "fm_chronos_daily_ragged",
        "fm_chronos_daily_complete",
        # A Chronos model added later is hidden too.
        "fm_chronos_daily_ragged_de",
    ]
    assert site_models(names) == [
        "randomwalk",
        "renewal",
        "fm_chronos_daily_ragged_ft",
    ]


def test_a_site_config_sets_the_shown_models(tmp_path: Path) -> None:
    path = tmp_path / "site.toml"
    path.write_text(
        '[scores]\nbaseline = "randomwalk"\n'
        '[models]\nhide = ["renewal*"]\nshow = ["renewalsp"]\n'
    )
    cfg = load_site_config(path)
    assert cfg.shows("renewalsp")
    assert not cfg.shows("renewal")
    assert cfg.shows("randomwalk")


def test_the_pages_list_only_shown_models() -> None:
    assert "fm_chronos_daily_ragged" not in report._page_models()
    assert "randomwalknowcast" in report._page_models()


def test_hidden_models_leave_the_germany_page(tmp_path: Path) -> None:
    from bdbvhubrenewal.reporting import germany

    pd.DataFrame(
        {
            "model": ["fm_chronos_daily_ragged", "fm_chronos_daily_ragged_ft"],
            "origin": ["2021-12-01", "2021-12-01"],
        }
    ).to_csv(tmp_path / "scores.csv.gz", index=False)
    got = germany._read(tmp_path, ("scores.csv.gz",))
    assert got is not None
    assert list(got["model"]) == ["fm_chronos_daily_ragged_ft"]


def test_hidden_models_leave_the_comparison_rows() -> None:
    rows = pd.DataFrame(
        {
            "model": ["fm_chronos_weekly_ragged", "renewal", "teamA-weekly"],
            "source": ["real", "real", "hub"],
        }
    )
    assert list(report._site_rows(rows)["model"]) == [
        "renewal",
        "teamA-weekly",
    ]


def test_hidden_models_leave_the_store_note(tmp_path: Path) -> None:
    import json

    from bdbvhubrenewal.reporting.store_note import write_store_note

    def row(model: str) -> dict:
        return {
            "model": model,
            "origin": "2026-11-01",
            "key": "k",
            "used": None,
            "commit": None,
            "current": False,
            "failed": False,
        }

    (tmp_path / "fit-store.json").write_text(
        json.dumps([row("fm_chronos_weekly_ragged"), row("renewal")])
    )
    write_store_note(tmp_path, tmp_path / "gen")
    text = (tmp_path / "gen" / "fit-store.md").read_text()
    assert "`renewal`" in text
    assert "fm_chronos_weekly_ragged" not in text


def test_hidden_models_leave_the_backtest_explorer() -> None:
    forecasts = pd.DataFrame(
        {
            "model": ["fm_chronos_daily_complete", "randomwalknowcast"] * 2,
            "kind": ["real", "real", "simulated", "simulated"],
        }
    )
    got = report._real_forecasts(forecasts, report._page_models())
    assert list(got["model"]) == ["randomwalknowcast"]


def test_hidden_models_leave_the_scenario_explorer(tmp_path: Path) -> None:
    import json

    from reporting.test_report import _scenario_results

    _scenario_results(tmp_path)
    path = tmp_path / "scenario-forecasts.csv.gz"
    samples = pd.read_csv(path)
    hidden = samples.loc[samples["model"] == "randomwalk"].assign(
        model="fm_chronos_daily_complete"
    )
    pd.concat([samples, hidden]).to_csv(path, index=False)
    report._scenario_section(
        tmp_path, tmp_path, ["randomwalk", "randomwalknowcast"], {}
    )
    for key in report._SCENARIOS:
        explorer = json.loads(
            (tmp_path / f"explorer-scenario-{key}.json").read_text()
        )
        assert [m["name"] for m in explorer["models"]] == [
            "randomwalk",
            "randomwalknowcast",
        ]
