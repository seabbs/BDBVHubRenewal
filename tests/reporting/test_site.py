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
    assert plots._metric_label("relative_wis") == (
        "WIS relative to randomwalknowcast"
    )
    assert plots._metric_label("wis") == "Mean WIS"


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
