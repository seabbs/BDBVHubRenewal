from __future__ import annotations

import pandas as pd

from bdbvhubrenewal.reporting.nowcast_plots import (
    plot_place_summary,
    plot_relative_overall,
    plot_truth_changes,
)


def test_plot_relative_overall_skips_models_missing_from_a_pool() -> None:
    table = pd.DataFrame(
        {
            "pool": ["national", "national", "zones"],
            "model": ["randomwalk", "renewal", "randomwalk"],
            "rel_wis": [1.0, 0.8, 1.0],
            "rel_log_wis": [1.0, 0.9, 1.0],
            "coverage_50": [0.5, 0.4, 0.45],
            "coverage_90": [0.9, 0.7, 0.85],
            "bias": [0.1, -0.2, 0.0],
        }
    )
    fig = plot_relative_overall(table, ["randomwalk", "renewal"])
    shown = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_ylabel() for ax in shown] == [
        "Relative WIS to randomwalk",
        "Relative WIS on the log scale to randomwalk",
        "50% coverage",
        "90% coverage",
        "Bias",
    ]
    wis, cover = shown[0], shown[3]
    # Three bars, as renewal has no zones.
    assert len(wis.patches) == 3
    assert [t.get_text() for t in wis.get_xticklabels()] == [
        "National",
        "Zones",
    ]
    assert cover.get_ylim() == (0, 1)
    assert shown[4].get_ylim() == (-1, 1)


def test_plot_truth_changes_has_a_pair_of_panels_per_place_with_cases() -> None:
    weeks = ["2026-10-10", "2026-10-17", "2026-10-24"]
    rows = [
        {
            "kind": "real",
            "vintage": vintage,
            "location": loc,
            "target_end_date": week,
            "total": total,
        }
        for vintage, extra in (("2026-10-24", 0), ("2026-11-01", 5))
        for loc, base in (("CD", 10), ("CD-SK", 0))
        for week in weeks
        for total in [base + extra if base else 0]
    ]
    fig = plot_truth_changes(pd.DataFrame(rows), {"CD": "DRC"})
    shown = [ax for ax in fig.axes if ax.get_visible()]
    # CD-SK has no cases, so only DRC has its counts and changes.
    assert len(shown) == 2
    assert shown[0].get_title() == "DRC"
    assert len(shown[0].get_lines()) == 2


def test_plot_place_summary_has_a_row_per_place_by_scale() -> None:
    table = pd.DataFrame(
        {
            "pool": ["provinces", "provinces", "zones", "national"],
            "location": ["CD-NK", "CD-IT", "CD-IT-01", "CD"],
            "model": ["a", "a", "a", "a"],
            "rel_wis": [0.8, 1.2, 0.9, 1.0],
        }
    )
    fig = plot_place_summary(
        table, ["a"], {"a": ("#000000", "#ffffff")}, {"CD-IT": "Ituri"}
    )
    provinces, zones = fig.axes
    assert provinces.get_title() == "Provinces"
    labels = [t.get_text() for t in provinces.get_yticklabels()]
    assert labels == ["CD-NK", "Ituri"]
    assert zones.get_title() == "Health zones"


def test_plot_place_summary_is_on_a_log_axis_without_zeros() -> None:
    table = pd.DataFrame(
        {
            "pool": ["provinces"] * 3,
            "location": ["CD-NK", "CD-IT", "CD-NU"],
            "model": ["a"] * 3,
            "rel_wis": [1.3, 0.0, 27.0],
        }
    )
    fig = plot_place_summary(table, ["a"], {"a": ("#000000", "#ffffff")}, {})
    (ax,) = fig.axes
    assert ax.get_xscale() == "log"
    (dots,) = ax.get_lines()[:1]
    assert list(dots.get_xdata()) == [1.3, 27.0]
