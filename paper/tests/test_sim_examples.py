"""Tests of the example simulations figure (`paperfigs.sim_examples`)."""

from __future__ import annotations

import pandas as pd


def _counts() -> pd.DataFrame:
    """Two outbreaks of 20 days, each reported over the next three days."""
    rows = []
    for outbreak in ("1-0001-1", "1-0002-1"):
        for day in range(20):
            ref = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
            for lag in range(3):
                rows.append(
                    {
                        "outbreak": outbreak,
                        "timestep": "day",
                        "location": "P01",
                        "reference_date": ref.date().isoformat(),
                        "report_date": (ref + pd.Timedelta(days=lag))
                        .date()
                        .isoformat(),
                        "count": 1,
                    }
                )
    return pd.DataFrame(rows)


def test_releases_count_only_what_was_reported_by_then() -> None:
    from paperfigs.sim_examples import as_of

    counts = _counts().query("outbreak == '1-0001-1'")
    final = as_of(counts, None)
    assert (final == 3).all()
    early = as_of(counts, pd.Timestamp("2026-01-10"))
    # The release date's own onsets have one report of three by then.
    assert early.loc[pd.Timestamp("2026-01-10")] == 1
    assert early.loc[pd.Timestamp("2026-01-08")] == 3
    assert early.index.max() == pd.Timestamp("2026-01-10")


def test_the_figure_has_a_panel_per_example_with_its_releases() -> None:
    from paperfigs.sim_examples import examples_figure

    settings = pd.DataFrame(
        {
            "outbreak": ["1-0001", "1-0002"],
            "trajectory": ["growth", "multi_wave"],
            "delay_change": ["static", "bunching"],
            "observation": ["negbin1d", "poisson"],
            "strata": ["single", "places"],
        }
    )
    picks = [("1-0001", "Growth"), ("1-0002", "Multi-wave")]
    fig = examples_figure(_counts(), settings, picks)
    drawn = [ax for ax in fig.axes if ax.has_data()]
    assert len(drawn) == 2
    assert drawn[1].get_title().startswith("Multi-wave")
    # The final counts and three releases.
    assert len(drawn[0].get_lines()) == 4
